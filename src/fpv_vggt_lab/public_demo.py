from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np
import typer


PUBLIC_SCHEMA_VERSION = "1.1.0"
PUBLIC_CATALOG_SCHEMA_VERSION = "wow-public-videos-v1"
FORBIDDEN_PUBLIC_SUFFIXES = {
    ".avi",
    ".bin",
    ".gif",
    ".glb",
    ".jpeg",
    ".jpg",
    ".mkv",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".ply",
    ".png",
    ".webm",
    ".webp",
}
TEXT_SUFFIXES = {".css", ".html", ".js", ".json", ".md", ".txt"}
ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?i)(?:\b[A-Z]:[\\/]|/(?:workspace|tmp|home|users|mnt/[a-z])/)"
)
EXTERNAL_RUNTIME_PATTERN = re.compile(r"(?i)(?:https?://|//cdn\.|@import\s+url\s*\()")
SECRET_PATTERN = re.compile(r"(?i)(?:sk-[A-Za-z0-9]{12,}|bearer\s+[A-Za-z0-9._-]{12,})")
SLUG_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
CATALOG_SLUG_PATTERN = re.compile(r"[a-z0-9]+(?:[_-][a-z0-9]+)*")
SOURCE_RECORD_URL_PATTERN = re.compile(
    r"https://www\.itamarweiss\.com/fpv/video/"
    r"(?P<slug>[a-z0-9]+(?:[_-][a-z0-9]+)*)/"
)
SAFE_RELATIVE_SCENE_URL_PATTERN = re.compile(
    r"scenes/[a-z0-9]+(?:[_-][a-z0-9]+)*/"
)
CATALOG_ANNOTATION_STATES = {
    "manual_ground_truth",
    "auto_generated",
    "unreviewed",
}
CATALOG_EDIT_TYPES = {
    "banner_start",
    "flight_start",
    "new_flight_start",
    "pause_start",
    "replay_start",
    "other",
    "end",
    "video_end",
    "uncertain",
}
BLOCKED_LOCAL_HOSTNAMES = {"localhost", "localdomain", "home.arpa"}
BLOCKED_LOCAL_HOST_SUFFIXES = (
    ".localhost",
    ".localdomain",
    ".local",
    ".internal",
    ".home.arpa",
    ".lan",
    ".home",
    ".corp",
    ".invalid",
    ".test",
    ".example",
)
FORBIDDEN_CATALOG_KEYS = {
    "frame",
    "frames",
    "frame_url",
    "frame_urls",
    "image_path",
    "image_url",
    "local_path",
    "media",
    "media_path",
    "media_url",
    "raw_path",
    "source_path",
    "thumbnail",
    "thumbnail_path",
    "thumbnail_url",
    "video_path",
    "video_url",
}
POINT_POSITION_ASSET = "vggt_omega_points.f32.bin"
POINT_COLOR_ASSET = "vggt_omega_colors.rgb8.bin"


class PublicDemoError(RuntimeError):
    """Raised when a private scene cannot be reduced to the public contract."""


@dataclass(frozen=True, slots=True)
class PublicDemoConfig:
    scene_root: Path
    archive_root: Path
    output_root: Path
    slug: str
    public_title: str
    catalog_path: Path | None = None
    path_sample_count: int = 96
    max_density_cells: int = 560
    density_grid: tuple[int, int, int] = (16, 16, 10)
    generated_at: str | None = None
    publish_real_point_cloud: bool = False
    point_cloud_authorization: str | None = None
    point_cloud_attribution: str | None = None

    def __post_init__(self) -> None:
        if not SLUG_PATTERN.fullmatch(self.slug):
            raise ValueError("slug must contain lowercase letters, digits, and single hyphens")
        if not self.public_title.strip():
            raise ValueError("public_title must not be empty")
        if self.path_sample_count < 8 or self.path_sample_count > 160:
            raise ValueError("path_sample_count must be between 8 and 160")
        if self.max_density_cells < 24 or self.max_density_cells > 900:
            raise ValueError("max_density_cells must be between 24 and 900")
        if len(self.density_grid) != 3 or any(value < 4 or value > 32 for value in self.density_grid):
            raise ValueError("density_grid must contain three values between 4 and 32")
        if self.publish_real_point_cloud:
            if not (self.point_cloud_authorization or "").strip():
                raise ValueError("point-cloud publication requires authorization provenance")
            if not (self.point_cloud_attribution or "").strip():
                raise ValueError("point-cloud publication requires attribution")


@dataclass(frozen=True, slots=True)
class PublicDemoBuildResult:
    status: str
    output_root: Path
    scene_entry: Path
    manifest: Path
    file_count: int
    total_bytes: int
    audit: Mapping[str, Any]


def build_public_demo(config: PublicDemoConfig) -> PublicDemoBuildResult:
    """Build an allowlisted public demonstration from private derived artifacts."""
    final_output_root = config.output_root.resolve()
    final_output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output_root.name or 'public-demo'}.stage-",
            dir=final_output_root.parent,
        )
    )
    try:
        staged_result, staged_relative_paths, managed_relative_paths = (
            _build_public_demo_stage(
                config,
                staging_root=staging_root,
                final_output_root=final_output_root,
            )
        )
        scene_relative = staged_result.scene_entry.relative_to(staging_root)
        manifest_relative = staged_result.manifest.relative_to(staging_root)
        _publish_staged_public_demo(
            staging_root=staging_root,
            output_root=final_output_root,
            staged_relative_paths=staged_relative_paths,
            managed_relative_paths=managed_relative_paths,
            slug=config.slug,
        )
        return PublicDemoBuildResult(
            status=staged_result.status,
            output_root=final_output_root,
            scene_entry=final_output_root / scene_relative,
            manifest=final_output_root / manifest_relative,
            file_count=staged_result.file_count,
            total_bytes=staged_result.total_bytes,
            audit=staged_result.audit,
        )
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _build_public_demo_stage(
    config: PublicDemoConfig,
    *,
    staging_root: Path,
    final_output_root: Path,
) -> tuple[PublicDemoBuildResult, tuple[str, ...], set[str]]:
    scene_root = config.scene_root.resolve()
    archive_root = config.archive_root.resolve()
    output_root = staging_root.resolve()
    viewer = scene_root / "viewer"
    diagnostics = scene_root / "diagnostics"
    visualization = scene_root / "visualization"

    scene_meta = _read_json(viewer / "scene_meta.json")
    calibration_state = str(scene_meta.get("calibration", {}).get("state", ""))
    display_units = str(scene_meta.get("display_units", ""))
    if calibration_state != "relative_only" or display_units != "relative units":
        raise PublicDemoError("public demo requires a relative_only scene in relative units")
    if scene_meta.get("pose_semantics") != "camera_pose_proxy":
        raise PublicDemoError("public demo requires camera_pose_proxy semantics")
    if scene_meta.get("body_attitude") != "unavailable":
        raise PublicDemoError("body attitude must remain unavailable")

    camera_path = _read_json(viewer / "camera_path.json")
    animation = _read_json(visualization / "pose_6dof_animation.json")
    profiles = _read_json(viewer / "trajectory_profiles.json")
    method_comparison = _read_json(viewer / "methods" / "method_comparison.json")
    match_graph = _read_json(diagnostics / "match_graph.json")
    failure_taxonomy = _read_json(diagnostics / "colmap_failure_taxonomy.json")

    if camera_path.get("scale_status") != "relative_only":
        raise PublicDemoError("camera paths must be relative_only")
    if animation.get("scale_status") != "relative_only":
        raise PublicDemoError("6DoF animation must be relative_only")

    display_transform = _display_transform(_raw_camera_path(camera_path))

    scene_dir = output_root / "scenes" / config.slug
    assets_dir = output_root / "assets"
    scene_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    generated_files: list[Path] = []
    authorized_binary_paths: tuple[str, ...] = ()
    point_cloud: dict[str, Any] | None = None
    if config.publish_real_point_cloud:
        point_cloud, point_files = _publish_authorized_point_cloud(
            viewer=viewer,
            scene_dir=scene_dir,
            scene_meta=scene_meta,
            display_transform=display_transform,
            authorization=config.point_cloud_authorization or "",
            attribution=config.point_cloud_attribution or "",
        )
        generated_files.extend(point_files)
        authorized_binary_paths = tuple(
            path.relative_to(output_root).as_posix() for path in point_files
        )

    generated_at = config.generated_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    catalog, selected_record = _build_public_catalog(
        config,
        scene_meta=scene_meta,
        generated_at=generated_at,
    )
    annotation = _public_scene_annotation(
        scene_meta,
        default_kind=str(selected_record.get("annotation_state", "unreviewed")),
    )
    edit_segments = _public_scene_edit_segments(scene_meta)
    methods = _public_methods(method_comparison)
    payload: dict[str, Any] = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "slug": config.slug,
        "title": config.public_title,
        "eyebrow": "Offline visual geometry / public research abstraction",
        "generated_at": generated_at,
        "scale_status": "relative_only",
        "display_units": "normalized relative units",
        "pose_semantics": "camera_pose_proxy",
        "body_attitude": "unavailable",
        "sample_count": config.path_sample_count,
        "catalog": {
            "date": selected_record["date"],
            "town": selected_record["town"],
            "annotation_state": selected_record["annotation_state"],
        },
        "annotation": annotation,
        "edit_segments": edit_segments,
        "quality": _public_quality(scene_meta.get("quality", {})),
        "geometry_density": _coarse_density(
            viewer / "points_preview.bin",
            grid=config.density_grid,
            max_cells=config.max_density_cells,
        ),
        "paths": _public_paths(
            camera_path,
            config.path_sample_count,
            display_transform=display_transform,
        ),
        "orientations": _public_orientations(animation, config.path_sample_count),
        "trajectory_profiles": _public_profiles(profiles, config.path_sample_count),
        "methods": methods,
        "match_connectivity": _public_match_connectivity(match_graph),
        "depth_summaries": {
            "r3": _summarize_r3(archive_root / "other_models" / "r3"),
            "lingbot_map": _summarize_lingbot(
                archive_root / "other_models" / "lingbot_map"
            ),
        },
        "failure_case": _public_failure_case(failure_taxonomy, methods),
        "provenance": {
            "source_type": "completed historical recorded clip",
            "catalog_date": _safe_catalog_field(scene_meta.get("source", {}).get("catalog_date")),
            "catalog_town": _safe_catalog_field(scene_meta.get("source", {}).get("town")),
            "interval_policy": "accepted manual interval",
            "public_derivation": (
                "authorized derived VGGT Omega point/color browser sample, coarse density "
                "aggregation, and normalized path resampling"
                if config.publish_real_point_cloud
                else "coarse density aggregation and normalized resampling"
            ),
            "coordinate_frame": "anonymous relative display frame",
            "original_media": "withheld",
            "raw_geometry": (
                "authorized derived browser sample published; full backend geometry withheld"
                if config.publish_real_point_cloud
                else "withheld"
            ),
            "scale_status": "relative_only",
        },
        "publication_boundary": {
            "real_point_sample_published": config.publish_real_point_cloud,
            "original_video_published": False,
            "source_frames_published": False,
            "recognizable_reprojection_published": False,
            "raw_npz_or_ply_published": False,
            "full_backend_arrays_published": False,
            "absolute_source_paths_published": False,
            "notice": (
                "Original media is not redistributed. One authorized derived VGGT Omega "
                "point/color browser sample is published in relative-only coordinates; "
                "source frames, reprojections, raw NPZ/PLY, and full backend arrays remain "
                "withheld."
                if config.publish_real_point_cloud
                else (
                    "Original media is not redistributed. This public scene contains only "
                    "coarse normalized aggregates and numeric summaries."
                )
            ),
        },
    }
    if point_cloud is not None:
        payload["point_cloud"] = point_cloud

    if point_cloud is not None:
        displayed_points = f'{int(point_cloud["point_count"]):,}'
        source_points = f'{int(point_cloud["source_point_count"]):,}'
        presentation_copy = {
            "{{GALLERY_META_DESCRIPTION}}": (
                "Offline visual-geometry research with an authorized colored VGGT Omega "
                "point sample in scale-free relative coordinates."
            ),
            "{{HERO_POINT_CLOUD_LABEL}}": (
                "Authorized colored VGGT Omega point sample with a relative camera-path overlay"
            ),
            "{{HERO_GEOMETRY_COPY}}": (
                "A completed historical clip reduced to relative camera motion, an authorized "
                "colored VGGT Omega point sample, model summaries, and visible failure states."
            ),
            "{{HERO_GEOMETRY_READOUT}}": f"{displayed_points} points · relative_only",
            "{{PROOF_TITLE}}": "The published sample is the artifact.",
            "{{PROOF_COPY}}": (
                f"This authorized colored VGGT Omega point sample publishes {displayed_points} "
                "derived points in relative-only coordinates. Video, source frames, recognizable "
                "reprojections, raw NPZ/PLY, full arrays, and machine paths remain withheld."
            ),
            "{{SCENE_STATUS}}": "authorized point sample loading",
            "{{POINT_CLOUD_CANVAS_LABEL}}": (
                "Authorized colored VGGT Omega relative point sample"
            ),
            "{{POINT_CLOUD_READOUT}}": (
                f"{displayed_points} displayed / {source_points} source points · relative_only"
            ),
            "{{MEDIA_NOTICE}}": (
                "original media is not redistributed. An authorized colored VGGT Omega point "
                "sample is published because the scene contract records publication approval "
                "and attribution. Video, source frames, recognizable reprojections, raw NPZ/PLY, "
                "full arrays, and machine paths remain withheld."
            ),
        }
    else:
        density_cells = f'{len(payload["geometry_density"]["cells"]):,}'
        presentation_copy = {
            "{{GALLERY_META_DESCRIPTION}}": (
                "Offline visual-geometry research presented as a privacy-reduced, scale-free "
                "density fallback."
            ),
            "{{HERO_POINT_CLOUD_LABEL}}": (
                "Point sample unavailable; coarse relative density fallback active"
            ),
            "{{HERO_GEOMETRY_COPY}}": (
                "A completed historical clip reduced to relative camera motion, a coarse density "
                "fallback, model summaries, and visible failure states."
            ),
            "{{HERO_GEOMETRY_READOUT}}": f"{density_cells} voxels · density fallback",
            "{{PROOF_TITLE}}": "The density fallback is the artifact.",
            "{{PROOF_COPY}}": (
                "This build publishes coarse normalized density and numeric summaries, not a "
                "colored point sample. Video, source frames, recognizable reprojections, raw "
                "NPZ/PLY, full arrays, and machine paths remain withheld."
            ),
            "{{SCENE_STATUS}}": "density fallback",
            "{{POINT_CLOUD_CANVAS_LABEL}}": (
                "Point sample unavailable; coarse relative density fallback active"
            ),
            "{{POINT_CLOUD_READOUT}}": (
                f"{density_cells} density voxels · relative_only fallback"
            ),
            "{{MEDIA_NOTICE}}": (
                "original media is not redistributed. No colored point sample is declared by "
                "this scene contract, so the viewer uses the coarse density fallback. Video, "
                "source frames, recognizable reprojections, raw NPZ/PLY, full arrays, and machine "
                "paths remain withheld."
            ),
        }

    asset_source = Path(__file__).with_name("public_demo_assets")
    for name in (
        "site.css",
        "point-cloud-webgl.js",
        "gallery.js",
        "catalog.js",
        "scene.js",
    ):
        source = asset_source / name
        if not source.is_file():
            raise PublicDemoError(f"missing public demo asset: {name}")
        destination = assets_dir / name
        shutil.copyfile(source, destination)
        generated_files.append(destination)

    safe_title = html.escape(config.public_title, quote=True)
    escaped_presentation_copy = {
        key: html.escape(value, quote=True) for key, value in presentation_copy.items()
    }
    replacements = {
        "{{PUBLIC_TITLE}}": safe_title,
        "{{SCENE_URL}}": f"scenes/{config.slug}/",
        "{{SCENE_DATA_URL}}": f"scenes/{config.slug}/scene.json",
        "{{GENERATED_AT}}": html.escape(generated_at, quote=True),
        **escaped_presentation_copy,
    }
    gallery_html = _render_template(asset_source / "gallery.html", replacements)
    scene_html = _render_template(
        asset_source / "scene.html",
        {"{{PUBLIC_TITLE}}": safe_title, **escaped_presentation_copy},
    )

    index_path = output_root / "index.html"
    scene_entry = scene_dir / "index.html"
    scene_json = scene_dir / "scene.json"
    catalog_json = output_root / "catalog.json"
    nojekyll = output_root / ".nojekyll"
    index_path.write_text(gallery_html, encoding="utf-8")
    scene_entry.write_text(scene_html, encoding="utf-8")
    scene_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    catalog_json.write_text(
        json.dumps(catalog, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    nojekyll.write_text("", encoding="utf-8")
    generated_files.extend(
        (index_path, scene_entry, scene_json, catalog_json, nojekyll)
    )

    managed_relative_paths = _managed_public_paths(
        (path.relative_to(output_root).as_posix() for path in generated_files),
        config.slug,
    )
    audit = _audit_effective_public_tree(
        staging_root=output_root,
        output_root=final_output_root,
        managed_relative_paths=managed_relative_paths,
        authorized_binary_paths=authorized_binary_paths,
    )
    if audit["status"] != "passed":
        reasons = "; ".join(item["reason"] for item in audit["findings"])
        raise PublicDemoError(f"public output audit failed: {reasons}")

    manifest_path = output_root / "build-manifest.json"
    manifest_payload = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_slug": config.slug,
        "scale_status": "relative_only",
        "publication_boundary": payload["publication_boundary"],
        "audit": audit,
        "files": [_file_record(path, output_root) for path in sorted(generated_files)],
    }
    if point_cloud is not None:
        manifest_payload["point_cloud"] = point_cloud
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    generated_files.append(manifest_path)

    final_audit = _audit_effective_public_tree(
        staging_root=output_root,
        output_root=final_output_root,
        managed_relative_paths=managed_relative_paths,
        authorized_binary_paths=authorized_binary_paths,
    )
    if final_audit["status"] != "passed":
        raise PublicDemoError("build manifest failed the final public output audit")
    staged_relative_paths = tuple(
        path.relative_to(output_root).as_posix() for path in generated_files
    )
    return (
        PublicDemoBuildResult(
            status="built",
            output_root=output_root,
            scene_entry=scene_entry,
            manifest=manifest_path,
            file_count=len(generated_files),
            total_bytes=sum(path.stat().st_size for path in generated_files),
            audit=final_audit,
        ),
        staged_relative_paths,
        managed_relative_paths,
    )


def audit_public_demo(
    root: Path,
    paths: Iterable[Path] | None = None,
    authorized_binary_paths: Iterable[str] = (),
) -> dict[str, Any]:
    """Fail closed on media, raw arrays, external runtimes, paths, and secrets."""
    root = root.resolve()
    candidates = list(paths) if paths is not None else _public_tree_files(root)
    binary_allowlist = set(authorized_binary_paths)
    findings: list[dict[str, str]] = []
    checked = 0
    for candidate in candidates:
        path = candidate.resolve()
        if not path.is_file():
            continue
        checked += 1
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            findings.append({"path": path.name, "reason": "file outside public root"})
            continue
        suffix = path.suffix.lower()
        if suffix == ".bin" and relative in binary_allowlist:
            continue
        if suffix in FORBIDDEN_PUBLIC_SUFFIXES:
            findings.append({"path": relative, "reason": "forbidden public file type"})
            continue
        if suffix not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if ABSOLUTE_PATH_PATTERN.search(text):
            findings.append({"path": relative, "reason": "absolute machine path"})
        external_scan_text = text
        if relative == "catalog.json":
            try:
                for allowed_url in _validated_catalog_external_urls(text):
                    external_scan_text = external_scan_text.replace(allowed_url, "")
            except PublicDemoError as exc:
                findings.append(
                    {"path": relative, "reason": f"invalid public catalog: {exc}"}
                )
        if EXTERNAL_RUNTIME_PATTERN.search(external_scan_text):
            findings.append({"path": relative, "reason": "external runtime dependency"})
        if SECRET_PATTERN.search(text):
            findings.append({"path": relative, "reason": "secret-like value"})
    return {
        "status": "failed" if findings else "passed",
        "checked_file_count": checked,
        "finding_count": len(findings),
        "findings": findings,
        "scope": "public demo files",
    }


def _public_tree_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        files.append(path)
    return files


def _managed_public_paths(staged_relative_paths: Iterable[str], slug: str) -> set[str]:
    paths = set(staged_relative_paths)
    paths.update(
        {
            "build-manifest.json",
            f"scenes/{slug}/geometry/{POINT_POSITION_ASSET}",
            f"scenes/{slug}/geometry/{POINT_COLOR_ASSET}",
        }
    )
    return paths


def _audit_effective_public_tree(
    *,
    staging_root: Path,
    output_root: Path,
    managed_relative_paths: set[str],
    authorized_binary_paths: Iterable[str],
) -> dict[str, Any]:
    staged_report = audit_public_demo(
        staging_root,
        _public_tree_files(staging_root),
        authorized_binary_paths=authorized_binary_paths,
    )
    existing_unmanaged = [
        path
        for path in _public_tree_files(output_root)
        if path.relative_to(output_root).as_posix() not in managed_relative_paths
    ]
    existing_report = audit_public_demo(output_root, existing_unmanaged)
    findings = [*staged_report["findings"], *existing_report["findings"]]
    return {
        "status": "failed" if findings else "passed",
        "checked_file_count": (
            staged_report["checked_file_count"] + existing_report["checked_file_count"]
        ),
        "finding_count": len(findings),
        "findings": findings,
        "scope": "effective public demo tree",
    }


def _publish_staged_public_demo(
    *,
    staging_root: Path,
    output_root: Path,
    staged_relative_paths: Iterable[str],
    managed_relative_paths: Iterable[str],
    slug: str,
) -> None:
    backup_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name or 'public-demo'}.backup-",
            dir=output_root.parent,
        )
    )
    backed_up: list[tuple[str, Path]] = []
    published: list[str] = []
    created_directories: set[Path] = set()
    backup_created_directories: set[Path] = set()
    retain_backup = False
    try:
        for relative in sorted(set(managed_relative_paths)):
            destination = _public_target(output_root, relative)
            if destination.is_file() or destination.is_symlink():
                backup = _public_target(backup_root, relative)
                _ensure_public_directory(
                    backup.parent,
                    output_root=backup_root,
                    created_directories=backup_created_directories,
                )
                os.replace(destination, backup)
                backed_up.append((relative, backup))
            elif destination.exists():
                raise PublicDemoError(f"managed public path is not a file: {relative}")

        for relative in sorted(set(staged_relative_paths)):
            source = _public_target(staging_root, relative)
            if not source.is_file():
                raise PublicDemoError(f"staged public file is missing: {relative}")
            destination = _public_target(output_root, relative)
            _ensure_public_directory(
                destination.parent,
                output_root=output_root,
                created_directories=created_directories,
            )
            os.replace(source, destination)
            published.append(relative)

        geometry_dir = _public_target(output_root, f"scenes/{slug}/geometry")
        if geometry_dir.is_dir() and not any(geometry_dir.iterdir()):
            geometry_dir.rmdir()
    except Exception as exc:
        rollback_errors: list[str] = []
        for relative in reversed(published):
            try:
                destination = _public_target(output_root, relative)
                if destination.is_file() or destination.is_symlink():
                    destination.unlink()
            except (OSError, PublicDemoError) as rollback_exc:
                rollback_errors.append(f"remove {relative}: {rollback_exc}")
        for relative, backup in reversed(backed_up):
            try:
                destination = _public_target(output_root, relative)
                _ensure_public_directory(
                    destination.parent,
                    output_root=output_root,
                    created_directories=created_directories,
                )
                os.replace(backup, destination)
            except (OSError, PublicDemoError) as rollback_exc:
                rollback_errors.append(f"restore {relative}: {rollback_exc}")
        for directory in sorted(
            created_directories,
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            try:
                _assert_safe_public_path(output_root, directory)
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()
            except (OSError, PublicDemoError) as rollback_exc:
                rollback_errors.append(f"remove directory {directory.name}: {rollback_exc}")
        if rollback_errors:
            retain_backup = True
            detail = "; ".join(rollback_errors)
            raise PublicDemoError(
                f"transactional public publish failed and rollback was incomplete: {detail}"
            ) from exc
        raise PublicDemoError("transactional public publish failed") from exc
    finally:
        if not retain_backup:
            shutil.rmtree(backup_root, ignore_errors=True)


def _public_target(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise PublicDemoError(f"invalid managed public path: {relative}")
    candidate = resolved_root / relative_path
    _assert_safe_public_path(resolved_root, candidate)
    return candidate


def _assert_safe_public_path(root: Path, candidate: Path) -> None:
    resolved_root = root.resolve()
    try:
        relative = candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise PublicDemoError("managed public path escapes output root") from exc

    current = resolved_root
    for part in relative.parts:
        current = current / part
        if _is_unsafe_public_link(current):
            raise PublicDemoError(f"unsafe public path component: {part}")
        if current.exists():
            try:
                resolved_component = current.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise PublicDemoError(
                    f"unable to resolve managed public path component: {part}"
                ) from exc
            if not resolved_component.is_relative_to(resolved_root):
                raise PublicDemoError(f"managed public path escapes output root: {part}")

    values = (candidate,) if candidate == resolved_root else (candidate.parent, candidate)
    for value in values:
        try:
            resolved_value = value.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise PublicDemoError("unable to resolve managed public path") from exc
        if not resolved_value.is_relative_to(resolved_root):
            raise PublicDemoError("managed public path escapes output root")


def _is_unsafe_public_link(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction()) if callable(is_junction) else False
    except OSError as exc:
        raise PublicDemoError(f"unable to inspect public path component: {path.name}") from exc


def _ensure_public_directory(
    directory: Path,
    *,
    output_root: Path,
    created_directories: set[Path],
) -> None:
    _assert_safe_public_path(output_root, directory)
    missing: list[Path] = []
    current = directory
    while not current.exists():
        if current == output_root.parent:
            raise PublicDemoError("managed public path escapes output root")
        missing.append(current)
        current = current.parent
    directory.mkdir(parents=True, exist_ok=True)
    _assert_safe_public_path(output_root, directory)
    created_directories.update(missing)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PublicDemoError(f"required derived artifact is missing: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PublicDemoError(f"expected a JSON object: {path.name}")
    return value


def _build_public_catalog(
    config: PublicDemoConfig,
    *,
    scene_meta: Mapping[str, Any],
    generated_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.catalog_path is None:
        catalog = _synthesized_public_catalog(
            config,
            scene_meta=scene_meta,
            generated_at=generated_at,
        )
        return catalog, catalog["records"][0]

    try:
        source_catalog = _read_json(config.catalog_path)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise PublicDemoError("catalog JSON could not be decoded") from exc
    catalog = _sanitize_public_catalog(source_catalog, generated_at=generated_at)

    source = scene_meta.get("source", {})
    if not isinstance(source, dict) or source.get("video_file") is None:
        raise PublicDemoError(
            "scene metadata must identify source.video_file for catalog overlay"
        )
    selected_video_file = _catalog_video_file(source.get("video_file"))
    selected: dict[str, Any] | None = None
    for record in catalog["records"]:
        if record["video_file"] != selected_video_file:
            continue
        record["research"] = {
            **record["research"],
            "public_scene_available": True,
            "public_scene_url": f"scenes/{config.slug}/",
            "calibration_state": "relative_only",
        }
        selected = record
        break
    if selected is None:
        raise PublicDemoError("scene source video is not present in catalog records")

    catalog["records"].sort(
        key=lambda record: (record["date"], record["slug"]),
        reverse=True,
    )
    catalog["counts"] = _public_catalog_counts(catalog["records"])
    return catalog, selected


def _sanitize_public_catalog(
    value: Mapping[str, Any],
    *,
    generated_at: str,
) -> dict[str, Any]:
    _assert_catalog_input_safe(value)
    allowed_top_level = {
        "schema_version",
        "generated_at",
        "source",
        "provenance",
        "publication_boundary",
        "counts",
        "records",
    }
    unknown = set(value) - allowed_top_level
    if unknown:
        raise PublicDemoError(f"unsupported catalog fields: {sorted(unknown)}")
    if value.get("schema_version") != PUBLIC_CATALOG_SCHEMA_VERSION:
        raise PublicDemoError(
            f"catalog schema_version must be {PUBLIC_CATALOG_SCHEMA_VERSION}"
        )

    publication = value.get("publication_boundary")
    if not isinstance(publication, dict):
        raise PublicDemoError("catalog publication_boundary must be an object")
    allowed_boundary = {
        "third_party_media_embedded",
        "real_media_assets_published",
        "source_record_links_only",
    }
    if set(publication) - allowed_boundary:
        raise PublicDemoError("catalog publication boundary contains unsupported fields")
    required_boundary = (
        "third_party_media_embedded",
        "real_media_assets_published",
    )
    if any(publication.get(key) is not False for key in required_boundary):
        raise PublicDemoError("catalog publication boundary must withhold real media")
    if publication.get("source_record_links_only") is not True:
        raise PublicDemoError("catalog publication boundary must allow source-record links only")

    raw_source = value.get("source")
    if not isinstance(raw_source, dict):
        raise PublicDemoError("catalog source metadata must be an object")
    allowed_source = {
        "dataset",
        "metadata_license",
        "source_repo",
        "source_commit",
        "retrieved_at",
        "kind",
        "selection_policy",
    }
    if set(raw_source) - allowed_source:
        raise PublicDemoError("catalog source metadata contains unsupported fields")
    source: dict[str, str] = {}
    for key in allowed_source:
        if key not in raw_source:
            continue
        source[key] = _catalog_text(
            raw_source[key],
            field=f"source.{key}",
            maximum=240,
        )
    for required_source_field in ("kind", "selection_policy"):
        if required_source_field not in source:
            raise PublicDemoError(
                f"catalog source.{required_source_field} is required"
            )
    if not re.fullmatch(r"[a-z0-9_]+", source["kind"]):
        raise PublicDemoError("catalog source.kind must be a safe identifier")

    provenance = _sanitize_catalog_provenance(value.get("provenance"))

    raw_records = value.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise PublicDemoError("catalog records must be a non-empty list")
    records = [_sanitize_catalog_record(record) for record in raw_records]
    if len({record["slug"] for record in records}) != len(records):
        raise PublicDemoError("catalog record slugs must be unique")
    if len({record["video_file"] for record in records}) != len(records):
        raise PublicDemoError("catalog video_file values must be unique")

    raw_counts = value.get("counts")
    if not isinstance(raw_counts, dict):
        raise PublicDemoError("catalog counts must be an object")
    for key, count in raw_counts.items():
        if not isinstance(key, str) or isinstance(count, bool) or not isinstance(count, int):
            raise PublicDemoError("catalog counts must contain integer values")
        if count < 0:
            raise PublicDemoError("catalog counts must be non-negative")
    if raw_counts.get("records") != len(records):
        raise PublicDemoError("catalog record count does not match records")

    records.sort(
        key=lambda record: (record["date"], record["slug"]),
        reverse=True,
    )
    return {
        "schema_version": PUBLIC_CATALOG_SCHEMA_VERSION,
        "generated_at": _catalog_text(
            generated_at,
            field="generated_at",
            maximum=80,
        ),
        "source": source,
        "provenance": provenance,
        "publication_boundary": {
            "third_party_media_embedded": False,
            "real_media_assets_published": False,
            "source_record_links_only": True,
        },
        "counts": _public_catalog_counts(records),
        "records": records,
    }


def _sanitize_catalog_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PublicDemoError("catalog provenance must be an object")
    allowed = {
        "generated_by",
        "annotation_files_considered",
        "gallery_manifest_merged",
        "scene_routes",
    }
    if set(value) != allowed:
        raise PublicDemoError("catalog provenance fields do not match the public contract")
    generated_by = _catalog_text(
        value.get("generated_by"),
        field="provenance.generated_by",
        maximum=120,
    )
    if not re.fullmatch(r"[a-z0-9_.-]+", generated_by):
        raise PublicDemoError("catalog provenance.generated_by is invalid")
    annotation_files = _nonnegative_catalog_int(
        value.get("annotation_files_considered"),
        field="provenance.annotation_files_considered",
    )
    gallery_merged = value.get("gallery_manifest_merged")
    if not isinstance(gallery_merged, bool):
        raise PublicDemoError("provenance.gallery_manifest_merged must be boolean")
    scene_routes = _catalog_text(
        value.get("scene_routes"),
        field="provenance.scene_routes",
        maximum=40,
    )
    if scene_routes != "explicit_only":
        raise PublicDemoError("provenance.scene_routes must be explicit_only")
    return {
        "generated_by": generated_by,
        "annotation_files_considered": annotation_files,
        "gallery_manifest_merged": gallery_merged,
        "scene_routes": scene_routes,
    }


def _sanitize_catalog_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PublicDemoError("catalog records must be objects")
    allowed = {
        "video_file",
        "slug",
        "date",
        "description",
        "town",
        "source_record_url",
        "annotation_state",
        "edit_segmentation",
        "research",
    }
    unknown = set(value) - allowed
    if unknown:
        raise PublicDemoError(f"unsupported catalog record fields: {sorted(unknown)}")

    slug = _catalog_slug(value.get("slug"))
    video_file = _catalog_video_file(value.get("video_file"))
    if Path(video_file).stem != slug:
        raise PublicDemoError("catalog video_file stem must match slug")
    annotation_state = _catalog_annotation_state(value.get("annotation_state"))
    record: dict[str, Any] = {
        "video_file": video_file,
        "slug": slug,
        "date": _catalog_date(value.get("date")),
        "description": _catalog_text(
            value.get("description"),
            field="description",
            maximum=320,
        ),
        "town": _catalog_text(value.get("town"), field="town", maximum=100),
        "annotation_state": annotation_state,
        "edit_segmentation": _sanitize_catalog_edit_segmentation(
            value.get("edit_segmentation")
        ),
        "research": _sanitize_catalog_research(value.get("research")),
    }
    if value.get("source_record_url") is not None:
        record["source_record_url"] = _catalog_source_record_url(
            value.get("source_record_url"),
            slug=slug,
        )
    return record


def _sanitize_catalog_edit_segmentation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PublicDemoError("edit_segmentation must be an object")
    allowed = {
        "segment_count",
        "manual_segment_count",
        "auto_segment_count",
        "source_span_seconds",
        "summary",
        "types",
        "segments",
    }
    if set(value) - allowed:
        raise PublicDemoError("edit_segmentation contains unsupported fields")
    raw_segments = value.get("segments")
    if not isinstance(raw_segments, list):
        raise PublicDemoError("edit_segmentation.segments must be a list")
    segments: list[dict[str, Any]] = []
    last_time = -math.inf
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            raise PublicDemoError("edit segmentation entries must be objects")
        allowed_segment = {
            "time",
            "start_s",
            "end_s",
            "type",
            "label",
            "review_status",
        }
        if set(raw_segment) - allowed_segment:
            raise PublicDemoError("edit segmentation entry contains unsupported fields")
        edit_type = _catalog_edit_type(raw_segment.get("type"))
        segment: dict[str, Any] = {"type": edit_type}
        if "time" in raw_segment:
            timestamp = _finite_catalog_number(
                raw_segment.get("time"),
                field="edit segment time",
                minimum=0.0,
            )
            if timestamp < last_time:
                raise PublicDemoError("edit segment times must be ordered")
            last_time = timestamp
            segment["time"] = timestamp
        elif "start_s" in raw_segment and "end_s" in raw_segment:
            start = _finite_catalog_number(
                raw_segment.get("start_s"),
                field="edit segment start_s",
                minimum=0.0,
            )
            end = _finite_catalog_number(
                raw_segment.get("end_s"),
                field="edit segment end_s",
                minimum=start,
            )
            if start < last_time:
                raise PublicDemoError("edit segments must be ordered")
            last_time = start
            segment.update({"start_s": start, "end_s": end})
        else:
            raise PublicDemoError("edit segment must contain time or start_s/end_s")
        if raw_segment.get("label") is not None:
            segment["label"] = _catalog_text(
                raw_segment["label"],
                field="edit segment label",
                maximum=120,
            )
        if raw_segment.get("review_status") is not None:
            segment["review_status"] = _catalog_annotation_state(
                raw_segment["review_status"]
            )
        segments.append(segment)

    segment_count = value.get("segment_count")
    if (
        isinstance(segment_count, bool)
        or not isinstance(segment_count, int)
        or segment_count != len(segments)
    ):
        raise PublicDemoError("edit segment_count must match segments")
    raw_types = value.get("types")
    if not isinstance(raw_types, list):
        raise PublicDemoError("edit segmentation types must be a list")
    types = [_catalog_edit_type(item) for item in raw_types]
    observed_types = [segment["type"] for segment in segments]
    if types != observed_types:
        raise PublicDemoError("edit segmentation types must match segment order")
    result: dict[str, Any] = {
        "segment_count": len(segments),
        "types": types,
        "segments": segments,
    }
    for key in ("manual_segment_count", "auto_segment_count"):
        if key in value:
            result[key] = _nonnegative_catalog_int(
                value[key],
                field=f"edit_segmentation.{key}",
            )
    if all(key in result for key in ("manual_segment_count", "auto_segment_count")):
        if result["manual_segment_count"] + result["auto_segment_count"] != len(segments):
            raise PublicDemoError(
                "manual and auto edit counts must add up to segment_count"
            )
    if "source_span_seconds" in value:
        span = _finite_catalog_number(
            value["source_span_seconds"],
            field="edit_segmentation.source_span_seconds",
            minimum=0.0,
        )
        observed_end = max(
            (
                segment.get("time", segment.get("end_s", 0.0))
                for segment in segments
            ),
            default=0.0,
        )
        if span + 1e-6 < observed_end:
            raise PublicDemoError("edit source_span_seconds does not cover segments")
        result["source_span_seconds"] = span
    if "summary" in value:
        result["summary"] = _catalog_text(
            value["summary"],
            field="edit_segmentation.summary",
            maximum=2048,
        )
    return result


def _sanitize_catalog_research(value: Any) -> dict[str, Any]:
    if value is None:
        return {"public_scene_available": False}
    if not isinstance(value, dict):
        raise PublicDemoError("research must be an object")
    allowed = {
        "public_scene_available",
        "public_scene_url",
        "backend",
        "calibration_state",
        "status",
        "hero_score",
        "method_status",
        "method_coverage",
        "warnings",
    }
    if set(value) - allowed:
        raise PublicDemoError("research contains unsupported fields")
    available = value.get("public_scene_available")
    if not isinstance(available, bool):
        raise PublicDemoError("research.public_scene_available must be boolean")
    research: dict[str, Any] = {"public_scene_available": available}
    if value.get("public_scene_url") is not None:
        research["public_scene_url"] = _catalog_public_scene_url(
            value.get("public_scene_url")
        )
    if available and "public_scene_url" not in research:
        raise PublicDemoError("available public scenes require public_scene_url")
    for key, maximum in (("backend", 120), ("status", 80)):
        if value.get(key) is not None:
            research[key] = _catalog_text(
                value[key],
                field=f"research.{key}",
                maximum=maximum,
            )
    if value.get("calibration_state") is not None:
        calibration = _catalog_text(
            value["calibration_state"],
            field="research.calibration_state",
            maximum=40,
        )
        if calibration != "relative_only":
            raise PublicDemoError("public catalog scenes must remain relative_only")
        research["calibration_state"] = calibration
    if value.get("hero_score") is not None:
        research["hero_score"] = _finite_catalog_number(
            value["hero_score"],
            field="research.hero_score",
            minimum=0.0,
            maximum=1.0,
        )
    if value.get("method_status") is not None:
        raw_statuses = value["method_status"]
        if not isinstance(raw_statuses, dict):
            raise PublicDemoError("research.method_status must be an object")
        statuses: dict[str, str] = {}
        for key, status in raw_statuses.items():
            if not isinstance(key, str) or not re.fullmatch(r"[a-z0-9_]+", key):
                raise PublicDemoError("research method identifiers must be safe")
            statuses[key] = _catalog_text(
                status,
                field=f"research.method_status.{key}",
                maximum=40,
            )
        research["method_status"] = statuses
    if value.get("method_coverage") is not None:
        research["method_coverage"] = _sanitize_catalog_method_coverage(
            value["method_coverage"]
        )
    if value.get("warnings") is not None:
        raw_warnings = value["warnings"]
        if not isinstance(raw_warnings, list) or len(raw_warnings) > 32:
            raise PublicDemoError("research.warnings must be a bounded list")
        research["warnings"] = [
            _catalog_text(
                warning,
                field="research.warnings",
                maximum=280,
            )
            for warning in raw_warnings
        ]
    return research


def _sanitize_catalog_method_coverage(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PublicDemoError("research.method_coverage must be an object")
    if set(value) != {"bucket", "done", "total", "ratio"}:
        raise PublicDemoError("research.method_coverage fields are invalid")
    bucket = _catalog_text(
        value.get("bucket"),
        field="research.method_coverage.bucket",
        maximum=40,
    )
    if not re.fullmatch(r"[a-z0-9_]+", bucket):
        raise PublicDemoError("research.method_coverage.bucket is invalid")
    done = _nonnegative_catalog_int(
        value.get("done"),
        field="research.method_coverage.done",
    )
    total = _nonnegative_catalog_int(
        value.get("total"),
        field="research.method_coverage.total",
    )
    if done > total:
        raise PublicDemoError("research method coverage done exceeds total")
    ratio = _finite_catalog_number(
        value.get("ratio"),
        field="research.method_coverage.ratio",
        minimum=0.0,
        maximum=1.0,
    )
    expected_ratio = done / total if total else 0.0
    if not math.isclose(ratio, expected_ratio, abs_tol=1e-6):
        raise PublicDemoError("research method coverage ratio is inconsistent")
    return {"bucket": bucket, "done": done, "total": total, "ratio": ratio}


def _synthesized_public_catalog(
    config: PublicDemoConfig,
    *,
    scene_meta: Mapping[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    source = scene_meta.get("source", {})
    source = source if isinstance(source, dict) else {}
    catalog_date = _optional_catalog_date(source.get("catalog_date"))
    if catalog_date is None:
        catalog_date = _catalog_date(generated_at[:10])
    town = _optional_catalog_text(source.get("town"), maximum=100) or "Undisclosed"
    video_file_value = source.get("video_file")
    if video_file_value is None:
        video_file = f"{config.slug}.mp4"
        record_slug = config.slug
    else:
        video_file = _catalog_video_file(video_file_value)
        record_slug = Path(video_file).stem
    annotation_source = scene_meta.get("annotation", {})
    annotation_kind = (
        annotation_source.get("kind")
        if isinstance(annotation_source, dict)
        else None
    )
    annotation_state = (
        _catalog_annotation_state(annotation_kind)
        if annotation_kind is not None
        else "unreviewed"
    )
    scene_edits = _public_scene_edit_segments(scene_meta)
    edit_segmentation = {
        "segment_count": len(scene_edits),
        "types": [item["type"] for item in scene_edits],
        "segments": scene_edits,
    }
    manual_count = sum(
        item["review_status"] == "manual_ground_truth" for item in scene_edits
    )
    auto_count = sum(
        item["review_status"] == "auto_generated" for item in scene_edits
    )
    if manual_count + auto_count == len(scene_edits):
        edit_segmentation.update(
            {
                "manual_segment_count": manual_count,
                "auto_segment_count": auto_count,
            }
        )
    source_span = max((item["end_s"] for item in scene_edits), default=0.0)
    edit_segmentation["source_span_seconds"] = round(source_span, 6)
    type_summary = " → ".join(edit_segmentation["types"]) or "none"
    edit_segmentation["summary"] = (
        f"{len(scene_edits)} intervals · {type_summary} · {source_span:.1f} source s"
    )
    record: dict[str, Any] = {
        "video_file": video_file,
        "slug": record_slug,
        "date": catalog_date,
        "description": config.public_title.strip(),
        "town": town,
        "annotation_state": annotation_state,
        "edit_segmentation": edit_segmentation,
        "research": {
            "public_scene_available": True,
            "public_scene_url": f"scenes/{config.slug}/",
            "calibration_state": "relative_only",
        },
    }
    candidate_url = source.get("source_record_url")
    if candidate_url is not None:
        record["source_record_url"] = _catalog_source_record_url(
            candidate_url,
            slug=record_slug,
        )
    records = [record]
    return {
        "schema_version": PUBLIC_CATALOG_SCHEMA_VERSION,
        "generated_at": _catalog_text(
            generated_at,
            field="generated_at",
            maximum=80,
        ),
        "source": {
            "kind": "synthesized_scene_metadata",
            "selection_policy": "one metadata-only record for the selected public scene",
        },
        "provenance": {
            "generated_by": "fpv_vggt_lab.public_demo",
            "annotation_files_considered": 1 if annotation_source else 0,
            "gallery_manifest_merged": False,
            "scene_routes": "explicit_only",
        },
        "publication_boundary": {
            "third_party_media_embedded": False,
            "real_media_assets_published": False,
            "source_record_links_only": True,
        },
        "counts": _public_catalog_counts(records),
        "records": records,
    }


def _public_catalog_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "records": len(records),
        "manual_ground_truth": sum(
            record.get("annotation_state") == "manual_ground_truth"
            for record in records
        ),
        "auto_generated": sum(
            record.get("annotation_state") == "auto_generated"
            for record in records
        ),
        "unreviewed": sum(
            record.get("annotation_state") == "unreviewed" for record in records
        ),
        "public_scenes": sum(
            record.get("research", {}).get("public_scene_available") is True
            for record in records
        ),
    }


def _public_scene_annotation(
    scene_meta: Mapping[str, Any],
    *,
    default_kind: str,
) -> dict[str, Any]:
    raw = scene_meta.get("annotation", {})
    raw = raw if isinstance(raw, dict) else {}
    kind = _catalog_annotation_state(raw.get("kind", default_kind))
    source_interval: dict[str, Any] | None = None
    intervals = raw.get("intervals", [])
    if intervals is not None and not isinstance(intervals, list):
        raise PublicDemoError("scene annotation intervals must be a list")
    if intervals:
        interval = intervals[0]
        if not isinstance(interval, dict):
            raise PublicDemoError("scene annotation interval must be an object")
        start = _finite_catalog_number(
            interval.get("start_s"),
            field="annotation interval start_s",
            minimum=0.0,
        )
        end = _finite_catalog_number(
            interval.get("end_s"),
            field="annotation interval end_s",
            minimum=start,
        )
        source_interval = {
            "segment_id": _catalog_text(
                interval.get("segment_id"),
                field="annotation interval segment_id",
                maximum=80,
            ),
            "start_s": start,
            "end_s": end,
        }
    return {"kind": kind, "source_interval": source_interval}


def _public_scene_edit_segments(scene_meta: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_segments = scene_meta.get("edit_segments", [])
    if raw_segments is None:
        return []
    if not isinstance(raw_segments, list):
        raise PublicDemoError("scene edit_segments must be a list")
    segments: list[dict[str, Any]] = []
    for raw in raw_segments:
        if not isinstance(raw, dict):
            raise PublicDemoError("scene edit segments must be objects")
        start = _finite_catalog_number(
            raw.get("start_s"),
            field="scene edit start_s",
            minimum=0.0,
        )
        end = _finite_catalog_number(
            raw.get("end_s"),
            field="scene edit end_s",
            minimum=start,
        )
        segment = {
            "start_s": start,
            "end_s": end,
            "type": _catalog_edit_type(raw.get("type")),
            "label": _catalog_text(
                raw.get("label", str(raw.get("type", "edit")).replace("_", " ")),
                field="scene edit label",
                maximum=120,
            ),
            "review_status": _catalog_annotation_state(
                raw.get("review_status", "unreviewed")
            ),
        }
        segments.append(segment)
    segments.sort(key=lambda segment: (segment["start_s"], segment["end_s"]))
    return segments


def _assert_catalog_input_safe(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise PublicDemoError("catalog keys must be strings")
            normalized = key.lower()
            if (
                normalized in FORBIDDEN_CATALOG_KEYS
                or normalized.endswith("_path")
                or normalized.endswith("_paths")
            ):
                raise PublicDemoError(
                    f"forbidden catalog field: {'.'.join((*path, key))}"
                )
            _assert_catalog_input_safe(child, (*path, key))
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _assert_catalog_input_safe(child, (*path, str(index)))
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise PublicDemoError(f"non-finite catalog value: {'.'.join(path)}")
    if isinstance(value, str):
        if value.startswith(("/", "\\")) or ABSOLUTE_PATH_PATTERN.search(value):
            raise PublicDemoError(f"absolute path in catalog: {'.'.join(path)}")
        if SECRET_PATTERN.search(value):
            raise PublicDemoError(f"secret-like catalog value: {'.'.join(path)}")


def _catalog_text(value: Any, *, field: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise PublicDemoError(f"{field} must be text")
    text = value.strip()
    if not text or len(text) > maximum:
        raise PublicDemoError(f"{field} has an invalid length")
    if any(ord(character) < 32 for character in text):
        raise PublicDemoError(f"{field} contains control characters")
    if (
        text.startswith(("/", "\\"))
        or ABSOLUTE_PATH_PATTERN.search(text)
        or EXTERNAL_RUNTIME_PATTERN.search(text)
        or SECRET_PATTERN.search(text)
    ):
        raise PublicDemoError(f"{field} contains a forbidden value")
    return text


def _optional_catalog_text(value: Any, *, maximum: int) -> str | None:
    if value is None:
        return None
    return _catalog_text(value, field="catalog metadata", maximum=maximum)


def _catalog_slug(value: Any) -> str:
    if not isinstance(value, str) or not CATALOG_SLUG_PATTERN.fullmatch(value):
        raise PublicDemoError("catalog slug is invalid")
    return value


def _catalog_video_file(value: Any) -> str:
    if not isinstance(value, str) or not value.lower().endswith(".mp4"):
        raise PublicDemoError("catalog video_file must be an MP4 basename")
    if Path(value).name != value or "/" in value or "\\" in value:
        raise PublicDemoError("catalog video_file must not contain a path")
    stem = Path(value).stem
    if not CATALOG_SLUG_PATTERN.fullmatch(stem):
        raise PublicDemoError("catalog video_file stem is invalid")
    return value


def _catalog_date(value: Any) -> str:
    if not isinstance(value, str):
        raise PublicDemoError("catalog date must be YYYY-MM-DD")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise PublicDemoError("catalog date must be YYYY-MM-DD") from exc
    return parsed.strftime("%Y-%m-%d")


def _optional_catalog_date(value: Any) -> str | None:
    if value is None:
        return None
    return _catalog_date(value)


def _catalog_annotation_state(value: Any) -> str:
    if value not in CATALOG_ANNOTATION_STATES:
        raise PublicDemoError("catalog annotation_state is invalid")
    return str(value)


def _catalog_edit_type(value: Any) -> str:
    if value not in CATALOG_EDIT_TYPES:
        raise PublicDemoError("catalog edit segment type is invalid")
    return str(value)


def _finite_catalog_number(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PublicDemoError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise PublicDemoError(f"{field} must be finite")
    if minimum is not None and number < minimum:
        raise PublicDemoError(f"{field} is below its minimum")
    if maximum is not None and number > maximum:
        raise PublicDemoError(f"{field} is above its maximum")
    return round(number, 6)


def _nonnegative_catalog_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PublicDemoError(f"{field} must be a non-negative integer")
    return value


def _catalog_source_record_url(value: Any, *, slug: str) -> str:
    if not isinstance(value, str):
        raise PublicDemoError("source_record_url must be text")
    match = SOURCE_RECORD_URL_PATTERN.fullmatch(value)
    if match is None or match.group("slug") != slug:
        raise PublicDemoError("source_record_url is not an approved Itamar record URL")
    return value


def _catalog_public_scene_url(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise PublicDemoError("public_scene_url must be text")
    if (
        "\\" in value
        or "%" in value
        or "?" in value
        or "#" in value
        or value.startswith("//")
    ):
        raise PublicDemoError("public_scene_url is not canonical")
    if SAFE_RELATIVE_SCENE_URL_PATTERN.fullmatch(value):
        return value
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise PublicDemoError("public_scene_url must be relative or HTTPS")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise PublicDemoError("public_scene_url contains unsafe URL components")
    try:
        port = parsed.port
    except ValueError as exc:
        raise PublicDemoError("public_scene_url contains an invalid port") from exc
    if port not in (None, 443):
        raise PublicDemoError("public_scene_url contains an unsafe port")
    hostname = parsed.hostname.lower()
    if hostname.endswith("."):
        raise PublicDemoError("public_scene_url hostname must not end with a dot")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is None:
        legacy_ipv4 = re.fullmatch(
            r"(?:0[xX][0-9A-Fa-f]+|[0-9]+)"
            r"(?:\.(?:0[xX][0-9A-Fa-f]+|[0-9]+)){0,3}",
            hostname,
        )
        if legacy_ipv4 is not None:
            raise PublicDemoError("public_scene_url uses a legacy numeric IPv4 form")
        if not re.fullmatch(r"[a-z0-9.-]+", hostname):
            raise PublicDemoError("public_scene_url hostname is invalid")
        if (
            "." not in hostname
            or hostname in BLOCKED_LOCAL_HOSTNAMES
            or hostname.endswith(BLOCKED_LOCAL_HOST_SUFFIXES)
        ):
            raise PublicDemoError("public_scene_url hostname is not public")
    elif not address.is_global:
        raise PublicDemoError("public_scene_url address is not public")
    if not re.fullmatch(r"/[A-Za-z0-9._~/-]+/?", parsed.path):
        raise PublicDemoError("public_scene_url path is invalid")
    if any(part in {".", ".."} for part in parsed.path.split("/")):
        raise PublicDemoError("public_scene_url contains traversal")
    return value


def _validated_catalog_external_urls(text: str) -> tuple[str, ...]:
    try:
        catalog = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PublicDemoError("catalog.json is invalid JSON") from exc
    if not isinstance(catalog, dict) or catalog.get("schema_version") != PUBLIC_CATALOG_SCHEMA_VERSION:
        raise PublicDemoError("catalog.json has an invalid public schema")
    generated_at = catalog.get("generated_at")
    if not isinstance(generated_at, str):
        raise PublicDemoError("catalog.json generated_at is invalid")
    canonical = _sanitize_public_catalog(catalog, generated_at=generated_at)
    if canonical != catalog:
        raise PublicDemoError("catalog.json is not canonical public metadata")
    catalog = canonical
    records = catalog.get("records")
    if not isinstance(records, list):
        raise PublicDemoError("catalog.json records are invalid")
    allowed: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            raise PublicDemoError("catalog.json record is invalid")
        slug = _catalog_slug(record.get("slug"))
        source_url = record.get("source_record_url")
        if source_url is not None:
            allowed.append(_catalog_source_record_url(source_url, slug=slug))
        research = record.get("research", {})
        if not isinstance(research, dict):
            raise PublicDemoError("catalog.json research is invalid")
        scene_url = research.get("public_scene_url")
        if scene_url is not None:
            validated = _catalog_public_scene_url(scene_url)
            if validated.startswith("https://"):
                allowed.append(validated)
    return tuple(allowed)


def _render_template(path: Path, replacements: Mapping[str, str]) -> str:
    if not path.is_file():
        raise PublicDemoError(f"missing public demo template: {path.name}")
    value = path.read_text(encoding="utf-8")
    for key, replacement in replacements.items():
        value = value.replace(key, replacement)
    if "{{" in value or "}}" in value:
        raise PublicDemoError(f"unresolved placeholder in template: {path.name}")
    return value


def _public_quality(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    components = source.get("components", {})
    allowed_components = {
        key: _unit_float(components.get(key, 0.0))
        for key in (
            "reconstruction_quality",
            "path_continuity",
            "match_connectivity",
            "contamination",
            "visual_clarity",
        )
    }
    return {
        "hero_score": _unit_float(source.get("hero_score", 0.0)),
        "score_confidence": _unit_float(source.get("score_confidence", 0.0)),
        "transparent_heuristic": bool(source.get("transparent_heuristic", True)),
        "components": allowed_components,
    }


def _raw_camera_path(camera_path: Mapping[str, Any]) -> np.ndarray:
    layers = camera_path.get("layers", {})
    if not isinstance(layers, dict) or "raw" not in layers:
        raise PublicDemoError("camera_path.json must include a raw layer")
    raw = np.asarray(layers["raw"], dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) < 2 or not np.isfinite(raw).all():
        raise PublicDemoError("invalid relative camera path layer: raw")
    return raw


def _display_transform(raw_path: np.ndarray) -> tuple[np.ndarray, float]:
    low = np.quantile(raw_path, 0.01, axis=0)
    high = np.quantile(raw_path, 0.99, axis=0)
    center = (low + high) * 0.5
    scale = max(float(np.max(high - low)) * 0.5, 1e-9)
    return center, scale


def _public_paths(
    camera_path: Mapping[str, Any],
    sample_count: int,
    *,
    display_transform: tuple[np.ndarray, float],
) -> dict[str, list[list[float]]]:
    layers = camera_path.get("layers", {})
    if not isinstance(layers, dict) or "raw" not in layers:
        raise PublicDemoError("camera_path.json must include a raw layer")
    arrays: dict[str, np.ndarray] = {}
    for key in ("raw", "bspline", "kalman", "rts"):
        if key not in layers:
            continue
        array = np.asarray(layers[key], dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3 or len(array) < 2 or not np.isfinite(array).all():
            raise PublicDemoError(f"invalid relative camera path layer: {key}")
        arrays[key] = _resample_matrix(array, sample_count)
    center, scale = display_transform
    result: dict[str, list[list[float]]] = {}
    for key, array in arrays.items():
        normalized = np.clip((array - center) / scale, -1.35, 1.35)
        result[key] = np.round(normalized, 5).tolist()
    return result


def _publish_authorized_point_cloud(
    *,
    viewer: Path,
    scene_dir: Path,
    scene_meta: Mapping[str, Any],
    display_transform: tuple[np.ndarray, float],
    authorization: str,
    attribution: str,
) -> tuple[dict[str, Any], list[Path]]:
    position_source = viewer / "points_preview.bin"
    color_source = viewer / "points_preview_colors.bin"
    if not position_source.is_file():
        raise PublicDemoError("authorized point positions are missing: points_preview.bin")
    if not color_source.is_file():
        raise PublicDemoError("authorized point colors are missing: points_preview_colors.bin")

    position_byte_count = position_source.stat().st_size
    if position_byte_count <= 0 or position_byte_count % 12:
        raise PublicDemoError(
            "points_preview.bin byte length must be positive and divisible by 12"
        )
    position_values = np.fromfile(position_source, dtype="<f4")
    if position_values.nbytes != position_byte_count:
        raise PublicDemoError(
            "points_preview.bin decoded byte count must match its source byte length"
        )
    if not len(position_values) or len(position_values) % 3:
        raise PublicDemoError("points_preview.bin must contain little-endian float32 XYZ rows")
    if not np.isfinite(position_values).all():
        raise PublicDemoError("points_preview.bin must contain only finite XYZ values")
    point_count = len(position_values) // 3
    expected_color_bytes = point_count * 3
    if color_source.stat().st_size != expected_color_bytes:
        raise PublicDemoError(
            "points_preview_colors.bin must contain one RGB8 triplet per XYZ point"
        )

    reconstruction = scene_meta.get("reconstruction", {})
    if not isinstance(reconstruction, dict):
        raise PublicDemoError("scene metadata must contain reconstruction provenance")
    source_point_count = reconstruction.get("point_count_source")
    if (
        isinstance(source_point_count, bool)
        or not isinstance(source_point_count, int)
        or source_point_count < point_count
    ):
        raise PublicDemoError(
            "scene metadata point_count_source must cover the published point sample"
        )
    viewer_point_count = reconstruction.get("point_count_viewer")
    if viewer_point_count is not None and viewer_point_count != point_count:
        raise PublicDemoError(
            "scene metadata point_count_viewer must match the published point sample"
        )

    geometry_dir = scene_dir / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    position_destination = geometry_dir / POINT_POSITION_ASSET
    color_destination = geometry_dir / POINT_COLOR_ASSET
    shutil.copyfile(position_source, position_destination)
    shutil.copyfile(color_source, color_destination)

    center, scale = display_transform
    position_record = _binary_record(
        position_destination,
        scene_dir,
        dtype="float32_le",
    )
    color_record = _binary_record(
        color_destination,
        scene_dir,
        dtype="uint8",
    )
    return (
        {
            "backend": "VGGT Omega",
            "scale_status": "relative_only",
            "point_count": point_count,
            "source_point_count": source_point_count,
            "positions": position_record,
            "colors": color_record,
            "display_transform": {
                "center": center.astype(float).tolist(),
                "scale": float(scale),
            },
            "authorization_provenance": authorization.strip(),
            "attribution": attribution.strip(),
        },
        [position_destination, color_destination],
    )


def _binary_record(path: Path, scene_dir: Path, *, dtype: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(scene_dir).as_posix(),
        "dtype": dtype,
        "components": 3,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _public_orientations(animation: Mapping[str, Any], sample_count: int) -> list[dict[str, Any]]:
    samples = animation.get("samples", [])
    if not isinstance(samples, list) or len(samples) < 2:
        raise PublicDemoError("pose animation must include at least two samples")
    quaternions = []
    confidence = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise PublicDemoError("pose animation samples must be objects")
        raw = sample.get("quaternion_wxyz") or sample.get("quaternion")
        quaternion = np.asarray(raw, dtype=np.float64)
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
            raise PublicDemoError("pose animation contains an invalid quaternion")
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-9:
            raise PublicDemoError("pose animation contains a zero quaternion")
        quaternion = quaternion / norm
        if quaternions and float(np.dot(quaternions[-1], quaternion)) < 0:
            quaternion = -quaternion
        quaternions.append(quaternion)
        confidence.append(_unit_float(sample.get("confidence", 0.0)))
    interpolated = _resample_matrix(np.asarray(quaternions), sample_count)
    norms = np.linalg.norm(interpolated, axis=1, keepdims=True)
    interpolated = interpolated / np.maximum(norms, 1e-9)
    confidence_values = _resample_vector(np.asarray(confidence), sample_count)
    time = np.linspace(0.0, 1.0, sample_count)
    return [
        {
            "time_normalized": round(float(time[index]), 6),
            "quaternion_wxyz": np.round(interpolated[index], 6).tolist(),
            "confidence": round(float(confidence_values[index]), 4),
        }
        for index in range(sample_count)
    ]


def _public_profiles(profiles: Mapping[str, Any], sample_count: int) -> dict[str, Any]:
    labels = {
        "speed_relative": "relative speed shape",
        "acceleration_relative": "relative acceleration shape",
        "curvature": "scale-free curvature shape",
        "jerk_proxy": "relative jerk proxy shape",
        "rts_residual": "RTS smoothing residual shape",
    }
    result: dict[str, Any] = {}
    for key, label in labels.items():
        source = np.asarray(profiles.get(key, []), dtype=np.float64)
        if source.ndim != 1 or len(source) < 2 or not np.isfinite(source).all():
            continue
        sampled = _resample_vector(source, sample_count)
        low, high = np.quantile(sampled, [0.05, 0.95])
        normalized = np.clip((sampled - low) / max(float(high - low), 1e-9), 0.0, 1.0)
        result[key] = {"label": label, "values": np.round(normalized, 5).tolist()}
    jumps = np.asarray(profiles.get("pose_jump", []), dtype=bool)
    if len(jumps) >= 2:
        indexes = np.rint(np.linspace(0, len(jumps) - 1, sample_count)).astype(int)
        result["pose_jump"] = {
            "label": "pose jump flag",
            "values": jumps[indexes].tolist(),
        }
    return result


def _coarse_density(
    path: Path,
    *,
    grid: tuple[int, int, int],
    max_cells: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise PublicDemoError("points_preview.bin is required for aggregate density")
    values = np.fromfile(path, dtype="<f4")
    if len(values) < 300 or len(values) % 3:
        raise PublicDemoError("points_preview.bin must contain finite XYZ float32 rows")
    points = values.reshape(-1, 3).astype(np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 100:
        raise PublicDemoError("not enough finite points for a public density abstraction")
    low = np.quantile(points, 0.02, axis=0)
    high = np.quantile(points, 0.98, axis=0)
    span = np.maximum(high - low, 1e-9)
    normalized = np.clip((points - low) / span, 0.0, 1.0 - np.finfo(float).eps)
    grid_array = np.asarray(grid, dtype=np.int64)
    bins = np.floor(normalized * grid_array).astype(np.int64)
    linear = np.ravel_multi_index(bins.T, grid)
    unique, counts = np.unique(linear, return_counts=True)
    order = np.argsort(counts)[::-1][:max_cells]
    unique = unique[order]
    counts = counts[order]
    coordinates = np.column_stack(np.unravel_index(unique, grid))
    centers = ((coordinates + 0.5) / grid_array) * 2.0 - 1.0
    maximum = max(int(counts.max()), 1)
    density = np.log1p(counts) / math.log1p(maximum)
    density = np.round(np.round(density * 7.0) / 7.0, 3)
    cells = [
        {
            "position": np.round(centers[index], 4).tolist(),
            "density": float(density[index]),
        }
        for index in range(len(centers))
    ]
    return {
        "kind": "coarse_normalized_voxel_density",
        "scale_status": "relative_only",
        "grid": list(grid),
        "cells": cells,
        "color_source_published": False,
        "raw_points_published": False,
        "notice": "Voxel centers are normalized aggregates, not a distributable point cloud.",
    }


def _public_methods(comparison: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = comparison.get("methods", {})
    if not isinstance(source, dict):
        raise PublicDemoError("method comparison must contain a methods object")
    labels = {
        "vggt_omega": "VGGT Omega",
        "hloc_lightglue_colmap": "HLoc + LightGlue + COLMAP",
        "r3": "R3",
        "lingbot_map": "LingBot-Map",
        "mast3r_sfm": "MASt3R SfM",
    }
    order = list(labels)
    rows = []
    for key in order:
        value = source.get(key)
        if not isinstance(value, dict):
            continue
        status = str(value.get("status", "unknown"))
        role = str(value.get("role", "method output retained for comparison"))
        metrics: list[dict[str, Any]] = []
        for metric, label in (
            ("input_frames", "input frames"),
            ("frame_count", "frames"),
            ("registered_images", "registered images"),
            ("pairs", "verified pairs"),
            ("points3D", "sparse points"),
            ("point_count", "source geometry points"),
        ):
            number = value.get(metric)
            if isinstance(number, (int, float)) and math.isfinite(float(number)):
                metrics.append({"label": label, "value": int(number)})
        inputs = ["accepted historical frame interval"]
        outputs = [role]
        if key in {"r3", "lingbot_map"}:
            outputs.append("aggregate relative depth and confidence summary")
        if key == "hloc_lightglue_colmap":
            outputs.append("aggregate match-connectivity matrix")
        if key == "mast3r_sfm" and status == "failed":
            outputs = ["no accepted output; failure retained without fabrication"]
        rows.append(
            {
                "id": key,
                "label": labels[key],
                "status": status,
                "role": role,
                "alignment": str(value.get("alignment", "not_aligned")),
                "scale_status": "relative_only",
                "inputs": inputs,
                "outputs": outputs,
                "metrics": metrics,
            }
        )
    return rows


def _public_match_connectivity(graph: Mapping[str, Any], bin_count: int = 20) -> dict[str, Any]:
    node_count = int(graph.get("node_count", 0))
    edge_count = int(graph.get("edge_count", 0))
    if node_count <= 0:
        raise PublicDemoError("match graph has no nodes")
    bins = min(bin_count, node_count)
    matrix = np.zeros((bins, bins), dtype=np.float64)
    degree = np.zeros(node_count, dtype=np.int64)
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = int(edge.get("source_id", 0))
        target = int(edge.get("target_id", 0))
        weight = max(float(edge.get("matches", 1)), 0.0)
        source_index = max(0, min(node_count - 1, source - 1 if source > 0 else source))
        target_index = max(0, min(node_count - 1, target - 1 if target > 0 else target))
        source_bin = min(bins - 1, source_index * bins // node_count)
        target_bin = min(bins - 1, target_index * bins // node_count)
        matrix[source_bin, target_bin] += weight
        matrix[target_bin, source_bin] += weight
        degree[source_index] += 1
        degree[target_index] += 1
    maximum = max(float(matrix.max()), 1.0)
    normalized = np.round(matrix / maximum, 4)
    histogram_counts, histogram_edges = np.histogram(degree, bins=min(10, max(3, node_count // 10)))
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "connected_component_count": int(graph.get("connected_component_count", 0)),
        "adjacent_pair_coverage": _unit_float(graph.get("adjacent_pair_coverage", 0.0)),
        "isolated_frame_count": len(graph.get("isolated_frames", [])),
        "matrix_bin_count": bins,
        "matrix": normalized.tolist(),
        "degree_histogram": {
            "counts": histogram_counts.astype(int).tolist(),
            "edges": np.round(histogram_edges, 3).tolist(),
        },
        "notice": "Frame identifiers and individual edges are withheld; cells aggregate connectivity.",
    }


def _summarize_r3(root: Path) -> dict[str, Any]:
    depth_paths = sorted((root / "depth").glob("*.npy"))
    confidence_paths = sorted((root / "conf").glob("*.npy"))
    if not depth_paths or not confidence_paths:
        return _missing_depth_summary("R3 numeric arrays unavailable")
    depth_values = _sample_npy_values(depth_paths)
    confidence_values = _sample_npy_values(confidence_paths)
    return _depth_summary("R3", len(depth_paths), depth_values, confidence_values)


def _summarize_lingbot(root: Path) -> dict[str, Any]:
    frame_paths = sorted((root / "frames").glob("*.npz"))
    if not frame_paths:
        return _missing_depth_summary("LingBot-Map numeric arrays unavailable")
    depth_chunks: list[np.ndarray] = []
    confidence_chunks: list[np.ndarray] = []
    for path in frame_paths:
        with np.load(path, allow_pickle=False) as archive:
            if "depth" not in archive or "depth_conf" not in archive:
                continue
            depth_chunks.append(_bounded_sample(np.asarray(archive["depth"], dtype=np.float64)))
            confidence_chunks.append(
                _bounded_sample(np.asarray(archive["depth_conf"], dtype=np.float64))
            )
    if not depth_chunks or not confidence_chunks:
        return _missing_depth_summary("LingBot-Map depth/confidence keys unavailable")
    return _depth_summary(
        "LingBot-Map",
        len(depth_chunks),
        np.concatenate(depth_chunks),
        np.concatenate(confidence_chunks),
    )


def _sample_npy_values(paths: Sequence[Path]) -> np.ndarray:
    chunks = []
    for path in paths:
        value = np.load(path, allow_pickle=False)
        chunks.append(_bounded_sample(np.asarray(value, dtype=np.float64)))
    return np.concatenate(chunks)


def _bounded_sample(value: np.ndarray, maximum: int = 4096) -> np.ndarray:
    flat = value.reshape(-1)
    stride = max(1, math.ceil(len(flat) / maximum))
    sampled = flat[::stride]
    return sampled[np.isfinite(sampled)]


def _depth_summary(
    label: str,
    frame_count: int,
    depth_values: np.ndarray,
    confidence_values: np.ndarray,
) -> dict[str, Any]:
    if not len(depth_values) or not len(confidence_values):
        return _missing_depth_summary(f"{label} finite values unavailable")
    positive = depth_values[depth_values > 0]
    median = float(np.median(positive)) if len(positive) else 1.0
    normalized_depth = positive / max(median, 1e-9)
    return {
        "label": label,
        "status": "summarized",
        "scale_status": "relative_only",
        "frames_summarized": frame_count,
        "sampled_value_count": int(len(depth_values)),
        "depth_normalized_to_scene_median": _quantiles(normalized_depth),
        "confidence_model_native": _quantiles(confidence_values),
        "source_arrays_published": False,
        "notice": "Depth is normalized within this scene and is not metric distance.",
    }


def _missing_depth_summary(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "scale_status": "relative_only",
        "reason": reason,
        "source_arrays_published": False,
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {"p05": 0.0, "p50": 0.0, "p95": 0.0}
    p05, p50, p95 = np.quantile(values, [0.05, 0.5, 0.95])
    return {"p05": round(float(p05), 5), "p50": round(float(p50), 5), "p95": round(float(p95), 5)}


def _public_failure_case(
    taxonomy: Mapping[str, Any],
    methods: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    hloc = next((method for method in methods if method.get("id") == "hloc_lightglue_colmap"), {})
    declared = next(
        (
            int(metric["value"])
            for metric in hloc.get("metrics", [])
            if metric.get("label") == "registered images"
        ),
        None,
    )
    status = str(taxonomy.get("status", "unclassified"))
    successful = bool(taxonomy.get("successful_reconstruction", False))
    if status == "completed_with_warnings" and successful:
        interpretation = (
            "Sparse reconstruction was retained. Bundle-adjustment logs show numerical "
            "instability, so the result remains a warning-bearing reference rather than "
            "a clean convergence claim."
        )
    else:
        interpretation = "Diagnostic evidence is retained without executing an automatic fallback."
    return {
        "status": status,
        "categories": [str(value).replace("_", " ") for value in taxonomy.get("categories", [])],
        "successful_reconstruction_retained": successful,
        "declared_registered_images": declared,
        "registered_frame_max_observed": _optional_int(taxonomy.get("registered_frame_max")),
        "ba_log_events": _optional_int(taxonomy.get("ba_invocations")),
        "fallback_eligible": bool(taxonomy.get("fallback_eligible", False)),
        "fallback_executed": bool(taxonomy.get("executes_fallback", False)),
        "recommendations": [
            str(value).replace("_", " ")
            for value in taxonomy.get("bounded_recommendations", [])
        ],
        "interpretation": interpretation,
        "secondary_method_failure": {
            "method": "MASt3R SfM",
            "status": "failed",
            "handling": "No accepted output was available; no comparison was fabricated.",
        },
    }


def _resample_matrix(value: np.ndarray, count: int) -> np.ndarray:
    source = np.linspace(0.0, 1.0, len(value))
    target = np.linspace(0.0, 1.0, count)
    columns = [np.interp(target, source, value[:, index]) for index in range(value.shape[1])]
    return np.column_stack(columns)


def _resample_vector(value: np.ndarray, count: int) -> np.ndarray:
    return np.interp(np.linspace(0.0, 1.0, count), np.linspace(0.0, 1.0, len(value)), value)


def _unit_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return round(min(1.0, max(0.0, number)), 5)


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_catalog_field(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or len(text) > 80:
        return None
    if ABSOLUTE_PATH_PATTERN.search(text) or EXTERNAL_RUNTIME_PATTERN.search(text):
        return None
    return text


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest,
    }


def main(
    scene_root: Path = typer.Option(..., "--scene-root", exists=True, file_okay=False),
    archive_root: Path = typer.Option(..., "--archive-root", exists=True, file_okay=False),
    output_root: Path = typer.Option(Path("build/public_demo"), "--output-root"),
    catalog_json: Path | None = typer.Option(
        None,
        "--catalog-json",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    slug: str = typer.Option(..., "--slug"),
    title: str = typer.Option(..., "--title"),
    path_samples: int = typer.Option(96, "--path-samples", min=8, max=160),
    density_cells: int = typer.Option(560, "--density-cells", min=24, max=900),
    publish_real_point_cloud: bool = typer.Option(False, "--publish-real-point-cloud"),
    point_cloud_authorization: str | None = typer.Option(
        None,
        "--point-cloud-authorization",
    ),
    point_cloud_attribution: str | None = typer.Option(
        None,
        "--point-cloud-attribution",
    ),
) -> None:
    """Create a privacy-reduced, dependency-free static research demonstration."""
    result = build_public_demo(
        PublicDemoConfig(
            scene_root=scene_root,
            archive_root=archive_root,
            output_root=output_root,
            catalog_path=catalog_json,
            slug=slug,
            public_title=title,
            path_sample_count=path_samples,
            max_density_cells=density_cells,
            publish_real_point_cloud=publish_real_point_cloud,
            point_cloud_authorization=point_cloud_authorization,
            point_cloud_attribution=point_cloud_attribution,
        )
    )
    typer.echo(
        json.dumps(
            {
                "status": result.status,
                "output_root": str(result.output_root),
                "scene_entry": str(result.scene_entry),
                "file_count": result.file_count,
                "total_bytes": result.total_bytes,
                "audit": result.audit,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    typer.run(main)
