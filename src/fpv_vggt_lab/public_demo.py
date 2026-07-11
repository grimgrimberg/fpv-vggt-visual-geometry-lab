from __future__ import annotations

import hashlib
import html
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import typer


PUBLIC_SCHEMA_VERSION = "1.1.0"
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


class PublicDemoError(RuntimeError):
    """Raised when a private scene cannot be reduced to the public contract."""


@dataclass(frozen=True, slots=True)
class PublicDemoConfig:
    scene_root: Path
    archive_root: Path
    output_root: Path
    slug: str
    public_title: str
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
    scene_root = config.scene_root.resolve()
    archive_root = config.archive_root.resolve()
    output_root = config.output_root.resolve()
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

    asset_source = Path(__file__).with_name("public_demo_assets")
    for name in ("site.css", "gallery.js", "scene.js"):
        source = asset_source / name
        if not source.is_file():
            raise PublicDemoError(f"missing public demo asset: {name}")
        destination = assets_dir / name
        shutil.copyfile(source, destination)
        generated_files.append(destination)

    safe_title = html.escape(config.public_title, quote=True)
    replacements = {
        "{{PUBLIC_TITLE}}": safe_title,
        "{{SCENE_URL}}": f"scenes/{config.slug}/",
        "{{SCENE_DATA_URL}}": f"scenes/{config.slug}/scene.json",
        "{{GENERATED_AT}}": html.escape(generated_at, quote=True),
    }
    gallery_html = _render_template(asset_source / "gallery.html", replacements)
    scene_html = _render_template(
        asset_source / "scene.html",
        {"{{PUBLIC_TITLE}}": safe_title},
    )

    index_path = output_root / "index.html"
    scene_entry = scene_dir / "index.html"
    scene_json = scene_dir / "scene.json"
    nojekyll = output_root / ".nojekyll"
    index_path.write_text(gallery_html, encoding="utf-8")
    scene_entry.write_text(scene_html, encoding="utf-8")
    scene_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    nojekyll.write_text("", encoding="utf-8")
    generated_files.extend((index_path, scene_entry, scene_json, nojekyll))

    audit = audit_public_demo(
        output_root,
        generated_files,
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
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    generated_files.append(manifest_path)

    final_audit = audit_public_demo(
        output_root,
        generated_files,
        authorized_binary_paths=authorized_binary_paths,
    )
    if final_audit["status"] != "passed":
        raise PublicDemoError("build manifest failed the final public output audit")
    return PublicDemoBuildResult(
        status="built",
        output_root=output_root,
        scene_entry=scene_entry,
        manifest=manifest_path,
        file_count=len(generated_files),
        total_bytes=sum(path.stat().st_size for path in generated_files),
        audit=final_audit,
    )


def audit_public_demo(
    root: Path,
    paths: Iterable[Path] | None = None,
    authorized_binary_paths: Iterable[str] = (),
) -> dict[str, Any]:
    """Fail closed on media, raw arrays, external runtimes, paths, and secrets."""
    root = root.resolve()
    candidates = list(paths) if paths is not None else [path for path in root.rglob("*") if path.is_file()]
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
        if EXTERNAL_RUNTIME_PATTERN.search(text):
            findings.append({"path": relative, "reason": "external runtime dependency"})
        if SECRET_PATTERN.search(text):
            findings.append({"path": relative, "reason": "secret-like value"})
    return {
        "status": "failed" if findings else "passed",
        "checked_file_count": checked,
        "finding_count": len(findings),
        "findings": findings,
        "scope": "generated public demo files",
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PublicDemoError(f"required derived artifact is missing: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PublicDemoError(f"expected a JSON object: {path.name}")
    return value


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

    position_values = np.fromfile(position_source, dtype="<f4")
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
    position_destination = geometry_dir / "vggt_omega_points.f32.bin"
    color_destination = geometry_dir / "vggt_omega_colors.rgb8.bin"
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
    output_root: Path = typer.Option(Path("docs"), "--output-root"),
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
