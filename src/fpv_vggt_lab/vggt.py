from __future__ import annotations

import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .frames import read_frame_manifest
from .schemas import BundleMetadata, VggtValidationReport, model_to_dict


SAFETY_WARNINGS = [
    "no geolocation",
    "no meters",
    "relative VGGT frame",
    "local-only media",
]


@dataclass(frozen=True)
class VggtBundle:
    path: Path
    metadata: BundleMetadata
    camera_centers: np.ndarray
    quaternions_xyzw: np.ndarray
    valid_pose_mask: np.ndarray
    pose_confidence: np.ndarray
    points: np.ndarray | None
    point_colors_rgb: np.ndarray | None = None
    point_confidence: np.ndarray | None = None
    point_depth: np.ndarray | None = None


def create_mock_bundle(frame_manifest_path: Path, output: Path) -> Path:
    manifest = read_frame_manifest(frame_manifest_path)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    frame_indices = [frame.frame_index for frame in manifest.frames]
    frame_timestamps = [frame.timestamp_sec for frame in manifest.frames]
    n_frames = len(frame_indices)

    t = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)
    camera_centers = np.column_stack(
        [
            t,
            0.15 * np.sin(t * np.pi * 2),
            0.08 * np.cos(t * np.pi * 2),
        ]
    ).astype(np.float32)
    quaternions = np.zeros((n_frames, 4), dtype=np.float32)
    quaternions[:, 3] = 1.0
    valid_pose_mask = np.ones(n_frames, dtype=bool)
    pose_confidence = np.full(n_frames, 0.92, dtype=np.float32)

    metadata = BundleMetadata(
        schema_version="v1",
        video_id=manifest.video_id,
        segment_id=manifest.segment_id,
        source_tool="mock",
        generated_at=datetime.now(timezone.utc).isoformat(),
        frame_indices=frame_indices,
        frame_timestamps_sec=frame_timestamps,
        warnings=["mocked VGGT output for synthetic tests only"],
    )

    (output / "metadata.json").write_text(
        json.dumps(model_to_dict(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    np.savez_compressed(
        output / "cameras.npz",
        camera_centers=camera_centers,
        quaternions_xyzw=quaternions,
        valid_pose_mask=valid_pose_mask,
        pose_confidence=pose_confidence,
    )
    np.savez_compressed(output / "points.npz", points=_mock_points(camera_centers))
    return output


def export_vggt_request(frame_manifest_path: Path, output: Path) -> Path:
    manifest = read_frame_manifest(frame_manifest_path)
    output = output.resolve()
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    copied_frames = []
    for ordinal, frame in enumerate(manifest.frames):
        source = Path(frame.path)
        destination = frames_dir / f"{ordinal:04d}_frame_{frame.frame_index:06d}{source.suffix}"
        shutil.copyfile(source, destination)
        copied_frames.append(
            {
                "frame_index": frame.frame_index,
                "timestamp_sec": frame.timestamp_sec,
                "filename": destination.name,
            }
        )

    request_manifest = {
        "video_id": manifest.video_id,
        "segment_id": manifest.segment_id,
        "frame_count": len(copied_frames),
        "frames": copied_frames,
        "warnings": [
            "local-only media-derived frames",
            "use only for offline VGGT reconstruction",
            "do not infer geolocation, meters, speed, standoff, or target coordinates",
        ],
    }
    (output / "request_manifest.json").write_text(
        json.dumps(request_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "README.md").write_text(_export_request_readme(manifest.video_id), encoding="utf-8")

    zip_path = output / "frames.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for frame_file in frames_dir.iterdir():
            archive.write(frame_file, arcname=frame_file.name)
    return output


def import_converted_bundle(
    source: Path,
    frame_manifest_path: Path,
    output: Path,
    source_tool: str,
    source_url_or_repo: str | None = None,
    source_commit_or_version: str | None = None,
    export_notes: str | None = None,
) -> Path:
    manifest = read_frame_manifest(frame_manifest_path)
    source = source.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    cameras_source = source / "cameras.npz"
    if not cameras_source.exists():
        raise ValueError(f"missing cameras.npz in converted source bundle: {source}")
    shutil.copyfile(cameras_source, output / "cameras.npz")

    points_source = source / "points.npz"
    if points_source.exists():
        shutil.copyfile(points_source, output / "points.npz")

    metadata = BundleMetadata(
        schema_version="v1",
        video_id=manifest.video_id,
        segment_id=manifest.segment_id,
        source_tool=source_tool,  # type: ignore[arg-type]
        source_url_or_repo=source_url_or_repo,
        source_commit_or_version=source_commit_or_version,
        export_notes=export_notes,
        generated_at=datetime.now(timezone.utc).isoformat(),
        frame_indices=[frame.frame_index for frame in manifest.frames],
        frame_timestamps_sec=[frame.timestamp_sec for frame in manifest.frames],
        warnings=[
            "Imported converted VGGT bundle; coordinates remain relative and scale ambiguous."
        ],
    )
    (output / "metadata.json").write_text(
        json.dumps(model_to_dict(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = validate_bundle(output)
    if not report.valid:
        raise ValueError("; ".join(report.errors))
    return output


def import_predictions_npz(
    predictions_path: Path,
    frame_manifest_path: Path,
    output: Path,
    source_tool: str = "huggingface-space",
    source_url_or_repo: str | None = None,
    source_commit_or_version: str | None = None,
    export_notes: str | None = None,
    max_points: int = 50000,
) -> Path:
    manifest = read_frame_manifest(frame_manifest_path)
    predictions_path = predictions_path.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    predictions = np.load(predictions_path, allow_pickle=False)
    frame_count = len(manifest.frames)
    if "extrinsic" not in predictions:
        raise ValueError("predictions.npz must include VGGT extrinsic array")
    extrinsic = np.asarray(predictions["extrinsic"], dtype=np.float32)
    camera_centers, quaternions = _camera_centers_and_quaternions(extrinsic, frame_count)
    valid_pose_mask = np.isfinite(camera_centers).all(axis=1)
    pose_confidence = _pose_confidence_from_predictions(predictions, frame_count)
    point_bundle = _point_bundle_from_predictions(predictions, max_points=max_points)

    np.savez_compressed(
        output / "cameras.npz",
        camera_centers=camera_centers,
        quaternions_xyzw=quaternions,
        valid_pose_mask=valid_pose_mask,
        pose_confidence=pose_confidence,
    )
    if point_bundle is not None:
        np.savez_compressed(output / "points.npz", **point_bundle)

    metadata = BundleMetadata(
        schema_version="v1",
        video_id=manifest.video_id,
        segment_id=manifest.segment_id,
        source_tool=source_tool,  # type: ignore[arg-type]
        source_url_or_repo=source_url_or_repo,
        source_commit_or_version=source_commit_or_version,
        export_notes=export_notes,
        generated_at=datetime.now(timezone.utc).isoformat(),
        frame_indices=[frame.frame_index for frame in manifest.frames],
        frame_timestamps_sec=[frame.timestamp_sec for frame in manifest.frames],
        warnings=[
            "Imported from VGGT predictions.npz; camera centers are relative and scale ambiguous.",
            "Extrinsic matrices are interpreted as OpenCV camera-from-world transforms.",
        ],
    )
    (output / "metadata.json").write_text(
        json.dumps(model_to_dict(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = validate_bundle(output)
    if not report.valid:
        raise ValueError("; ".join(report.errors))
    return output


def import_cloud_job_bundles(
    source: Path,
    output_root: Path,
    report_path: Path,
    overwrite: bool = False,
    expected_bundles: set[tuple[str, str]] | None = None,
    dry_run: bool = False,
) -> dict:
    source = source.resolve()
    source_was_zip = source.suffix.lower() == ".zip"
    extracted_to: Path | None = None
    cloud_summary = _load_adjacent_cloud_summary(source)
    cloud_log = _adjacent_cloud_log(source)
    cloud_summary_audit = _audit_cloud_summary(cloud_summary, expected_bundles)
    if source_was_zip:
        extracted_to = _extract_bundle_zip(source, report_path)
        source = extracted_to
    if (source / "bundles").is_dir():
        source = source / "bundles"
    output_root = output_root.resolve()
    report_path = report_path.resolve()

    bundle_paths = _discover_bundle_paths(source)
    clips = []
    imported_count = 0
    would_import_count = 0
    skipped_unexpected_count = 0
    satisfied_expected: set[tuple[str, str]] = set()
    status = "done"
    cloud_summary_needs_review = cloud_summary_audit["status"] == "needs_review"
    if cloud_summary_needs_review:
        status = "failed_soft"

    for bundle_path in bundle_paths:
        validation = validate_bundle(bundle_path)
        destination = None
        copied = False
        would_copy = False
        errors = list(validation.errors)
        metadata = validation.metadata
        skipped_reason = None
        if validation.valid and metadata is not None:
            bundle_key = (metadata["video_id"], metadata["segment_id"])
            if expected_bundles is not None and bundle_key not in expected_bundles:
                skipped_reason = "unexpected bundle for this run"
                skipped_unexpected_count += 1
            else:
                destination = output_root / metadata["video_id"] / metadata["segment_id"]
                if destination.exists() and not overwrite:
                    errors.append(f"destination already exists: {destination}")
                if cloud_summary_needs_review and not dry_run:
                    errors.append("cloud_summary_audit needs review; run dry-run first")
                if not errors:
                    if dry_run:
                        would_copy = True
                        would_import_count += 1
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if destination.exists():
                            shutil.rmtree(destination)
                        shutil.copytree(bundle_path, destination)
                        copied = True
                        imported_count += 1
                    satisfied_expected.add(bundle_key)
        if errors:
            status = "failed_soft"
        clips.append(
            {
                "source": str(bundle_path),
                "destination": str(destination) if destination is not None else None,
                "copied": copied,
                "would_copy": would_copy,
                "valid": validation.valid,
                "errors": errors,
                "warnings": validation.warnings,
                "metadata": metadata,
                "skipped_reason": skipped_reason,
            }
        )

    if not bundle_paths:
        status = "failed_soft"
    missing_expected = []
    if expected_bundles is not None:
        missing_expected = _bundle_key_dicts(expected_bundles - satisfied_expected)
        if missing_expected:
            status = "failed_soft"
    report = {
        "status": status,
        "dry_run": dry_run,
        "source": str(source),
        "source_was_zip": source_was_zip,
        "extracted_to": str(extracted_to) if extracted_to is not None else None,
        "cloud_summary": cloud_summary,
        "cloud_summary_audit": cloud_summary_audit,
        "cloud_log": str(cloud_log) if cloud_log is not None else None,
        "output_root": str(output_root),
        "discovered_count": len(bundle_paths),
        "imported_count": imported_count,
        "would_import_count": would_import_count,
        "skipped_unexpected_count": skipped_unexpected_count,
        "expected_bundles": _bundle_key_dicts(expected_bundles) if expected_bundles is not None else None,
        "missing_expected_bundles": missing_expected,
        "warnings": SAFETY_WARNINGS,
        "clips": clips,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def expected_bundles_from_run_summary(summary_path: Path) -> set[tuple[str, str]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    inputs = summary.get("inputs", {})
    video_ids = inputs.get("video_ids")
    segment_id = inputs.get("segment_id", "segment-001")
    if isinstance(video_ids, list) and video_ids:
        return {(str(video_id), str(segment_id)) for video_id in video_ids}

    bundle_paths = inputs.get("bundles") or summary.get("artifacts", {}).get("bundles")
    if isinstance(bundle_paths, list) and bundle_paths:
        expected = set()
        for bundle_path in bundle_paths:
            parts = Path(str(bundle_path)).parts
            if len(parts) < 2:
                raise ValueError(f"cannot derive video_id/segment_id from bundle path: {bundle_path}")
            expected.add((parts[-2], parts[-1]))
        return expected

    raise ValueError(
        "run summary does not contain inputs.video_ids or bundle paths for expected bundle filtering"
    )


def _bundle_key_dicts(bundle_keys: set[tuple[str, str]]) -> list[dict[str, str]]:
    return [
        {"video_id": video_id, "segment_id": segment_id}
        for video_id, segment_id in sorted(bundle_keys)
    ]


def _audit_cloud_summary(
    cloud_summary: dict | None,
    expected_bundles: set[tuple[str, str]] | None,
) -> dict[str, object]:
    if cloud_summary is None:
        return {"status": "not_provided", "issues": []}

    issues: list[str] = []
    summary_status = cloud_summary.get("status")
    if summary_status != "done":
        issues.append(f"cloud_summary status is {summary_status}")

    clips_by_key = {}
    for clip in cloud_summary.get("clips", []):
        video_id = clip.get("video_id")
        segment_id = clip.get("segment_id")
        if video_id is None or segment_id is None:
            continue
        clips_by_key[(str(video_id), str(segment_id))] = clip

    if expected_bundles is not None and clips_by_key:
        for video_id, segment_id in sorted(expected_bundles):
            clip = clips_by_key.get((video_id, segment_id))
            if clip is None:
                issues.append(f"cloud_summary missing expected clip {video_id}/{segment_id}")
                continue
            clip_status = clip.get("status")
            bundle_valid = clip.get("bundle_valid")
            if clip_status != "done" or bundle_valid is not True:
                message = (
                    f"expected clip {video_id}/{segment_id} status is {clip_status} "
                    f"with bundle_valid={bundle_valid}"
                )
                if clip.get("error"):
                    message += f": {clip['error']}"
                issues.append(message)

    return {
        "status": "needs_review" if issues else "consistent",
        "issues": issues,
    }


def load_bundle(bundle_path: Path) -> VggtBundle:
    bundle_path = bundle_path.resolve()
    metadata_path = bundle_path / "metadata.json"
    cameras_path = bundle_path / "cameras.npz"

    if not metadata_path.exists():
        raise ValueError(f"missing metadata.json in {bundle_path}")
    if not cameras_path.exists():
        raise ValueError(f"missing cameras.npz in {bundle_path}")

    metadata = BundleMetadata.model_validate(
        json.loads(metadata_path.read_text(encoding="utf-8"))
    )
    cameras = np.load(cameras_path)

    required_arrays = [
        "camera_centers",
        "quaternions_xyzw",
        "valid_pose_mask",
        "pose_confidence",
    ]
    for name in required_arrays:
        if name not in cameras:
            raise ValueError(f"missing {name} in cameras.npz")

    camera_centers = np.asarray(cameras["camera_centers"], dtype=np.float32)
    quaternions = np.asarray(cameras["quaternions_xyzw"], dtype=np.float32)
    valid_pose_mask = np.asarray(cameras["valid_pose_mask"], dtype=bool)
    pose_confidence = np.asarray(cameras["pose_confidence"], dtype=np.float32)

    n_frames = len(metadata.frame_indices)
    if camera_centers.shape != (n_frames, 3):
        raise ValueError("camera_centers must have shape (frame_count, 3)")
    if quaternions.shape != (n_frames, 4):
        raise ValueError("quaternions_xyzw must have shape (frame_count, 4)")
    if valid_pose_mask.shape != (n_frames,):
        raise ValueError("valid_pose_mask must have shape (frame_count,)")
    if pose_confidence.shape != (n_frames,):
        raise ValueError("pose_confidence must have shape (frame_count,)")
    _require_finite("camera_centers", camera_centers)
    _require_finite("quaternions_xyzw", quaternions)
    _require_finite("pose_confidence", pose_confidence)

    points = None
    point_colors_rgb = None
    point_confidence = None
    point_depth = None
    points_path = bundle_path / "points.npz"
    if points_path.exists():
        point_data = np.load(points_path, allow_pickle=False)
        if "points" not in point_data:
            raise ValueError("missing points in points.npz")
        points = np.asarray(point_data["points"], dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (point_count, 3)")
        _require_finite("points", points)
        point_count = len(points)
        point_colors_rgb = _optional_point_colors(point_data, point_count)
        point_confidence = _optional_point_vector(
            point_data,
            ["point_confidence", "point_conf", "confidence", "conf"],
            point_count,
            "point_confidence",
        )
        point_depth = _optional_point_vector(
            point_data,
            ["point_depth", "depth", "relative_depth"],
            point_count,
            "point_depth",
        )

    return VggtBundle(
        path=bundle_path,
        metadata=metadata,
        camera_centers=camera_centers,
        quaternions_xyzw=quaternions,
        valid_pose_mask=valid_pose_mask,
        pose_confidence=pose_confidence,
        points=points,
        point_colors_rgb=point_colors_rgb,
        point_confidence=point_confidence,
        point_depth=point_depth,
    )


def validate_bundle(bundle_path: Path) -> VggtValidationReport:
    bundle_path = bundle_path.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    metadata_dict: dict | None = None

    metadata_path = bundle_path / "metadata.json"
    cameras_path = bundle_path / "cameras.npz"
    if not metadata_path.exists():
        errors.append(f"missing metadata.json in {bundle_path}")
    else:
        try:
            metadata_dict = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata = BundleMetadata.model_validate(metadata_dict)
            for issue in _provenance_warnings(metadata):
                if "secret-looking" in issue:
                    errors.append(issue)
                else:
                    warnings.append(issue)
        except Exception as exc:
            errors.append(f"invalid metadata.json: {exc}")

    if not cameras_path.exists():
        errors.append(f"missing cameras.npz in {bundle_path}")

    if not errors:
        try:
            load_bundle(bundle_path)
        except Exception as exc:
            errors.append(str(exc))

    return VggtValidationReport(
        valid=not errors,
        bundle_path=bundle_path,
        errors=errors,
        warnings=warnings,
        metadata=metadata_dict,
    )


def format_bundle_inspection(bundle_path: Path) -> str:
    report = validate_bundle(bundle_path)
    lines = [
        "VGGT bundle inspection",
        f"Bundle: {bundle_path.resolve()}",
        f"Validation: {'valid' if report.valid else 'invalid'}",
    ]
    if not report.valid:
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in report.errors)
        lines.append("Safety: No meters, no geolocation; VGGT coordinates are relative.")
        return "\n".join(lines)

    bundle = load_bundle(bundle_path)
    metadata = bundle.metadata
    lines.extend(
        [
            f"Video: {metadata.video_id}",
            f"Segment: {metadata.segment_id}",
            f"Source tool: {metadata.source_tool}",
            f"Frames: {len(metadata.frame_indices)}",
            f"Coordinate frame: {metadata.coordinate_frame}",
            "",
            "Camera arrays:",
            _array_status("camera_centers", bundle.camera_centers),
            _array_status("quaternions_xyzw", bundle.quaternions_xyzw),
            _array_status("valid_pose_mask", bundle.valid_pose_mask),
            _array_status("pose_confidence", bundle.pose_confidence),
            "",
            "Point cloud arrays:",
            _array_status("points", bundle.points),
            _array_status("point_colors_rgb", bundle.point_colors_rgb),
            _array_status("point_confidence", bundle.point_confidence),
            _array_status("point_depth", bundle.point_depth),
            "",
            "Review behavior:",
            _rgb_behavior(bundle),
            _confidence_behavior(bundle),
            _depth_behavior(bundle),
            "6DoF pose view: retrospective VGGT-frame camera pose only",
            "Safety: No meters, no geolocation; no speed, standoff, targeting, or guidance claims.",
        ]
    )
    return "\n".join(lines)


def _array_status(name: str, values: np.ndarray | None) -> str:
    if values is None:
        return f"- {name}: missing"
    finite = "yes" if _array_is_finite(values) else "no"
    return f"- {name}: present shape={values.shape} finite={finite}"


def _array_is_finite(values: np.ndarray) -> bool:
    if values.dtype == bool:
        return True
    return bool(np.isfinite(values).all())


def _rgb_behavior(bundle: VggtBundle) -> str:
    if bundle.point_colors_rgb is not None:
        return "RGB point colors: available"
    return "RGB point colors: unavailable; review uses depth/fallback coloring"


def _confidence_behavior(bundle: VggtBundle) -> str:
    if bundle.point_confidence is not None:
        return "Confidence filter: uses point_confidence"
    return "Confidence filter: no per-point confidence; all points pass confidence filtering"


def _depth_behavior(bundle: VggtBundle) -> str:
    if bundle.point_depth is not None:
        return "Depth filter: uses point_depth"
    return "Depth filter: uses relative-depth fallback from first camera center"


def _require_finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")


def _optional_point_colors(
    point_data: np.lib.npyio.NpzFile,
    point_count: int,
) -> np.ndarray | None:
    array = _first_present_array(
        point_data,
        ["point_colors_rgb", "colors_rgb", "point_rgb", "rgb", "colors"],
    )
    if array is None:
        return None
    colors = np.asarray(array)
    if colors.ndim > 2:
        colors = colors.reshape(-1, colors.shape[-1])
    if colors.ndim != 2 or colors.shape != (point_count, 3):
        raise ValueError("point_colors_rgb must have shape (point_count, 3)")
    colors = colors.astype(np.float32)
    _require_finite("point_colors_rgb", colors)
    if colors.size and float(colors.max()) <= 1.0:
        colors = colors * 255.0
    return np.clip(np.rint(colors), 0, 255).astype(np.uint8)


def _optional_point_vector(
    point_data: np.lib.npyio.NpzFile,
    names: list[str],
    point_count: int,
    label: str,
) -> np.ndarray | None:
    array = _first_present_array(point_data, names)
    if array is None:
        return None
    values = np.asarray(array, dtype=np.float32).reshape(-1)
    if values.shape != (point_count,):
        raise ValueError(f"{label} must have shape (point_count,)")
    _require_finite(label, values)
    return values.astype(np.float32)


def _first_present_array(
    data: np.lib.npyio.NpzFile,
    names: list[str],
) -> np.ndarray | None:
    for name in names:
        if name in data:
            return data[name]
    return None


def load_metadata_if_present(bundle_path: Path) -> BundleMetadata | None:
    metadata_path = bundle_path / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        return BundleMetadata.model_validate(
            json.loads(metadata_path.read_text(encoding="utf-8"))
        )
    except Exception:
        return None


def _discover_bundle_paths(source: Path) -> list[Path]:
    if not source.exists():
        return []
    if (source / "metadata.json").exists():
        return [source]
    return sorted(path.parent for path in source.rglob("metadata.json"))


def _extract_bundle_zip(source: Path, report_path: Path) -> Path:
    if not source.exists():
        return report_path.parent / f"{source.stem}_extracted"
    extraction_root = report_path.parent / f"{source.stem}_extracted"
    if extraction_root.exists():
        shutil.rmtree(extraction_root)
    extraction_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"unsafe path in bundle zip: {member.filename}")
        archive.extractall(extraction_root)
    return extraction_root


def _load_adjacent_cloud_summary(source: Path) -> dict | None:
    summary_path = source.with_name("cloud_summary.json")
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _adjacent_cloud_log(source: Path) -> Path | None:
    log_path = source.with_name("cloud_run.log")
    return log_path if log_path.exists() else None


def _mock_points(camera_centers: np.ndarray) -> np.ndarray:
    offsets = np.array(
        [
            [-0.2, -0.1, 0.1],
            [-0.1, 0.2, -0.1],
            [0.0, -0.2, 0.2],
            [0.1, 0.1, -0.2],
            [0.2, -0.05, 0.05],
        ],
        dtype=np.float32,
    )
    return (camera_centers[:, None, :] + offsets[None, :, :]).reshape(-1, 3)


def _camera_centers_and_quaternions(
    extrinsic: np.ndarray, frame_count: int
) -> tuple[np.ndarray, np.ndarray]:
    if extrinsic.shape == (frame_count, 3, 4):
        matrices = extrinsic
    elif extrinsic.shape == (frame_count, 4, 4):
        matrices = extrinsic[:, :3, :]
    else:
        raise ValueError("extrinsic must have shape (frame_count, 3, 4) or (frame_count, 4, 4)")

    centers = []
    quaternions = []
    for matrix in matrices:
        rotation_world_to_camera = matrix[:3, :3]
        translation = matrix[:3, 3]
        rotation_camera_to_world = rotation_world_to_camera.T
        center = -rotation_camera_to_world @ translation
        centers.append(center)
        quaternions.append(_rotation_matrix_to_quaternion_xyzw(rotation_camera_to_world))
    return np.asarray(centers, dtype=np.float32), np.asarray(quaternions, dtype=np.float32)


def _rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray([x, y, z, w], dtype=np.float32)
    norm = float(np.linalg.norm(quaternion))
    return quaternion / norm if norm > 0 else np.asarray([0, 0, 0, 1], dtype=np.float32)


def _pose_confidence_from_predictions(predictions: np.lib.npyio.NpzFile, frame_count: int) -> np.ndarray:
    for key in ["pose_conf", "camera_conf", "depth_conf", "point_conf"]:
        if key not in predictions:
            continue
        values = np.asarray(predictions[key], dtype=np.float32)
        if values.shape[0] != frame_count:
            continue
        if values.ndim == 1:
            return values
        return values.reshape(frame_count, -1).mean(axis=1).astype(np.float32)
    return np.ones(frame_count, dtype=np.float32)


def _points_from_predictions(
    predictions: np.lib.npyio.NpzFile, max_points: int
) -> np.ndarray | None:
    point_bundle = _point_bundle_from_predictions(predictions, max_points=max_points)
    if point_bundle is None:
        return None
    return point_bundle["points"]


def _point_bundle_from_predictions(
    predictions: np.lib.npyio.NpzFile,
    max_points: int,
) -> dict[str, np.ndarray] | None:
    for key in ["world_points_from_depth", "point_map", "points", "world_points"]:
        if key not in predictions:
            continue
        source = np.asarray(predictions[key], dtype=np.float32)
        points = source.reshape(-1, 3) if source.ndim > 2 else source
        if points.ndim != 2 or points.shape[1] != 3:
            continue
        finite = np.isfinite(points).all(axis=1)
        colors = _prediction_point_colors(predictions, len(points))
        confidence = _prediction_point_vector(
            predictions,
            ["point_confidence", "point_conf", "depth_conf", "confidence"],
            len(points),
        )
        depth = _prediction_point_vector(
            predictions,
            ["point_depth", "depth", "depth_map"],
            len(points),
        )
        points = points[finite]
        if colors is not None:
            colors = colors[finite]
        if confidence is not None:
            confidence = confidence[finite]
        if depth is not None:
            depth = depth[finite]
        if len(points) > max_points:
            indices = np.linspace(0, len(points) - 1, max_points).round().astype(int)
            points = points[indices]
            if colors is not None:
                colors = colors[indices]
            if confidence is not None:
                confidence = confidence[indices]
            if depth is not None:
                depth = depth[indices]
        bundle: dict[str, np.ndarray] = {"points": points.astype(np.float32)}
        if colors is not None:
            bundle["point_colors_rgb"] = colors
        if confidence is not None:
            bundle["point_confidence"] = confidence.astype(np.float32)
        if depth is not None:
            bundle["point_depth"] = depth.astype(np.float32)
        return bundle
    return None


def _prediction_point_colors(
    predictions: np.lib.npyio.NpzFile,
    point_count: int,
) -> np.ndarray | None:
    for key in ["point_colors_rgb", "colors_rgb", "rgb", "images", "image"]:
        if key not in predictions:
            continue
        colors = np.asarray(predictions[key])
        if colors.ndim > 2:
            colors = colors.reshape(-1, colors.shape[-1])
        if colors.ndim != 2 or colors.shape != (point_count, 3):
            continue
        colors = colors.astype(np.float32)
        if not np.isfinite(colors).all():
            continue
        if colors.size and float(colors.max()) <= 1.0:
            colors = colors * 255.0
        return np.clip(np.rint(colors), 0, 255).astype(np.uint8)
    return None


def _prediction_point_vector(
    predictions: np.lib.npyio.NpzFile,
    names: list[str],
    point_count: int,
) -> np.ndarray | None:
    for key in names:
        if key not in predictions:
            continue
        values = np.asarray(predictions[key], dtype=np.float32).reshape(-1)
        if values.shape != (point_count,):
            continue
        if not np.isfinite(values).all():
            continue
        return values.astype(np.float32)
    return None


def _export_request_readme(video_id: str) -> str:
    return "\n".join(
        [
            "# Hugging Face VGGT Request Package",
            "",
            f"Video id: `{video_id}`",
            "",
            "Upload `frames.zip` or the files in `frames/` to a VGGT tool such as",
            "the Hugging Face VGGT Space. Convert the returned camera and point",
            "outputs into this repository's VGGT prediction bundle format:",
            "",
            "- `metadata.json`",
            "- `cameras.npz` with `camera_centers`, `quaternions_xyzw`,",
            "  `valid_pose_mask`, and `pose_confidence`",
            "- optional `points.npz` with `points` plus aligned",
            "  `point_colors_rgb`, `point_confidence`, and `point_depth` when",
            "  the VGGT/export source provides them",
            "",
            "Safety boundaries: local-only frames, no geolocation, no meters, no",
            "speed/standoff/dive-angle claims, and no tactical interpretation.",
            "",
        ]
    )


def _provenance_warnings(metadata: BundleMetadata) -> list[str]:
    warnings: list[str] = []
    text_parts = [
        metadata.source_url_or_repo or "",
        metadata.source_commit_or_version or "",
        metadata.export_notes or "",
    ]
    secret_pattern = re.compile(r"(api[_-]?key|token|secret|sk-[A-Za-z0-9])", re.IGNORECASE)
    if any(secret_pattern.search(part) for part in text_parts):
        warnings.append("provenance contains secret-looking text")
    if metadata.coordinate_frame.lower().find("relative") == -1:
        warnings.append("coordinate_frame should explicitly state relative coordinates")
    return warnings
