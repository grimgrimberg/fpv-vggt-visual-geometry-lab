from __future__ import annotations

import html
import json
import os
from pathlib import Path

import numpy as np

from .frames import read_frame_manifest
from .heatmaps import read_heatmap_manifest
from .method_contract import expanded_method_matrix
from .schemas import model_to_dict
from .segment_qa import frame_manifest_cut_warnings
from .vggt import load_bundle


SAFETY_WARNINGS = [
    "no geolocation",
    "no meters",
    "relative VGGT frame",
    "local-only media",
]


def _json_for_script(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _html_value(value: object) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def render_review_html(
    frame_manifest_path: Path,
    bundle_path: Path,
    summary_path: Path,
    output: Path,
    smoothed_poses_path: Path | None = None,
    heatmap_manifest_path: Path | None = None,
    glb_scene_path: Path | None = None,
    segment_qa_path: Path | None = None,
) -> Path:
    manifest = read_frame_manifest(frame_manifest_path)
    bundle = load_bundle(bundle_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output_parent = output.resolve().parent
    point_payload = _point_payload(bundle)
    smoothed_payload = _smoothed_path_payload(bundle, smoothed_poses_path)
    heatmap_payload = _heatmap_payload(heatmap_manifest_path, output_parent)
    glb_scene_payload = _optional_artifact_payload(glb_scene_path, output_parent)
    pose_jump_indices = _pose_jump_indices(bundle.camera_centers)
    segment_quality_payload = _segment_quality_payload(frame_manifest_path, segment_qa_path, output_parent)
    player_diagnosis_payload = _player_diagnosis_payload(
        frame_count=len(manifest.frames),
        camera_count=len(bundle.camera_centers),
        point_stats=point_payload["pointCloudStats"],
        pose_jump_indices=pose_jump_indices,
        summary=summary,
        glb_scene=glb_scene_payload,
    )

    payload = {
        "frames": [_frame_payload(frame, output_parent) for frame in manifest.frames],
        "sourceVideo": _relative_or_uri(manifest.source_video, output_parent),
        "playbackStartSec": float(manifest.frames[0].timestamp_sec),
        "playbackEndSec": float(manifest.frames[-1].timestamp_sec),
        "cameraPath": bundle.camera_centers.tolist(),
        "cameraOrientations": bundle.quaternions_xyzw.tolist(),
        "poseJumpIndices": pose_jump_indices,
        "cameraConvention": {
            "right": [1, 0, 0],
            "visual_up": [0, -1, 0],
            "forward": [0, 0, 1],
            "label": "camera +Z forward, +X right, visual up is -Y",
        },
        **point_payload,
        **smoothed_payload,
        **heatmap_payload,
        "glbScene": glb_scene_payload,
        "sixDofStates": _six_dof_states(bundle),
        "reliabilityTimeline": [
            {
                "frame_index": frame_index,
                "timestamp_sec": timestamp,
                "pose_confidence": float(confidence),
                "valid_pose": bool(valid),
            }
            for frame_index, timestamp, confidence, valid in zip(
                bundle.metadata.frame_indices,
                bundle.metadata.frame_timestamps_sec,
                bundle.pose_confidence,
                bundle.valid_pose_mask,
            )
        ],
        "metadata": model_to_dict(bundle.metadata),
        "summary": summary,
        "playerDiagnosis": player_diagnosis_payload,
        "bundleQuality": _bundle_quality_payload(
            point_payload["pointCloudStats"],
            smoothed_payload["smoothedPathStatus"],
        ),
        "segmentQuality": segment_quality_payload,
        "methodStatus": _method_status_payload(
            summary,
            glb_scene_payload,
            summary_path=summary_path,
            output_parent=output_parent,
        ),
        "warnings": SAFETY_WARNINGS,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_review_template(payload), encoding="utf-8")
    return output


def _segment_quality_payload(
    frame_manifest_path: Path,
    segment_qa_path: Path | None,
    output_parent: Path,
) -> dict:
    sampled = frame_manifest_cut_warnings(frame_manifest_path)
    report = None
    if segment_qa_path is not None and segment_qa_path.exists():
        try:
            report = json.loads(segment_qa_path.read_text(encoding="utf-8"))
        except Exception:
            report = {"status": "failed_soft", "warnings": ["segment QA report could not be read"]}
    warnings = list(sampled.get("warnings") or [])
    if report:
        warnings.extend(str(item) for item in report.get("warnings", []) if str(item) not in warnings)
    payload = {
        "status": "needs_human_review" if warnings else "no_cut_warning",
        "sampledFrameCheck": sampled,
        "report": report,
        "warnings": warnings,
        "reportHref": None,
    }
    if segment_qa_path is not None and segment_qa_path.exists():
        payload["reportHref"] = _relative_or_uri(segment_qa_path, output_parent)
    return payload


def _optional_artifact_payload(path: Path | None, output_parent: Path) -> dict:
    if path is None:
        return {"status": "missing", "href": None}
    try:
        resolved = path.resolve()
        if not resolved.exists():
            return {"status": "missing", "href": None}
        return {
            "status": "available",
            "href": _relative_or_uri(resolved, output_parent),
            "name": resolved.name,
            "size_bytes": resolved.stat().st_size,
        }
    except Exception:
        return {"status": "failed_soft", "href": None}


def _frame_payload(frame: object, output_parent: Path) -> dict:
    payload = model_to_dict(frame)
    frame_path = Path(payload["path"]).resolve()
    try:
        payload["path"] = Path(os.path.relpath(frame_path, output_parent)).as_posix()
    except ValueError:
        payload["path"] = frame_path.as_uri()
    return payload


def _heatmap_payload(path: Path | None, output_parent: Path) -> dict:
    if path is None:
        return {
            "heatmapLayers": [],
            "heatmapStatus": "missing",
            "heatmapWarnings": ["heatmaps were not generated for this review"],
        }
    try:
        manifest = read_heatmap_manifest(path)
        rows = []
        for layer in manifest.layers:
            row = model_to_dict(layer)
            heatmap_path = Path(row["heatmap_path"]).resolve()
            source_path = Path(row["source_frame_path"]).resolve()
            try:
                row["heatmap_path"] = Path(
                    os.path.relpath(heatmap_path, output_parent)
                ).as_posix()
                row["source_frame_path"] = Path(
                    os.path.relpath(source_path, output_parent)
                ).as_posix()
            except ValueError:
                row["heatmap_path"] = heatmap_path.as_uri()
                row["source_frame_path"] = source_path.as_uri()
            rows.append(row)
        return {
            "heatmapLayers": rows,
            "heatmapStatus": "available",
            "heatmapWarnings": manifest.warnings,
        }
    except Exception:
        return {
            "heatmapLayers": [],
            "heatmapStatus": "failed_soft",
            "heatmapWarnings": [
                "heatmap manifest unavailable: failed to read or validate heatmap manifest"
            ],
        }


def _pose_jump_indices(camera_centers: np.ndarray) -> list[int]:
    centers = np.asarray(camera_centers, dtype=np.float32)
    if len(centers) < 3:
        return []
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    median = float(np.median(steps))
    if median <= 1e-9:
        return []
    mad = float(np.median(np.abs(steps - median)))
    threshold = max(median * 3.0, median + 6.0 * max(mad, 1e-9))
    return [int(index + 1) for index, step in enumerate(steps) if float(step) > threshold]


def _player_diagnosis_payload(
    frame_count: int,
    camera_count: int,
    point_stats: dict,
    pose_jump_indices: list[int],
    summary: dict,
    glb_scene: dict,
) -> dict:
    frame_pose_aligned = frame_count == camera_count and frame_count > 0
    has_points = int(point_stats.get("count") or 0) > 0
    has_rgb = bool(point_stats.get("has_rgb"))
    has_depth = bool(point_stats.get("has_depth")) or bool(point_stats.get("uses_relative_depth_fallback"))
    glb_available = glb_scene.get("status") == "available"
    player_ready = frame_pose_aligned and has_points
    failure_flags = summary.get("reliability", {}).get("failure_flags", [])
    reconstruction_gated = bool(pose_jump_indices) or "pose_jumps_detected" in failure_flags

    if player_ready and reconstruction_gated:
        verdict = "Player is now instrumented; remaining visible discontinuity is likely reconstruction continuity, not just playback polish."
    elif player_ready:
        verdict = "Player instrumentation is ready; no pose-jump gate is currently active. Inspect POV overlap for convention/export issues."
    else:
        verdict = "Player/export inputs are incomplete, so visual quality cannot be attributed to reconstruction yet."

    checks = [
        {
            "label": "frame-to-pose alignment",
            "status": "pass" if frame_pose_aligned else "needs_review",
            "detail": f"{frame_count} frames / {camera_count} camera poses",
        },
        {
            "label": "point cloud available",
            "status": "pass" if has_points else "needs_review",
            "detail": f"{int(point_stats.get('count') or 0):,} points",
        },
        {
            "label": "RGB/depth rendering data",
            "status": "pass" if has_rgb and has_depth else "fallback",
            "detail": f"rgb={has_rgb}, depth_or_fallback={has_depth}",
        },
        {
            "label": "HF-style scene artifact",
            "status": "pass" if glb_available else "missing",
            "detail": glb_scene.get("name") or "scene.glb unavailable",
        },
        {
            "label": "pose continuity gate",
            "status": "needs_review" if reconstruction_gated else "pass",
            "detail": f"{len(pose_jump_indices)} pose jumps detected",
        },
    ]

    return {
        "player_status": "ready" if player_ready else "needs_review",
        "reconstruction_status": "pose_jump_gated" if reconstruction_gated else "continuous_by_current_gate",
        "verdict": verdict,
        "checks": checks,
        "warnings": [
            "POV replay is a visual consistency diagnostic only",
            "no geolocation",
            "no meters",
            "relative VGGT frame",
        ],
    }


def _six_dof_states(bundle: object) -> list[dict]:
    frame_indices = list(getattr(bundle.metadata, "frame_indices"))
    timestamps = np.asarray(getattr(bundle.metadata, "frame_timestamps_sec"), dtype=np.float32)
    positions = np.asarray(getattr(bundle, "camera_centers"), dtype=np.float32)
    quaternions = np.asarray(getattr(bundle, "quaternions_xyzw"), dtype=np.float32)
    confidence = np.asarray(getattr(bundle, "pose_confidence"), dtype=np.float32)
    valid_mask = np.asarray(getattr(bundle, "valid_pose_mask"), dtype=bool)
    states = []
    for index, (frame_index, timestamp, position, quaternion, pose_confidence, valid) in enumerate(
        zip(frame_indices, timestamps, positions, quaternions, confidence, valid_mask)
    ):
        euler = _quaternion_to_euler_rpy(quaternion)
        velocity = _finite_difference_vector(positions, timestamps, index)
        angular_rate = _finite_difference_angle(quaternions, timestamps, index)
        state_vector = [
            float(position[0]),
            float(position[1]),
            float(position[2]),
            float(euler[0]),
            float(euler[1]),
            float(euler[2]),
        ]
        states.append(
            {
                "frame_index": int(frame_index),
                "timestamp_sec": float(timestamp),
                "position": position.tolist(),
                "quaternion_xyzw": quaternion.tolist(),
                "euler_rpy_rad": euler.tolist(),
                "state_vector": state_vector,
                "velocity_relative_per_sec": velocity.tolist(),
                "angular_rate_relative_rad_per_sec": float(angular_rate),
                "pose_confidence": float(pose_confidence),
                "valid_pose": bool(valid),
                "coordinate_convention": "camera +Z forward, +X right, visual up is -Y",
            }
        )
    return states


def _finite_difference_vector(values: np.ndarray, timestamps: np.ndarray, index: int) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros(3, dtype=np.float32)
    previous = max(0, index - 1)
    if index == 0:
        previous = 0
        current = 1
    else:
        current = index
    dt = float(timestamps[current] - timestamps[previous])
    if dt <= 1e-6:
        return np.zeros(3, dtype=np.float32)
    return ((values[current] - values[previous]) / dt).astype(np.float32)


def _finite_difference_angle(quaternions: np.ndarray, timestamps: np.ndarray, index: int) -> float:
    if len(quaternions) <= 1 or index == 0:
        return 0.0
    dt = float(timestamps[index] - timestamps[index - 1])
    if dt <= 1e-6:
        return 0.0
    previous = _normalize_quaternion(quaternions[index - 1])
    current = _normalize_quaternion(quaternions[index])
    dot = float(abs(np.dot(previous, current)))
    dot = max(-1.0, min(1.0, dot))
    angle = 2.0 * np.arccos(dot)
    return float(angle / dt)


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0:
        return np.asarray([0, 0, 0, 1], dtype=np.float32)
    return (quaternion / norm).astype(np.float32)


def _quaternion_to_euler_rpy(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quaternion(np.asarray(quaternion, dtype=np.float32))
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = 2.0 * (w * y - z * x)
    pitch = np.arcsin(max(-1.0, min(1.0, float(pitch_arg))))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.asarray([roll, pitch, yaw], dtype=np.float32)


def _point_payload(bundle: object) -> dict:
    points = getattr(bundle, "points", None)
    colors = getattr(bundle, "point_colors_rgb", None)
    confidence = getattr(bundle, "point_confidence", None)
    depth = getattr(bundle, "point_depth", None)
    if points is None:
        return {
            "pointCloud": [],
            "pointColors": [],
            "pointConfidence": [],
            "pointDepth": [],
            "pointCloudStats": {
                "count": 0,
                "has_rgb": False,
                "has_confidence": False,
                "has_depth": False,
                "uses_relative_depth_fallback": False,
            },
        }

    point_array = np.asarray(points, dtype=np.float32)
    depth_array: np.ndarray | None
    uses_relative_depth_fallback = depth is None
    if depth is None:
        if len(point_array):
            camera_centers = np.asarray(getattr(bundle, "camera_centers"), dtype=np.float32)
            origin = camera_centers[0] if len(camera_centers) else point_array.mean(axis=0)
            depth_array = np.linalg.norm(point_array - origin[None, :], axis=1).astype(np.float32)
        else:
            depth_array = np.asarray([], dtype=np.float32)
    else:
        depth_array = np.asarray(depth, dtype=np.float32)

    stats: dict[str, object] = {
        "count": int(len(point_array)),
        "has_rgb": colors is not None,
        "has_confidence": confidence is not None,
        "has_depth": depth is not None,
        "uses_relative_depth_fallback": uses_relative_depth_fallback,
    }
    if len(depth_array):
        stats["depth_min"] = float(np.nanmin(depth_array))
        stats["depth_max"] = float(np.nanmax(depth_array))
    if confidence is not None and len(confidence):
        confidence_array = np.asarray(confidence, dtype=np.float32)
        stats["confidence_min"] = float(np.nanmin(confidence_array))
        stats["confidence_max"] = float(np.nanmax(confidence_array))

    return {
        "pointCloud": point_array.tolist(),
        "pointColors": colors.tolist() if colors is not None else [],
        "pointConfidence": confidence.tolist() if confidence is not None else [],
        "pointDepth": depth_array.tolist(),
        "pointCloudStats": stats,
    }


def _smoothed_path_payload(bundle: object, smoothed_poses_path: Path | None) -> dict:
    if smoothed_poses_path is None:
        return {
            "smoothedCameraPath": [],
            "smoothedPathStatus": "missing",
            "smoothedPathMessage": "Smoothed pose path is unavailable for this clip.",
        }
    try:
        data = np.load(smoothed_poses_path, allow_pickle=False)
        if "smoothed_camera_centers" not in data:
            raise ValueError("missing smoothed_camera_centers")
        path = np.asarray(data["smoothed_camera_centers"], dtype=np.float32)
        raw = np.asarray(getattr(bundle, "camera_centers"), dtype=np.float32)
        if path.shape != raw.shape:
            raise ValueError(f"smoothed path shape {path.shape} does not match raw path {raw.shape}")
        if not np.isfinite(path).all():
            raise ValueError("smoothed path contains non-finite values")
        return {
            "smoothedCameraPath": path.tolist(),
            "smoothedPathStatus": "done",
            "smoothedPathMessage": "Smoothed pose path is available as a reliability-gated diagnostic.",
        }
    except Exception as exc:
        return {
            "smoothedCameraPath": [],
            "smoothedPathStatus": "failed_soft",
            "smoothedPathMessage": f"Smoothed pose path unavailable: {exc}",
        }


def _bundle_quality_payload(point_stats: dict, smoothed_path_status: str) -> dict:
    has_rgb = bool(point_stats.get("has_rgb"))
    has_confidence = bool(point_stats.get("has_confidence"))
    has_depth = bool(point_stats.get("has_depth"))
    uses_depth_fallback = bool(point_stats.get("uses_relative_depth_fallback"))
    notes = []
    if has_rgb:
        notes.append("RGB point colors: available.")
    else:
        notes.append("RGB point colors: unavailable; color mode uses relative-depth fallback.")
    if has_confidence:
        notes.append("Confidence filter: uses point_confidence.")
    else:
        notes.append("Confidence filter: no per-point confidence; all points pass confidence filtering.")
    if has_depth:
        notes.append("Depth filter: uses point_depth.")
    elif uses_depth_fallback:
        notes.append("Depth filter: relative-depth fallback from first camera center; not physical range.")
    else:
        notes.append("Depth filter: unavailable.")
    if smoothed_path_status == "done":
        notes.append("Smoothed pose path: available as a reliability-gated diagnostic.")
    else:
        notes.append("Smoothed pose path: unavailable on this clip.")
    return {
        "point_count": int(point_stats.get("count", 0)),
        "rgb_status": "available" if has_rgb else "unavailable",
        "confidence_status": "available" if has_confidence else "fallback-all-points",
        "depth_status": "available" if has_depth else ("relative-depth fallback" if uses_depth_fallback else "unavailable"),
        "uses_relative_depth_fallback": uses_depth_fallback,
        "smoothed_path": "available" if smoothed_path_status == "done" else "unavailable",
        "notes": notes,
    }



def _artifact_context_dirs(summary_path: Path, output_parent: Path) -> list[Path]:
    candidates: list[Path] = []
    for base in [summary_path.resolve().parent, output_parent.resolve()]:
        candidates.append(base)
        candidates.extend(parent for parent in base.parents[:6])
        for name in [
            "expanded_artifacts",
            "h100_return_extracted",
            "h100-helper-smoke-video_return",
            "h100-helper-smoke-video_h100_return_inspect",
        ]:
            candidates.append(base / name)
            candidates.append(base.parent / name)
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def _safe_json_file(path: Path | None) -> dict:
    if path is None or not path.exists() or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _find_artifact_file(name: str, roots: list[Path]) -> Path | None:
    for root in roots:
        for candidate in [root / name, root / "expanded_artifacts" / name]:
            try:
                if candidate.exists() and candidate.is_file():
                    return candidate.resolve()
            except Exception:
                continue
    return None


def _artifact_link_payload(path: Path, output_parent: Path) -> dict:
    suffix = path.suffix.lower().lstrip(".") or "file"
    payload = {
        "name": path.name,
        "href": _relative_or_uri(path, output_parent),
        "size_bytes": path.stat().st_size,
        "kind": suffix,
    }
    if suffix in {"json", "md", "txt", "log", "csv"} and path.stat().st_size <= 2_000_000:
        try:
            preview = path.read_text(encoding="utf-8", errors="replace")[:1400]
        except Exception:
            preview = ""
        if preview:
            payload["preview_text"] = preview
    return payload


def _artifact_context_payload(summary_path: Path, output_parent: Path) -> dict:
    roots = _artifact_context_dirs(summary_path, output_parent)
    report_names = [
        "method_stage_report.json",
        "import_report.json",
        "summary.json",
        "OPTIONAL_METHOD_REPORT.md",
        "METHOD_CONTRACT.md",
        "expanded_runtime_check.json",
        "quality_report.json",
        "points_ply_report.json",
        "registration_report.json",
        "trajectory_qc.json",
        "showcase_manifest.json",
        "openmvs_dense_mesh_manifest.json",
        "odm_relative_artifacts_manifest.json",
        "depth_overlay_manifest.json",
        "research_methods_manifest.json",
        "vggetr_feasibility_report.json",
        "r3_reconstruction_manifest.json",
        "lingbot_map_manifest.json",
    ]
    reports = []
    for name in report_names:
        path = _find_artifact_file(name, roots)
        if path is not None:
            reports.append(_artifact_link_payload(path, output_parent))
    method_stage_path = _find_artifact_file("method_stage_report.json", roots)
    import_report_path = _find_artifact_file("import_report.json", roots)
    h100_summary_path = None
    for root in roots:
        candidate = root / "summary.json"
        if candidate.exists() and candidate.resolve() != summary_path.resolve():
            h100_summary_path = candidate.resolve()
            break
    import_report = _safe_json_file(import_report_path)
    h100_summary = _safe_json_file(h100_summary_path)
    method_stage_report = _safe_json_file(method_stage_path)
    method_artifact_audit = list(import_report.get("method_artifact_audit") or h100_summary.get("method_artifact_audit") or [])
    expanded_artifacts = list(import_report.get("expanded_artifacts") or [])
    return {
        "roots_checked": [str(path) for path in roots[:10]],
        "reports": reports,
        "method_stage_report": method_stage_report,
        "method_artifact_audit": method_artifact_audit,
        "expanded_artifacts": expanded_artifacts,
    }


def _method_status_payload(
    summary: dict,
    glb_scene: dict,
    summary_path: Path,
    output_parent: Path,
) -> dict:
    context = _artifact_context_payload(summary_path, output_parent)
    stage_report = (
        summary.get("method_stage_report")
        or summary.get("expanded_method_stage")
        or context.get("method_stage_report")
        or {}
    )
    stage_rows = stage_report.get("stages") if isinstance(stage_report, dict) else []
    stages = {
        str(row.get("method_id")): row
        for row in stage_rows or []
        if isinstance(row, dict) and row.get("method_id")
    }
    audit_rows = list(summary.get("method_artifact_audit") or context.get("method_artifact_audit") or [])
    audit = {
        str(row.get("method_id")): row
        for row in audit_rows
        if isinstance(row, dict) and row.get("method_id")
    }
    roots = _artifact_context_dirs(summary_path, output_parent)
    label_overrides = {
        "vggt_feedforward_full": "VGGT feed-forward",
        "vggt_colmap_ba_windowed": "VGGT + COLMAP BA",
        "colmap_classical_sift_sequential": "Classical COLMAP",
        "trajectory_preprocess_evo_eval": "Trajectory / EVO QA",
        "openmvs_dense_mesh_baseline": "OpenMVS dense mesh",
        "odm_offline_photogrammetry_baseline": "ODM baseline",
        "gsplat_nerfstudio_showcase": "Nerfstudio / gSplat",
        "mast3r_dust3r_vggetr_single_run": "MASt3R / DUSt3R / VGGeTR",
        "vggetr_candidate_unresolved": "VGGeTR candidate",
        "relative_depth_overlay_optional": "Relative-depth overlays",
        "r3_relative_regression_optional": "R3 relative regression",
        "lingbot_map_streaming_optional": "LingBot-Map",
    }
    rows = []
    for method in expanded_method_matrix():
        method_id = str(method["method_id"])
        label = label_overrides.get(method_id, method_id)
        expected = list(method.get("return_contract") or [])
        found = []
        for artifact_name in expected:
            if artifact_name == "compact_bundles.zip":
                continue
            artifact_path = _find_artifact_file(str(artifact_name), roots)
            if artifact_path is not None:
                found.append(_artifact_link_payload(artifact_path, output_parent))
        stage = stages.get(method_id)
        audit_row = audit.get(method_id)
        if audit_row:
            status = str(audit_row.get("status") or "reported")
            detail = f"artifact audit: {audit_row.get('found_count', len(found))}/{audit_row.get('expected_count', len(expected))} expected files present"
        elif stage:
            status = str(stage.get("status") or "reported")
            detail = str(stage.get("reason") or stage.get("message") or stage.get("warning") or "method stage reported by the cloud run")
        elif found:
            status = "artifact_available"
            detail = f"{len(found)}/{len(expected)} expected artifact files found near this review"
        elif method_id == "vggt_feedforward_full":
            status = "available"
            detail = "this review page was rendered from the compact VGGT bundle"
        elif method_id == "gsplat_nerfstudio_showcase" and glb_scene.get("status") == "available":
            status = "available"
            detail = f"local scene artifact available: {glb_scene.get('name') or 'scene.glb'}"
        else:
            status = "missing_artifact"
            detail = "not present in this local review artifact; import the expanded method outputs before judging it here"
        rows.append(
            {
                "method_id": method_id,
                "label": label,
                "status": status,
                "detail": detail,
                "role": method.get("role", ""),
                "safe_use": method.get("safe_use", "relative diagnostic only"),
                "expected_artifacts": expected,
                "found_artifacts": found,
                "found_count": len(found),
                "expected_count": len(expected),
            }
        )
    return {
        "status": "diagnostic",
        "note": "Artifact previews report local evidence and links. They do not prove metric accuracy or world alignment.",
        "reports": context.get("reports", []),
        "methods": rows,
    }

def render_comparison_html(
    summary_paths: list[Path],
    output: Path,
    review_paths: list[Path] | None = None,
) -> Path:
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
    output.parent.mkdir(parents=True, exist_ok=True)
    output_parent = output.resolve().parent
    review_paths = review_paths or []
    comparison_payload = []
    for index, summary in enumerate(summaries):
        descriptors = summary.get("descriptors", {})
        reliability = summary.get("reliability", {})
        review_href = None
        if index < len(review_paths):
            review_href = _relative_or_uri(review_paths[index], output_parent)
        comparison_payload.append(
            {
                "video_id": summary.get("video_id", ""),
                "segment_id": summary.get("segment_id", ""),
                "review_href": review_href,
                "reliability": {
                    "label": reliability.get("label", ""),
                    "score": reliability.get("score"),
                    "failure_flags": reliability.get("failure_flags", []),
                },
                "descriptors": {
                    "normalized_path_length": descriptors.get("normalized_path_length"),
                    "displacement_ratio": descriptors.get("displacement_ratio"),
                    "pose_jump_count": descriptors.get("pose_jump_count"),
                },
            }
        )
    focus_items = _focus_queue_items(comparison_payload)
    rows = [_comparison_row(row) for row in comparison_payload]
    output.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html lang='en'><head><meta charset='utf-8'>",
                "<meta name='viewport' content='width=device-width, initial-scale=1'>",
                "<title>Three-Clip Reconstruction Comparison</title>",
                "<style>:root{--paper:#f4f1e8;--ink:#1e2421;--muted:#626b66;--line:#d8d2c4;--panel:#fffdf7;--accent:#287f7a;--accent-2:#c59b35;--danger:#b55e55;--ok:#4f8d65;--shadow:0 18px 44px rgba(31,35,31,.12)}"
                "*{box-sizing:border-box}"
                "body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:linear-gradient(180deg,#ebe6d8 0,#f7f4ec 220px,#f4f1e8 100%);color:var(--ink);letter-spacing:0}"
                "header{padding:24px 28px;background:#1f2a25;color:#f9f6ee;border-bottom:1px solid rgba(255,255,255,.12)}"
                "header h1{margin:0 0 8px 0;font-size:26px;line-height:1.12}"
                "header p{margin:0;color:#d9d1c0;font-size:13px;line-height:1.5}"
                "main{padding:20px 24px;max-width:1400px;margin:0 auto}"
                ".warning{margin:0 0 16px 0;padding:12px 14px;border:1px solid #d7c36a;background:#fff7c7;border-radius:8px;color:#4d4120}"
                ".comparison-bars{display:grid;gap:12px;margin:0 0 18px 0;padding:16px;background:rgba(255,253,247,.96);border:1px solid var(--line);border-radius:8px;box-shadow:var(--shadow)}"
                ".metric-row{display:grid;grid-template-columns:220px 1fr;gap:12px;align-items:center}"
                ".metric-label{font-size:13px;font-weight:760;color:#303833;overflow:hidden;text-overflow:ellipsis}"
                ".metric-track{display:flex;gap:7px;align-items:end;min-height:40px;border-left:1px solid var(--line);padding-left:10px}"
                ".metric-bar{min-width:48px;border:0;border-radius:4px 4px 0 0;background:var(--ok);color:#10201a;font-size:11px;font-weight:760;padding:2px;cursor:pointer}"
                ".metric-bar.secondary{background:#7a8fbd}.metric-bar.warn{background:var(--accent-2)}.metric-bar.bad{background:var(--danger);color:white}"
                ".focus-queue{margin:0 0 18px 0;padding:16px;background:rgba(255,253,247,.96);border:1px solid var(--line);border-radius:8px;box-shadow:var(--shadow)}"
                ".focus-queue h2{margin:0 0 4px 0;font-size:18px}.focus-queue p{margin:0 0 12px 0;color:var(--muted);font-size:13px;line-height:1.45}"
                ".focus-list{display:grid;gap:8px;margin:0;padding:0;list-style:none}"
                ".focus-item{display:grid;grid-template-columns:48px minmax(0,1fr) max-content;gap:12px;align-items:center;padding:10px 12px;border:1px solid var(--line);border-radius:6px;background:#f9f5ec}"
                ".focus-rank{display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;border-radius:50%;background:#1f2a25;color:#f9f6ee;font-weight:800}"
                ".focus-title{font-weight:800}.focus-reasons{display:block;margin-top:3px;color:var(--muted);font-size:12px;line-height:1.35}"
                ".focus-score{font-size:12px;font-weight:760;color:#38413d;text-align:right}"
                ".comparison-table{overflow:auto;border:1px solid var(--line);border-radius:8px;background:var(--panel);box-shadow:var(--shadow)}"
                "table{border-collapse:collapse;width:100%;min-width:900px;background:var(--panel)}"
                "td,th{border-bottom:1px solid var(--line);padding:10px 12px;text-align:left;font-size:13px}"
                "th{background:#ede7da;color:#303833;font-size:11px;text-transform:uppercase;letter-spacing:.04em}"
                "tr:last-child td{border-bottom:0}"
                "a{color:var(--accent);font-weight:700}"
                "@media(max-width:760px){main{padding:14px}.metric-row{grid-template-columns:1fr}.metric-track{border-left:0;padding-left:0}.focus-item{grid-template-columns:36px minmax(0,1fr)}}"
                "</style>",
                "</head><body>",
                "<header><h1>Three-Clip Reconstruction Comparison</h1>"
                "<p>Local-only comparison. No geolocation, no meters, relative VGGT frame, local-only media.</p>"
                "</header>",
                "<main>",
                "<p class='warning'>Diagnostics are relative VGGT-frame review signals, not physical truth claims.</p>",
                _focus_queue_html(focus_items),
                "<section id='comparison-bars' class='comparison-bars' aria-label='Cross-clip diagnostic bars'></section>",
                "<div class='comparison-table'>",
                "<table><thead><tr><th>video_id</th><th>segment_id</th>"
                "<th>reliability</th><th>score</th><th>normalized_path_length</th>"
                "<th>displacement_ratio</th><th>pose_jump_count</th><th>review</th></tr></thead><tbody>",
                *rows,
                "</tbody></table>",
                "</div>",
                "</main>",
                f"<script>const comparisonPayload = {_json_for_script(comparison_payload)};",
                f"const focusQueuePayload = {_json_for_script(focus_items)};",
                "const comparisonMetrics=[['Reliability score','reliability.score','score'],['Normalized path length','descriptors.normalized_path_length','secondary'],['Displacement ratio','descriptors.displacement_ratio','secondary'],['Pose jump count','descriptors.pose_jump_count','warn']];",
                "function metricValue(row,path){return path.split('.').reduce((acc,key)=>acc&&acc[key],row);}",
                "function barClass(row,kind){if(kind==='score'&&Number(metricValue(row,'reliability.score'))<0.5)return 'metric-bar bad'; if(kind==='warn'&&Number(metricValue(row,'descriptors.pose_jump_count'))>0)return 'metric-bar warn'; return `metric-bar ${kind==='secondary'?'secondary':''}`;}",
                "function renderComparisonBars(){const root=document.getElementById('comparison-bars'); root.innerHTML=''; comparisonMetrics.forEach(([label,path,kind])=>{",
                "const values=comparisonPayload.map(row=>Number(metricValue(row,path))||0); const max=Math.max(...values,1e-6); const row=document.createElement('div'); row.className='metric-row'; row.setAttribute('data-comparison-metric',path);",
                "const name=document.createElement('div'); name.className='metric-label'; name.textContent=label; const track=document.createElement('div'); track.className='metric-track';",
                "comparisonPayload.forEach((clip,idx)=>{const value=values[idx]; const bar=document.createElement('button'); bar.type='button'; bar.className=barClass(clip,kind); bar.style.height=`${Math.max(12,Math.round((value/max)*32))}px`; bar.title=`${clip.video_id}: ${label} ${value}`; bar.textContent=Number.isFinite(value)?value.toFixed(value>=10?0:2):''; if(clip.review_href){bar.addEventListener('click',()=>{window.location.href=clip.review_href;});} track.appendChild(bar);});",
                "row.appendChild(name); row.appendChild(track); root.appendChild(row);});} renderComparisonBars();</script>",
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )
    return output


def render_run_landing_html(report: dict, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output_parent = output.resolve().parent
    clips = list(report.get("clips", []))
    payload = []
    summary_rows = []
    for clip in clips:
        summary = _read_optional_summary(clip.get("summary"))
        review_href = (
            _relative_or_uri(Path(str(clip["review_html"])), output_parent)
            if clip.get("review_html")
            else None
        )
        artifact_status = _clip_artifact_status(clip)
        if summary:
            descriptors = summary.get("descriptors", {})
            reliability = summary.get("reliability", {})
            item = {
                "video_id": summary.get("video_id", clip.get("video_id", "")),
                "segment_id": summary.get("segment_id", clip.get("segment_id", "")),
                "review_href": review_href,
                "reliability": {
                    "label": reliability.get("label", ""),
                    "score": reliability.get("score"),
                    "failure_flags": reliability.get("failure_flags", []),
                },
                "descriptors": {
                    "normalized_path_length": descriptors.get("normalized_path_length"),
                    "displacement_ratio": descriptors.get("displacement_ratio"),
                    "pose_jump_count": descriptors.get("pose_jump_count"),
                },
                "artifact_status": artifact_status,
            }
        else:
            item = {
                "video_id": clip.get("video_id", ""),
                "segment_id": clip.get("segment_id", ""),
                "review_href": review_href,
                "reliability": {"label": "missing", "score": None, "failure_flags": []},
                "descriptors": {"pose_jump_count": None},
                "artifact_status": artifact_status,
            }
        payload.append(item)
        summary_rows.append(_landing_clip_row(item, clip, output_parent))

    focus_items = _focus_queue_items(payload)
    comparison_href = (
        _relative_or_uri(Path(str(report["comparison_html"])), output_parent)
        if report.get("comparison_html")
        else None
    )
    run_report_href = _relative_or_uri(output.parent / "run_report.json", output_parent)
    warnings = report.get("warnings", SAFETY_WARNINGS)
    output.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html lang='en'><head><meta charset='utf-8'>",
                "<meta name='viewport' content='width=device-width, initial-scale=1'>",
                "<title>FPV Review Run</title>",
                "<style>:root{--paper:#f4f1e8;--ink:#1e2421;--muted:#626b66;--line:#d8d2c4;--panel:#fffdf7;--accent:#287f7a;--accent-2:#c59b35;--danger:#b55e55;--ok:#4f8d65;--shadow:0 18px 44px rgba(31,35,31,.12)}"
                "*{box-sizing:border-box}"
                "body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:linear-gradient(180deg,#ebe6d8 0,#f7f4ec 220px,#f4f1e8 100%);color:var(--ink);letter-spacing:0}"
                "header{padding:24px 28px;background:#1f2a25;color:#f9f6ee;border-bottom:1px solid rgba(255,255,255,.12)}"
                "header h1{margin:0 0 8px 0;font-size:28px;line-height:1.1}"
                "header p{margin:0;color:#d9d1c0;font-size:13px;line-height:1.5}"
                "main{padding:20px 24px;max-width:1400px;margin:0 auto;display:grid;gap:18px}"
                ".warning{padding:12px 14px;border:1px solid #d7c36a;background:#fff7c7;border-radius:8px;color:#4d4120}"
                ".toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center}"
                ".button{display:inline-flex;align-items:center;gap:8px;padding:9px 12px;border-radius:6px;border:1px solid #bfb6a4;background:#fffdf7;color:var(--accent);font-weight:800;text-decoration:none}"
                ".cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}"
                ".card{background:rgba(255,253,247,.96);border:1px solid var(--line);border-radius:8px;padding:14px;box-shadow:var(--shadow)}"
                ".card span{display:block;color:var(--muted);font-size:11px;font-weight:800;text-transform:uppercase}"
                ".card strong{display:block;margin-top:4px;font-size:22px}"
                ".focus-queue,.clip-table{background:rgba(255,253,247,.96);border:1px solid var(--line);border-radius:8px;padding:16px;box-shadow:var(--shadow)}"
                ".focus-queue h2,.clip-table h2{margin:0 0 4px 0;font-size:18px}"
                ".focus-queue p,.clip-table p{margin:0 0 12px 0;color:var(--muted);font-size:13px;line-height:1.45}"
                ".focus-list{display:grid;gap:8px;margin:0;padding:0;list-style:none}"
                ".focus-item{display:grid;grid-template-columns:48px minmax(0,1fr) max-content;gap:12px;align-items:center;padding:10px 12px;border:1px solid var(--line);border-radius:6px;background:#f9f5ec}"
                ".focus-rank{display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;border-radius:50%;background:#1f2a25;color:#f9f6ee;font-weight:800}"
                ".focus-title{font-weight:800}.focus-reasons{display:block;margin-top:3px;color:var(--muted);font-size:12px;line-height:1.35}"
                ".focus-score{font-size:12px;font-weight:760;color:#38413d;text-align:right}"
                ".table-wrap{overflow:auto}"
                "table{border-collapse:collapse;width:100%;min-width:900px;background:var(--panel)}"
                "td,th{border-bottom:1px solid var(--line);padding:10px 12px;text-align:left;font-size:13px}"
                "th{background:#ede7da;color:#303833;font-size:11px;text-transform:uppercase;letter-spacing:.04em}"
                "tr:last-child td{border-bottom:0}a{color:var(--accent);font-weight:800}"
                "@media(max-width:760px){main{padding:14px}.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.focus-item{grid-template-columns:36px minmax(0,1fr)}}"
                "</style></head><body>",
                "<header><h1>FPV Review Run</h1>"
                "<p>Local landing page for review artifacts, focus queue, provenance, and conservative diagnostics.</p></header>",
                "<main>",
                "<div class='warning'><strong>Safety boundary:</strong> "
                + html.escape(", ".join(_display_safety_warning(warning) for warning in warnings))
                + ". Diagnostics are relative review signals, not physical truth claims.</div>",
                "<div class='toolbar'>",
                _link_button("Open comparison", comparison_href),
                _link_button("Run report JSON", run_report_href),
                "</div>",
                "<section class='cards' aria-label='Run status cards'>",
                _landing_metric_card("Status", report.get("status", "unknown")),
                _landing_metric_card("Clips", len(clips)),
                _landing_metric_card("Review pages", sum(1 for clip in clips if clip.get("review_html"))),
                _landing_metric_card("Side-by-side MP4s", sum(1 for clip in clips if clip.get("side_by_side_video"))),
                "</section>",
                _focus_queue_html(focus_items),
                "<section class='clip-table'><h2>Artifacts</h2><p>Per-clip local outputs and diagnostic statuses.</p><div class='table-wrap'>",
                "<table><thead><tr><th>clip</th><th>reliability</th><th>score</th><th>pose jumps</th><th>heatmaps</th><th>smoothing</th><th>mp4</th><th>review</th></tr></thead><tbody>",
                *summary_rows,
                "</tbody></table></div></section>",
                "</main>",
                f"<script>const runLandingPayload = {_json_for_script(payload)};",
                f"const focusQueuePayload = {_json_for_script(focus_items)};</script>",
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )
    return output


def _comparison_row(row: dict) -> str:
    descriptors = row.get("descriptors", {})
    reliability = row.get("reliability", {})
    review_href = row.get("review_href")
    review_cell = (
        f'<a href="{html.escape(str(review_href), quote=True)}">Open review</a>'
        if review_href
        else ""
    )
    return (
        "<tr>"
        f"<td>{_html_value(row.get('video_id', ''))}</td>"
        f"<td>{_html_value(row.get('segment_id', ''))}</td>"
        f"<td>{_html_value(reliability.get('label', ''))}</td>"
        f"<td>{_html_value(reliability.get('score', ''))}</td>"
        f"<td>{_html_value(descriptors.get('normalized_path_length', ''))}</td>"
        f"<td>{_html_value(descriptors.get('displacement_ratio', ''))}</td>"
        f"<td>{_html_value(descriptors.get('pose_jump_count', ''))}</td>"
        f"<td>{review_cell}</td>"
        "</tr>"
    )


def _focus_queue_items(rows: list[dict]) -> list[dict]:
    items = []
    for row in rows:
        reliability = row.get("reliability", {})
        descriptors = row.get("descriptors", {})
        artifact_status = row.get("artifact_status", {})
        priority = 0
        reasons: list[str] = []
        score = _float_or_none(reliability.get("score"))
        label = str(reliability.get("label") or "unknown")
        if score is None:
            priority += 3
            reasons.append("missing reliability score")
        elif score < 0.5:
            priority += 4
            reasons.append("low reliability score")
        elif score < 0.75:
            priority += 2
            reasons.append("medium reliability score")
        else:
            reasons.append("high reliability; lower review priority")
        failure_flags = reliability.get("failure_flags") or []
        if failure_flags:
            priority += min(3, len(failure_flags))
            reasons.append("reliability flags: " + ", ".join(map(str, failure_flags[:3])))
        pose_jumps = _int_or_none(descriptors.get("pose_jump_count"))
        if pose_jumps and pose_jumps > 0:
            priority += 2
            reasons.append(f"pose-jump diagnostics: {pose_jumps}")
        heatmap_status = artifact_status.get("heatmaps")
        if heatmap_status == "failed_soft":
            priority += 1
            reasons.append("heatmap QA failed softly")
        smoothing_status = artifact_status.get("smoothing")
        if smoothing_status in {"skipped", "failed_soft"}:
            reasons.append("smoothing unavailable or gated")
        if not reasons:
            reasons.append("ready for standard review")
        items.append(
            {
                "video_id": row.get("video_id", ""),
                "segment_id": row.get("segment_id", ""),
                "review_href": row.get("review_href"),
                "priority": priority,
                "reliability_label": label,
                "reliability_score": score,
                "pose_jump_count": pose_jumps,
                "reasons": reasons,
            }
        )
    return sorted(
        items,
        key=lambda item: (
            -int(item["priority"]),
            float(item["reliability_score"]) if item["reliability_score"] is not None else -1.0,
            str(item["video_id"]),
        ),
    )


def _focus_queue_html(items: list[dict]) -> str:
    if not items:
        return (
            "<section id='review-focus-queue' class='focus-queue'><h2>Review Focus Queue</h2>"
            "<p>No clips were available for queueing.</p></section>"
        )
    rows = []
    for index, item in enumerate(items, start=1):
        title = f"{item.get('video_id', '')} / {item.get('segment_id', '')}"
        href = item.get("review_href")
        title_html = (
            f"<a href='{html.escape(str(href), quote=True)}'>{html.escape(title)}</a>"
            if href
            else html.escape(title)
        )
        reasons = "; ".join(str(reason) for reason in item.get("reasons", []))
        score = _format_metric_value(item.get("reliability_score"))
        rows.append(
            "<li class='focus-item'>"
            f"<span class='focus-rank'>{index}</span>"
            f"<span><span class='focus-title'>{title_html}</span>"
            f"<span class='focus-reasons'>{html.escape(reasons)}</span></span>"
            f"<span class='focus-score'>priority {item.get('priority', 0)}<br>score {html.escape(score)}</span>"
            "</li>"
        )
    return (
        "<section id='review-focus-queue' class='focus-queue'>"
        "<h2>Review Focus Queue</h2>"
        "<p>Non-operational triage using reliability, pose-jump diagnostics, and artifact readiness only.</p>"
        "<ol class='focus-list'>"
        + "".join(rows)
        + "</ol></section>"
    )


def _read_optional_summary(path_value: object) -> dict | None:
    if not path_value:
        return None
    try:
        return json.loads(Path(str(path_value)).read_text(encoding="utf-8"))
    except Exception:
        return None


def _clip_artifact_status(clip: dict) -> dict:
    return {
        "heatmaps": clip.get("heatmap_status") or "not_requested",
        "smoothing": clip.get("smoothing_status") or "not_requested",
        "side_by_side_video": clip.get("side_by_side_video_status") or "not_requested",
    }


def _landing_clip_row(item: dict, clip: dict, output_parent: Path) -> str:
    reliability = item.get("reliability", {})
    descriptors = item.get("descriptors", {})
    artifact_status = item.get("artifact_status", {})
    review_href = item.get("review_href")
    review_cell = (
        f'<a href="{html.escape(str(review_href), quote=True)}">Open review</a>'
        if review_href
        else ""
    )
    mp4_href = (
        _relative_or_uri(Path(str(clip["side_by_side_video"])), output_parent)
        if clip.get("side_by_side_video")
        else None
    )
    mp4_cell = (
        f'<a href="{html.escape(str(mp4_href), quote=True)}">Open MP4</a>'
        if mp4_href
        else _html_value(artifact_status.get("side_by_side_video", ""))
    )
    title = f"{item.get('video_id', '')} / {item.get('segment_id', '')}"
    return (
        "<tr>"
        f"<td>{html.escape(title)}</td>"
        f"<td>{_html_value(reliability.get('label', ''))}</td>"
        f"<td>{_html_value(reliability.get('score', ''))}</td>"
        f"<td>{_html_value(descriptors.get('pose_jump_count', ''))}</td>"
        f"<td>{_html_value(artifact_status.get('heatmaps', ''))}</td>"
        f"<td>{_html_value(artifact_status.get('smoothing', ''))}</td>"
        f"<td>{mp4_cell}</td>"
        f"<td>{review_cell}</td>"
        "</tr>"
    )


def _landing_metric_card(label: str, value: object) -> str:
    return (
        "<div class='card'>"
        f"<span>{html.escape(label)}</span>"
        f"<strong>{_html_value(value)}</strong>"
        "</div>"
    )


def _link_button(label: str, href: str | None) -> str:
    if not href:
        return ""
    return f'<a class="button" href="{html.escape(href, quote=True)}">{html.escape(label)}</a>'


def _display_safety_warning(value: object) -> str:
    labels = {
        "no geolocation": "No geolocation",
        "no meters": "No meters",
        "relative vggt frame": "Relative VGGT frame",
        "local-only media": "Local-only media",
    }
    text = str(value)
    return labels.get(text.lower(), text)


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _relative_or_uri(path: Path, output_parent: Path) -> str:
    resolved = path.resolve()
    try:
        return Path(os.path.relpath(resolved, output_parent)).as_posix()
    except ValueError:
        return resolved.as_uri()


def _glb_scene_links(payload: dict) -> str:
    scene = payload.get("glbScene", {})
    if scene.get("status") != "available" or not scene.get("href"):
        return (
            "<div class='artifact-links'>"
            "<p class='artifact-note'>HF-style GLB scene unavailable for this review. "
            "Run <code>fpv viz export-glb</code> to create a Model3D-style artifact.</p>"
            "</div>"
        )
    size_mb = float(scene.get("size_bytes") or 0) / 1024 / 1024
    return (
        "<div class='artifact-links'>"
        f"<a class='button' href='{html.escape(scene['href'], quote=True)}'>Open HF-style Scene GLB</a>"
        f"<span class='artifact-note'>{html.escape(scene.get('name', 'scene.glb'))} | {size_mb:.2f} MB | "
        "colored points + camera cones + jump markers; relative VGGT frame only</span>"
        "</div>"
    )


def _player_diagnosis_panel(payload: dict) -> str:
    diagnosis = payload.get("playerDiagnosis", {})
    checks = diagnosis.get("checks", [])
    check_items = "".join(
        "<li>"
        f"<span class='{html.escape(str(check.get('status', '')))}'>"
        f"{html.escape(str(check.get('status', 'unknown')))}</span> "
        f"{html.escape(str(check.get('label', 'check')))}: "
        f"{html.escape(str(check.get('detail', '')))}</li>"
        for check in checks
    )
    return (
        "<section id='player-diagnosis-panel' class='analysis-card'>"
        "<div class='section-heading'><div><h2>Player vs Reconstruction</h2>"
        "<p>Separates display/convention problems from VGGT pose-continuity problems.</p></div></div>"
        f"<p class='diagnostic-verdict'>{html.escape(str(diagnosis.get('verdict', 'No verdict available.')))}</p>"
        "<dl class='status-grid'>"
        f"<dt>Player</dt><dd>{html.escape(str(diagnosis.get('player_status', 'unknown')))}</dd>"
        f"<dt>Reconstruction</dt><dd>{html.escape(str(diagnosis.get('reconstruction_status', 'unknown')))}</dd>"
        "</dl>"
        f"<ul class='diagnostic-list'>{check_items}</ul>"
        "<dl id='player-diagnosis-current' class='status-grid'></dl>"
        "</section>"
    )


def _segment_quality_panel(payload: dict) -> str:
    quality = payload.get("segmentQuality", {})
    warnings = quality.get("warnings", [])
    report_href = quality.get("reportHref")
    items = "".join(f"<li>{html.escape(str(warning))}</li>" for warning in warnings) or "<li>No sampled-frame cut warning from this manifest.</li>"
    link = (
        f"<a class='button' href='{html.escape(str(report_href), quote=True)}'>Open Segment QA JSON</a>"
        if report_href else
        "<span class='artifact-note'>No external segment QA report linked; using sampled-frame checks.</span>"
    )
    sampled = quality.get("sampledFrameCheck", {})
    return (
        "<section id='segment-quality-panel' class='analysis-card diagnostic-card'>"
        "<div class='section-heading'><h2>Segment QA</h2>"
        "<p>Pre-VGGT cut diagnostics: title cards, end cards, smoke/static, and sampled-frame contamination.</p></div>"
        "<dl class='status-grid'>"
        f"<dt>Status</dt><dd>{html.escape(str(quality.get('status', 'unknown')))}</dd>"
        f"<dt>Checked samples</dt><dd>{html.escape(str(sampled.get('checked_frame_count', 0)))}</dd>"
        f"<dt>Flagged samples</dt><dd>{html.escape(str(sampled.get('flagged_frame_count', 0)))}</dd>"
        "</dl>"
        f"<div class='artifact-links'>{link}</div>"
        f"<ul class='diagnostic-list'>{items}</ul>"
        "</section>"
    )


def _bundle_quality_panel(payload: dict) -> str:
    quality = payload["bundleQuality"]
    notes = "".join(
        f"<li>{html.escape(note)}</li>" for note in quality.get("notes", [])
    )
    return (
        "<section id='bundle-quality-panel' class='analysis-card diagnostic-card'>"
        "<div class='section-heading'><h2>Bundle Quality</h2>"
        "<p>Array availability and fallback behavior for this imported VGGT bundle.</p></div>"
        "<dl class='status-grid'>"
        f"<dt>Point count</dt><dd>{quality['point_count']}</dd>"
        f"<dt>RGB point colors</dt><dd>{html.escape(quality['rgb_status'])}</dd>"
        f"<dt>Point confidence</dt><dd>{html.escape(quality['confidence_status'])}</dd>"
        f"<dt>Point depth</dt><dd>{html.escape(quality['depth_status'])}</dd>"
        f"<dt>Smoothed pose path</dt><dd>{html.escape(quality['smoothed_path'])}</dd>"
        "</dl>"
        f"<ul class='diagnostic-list'>{notes}</ul>"
        "</section>"
    )



def _method_status_panel(payload: dict) -> str:
    method_status = payload.get("methodStatus", {})
    rows = method_status.get("methods", [])
    report_links = []
    report_previews = []
    for report in method_status.get("reports", []):
        name = str(report.get('name', 'report'))
        report_links.append(
            "<a class='artifact-pill report-pill' "
            f"href='{html.escape(str(report.get('href', '#')), quote=True)}'>"
            f"{html.escape(name)}</a>"
        )
        preview = report.get("preview_text")
        if preview:
            report_previews.append(
                "<details class='artifact-preview report-preview'><summary>Preview "
                f"{html.escape(name)}</summary><pre>{html.escape(str(preview))}</pre></details>"
            )
    cards = []
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_class = "status-" + "".join(ch if ch.isalnum() else "-" for ch in status.lower())
        found = int(row.get("found_count") or 0)
        expected = int(row.get("expected_count") or 0)
        artifact_buttons = []
        preview_blocks = []
        for artifact in row.get("found_artifacts", []) or []:
            name = str(artifact.get("name", "artifact"))
            href = str(artifact.get("href", "#"))
            size_mb = float(artifact.get("size_bytes") or 0) / (1024 * 1024)
            artifact_buttons.append(
                "<a class='artifact-pill' "
                f"href='{html.escape(href, quote=True)}' title='{html.escape(name, quote=True)}'>"
                f"{html.escape(name)} <small>{size_mb:.1f} MB</small></a>"
            )
            preview = artifact.get("preview_text")
            if preview:
                preview_blocks.append(
                    "<details class='artifact-preview'><summary>Preview "
                    f"{html.escape(name)}</summary><pre>{html.escape(str(preview))}</pre></details>"
                )
        if not artifact_buttons:
            expected_names = ", ".join(str(name) for name in row.get("expected_artifacts", [])[:4])
            if len(row.get("expected_artifacts", [])) > 4:
                expected_names += ", ..."
            artifact_buttons.append(
                "<span class='artifact-pill missing-artifact-pill'>No local files yet"
                + (f"<small>expects {html.escape(expected_names)}</small>" if expected_names else "")
                + "</span>"
            )
        cards.append(
            "<article class='method-card'>"
            "<div class='method-card-head'>"
            f"<strong>{html.escape(str(row.get('label', row.get('method_id', 'method'))))}</strong>"
            f"<span class='method-status {status_class}'>{html.escape(status)}</span>"
            "</div>"
            f"<p class='method-detail'>{html.escape(str(row.get('detail', '')))}</p>"
            f"<p class='method-role'>{html.escape(str(row.get('role', '')))}</p>"
            f"<div class='artifact-count'>{found}/{expected} expected artifacts linked</div>"
            "<div class='artifact-pill-row'>" + "".join(artifact_buttons) + "</div>"
            + "".join(preview_blocks[:2])
            + f"<p class='artifact-safe-use'>{html.escape(str(row.get('safe_use', 'relative diagnostic only')))}</p>"
            "</article>"
        )
    reports_html = ("".join(report_links) + "".join(report_previews[:4])) or "<span class='artifact-note'>No expanded-method reports found near this page yet.</span>"
    return (
        "<section id='method-status-panel' class='analysis-card diagnostic-card method-artifacts-panel'>"
        "<div class='section-heading'><h2>Method Artifacts</h2>"
        "<p>Preview evidence for COLMAP, OpenMVS, Nerfstudio/gSplat, research baselines, and compact VGGT outputs.</p></div>"
        f"<p class='diagnostic-note'>{html.escape(str(method_status.get('note', 'Artifact presence only.')))}</p>"
        f"<div class='artifact-report-row'>{reports_html}</div>"
        f"<div class='method-card-grid'>{''.join(cards)}</div>"
        "</section>"
    )

def _review_metric_strip(payload: dict) -> str:
    summary = payload["summary"]
    reliability = summary.get("reliability", {})
    descriptors = summary.get("descriptors", {})
    score_text = _format_metric_value(reliability.get("score"))
    items = [
        (
            "Reliability",
            f"{html.escape(str(reliability.get('label', 'unknown')))} / {score_text}",
            "diagnostic, not truth",
        ),
        ("Sampled frames", str(len(payload.get("frames", []))), "deterministic review sample"),
        (
            "Point cloud",
            f"{payload.get('pointCloudStats', {}).get('count', 0):,}",
            "VGGT-relative points",
        ),
        (
            "Heatmaps",
            html.escape(str(payload.get("heatmapStatus", "missing"))),
            "image-space only",
        ),
        (
            "Path length",
            html.escape(_format_metric_value(descriptors.get("normalized_path_length"))),
            "scale-free descriptor",
        ),
    ]
    cards = []
    for label, value, note in items:
        cards.append(
            "<div class='metric-card'>"
            f"<span>{html.escape(label)}</span>"
            f"<strong>{value}</strong>"
            f"<small>{html.escape(note)}</small>"
            "</div>"
        )
    return "<div id='review-metric-strip' class='metric-strip'>" + "".join(cards) + "</div>"


def _format_metric_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float):
        return f"{float(value):.3f}".rstrip("0").rstrip(".")
    return str(value)


def _review_template(payload: dict) -> str:
    data = _json_for_script(payload)
    summary = payload["summary"]
    descriptors = summary.get("descriptors", {})
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang='en'><head><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'>",
            "<title>Relative VGGT Review</title>",
            "<style>",
            ":root{--paper:#f4f1e8;--ink:#1e2421;--muted:#626b66;--line:#d8d2c4;--panel:#fffdf7;--panel-strong:#f8f4ea;--night:#111817;--accent:#287f7a;--accent-2:#c59b35;--danger:#b55e55;--ok:#4f8d65;--shadow:0 18px 44px rgba(31,35,31,.12)}",
            "*{box-sizing:border-box}",
            "body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:linear-gradient(180deg,#ebe6d8 0,#f7f4ec 220px,#f4f1e8 100%);color:var(--ink);letter-spacing:0}",
            "body::before{content:'';position:fixed;inset:0;background:linear-gradient(90deg,rgba(40,127,122,.08),transparent 38%,rgba(197,155,53,.08));pointer-events:none}",
            "header{position:relative;padding:22px 28px 18px;background:#1f2a25;color:#f9f6ee;border-bottom:1px solid rgba(255,255,255,.12)}",
            "header h1{margin:0 0 8px 0;font-size:26px;line-height:1.12;font-weight:760}",
            "header p{margin:0;color:#d9d1c0;font-size:13px;line-height:1.5}",
            ".review-shell{position:relative;display:grid;grid-template-columns:minmax(0,1.16fr) minmax(420px,.84fr);gap:18px;padding:18px;max-width:1600px;margin:0 auto;min-width:0}",
            ".review-column{display:grid;gap:18px;align-content:start;min-width:0}",
            "section,.analysis-card{background:rgba(255,253,247,.94);border:1px solid var(--line);border-radius:8px;padding:16px;box-shadow:var(--shadow);min-width:0}",
            ".artifact-links{display:flex;flex-wrap:wrap;gap:8px;padding:0 16px 14px;align-items:center}",
            ".button{display:inline-flex;align-items:center;justify-content:center;border:1px solid #bfb6a4;background:#f7f2e8;color:var(--ink);border-radius:6px;padding:8px 11px;min-height:34px;text-decoration:none;font-size:12px;font-weight:760}",
            ".button:hover{border-color:var(--accent);background:#eef7f4}",
            ".artifact-note{margin:0;color:var(--muted);font-size:12px;line-height:1.45}",
            ".method-artifacts-panel{padding-bottom:16px}",
            ".artifact-report-row{display:flex;flex-wrap:wrap;gap:8px;margin:0 16px 14px}",
            ".method-card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;padding:0 16px}",
            ".method-card{border:1px solid var(--line);border-radius:8px;background:#f9f5ec;padding:12px;min-width:0}",
            ".method-card-head{display:flex;align-items:flex-start;justify-content:space-between;gap:10px;margin-bottom:8px}",
            ".method-card-head strong{font-size:13px;line-height:1.25;color:#222b27}",
            ".method-status{display:inline-flex;align-items:center;border-radius:999px;padding:4px 7px;font-size:10px;font-weight:820;background:#e8e0cf;color:#3a3328;white-space:nowrap}",
            ".status-available,.status-artifact-complete,.status-artifact-available,.status-primary-production{background:#dcefe3;color:#214a2c}",
            ".status-artifact-partial,.status-staged-artifacts-partial,.status-ready-pending-execution,.status-reported{background:#fff1c7;color:#6b4e12}",
            ".status-missing-artifact,.status-artifact-missing,.status-skipped-missing-dependency{background:#f4d9d6;color:#73342f}",
            ".method-detail,.method-role,.artifact-safe-use{margin:6px 0;color:#4f5b54;font-size:12px;line-height:1.42}",
            ".method-role{color:#303833;font-weight:650}",
            ".artifact-count{margin:8px 0;color:#6a645a;font-size:11px;font-weight:760;text-transform:uppercase}",
            ".artifact-pill-row{display:flex;flex-wrap:wrap;gap:7px;margin-top:8px}",
            ".artifact-pill{display:inline-flex;gap:6px;align-items:center;border:1px solid #cfc5b3;border-radius:999px;background:#fffdf7;color:#1f514e;padding:5px 8px;font-size:11px;font-weight:760;text-decoration:none;max-width:100%}",
            ".artifact-pill small{color:#746c5f;font-weight:650}",
            ".report-pill{background:#eef7f4;border-color:#abd4cc}",
            ".missing-artifact-pill{color:#746c5f;background:#f1eadf}",
            ".missing-artifact-pill small{display:block;margin-left:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:210px}",
            ".artifact-preview{margin-top:8px;border-top:1px solid var(--line);padding-top:8px}",
            ".artifact-preview summary{cursor:pointer;color:#1f514e;font-size:12px;font-weight:760}",
            ".media-card,.viz-card{padding:0;overflow:hidden;min-width:0}",
            ".section-heading{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin:0 0 12px 0;padding:16px 16px 0}",
            ".section-heading h2{margin:0;font-size:17px;line-height:1.2;font-weight:760}",
            ".section-heading p{margin:4px 0 0;color:var(--muted);font-size:12px;line-height:1.45}",
            ".safety-strip{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:12px 0 0}",
            ".safety-chip,.status-chip{display:inline-flex;align-items:center;gap:6px;border:1px solid rgba(255,255,255,.18);border-radius:999px;padding:5px 9px;font-size:12px;font-weight:650;background:rgba(255,255,255,.08);color:#f5efe3}",
            ".metric-strip{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin-top:14px}",
            ".metric-card{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.16);border-radius:8px;padding:10px 12px;min-width:0}",
            ".metric-card span{display:block;color:#c8c0ad;font-size:11px;font-weight:700;text-transform:uppercase}",
            ".metric-card strong{display:block;margin:4px 0;color:#fff;font-size:18px;line-height:1.1;overflow:hidden;text-overflow:ellipsis}",
            ".metric-card small{display:block;color:#d9d1c0;font-size:11px;line-height:1.25}",
            ".video-stage{position:relative;background:var(--night);border-top:1px solid rgba(0,0,0,.3);border-bottom:1px solid rgba(0,0,0,.3)}",
            ".video-stage video{border:0;display:block;width:100%;max-height:520px;background:#0b0f0f}",
            "#heatmap-overlay{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;pointer-events:none;display:none;mix-blend-mode:screen}",
            ".control-deck{display:grid;gap:12px;padding:14px 16px 16px;background:var(--panel)}",
            ".controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0;min-width:0}",
            ".controls button{border:1px solid #bfb6a4;background:#f7f2e8;border-radius:6px;padding:7px 10px;min-height:34px;color:var(--ink);font-size:12px;font-weight:700;cursor:pointer;max-width:100%}",
            ".controls button:hover{border-color:var(--accent);background:#eef7f4}",
            ".controls button:focus-visible,.controls select:focus-visible,.controls input:focus-visible{outline:3px solid rgba(40,127,122,.28);outline-offset:2px}",
            ".controls label{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:#38413d;font-weight:650;max-width:100%;min-width:0}",
            ".controls select{border:1px solid #bfb6a4;border-radius:6px;background:#fffdf8;padding:6px 8px;color:var(--ink);font-size:12px}",
            ".controls input[type='range']{width:148px;accent-color:var(--accent)}",
            ".wide-range{width:100%;accent-color:var(--accent)}",
            ".review-progress-label,.diagnostic-note{margin:0;color:var(--muted);font-size:12px;line-height:1.45}",
            ".frame-review{display:grid;grid-template-columns:minmax(180px,260px) 1fr;gap:12px;align-items:start;margin-top:2px}",
            ".frame-review img{width:100%;max-height:190px;object-fit:contain;border:1px solid var(--line);border-radius:6px;background:#111}",
            ".pov-replay-grid{display:grid;grid-template-columns:minmax(220px,.46fr) minmax(280px,.54fr);gap:12px;align-items:stretch}",
            ".pov-replay-grid .frame-review{display:grid;grid-template-columns:minmax(0,1fr);align-content:start;margin:0}",
            ".pov-replay-panel{display:grid;grid-template-rows:auto min-content;gap:8px;min-width:0}",
            "#pov-replay-canvas{width:100%;height:230px;border:1px solid #202a28;border-radius:6px;background:#0b0f0f}",
            ".diagnostic-verdict{margin:8px 0 10px;color:#303833;font-size:13px;line-height:1.5;font-weight:650}",
            ".diagnostic-list .pass{color:var(--ok);font-weight:760}.diagnostic-list .needs_review,.diagnostic-list .fallback{color:var(--accent-2);font-weight:760}.diagnostic-list .missing{color:var(--danger);font-weight:760}",
            "canvas{width:100%;height:380px;border:1px solid #202a28;background:#101615}",
            ".geometry-grid{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(320px,.75fr);gap:12px;align-items:start;padding:0 16px 16px}",
            ".geometry-grid canvas{height:430px;border-radius:6px;background:radial-gradient(circle at 50% 42%,#14201e 0,#0d1413 64%,#090d0c 100%)}",
            ".canvas-caption{margin:8px 0 0;color:#5e6862;font-size:12px;line-height:1.45}",
            ".status-grid,dl{display:grid;grid-template-columns:max-content 1fr;gap:6px 12px;margin:8px 0}",
            ".section-heading .status-chip{border-color:#d8d2c4;background:#eef7f4;color:#276b67}",
            ".viz-card>.controls{padding:0 16px 12px}",
            "dt{font-weight:760;color:#303833}dd{margin:0;color:#59645e;min-width:0}",
            ".diagnostic-list{margin:12px 0 0 18px;padding:0;color:#4f5b54;font-size:13px;line-height:1.5}",
            ".timeline-bars{display:flex;align-items:end;gap:3px;height:96px;margin:8px 0 10px 0;padding:10px;background:#ede7da;border:1px solid var(--line);border-radius:6px;overflow-x:auto}",
            ".reliability-bar{min-width:18px;border:0;border-radius:4px 4px 0 0;background:var(--ok);cursor:pointer;opacity:.72}",
            ".reliability-bar.invalid{background:var(--danger)}",
            ".reliability-bar.low{background:var(--accent-2)}",
            ".reliability-bar.active{outline:3px solid #27302c;opacity:1}",
            "pre{white-space:pre-wrap;font-size:12px;line-height:1.45;background:#f4efe4;border:1px solid var(--line);border-radius:6px;padding:10px;max-height:420px;overflow:auto}",
            "a{color:var(--accent);font-weight:700}",
            "@media(max-width:1100px){.review-shell{grid-template-columns:minmax(0,1fr)}.metric-strip{grid-template-columns:repeat(2,minmax(0,1fr))}.geometry-grid{grid-template-columns:minmax(0,1fr)}}",
            "@media(max-width:720px){header{padding:18px}.review-shell{padding:12px}.metric-strip{grid-template-columns:minmax(0,1fr)}.frame-review{grid-template-columns:minmax(0,1fr)}.controls input[type='range']{width:100%}.section-heading{display:block}}",
            "</style></head><body>",
            "<header><h1>Relative VGGT Review</h1>"
            "<p>Local diagnostic workbench for synchronized FPV playback, VGGT-relative geometry, heatmaps, and reconstruction reliability.</p>"
            "<div class='safety-strip'>"
            "<span class='safety-chip'>No geolocation</span>"
            "<span class='safety-chip'>No meters</span>"
            "<span class='safety-chip'>Relative VGGT frame</span>"
            "<span class='safety-chip'>Local-only media</span>"
            "</div>"
            + _review_metric_strip(payload)
            + "</header>",
            "<main class='review-shell'>",
            "<div class='review-column review-primary'>",
            "<section class='media-card'><div class='section-heading'><div><h2>FPV Playback</h2>"
            "<p>Synchronized local video, sampled frames, and image-space overlays.</p></div>"
            "<span class='status-chip'>review artifact</span></div>"
            "<div class='video-stage'>"
            f"<video id='source-video' controls preload='metadata'><source src='{html.escape(payload['sourceVideo'], quote=True)}' type='video/mp4'></video>"
            "<img id='heatmap-overlay' alt='image-space heatmap overlay'>"
            "</div>"
            "<div class='control-deck'>"
            "<div class='controls'>"
            "<button id='play-segment' type='button'>Play Segment</button>"
            "<button id='segment-start' type='button'>Segment Start</button>"
            "<button id='segment-end' type='button'>Segment End</button>"
            "<button id='prev-sample' type='button'>Prev Sample</button>"
            "<button id='next-sample' type='button'>Next Sample</button>"
            "<button id='sync-video-frame' type='button'>Sync To Sample</button>"
            "<label><input id='sync-slider-video' type='checkbox' checked> Sync scrubber</label>"
            "<label><input id='loop-segment' type='checkbox'> Loop segment</label>"
            "<label>Rate <select id='playback-rate'>"
            "<option value='0.25'>0.25x</option><option value='0.5'>0.5x</option>"
            "<option value='1' selected>1x</option><option value='1.5'>1.5x</option><option value='2'>2x</option>"
            "</select></label>"
            "<label>Heatmap <select id='heatmap-layer-select'>"
            "<option value='none'>None</option>"
            "<option value='optical_flow'>Optical flow</option>"
            "<option value='frame_difference'>Frame difference</option>"
            "<option value='blur'>Focus/edges</option>"
            "<option value='visibility_change'>Visibility change</option>"
            "</select></label>"
            "<label>Heatmap opacity <input id='heatmap-opacity' type='range' min='0' max='100' step='1' value='62'></label>"
            "<label>POV <select id='pov-convention'>"
            "<option value='auto' selected>Auto</option>"
            "<option value='vggt-z-forward'>VGGT +Z forward</option>"
            "<option value='opengl-neg-z'>OpenGL -Z check</option>"
            "<option value='flip-up'>Flip-up check</option>"
            "<option value='position-only'>Position-only debug only</option>"
            "</select></label>"
            "<label>POV points <input id='pov-point-budget' type='range' min='500' max='20000' step='500' value='5000'></label>"
            "<label>POV focal <input id='pov-focal' type='range' min='140' max='820' step='10' value='360'></label>"
            "</div>"
            "<input id='video-progress' class='wide-range' type='range' min='0' value='0' step='0.01'>"
            "<p id='video-time-label' class='review-progress-label'></p>"
            "<p id='heatmap-status-label' class='diagnostic-note'>image-space heatmaps only; no geolocation, no meters</p>"
            "<input id='frame-slider' class='wide-range' type='range' min='0' value='0'>"
            "<div class='pov-replay-grid'>"
            "<div class='frame-review'><img id='frame-image' alt='sampled frame'><p id='frame-label' class='review-progress-label'></p></div>"
            "<div class='pov-replay-panel'><canvas id='pov-replay-canvas' width='720' height='405'></canvas>"
            "<p id='pov-status-label' class='diagnostic-note'>VGGT POV replay is a visual consistency diagnostic only.</p></div>"
            "</div>"
            "</div></section>",
            "<section class='viz-card'><div class='section-heading'><div><h2>3D Relative Geometry</h2>"
            "<p>Orbitable VGGT-frame path, point cloud, camera frustums, and retrospective 6DoF glyph.</p></div>"
            "<span class='status-chip'>relative coordinates</span></div>"
            + _glb_scene_links(payload)
            + "<div class='controls'>"
            "<button id='animate-drone' type='button'>Animate Drone</button>"
            "<button id='reset-orbit' type='button'>Reset View</button>"
            "<button id='fit-scene' type='button'>Fit Scene</button>"
            "<button id='view-top' type='button'>Top</button>"
            "<button id='view-side' type='button'>Side</button>"
            "<button id='view-chase' type='button'>Chase</button>"
            "<label><input id='show-points' type='checkbox' checked> Points</label>"
            "<label><input id='show-frustums' type='checkbox' checked> Frustums</label>"
            "<label><input id='use-smoothed-path' type='checkbox' disabled> Smoothed pose path</label>"
            "<label><input id='presentation-smooth-path' type='checkbox'> Display smoother</label>"
            "<label>View frame <select id='view-transform'>"
            "<option value='vggt-raw' selected>Raw VGGT</option>"
            "<option value='z-up'>Z-up check</option>"
            "<option value='invert-z'>Invert Z check</option>"
            "<option value='swap-yz'>Swap Y/Z check</option>"
            "<option value='flip-yz'>Flip Y/Z check</option>"
            "</select></label>"
            "<label>Color <select id='point-color-mode'>"
            "<option value='rgb'>RGB / fallback</option>"
            "<option value='confidence'>Confidence</option>"
            "<option value='depth'>Depth</option>"
            "</select></label>"
            "<label>Point budget <input id='point-budget' type='range' min='500' max='50000' step='500' value='8000'></label>"
            "<label>Min confidence <input id='confidence-min' type='range' min='0' max='100' step='1' value='0'></label>"
            "<label>Max relative depth <input id='relative-depth-max' type='range' min='1' max='1000' step='1' value='1000'></label>"
            "</div>"
            "<div class='geometry-grid'>"
            "<div><canvas id='geometry' width='720' height='420'></canvas>"
            "<p id='view-transform-note' class='canvas-caption'>Drag to orbit, wheel to zoom. View-frame presets are display-only checks, not world axes.</p></div>"
            "<div id='sixdof-drone-animation-panel'><canvas id='drone-state-canvas' width='520' height='420'></canvas>"
            "<p class='canvas-caption'>State-space drone animation from the same 6DoF diagnostic vector; orientation is VGGT-relative only.</p></div>"
            "</div></section>",
            "</div><div class='review-column review-secondary'>",
            "<section id='sixdof-state-panel' class='analysis-card'><div class='section-heading'><div><h2>6DoF State-Space Diagnostic</h2>"
            "<p>Relative VGGT-frame state vector only; not physical drone dynamics.</p></div></div>"
            "<dl id='sixdof-state-current'></dl></section>",
            _segment_quality_panel(payload),
            _bundle_quality_panel(payload),
            _player_diagnosis_panel(payload),
            _method_status_panel(payload),
            "<section id='provenance-panel' class='analysis-card'><div class='section-heading'><div><h2>Provenance</h2>"
            "<p>Source bundle metadata carried into this local review page.</p></div></div><pre>"
            f"{html.escape(json.dumps(payload['metadata'], indent=2))}</pre></section>",
            "<section id='reliability-timeline' class='analysis-card'><div class='section-heading'><div><h2>Reliability Timeline</h2>"
            "<p>Per-sample pose validity and confidence, used as diagnostic context.</p></div></div>"
            "<div id='reliability-bars' class='timeline-bars' aria-label='Frame reliability timeline'></div>"
            "<p id='reliability-label'></p>"
            "<pre>"
            f"{html.escape(json.dumps(payload['reliabilityTimeline'], indent=2))}</pre></section>",
            "<section id='descriptor-panel' class='analysis-card'><div class='section-heading'><div><h2>Scale-Free Descriptors</h2>"
            "<p>Conservative relative descriptors only; no physical units.</p></div></div><pre>"
            f"{html.escape(json.dumps(descriptors, indent=2))}</pre></section>",
            "<section class='analysis-card'><div class='section-heading'><div><h2>Safety Warnings</h2>"
            "<p>These labels apply to every interpretation of this review artifact.</p></div></div><ul class='diagnostic-list'>"
            + "".join(f"<li>{html.escape(warning)}</li>" for warning in payload["warnings"])
            + "</ul></section>",
            "</div></main>",
            f"<script>const reviewPayload = {data};",
            "const frames=reviewPayload.frames; const slider=document.getElementById('frame-slider');",
            "slider.max=Math.max(frames.length-1,0); const img=document.getElementById('frame-image');",
            "const label=document.getElementById('frame-label');",
            "const geometry=document.getElementById('geometry'); const canvas=geometry; const ctx=canvas.getContext('2d');",
            "const droneStateCanvas=document.getElementById('drone-state-canvas'); const droneCtx=droneStateCanvas.getContext('2d');",
            "const video=document.getElementById('source-video'); const playSegment=document.getElementById('play-segment');",
            "const segmentStart=document.getElementById('segment-start'); const segmentEnd=document.getElementById('segment-end');",
            "const prevSample=document.getElementById('prev-sample'); const nextSample=document.getElementById('next-sample');",
            "const syncVideoFrame=document.getElementById('sync-video-frame'); const syncSliderVideo=document.getElementById('sync-slider-video');",
            "const loopSegment=document.getElementById('loop-segment'); const playbackRate=document.getElementById('playback-rate');",
            "const videoProgress=document.getElementById('video-progress'); const videoTimeLabel=document.getElementById('video-time-label');",
            "const povCanvas=document.getElementById('pov-replay-canvas'); const povCtx=povCanvas.getContext('2d');",
            "const povConventionSelect=document.getElementById('pov-convention'); const povPointBudget=document.getElementById('pov-point-budget');",
            "const povFocal=document.getElementById('pov-focal'); const povStatusLabel=document.getElementById('pov-status-label');",
            "const heatmapOverlay=document.getElementById('heatmap-overlay'); const heatmapLayerSelect=document.getElementById('heatmap-layer-select');",
            "const heatmapOpacity=document.getElementById('heatmap-opacity'); const heatmapStatusLabel=document.getElementById('heatmap-status-label');",
            "const animateDrone=document.getElementById('animate-drone'); const resetOrbit=document.getElementById('reset-orbit');",
            "const fitScene=document.getElementById('fit-scene'); const viewTop=document.getElementById('view-top'); const viewSide=document.getElementById('view-side'); const viewChase=document.getElementById('view-chase');",
            "const showPoints=document.getElementById('show-points'); const showFrustums=document.getElementById('show-frustums'); const useSmoothedPath=document.getElementById('use-smoothed-path');",
            "const presentationSmoothPath=document.getElementById('presentation-smooth-path'); const viewTransformSelect=document.getElementById('view-transform'); const viewTransformNote=document.getElementById('view-transform-note');",
            "const pointBudget=document.getElementById('point-budget'); const pointColorMode=document.getElementById('point-color-mode');",
            "const confidenceMin=document.getElementById('confidence-min'); const relativeDepthMax=document.getElementById('relative-depth-max');",
            "const reliabilityTimeline=reviewPayload.reliabilityTimeline||[];",
            "const heatmapLayers=reviewPayload.heatmapLayers||[];",
            "const reliabilityBars=document.getElementById('reliability-bars');",
            "const reliabilityLabel=document.getElementById('reliability-label');",
            "const sixDofStateCurrent=document.getElementById('sixdof-state-current'); const playerDiagnosisCurrent=document.getElementById('player-diagnosis-current');",
            "const pointStats=reviewPayload.pointCloudStats||{};",
            "const cameraConvention=reviewPayload.cameraConvention||{right:[1,0,0],visual_up:[0,-1,0],forward:[0,0,1]};",
            "if(!pointStats.has_rgb){pointColorMode.value='depth';}",
            "if(reviewPayload.smoothedPathStatus==='done'){useSmoothedPath.disabled=false; useSmoothedPath.title='Show reliability-gated smoothed diagnostic path';} else {useSmoothedPath.title=reviewPayload.smoothedPathMessage||'Smoothed path unavailable';}",
            "videoProgress.min=String(Number(reviewPayload.playbackStartSec||0)); videoProgress.max=String(Number(reviewPayload.playbackEndSec||0)); videoProgress.value=videoProgress.min;",
            "let animationTimer=null;",
            "let orbitState={yaw:-0.65,pitch:-0.45,zoom:1,panX:0,panY:0,dragging:false,lastX:0,lastY:0};",
            "function clamp(value,min,max){return Math.max(min,Math.min(max,value));}",
            "function resetOrbitState(){orbitState={yaw:-0.65,pitch:-0.45,zoom:1,panX:0,panY:0,dragging:false,lastX:0,lastY:0}; update();}",
            "function setViewPreset(yaw,pitch,zoom=1){orbitState={yaw,pitch,zoom,panX:0,panY:0,dragging:false,lastX:0,lastY:0}; update();}",
            "function chaseActiveView(){const idx=Number(slider.value)||0; const state=(reviewPayload.sixDofStates||[])[idx]; const q=(state&&state.quaternion_xyzw)||[0,0,0,1]; const f=transformVectorForView(rotateByQuaternion(cameraConvention.forward||[0,0,1],q)); const yaw=Math.atan2(f[0],f[2]||1e-6); setViewPreset(yaw,-0.28,1.25);}",
            "function rotateByQuaternion(v,q){const x=q[0],y=q[1],z=q[2],w=q[3]; const uv=[y*v[2]-z*v[1], z*v[0]-x*v[2], x*v[1]-y*v[0]]; const uuv=[y*uv[2]-z*uv[1], z*uv[0]-x*uv[2], x*uv[1]-y*uv[0]]; return [v[0]+2*(w*uv[0]+uuv[0]), v[1]+2*(w*uv[1]+uuv[1]), v[2]+2*(w*uv[2]+uuv[2])];}",
            "const VIEW_TRANSFORMS={",
            "'vggt-raw':{label:'Raw VGGT frame',matrix:[[1,0,0],[0,1,0],[0,0,1]]},",
            "'z-up':{label:'Z-up display check',matrix:[[1,0,0],[0,0,1],[0,-1,0]]},",
            "'invert-z':{label:'Invert Z display check',matrix:[[1,0,0],[0,1,0],[0,0,-1]]},",
            "'swap-yz':{label:'Swap Y/Z display check',matrix:[[1,0,0],[0,0,1],[0,1,0]]},",
            "'flip-yz':{label:'Flip Y/Z display check',matrix:[[1,0,0],[0,0,-1],[0,-1,0]]}",
            "};",
            "function currentViewTransform(){return VIEW_TRANSFORMS[viewTransformSelect.value]||VIEW_TRANSFORMS['vggt-raw'];}",
            "function transformVectorForView(v){const m=currentViewTransform().matrix; return [m[0][0]*v[0]+m[0][1]*v[1]+m[0][2]*v[2],m[1][0]*v[0]+m[1][1]*v[1]+m[1][2]*v[2],m[2][0]*v[0]+m[2][1]*v[1]+m[2][2]*v[2]];}",
            "function transformPointForView(p){return transformVectorForView([Number(p[0])||0,Number(p[1])||0,Number(p[2])||0]);}",
            "function transformPathForView(path){return (path||[]).map(transformPointForView);}",
            "function chaikinSmoothPath(points,iterations=2){let out=(points||[]).map(p=>[p[0],p[1],p[2]]); for(let k=0;k<iterations;k++){if(out.length<3)break; const next=[out[0]]; for(let i=0;i<out.length-1;i++){const a=out[i], b=out[i+1]; next.push([a[0]*.75+b[0]*.25,a[1]*.75+b[1]*.25,a[2]*.75+b[2]*.25]); next.push([a[0]*.25+b[0]*.75,a[1]*.25+b[1]*.75,a[2]*.25+b[2]*.75]);} next.push(out[out.length-1]); out=next;} return out;}",
            "function updateViewTransformNote(){const vt=currentViewTransform(); viewTransformNote.textContent=`${vt.label} | display-only axis/convention check | drag to orbit, wheel to zoom | no meters`;}",
            "function quantile(sorted,q){if(!sorted.length)return 0; const pos=(sorted.length-1)*q; const lo=Math.floor(pos), hi=Math.ceil(pos); const mix=pos-lo; return sorted[lo]*(1-mix)+sorted[hi]*mix;}",
            "function robustMinMax(values){const sorted=values.filter(Number.isFinite).sort((a,b)=>a-b); if(sorted.length<8)return [Math.min(...values),Math.max(...values)]; return [quantile(sorted,.03),quantile(sorted,.97)];}",
            "function sceneBounds(points){if(!points.length)points=[[0,0,0]]; const xs=points.map(p=>Number(p[0])||0), ys=points.map(p=>Number(p[1])||0), zs=points.map(p=>Number(p[2])||0); const [minX,maxX]=robustMinMax(xs), [minY,maxY]=robustMinMax(ys), [minZ,maxZ]=robustMinMax(zs); const span=Math.max(maxX-minX,maxY-minY,maxZ-minZ,1e-6); return {center:[(minX+maxX)/2,(minY+maxY)/2,(minZ+maxZ)/2], span, scale:Math.min(canvas.width,canvas.height)*0.86/span, cameraDistance:span*2.35};}",
            "function orbitProject(p,b){const x=(Number(p[0])||0)-b.center[0], y=(Number(p[1])||0)-b.center[1], z=(Number(p[2])||0)-b.center[2]; const cy=Math.cos(orbitState.yaw), sy=Math.sin(orbitState.yaw), cp=Math.cos(orbitState.pitch), sp=Math.sin(orbitState.pitch); const x1=cy*x+sy*z, z1=-sy*x+cy*z; const y1=cp*y-sp*z1, z2=sp*y+cp*z1; const depth=clamp((z2+b.cameraDistance)/(b.cameraDistance*2.15),.05,1.65); const persp=1/(.56+depth*.42); const s=b.scale*orbitState.zoom*persp; return [canvas.width/2+orbitState.panX+x1*s, canvas.height/2+orbitState.panY-y1*s, z2, depth, persp];}",
            "function activeCameraPath(){const smooth=reviewPayload.smoothedCameraPath||[]; if(useSmoothedPath.checked&&smooth.length===(reviewPayload.cameraPath||[]).length)return smooth; return reviewPayload.cameraPath||[];}",
            "function displayCameraPath(){const path=transformPathForView(activeCameraPath()); return presentationSmoothPath.checked?chaikinSmoothPath(path,2):path;}",
            "function activePosePositionRaw(active){const path=activeCameraPath(); const state=(reviewPayload.sixDofStates||[])[active]; return path[active]||(state&&state.position)||[0,0,0];}",
            "function activePosePosition(active){return transformPointForView(activePosePositionRaw(active));}",
            "function fmtNumber(value,digits=3){const n=Number(value); return Number.isFinite(n)?n.toFixed(digits):'n/a';}",
            "function fmtVector(values,digits=3){return (values||[]).map(v=>fmtNumber(v,digits)).join(', ');}",
            "function updateSixDofStatePanel(active){const state=(reviewPayload.sixDofStates||[])[active]; if(!state){sixDofStateCurrent.innerHTML=''; return;} const position=activePosePosition(active); const stateVector=[position[0],position[1],position[2],...(state.euler_rpy_rad||[0,0,0])]; sixDofStateCurrent.innerHTML=`<dt>Frame</dt><dd>${state.frame_index} @ ${fmtNumber(state.timestamp_sec,3)} sec</dd><dt>State vector [x y z roll pitch yaw]</dt><dd>${fmtVector(stateVector,4)}</dd><dt>Quaternion xyzw</dt><dd>${fmtVector(state.quaternion_xyzw,4)}</dd><dt>Relative motion / sec</dt><dd>${fmtVector(state.velocity_relative_per_sec,4)}</dd><dt>Angular delta / sec</dt><dd>${fmtNumber(state.angular_rate_relative_rad_per_sec,4)} rad</dd><dt>Pose confidence</dt><dd>${fmtNumber(state.pose_confidence,3)} | valid ${state.valid_pose}</dd><dt>Convention</dt><dd>${state.coordinate_convention}</dd>`;}",
            "function selectedPointRows(){const pts=reviewPayload.pointCloud||[]; const colors=reviewPayload.pointColors||[]; const conf=reviewPayload.pointConfidence||[]; const depths=reviewPayload.pointDepth||[]; const minConf=Number(confidenceMin.value)/100; const depthLimit=(Number(relativeDepthMax.value)/1000)*Number(pointStats.depth_max||0); const budget=Math.min(pts.length, Number(pointBudget.value)||8000); if(!showPoints.checked||budget<=0)return []; const step=Math.max(1,Math.ceil(pts.length/budget)); const out=[]; for(let i=0;i<pts.length;i+=step){const c=conf.length?Number(conf[i]):1; const d=depths.length?Number(depths[i]):0; if(conf.length&&c<minConf)continue; if(depths.length&&depthLimit>0&&d>depthLimit)continue; out.push({p:pts[i], color:colors[i]||null, confidence:c, depth:d, index:i}); if(out.length>=budget)break;} return out;}",
            "function gradientColor(value){const v=clamp(Number(value)||0,0,1); const r=Math.round(45+190*v), g=Math.round(180-110*v), b=Math.round(210-150*v); return `rgb(${r},${g},${b})`;}",
            "function colorForPoint(row){const mode=pointColorMode.value; if(mode==='rgb'&&row.color){return `rgb(${row.color[0]},${row.color[1]},${row.color[2]})`;} if(mode==='confidence'){return gradientColor(row.confidence);} const maxDepth=Number(pointStats.depth_max||1); return gradientColor(1-(row.depth/Math.max(maxDepth,1e-6)));}",
            "function vecDot(a,b){return (Number(a[0])||0)*(Number(b[0])||0)+(Number(a[1])||0)*(Number(b[1])||0)+(Number(a[2])||0)*(Number(b[2])||0);}",
            "function selectedPovRows(){const pts=reviewPayload.pointCloud||[]; const colors=reviewPayload.pointColors||[]; const conf=reviewPayload.pointConfidence||[]; const depths=reviewPayload.pointDepth||[]; const minConf=Number(confidenceMin.value)/100; const depthLimit=(Number(relativeDepthMax.value)/1000)*Number(pointStats.depth_max||0); const budget=Math.min(pts.length, Number(povPointBudget.value)||5000); if(budget<=0)return []; const step=Math.max(1,Math.ceil(pts.length/budget)); const out=[]; for(let i=0;i<pts.length;i+=step){const c=conf.length?Number(conf[i]):1; const d=depths.length?Number(depths[i]):0; if(conf.length&&c<minConf)continue; if(depths.length&&depthLimit>0&&d>depthLimit)continue; out.push({p:pts[i], color:colors[i]||null, confidence:c, depth:d}); if(out.length>=budget)break;} return out;}",
            "function povBasisForMode(state,mode){const q=(state&&state.quaternion_xyzw)||[0,0,0,1]; const r=rotateByQuaternion([1,0,0],q), upVisual=rotateByQuaternion([0,-1,0],q), upPositive=rotateByQuaternion([0,1,0],q), f=rotateByQuaternion([0,0,1],q), nf=rotateByQuaternion([0,0,-1],q); if(mode==='opengl-neg-z')return {right:r,up:upPositive,forward:nf,label:'OpenGL -Z forward check'}; if(mode==='flip-up')return {right:r,up:upPositive,forward:f,label:'VGGT +Z with +Y up check'}; if(mode==='position-only')return {right:[1,0,0],up:[0,-1,0],forward:[0,0,1],label:'Position-only debug only'}; return {right:r,up:upVisual,forward:f,label:'VGGT +Z forward'};}",
            "function povProject(row,origin,basis,width,height,focal){const p=row.p||[0,0,0]; const v=[(Number(p[0])||0)-origin[0],(Number(p[1])||0)-origin[1],(Number(p[2])||0)-origin[2]]; const x=vecDot(v,basis.right), y=vecDot(v,basis.up), z=vecDot(v,basis.forward); if(!Number.isFinite(z)||z<=1e-5)return null; const sx=width/2+(x/z)*focal, sy=height/2-(y/z)*focal; return {x:sx,y:sy,z:z,row:row};}",
            "function povProjectionStats(active,rows,mode){const states=reviewPayload.sixDofStates||[]; const state=states[active]; if(!state||!rows.length)return {mode,modeLabel:mode,visible:0,inside:0,coverage:0,score:0}; const basis=povBasisForMode(state,mode); const origin=activePosePositionRaw(active); const width=povCanvas.width, height=povCanvas.height, focal=Number(povFocal.value)||360; let visible=0, inside=0, minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity; for(const row of rows){const pr=povProject(row,origin,basis,width,height,focal); if(!pr)continue; visible++; if(pr.x>=0&&pr.x<=width&&pr.y>=0&&pr.y<=height){inside++; minX=Math.min(minX,pr.x); minY=Math.min(minY,pr.y); maxX=Math.max(maxX,pr.x); maxY=Math.max(maxY,pr.y);}} const coverage=inside>1?((maxX-minX)*(maxY-minY))/(width*height):0; return {mode,modeLabel:basis.label,visible,inside,coverage,score:inside+visible*.08+coverage*250};}",
            "function choosePovMode(active,rows){const requested=povConventionSelect.value; if(requested!=='auto')return povProjectionStats(active,rows,requested); const modes=['vggt-z-forward','opengl-neg-z','flip-up']; let best=povProjectionStats(active,rows,modes[0]); for(const mode of modes.slice(1)){const stats=povProjectionStats(active,rows,mode); if(stats.score>best.score)best=stats;} return best;}",
            "function drawPovReplay(active){const rows=selectedPovRows(); const stats=choosePovMode(active,rows); const states=reviewPayload.sixDofStates||[]; const state=states[active]; const width=povCanvas.width, height=povCanvas.height, focal=Number(povFocal.value)||360; povCtx.clearRect(0,0,width,height); povCtx.fillStyle='#0b0f0f'; povCtx.fillRect(0,0,width,height); povCtx.strokeStyle='rgba(255,255,255,.16)'; povCtx.lineWidth=1; povCtx.strokeRect(16,16,width-32,height-32); povCtx.strokeStyle='rgba(255,255,255,.18)'; povCtx.beginPath(); povCtx.moveTo(width/2-18,height/2); povCtx.lineTo(width/2+18,height/2); povCtx.moveTo(width/2,height/2-18); povCtx.lineTo(width/2,height/2+18); povCtx.stroke(); povCtx.fillStyle='#d9d9d9'; povCtx.font='12px Arial'; povCtx.fillText('VGGT POV replay | projected point cloud | no meters',14,22); if(!state){povCtx.fillText('No pose for this frame',14,44); return {...stats,projected:0};} const basis=povBasisForMode(state,stats.mode); const origin=activePosePositionRaw(active); const projected=[]; for(const row of rows){const pr=povProject(row,origin,basis,width,height,focal); if(pr&&pr.x>=-8&&pr.x<=width+8&&pr.y>=-8&&pr.y<=height+8)projected.push(pr);} projected.sort((a,b)=>b.z-a.z); for(const pr of projected){const size=clamp(2.8/Math.sqrt(Math.max(pr.z,.02)),1,3.4); povCtx.globalAlpha=.78; povCtx.fillStyle=colorForPoint(pr.row); povCtx.fillRect(pr.x,pr.y,size,size);} povCtx.globalAlpha=1; const activeJump=jumpSet().has(active); if(activeJump){povCtx.strokeStyle='#ff5a5a'; povCtx.lineWidth=5; povCtx.strokeRect(3,3,width-6,height-6); povCtx.fillStyle='#ffb8b0'; povCtx.fillText(`pose jump marker at sample ${active}`,14,44);} povCtx.fillStyle='#d9d9d9'; povCtx.fillText(`${stats.modeLabel} | visible ${stats.visible}/${rows.length} | in-frame ${stats.inside} | coverage ${(stats.coverage*100).toFixed(1)}%`,14,height-18); povStatusLabel.textContent=`POV replay: ${stats.modeLabel}; projected ${projected.length} points; active pose jump ${activeJump?'yes':'no'}.`; return {...stats,projected:projected.length,activeJump};}",
            "function updatePlayerDiagnosis(active,povStats){if(!playerDiagnosisCurrent)return; const diagnosis=reviewPayload.playerDiagnosis||{}; const checks=diagnosis.checks||[]; const failed=checks.filter(c=>c.status!=='pass').map(c=>`${c.label}: ${c.detail}`).join(' | ')||'none'; playerDiagnosisCurrent.innerHTML=`<dt>Current POV mode</dt><dd>${povStats.modeLabel||povStats.mode}</dd><dt>Projected points</dt><dd>${povStats.projected||0} in current POV, ${povStats.inside||0} inside frame</dd><dt>Coverage</dt><dd>${fmtNumber((povStats.coverage||0)*100,1)}% of POV canvas</dd><dt>Active pose jump</dt><dd>${povStats.activeJump?'yes - reconstruction discontinuity visible':'no'}</dd><dt>Non-pass checks</dt><dd>${failed}</dd>`;}",
            "function drawLine3(a,boundsPoint,b,color,width=1.5){const p0=orbitProject(a,b), p1=orbitProject(boundsPoint,b); ctx.strokeStyle=color; ctx.lineWidth=width; ctx.beginPath(); ctx.moveTo(p0[0],p0[1]); ctx.lineTo(p1[0],p1[1]); ctx.stroke();}",
            "function addVec(a,b,scale=1){return [a[0]+b[0]*scale,a[1]+b[1]*scale,a[2]+b[2]*scale];}",
            "function droneAttitudeProject(p){const yaw=-0.72,pitch=-0.36,scale=Math.min(droneStateCanvas.width,droneStateCanvas.height)*0.22; const x=Number(p[0])||0, y=Number(p[1])||0, z=Number(p[2])||0; const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch); const x1=cy*x+sy*z, z1=-sy*x+cy*z; const y1=cp*y-sp*z1; return [droneStateCanvas.width/2+x1*scale, droneStateCanvas.height*.52-y1*scale];}",
            "function drawDroneStateLine(a,b,color,width=2){const p0=droneAttitudeProject(a), p1=droneAttitudeProject(b); droneCtx.strokeStyle=color; droneCtx.lineWidth=width; droneCtx.beginPath(); droneCtx.moveTo(p0[0],p0[1]); droneCtx.lineTo(p1[0],p1[1]); droneCtx.stroke();}",
            "function drawDroneStateFrustum(q){const origin=[0,0,0], size=.7; const forward=transformVectorForView(rotateByQuaternion(cameraConvention.forward||[0,0,1],q)), right=transformVectorForView(rotateByQuaternion(cameraConvention.right||[1,0,0],q)), up=transformVectorForView(rotateByQuaternion(cameraConvention.visual_up||[0,-1,0],q)); const center=addVec(origin,forward,size*1.85); const corners=[addVec(addVec(center,right,size),up,size*.58),addVec(addVec(center,right,-size),up,size*.58),addVec(addVec(center,right,-size),up,-size*.58),addVec(addVec(center,right,size),up,-size*.58)]; corners.forEach(c=>drawDroneStateLine(origin,c,'rgba(255,235,59,.68)',1.5)); for(let i=0;i<corners.length;i++){drawDroneStateLine(corners[i],corners[(i+1)%corners.length],'#ffeb3b',2);} drawDroneStateLine(origin,addVec(origin,forward,size*2.35),'#60a8ff',3);}",
            "function drawDroneStateAnimation(active){const states=reviewPayload.sixDofStates||[]; const state=states[active]; droneCtx.clearRect(0,0,droneStateCanvas.width,droneStateCanvas.height); droneCtx.fillStyle='#111'; droneCtx.fillRect(0,0,droneStateCanvas.width,droneStateCanvas.height); droneCtx.fillStyle='#ddd'; droneCtx.font='12px Arial'; droneCtx.fillText('VGGT-frame 6DoF state-space drone | display-frame diagnostic | no meters',12,20); if(!state){droneCtx.fillText('No 6DoF state for this frame',12,44); return;} const q=state.quaternion_xyzw||[0,0,0,1]; const origin=[0,0,0]; const right=transformVectorForView(rotateByQuaternion(cameraConvention.right||[1,0,0],q)), up=transformVectorForView(rotateByQuaternion(cameraConvention.visual_up||[0,-1,0],q)), forward=transformVectorForView(rotateByQuaternion(cameraConvention.forward||[0,0,1],q)); const arm=.92; drawDroneStateFrustum(q); drawDroneStateLine(addVec(origin,right,-arm),addVec(origin,right,arm),'#ffd54a',4); drawDroneStateLine(addVec(origin,up,-arm*.72),addVec(origin,up,arm*.72),'#ffd54a',4); drawDroneStateLine(addVec(origin,forward,-arm*.55),addVec(origin,forward,arm*1.35),'#ffeb3b',5); drawDroneStateLine(origin,addVec(origin,right,1.35),'#ff5a5a',2); drawDroneStateLine(origin,addVec(origin,up,1.35),'#63d46b',2); drawDroneStateLine(origin,addVec(origin,forward,1.55),'#60a8ff',2); const c=droneAttitudeProject(origin); droneCtx.fillStyle='#ffeb3b'; droneCtx.beginPath(); droneCtx.arc(c[0],c[1],6,0,Math.PI*2); droneCtx.fill(); droneCtx.strokeStyle='#111'; droneCtx.stroke(); const e=state.euler_rpy_rad||[0,0,0]; const deg=e.map(v=>fmtNumber((Number(v)||0)*180/Math.PI,1)); const pos=activePosePosition(active); const v=state.velocity_relative_per_sec||[0,0,0]; droneCtx.fillStyle='#d9d9d9'; droneCtx.font='12px Arial'; const lines=[`frame ${state.frame_index} @ ${fmtNumber(state.timestamp_sec,3)} sec`,`display position + raw r/p/y ${fmtVector([pos[0],pos[1],pos[2],...(state.euler_rpy_rad||[0,0,0])],3)}`,`r/p/y degrees ${deg.join(', ')}`,`relative motion/sec ${fmtVector(v,3)}`,`confidence ${fmtNumber(state.pose_confidence,3)} | valid ${state.valid_pose}`,`view frame: ${currentViewTransform().label}`]; lines.forEach((line,i)=>droneCtx.fillText(line,12,droneStateCanvas.height-104+i*16));}",
            "function drawCameraFrustum(state,b,color,width=1,originOverride=null){if(!state)return; const origin=originOverride||transformPointForView(state.position||[0,0,0]); const q=state.quaternion_xyzw; const size=Math.max(b.span*0.045,0.025); const forward=transformVectorForView(rotateByQuaternion(cameraConvention.forward||[0,0,1],q)), right=transformVectorForView(rotateByQuaternion(cameraConvention.right||[1,0,0],q)), up=transformVectorForView(rotateByQuaternion(cameraConvention.visual_up||[0,-1,0],q)); const center=addVec(origin,forward,size*2.2); const corners=[addVec(addVec(center,right,size),up,size*.65),addVec(addVec(center,right,-size),up,size*.65),addVec(addVec(center,right,-size),up,-size*.65),addVec(addVec(center,right,size),up,-size*.65)]; corners.forEach(c=>drawLine3(origin,c,b,color,width)); for(let i=0;i<corners.length;i++){drawLine3(corners[i],corners[(i+1)%corners.length],b,color,width);}}",
            "function drawDroneBody(origin,q,b){const size=Math.max(b.span*0.038,0.02); const right=transformVectorForView(rotateByQuaternion(cameraConvention.right||[1,0,0],q)), up=transformVectorForView(rotateByQuaternion(cameraConvention.visual_up||[0,-1,0],q)), forward=transformVectorForView(rotateByQuaternion(cameraConvention.forward||[0,0,1],q)); const nose=addVec(origin,forward,size*1.6); const tail=addVec(origin,forward,-size*.75); const left=addVec(origin,right,-size); const rightPoint=addVec(origin,right,size); const top=addVec(origin,up,size*.75); const bottom=addVec(origin,up,-size*.75); drawLine3(left,rightPoint,b,'#ffd54a',3); drawLine3(top,bottom,b,'#ffd54a',3); drawLine3(tail,nose,b,'#ffeb3b',4); const p=orbitProject(nose,b); ctx.fillStyle='#ffeb3b'; ctx.beginPath(); ctx.arc(p[0],p[1],4,0,Math.PI*2); ctx.fill();}",
            "function drawDronePose(active,b){const state=(reviewPayload.sixDofStates||[])[active]; if(!state)return; const origin=activePosePosition(active); const q=state.quaternion_xyzw; const center=orbitProject(origin,b); drawDroneBody(origin,q,b); ctx.fillStyle='#ffeb3b'; ctx.beginPath(); ctx.arc(center[0],center[1],5,0,Math.PI*2); ctx.fill(); ctx.strokeStyle='#111'; ctx.stroke(); const axisScale=Math.max(b.span*0.07,0.035); drawLine3(origin,addVec(origin,transformVectorForView(rotateByQuaternion(cameraConvention.right||[1,0,0],q)),axisScale),b,'#ff5a5a',2); drawLine3(origin,addVec(origin,transformVectorForView(rotateByQuaternion(cameraConvention.visual_up||[0,-1,0],q)),axisScale),b,'#63d46b',2); drawLine3(origin,addVec(origin,transformVectorForView(rotateByQuaternion(cameraConvention.forward||[0,0,1],q)),axisScale),b,'#60a8ff',2);}",
            "function jumpSet(){return new Set(reviewPayload.poseJumpIndices||[]);}",
            "function drawPath(path,b,color,width,alpha=1,useJumps=true){const jumps=useJumps?jumpSet():new Set(); ctx.save(); ctx.globalAlpha=alpha; ctx.strokeStyle=color; ctx.lineWidth=width; ctx.lineJoin='round'; ctx.lineCap='round'; ctx.shadowColor=color; ctx.shadowBlur=6; ctx.beginPath(); path.forEach((p,i)=>{const q=orbitProject(p,b); if(i===0||jumps.has(i))ctx.moveTo(q[0],q[1]); else ctx.lineTo(q[0],q[1]);}); ctx.stroke(); ctx.restore();}",
            "function drawSceneGrid(b){const y=b.center[1]-b.span*.42; const step=b.span/6; ctx.save(); ctx.strokeStyle='rgba(255,255,255,.055)'; ctx.lineWidth=1; for(let i=-6;i<=6;i++){drawLine3([b.center[0]-b.span,b.center[1]+i*0,b.center[2]+i*step],[b.center[0]+b.span,y,b.center[2]+i*step],b,'rgba(255,255,255,.055)',1); drawLine3([b.center[0]+i*step,y,b.center[2]-b.span],[b.center[0]+i*step,y,b.center[2]+b.span],b,'rgba(255,255,255,.045)',1);} ctx.restore();}",
            "function drawJumpMarkers(path,b){const jumps=jumpSet(); if(!jumps.size)return; ctx.save(); ctx.strokeStyle='#ff5a5a'; ctx.fillStyle='rgba(255,90,90,.18)'; ctx.lineWidth=2; jumps.forEach(i=>{const p=path[i]; if(!p)return; const q=orbitProject(p,b); ctx.beginPath(); ctx.arc(q[0],q[1],8,0,Math.PI*2); ctx.fill(); ctx.stroke(); ctx.fillStyle='#ffb8b0'; ctx.font='11px Arial'; ctx.fillText(`jump ${i}`,q[0]+10,q[1]-8); ctx.fillStyle='rgba(255,90,90,.18)';}); ctx.restore();}",
            "function drawAxisGizmo(b){const origin=b.center, s=b.span*.12; const axes=[[[s,0,0],'#ff5a5a','+X'],[[0,s,0],'#63d46b','+Y'],[[0,0,s],'#60a8ff','+Z']]; ctx.font='11px Arial'; axes.forEach(([vec,color,label])=>{const end=addVec(origin,vec,1); drawLine3(origin,end,b,color,2); const p=orbitProject(end,b); ctx.fillStyle=color; ctx.fillText(label,p[0]+4,p[1]-4);});}",
            "function drawGeometry(active){updateViewTransformNote(); const rawPathView=transformPathForView(reviewPayload.cameraPath||[]); const basePathView=transformPathForView(activeCameraPath()); const path=displayCameraPath(); const rows=selectedPointRows().map(r=>({...r,p:transformPointForView(r.p)})); const all=rawPathView.concat(path).concat(rows.length?rows.map(r=>r.p):[[0,0,0]]); const b=sceneBounds(all); ctx.clearRect(0,0,canvas.width,canvas.height); ctx.fillStyle='#111'; ctx.fillRect(0,0,canvas.width,canvas.height); ctx.fillStyle='#ddd'; ctx.font='12px Arial'; const smoothLabel=presentationSmoothPath.checked?'display-smoothed path':(useSmoothedPath.checked?'gated smoothed path':'raw path'); ctx.fillText(`${currentViewTransform().label} | ${smoothLabel} | no meters`,12,20);",
            "drawSceneGrid(b); const sortedRows=rows.map(row=>({row,q:orbitProject(row.p,b)})).sort((a,b)=>a.q[2]-b.q[2]); for(const item of sortedRows){const q=item.q,row=item.row; const alpha=clamp(.28+q[3]*.55,.22,.86); const size=clamp(1.2+q[4]*1.45,1.1,4.2); ctx.fillStyle=colorForPoint(row); ctx.globalAlpha=alpha; ctx.beginPath(); ctx.arc(q[0],q[1],size,0,Math.PI*2); ctx.fill();} ctx.globalAlpha=1; if(useSmoothedPath.checked||presentationSmoothPath.checked){drawPath(rawPathView,b,'#6b7f86',1.2,.5,true); drawPath(path,b,'#62d97d',3.2,1,!presentationSmoothPath.checked);} else {drawPath(path,b,'#00c2ff',2.7,1,true);} drawJumpMarkers(rawPathView,b); drawAxisGizmo(b); if(showFrustums.checked){const states=reviewPayload.sixDofStates||[]; const stride=Math.max(1,Math.ceil(states.length/14)); states.forEach((state,i)=>{if(i%stride===0)drawCameraFrustum(state,b,'rgba(255,255,255,.26)',1,basePathView[i]||transformPointForView(state.position||[0,0,0]));}); drawCameraFrustum(states[active],b,'#ffeb3b',2.3,activePosePosition(active));} drawDronePose(active,b); const activeQ=orbitProject(activePosePosition(active),b); ctx.fillStyle='#f2efe6'; ctx.font='12px Arial'; ctx.fillText(`points ${rows.length}/${(reviewPayload.pointCloud||[]).length} | wheel zoom ${orbitState.zoom.toFixed(2)}x | active depth ${activeQ[3].toFixed(2)}`,12,canvas.height-14);}",
            "function reliabilityClass(item){if(!item.valid_pose)return 'invalid'; if(Number(item.pose_confidence)<0.5)return 'low'; return '';}",
            "function renderReliabilityTimeline(){reliabilityBars.innerHTML=''; reliabilityTimeline.forEach((item,idx)=>{",
            "const bar=document.createElement('button'); bar.type='button'; bar.className=`reliability-bar ${reliabilityClass(item)}`;",
            "const confidence=Math.max(0.08, Math.min(1, Number(item.pose_confidence)||0)); bar.style.height=`${Math.round(confidence*80)}px`;",
            "bar.setAttribute('data-frame-index', item.frame_index ?? ''); bar.setAttribute('aria-label', `frame ${item.frame_index}, confidence ${confidence.toFixed(2)}`);",
            "bar.title=`frame ${item.frame_index} @ ${item.timestamp_sec}s | confidence ${confidence.toFixed(2)} | valid ${item.valid_pose}`;",
            "bar.addEventListener('click',()=>{slider.value=String(idx); update();}); reliabilityBars.appendChild(bar);});}",
            "function setActiveReliabilityBar(active){Array.from(reliabilityBars.children).forEach((bar,idx)=>bar.classList.toggle('active',idx===active));",
            "const item=reliabilityTimeline[active]; reliabilityLabel.textContent=item?`reliability frame ${item.frame_index}: confidence ${Number(item.pose_confidence).toFixed(3)}, valid ${item.valid_pose}`:'';}",
            "function seekVideoToFrame(idx){const f=frames[idx]||{}; if(video&&Number.isFinite(Number(f.timestamp_sec))){video.currentTime=Number(f.timestamp_sec);}}",
            "function seekVideoTime(time){if(video&&Number.isFinite(Number(time))){video.currentTime=Number(time); updateVideoTimeLabel();}}",
            "function formatSeconds(time){const value=Number(time)||0; const minutes=Math.floor(value/60); const seconds=(value-minutes*60).toFixed(2).padStart(5,'0'); return `${minutes}:${seconds}`;}",
            "function updateVideoTimeLabel(){const start=Number(reviewPayload.playbackStartSec||0), end=Number(reviewPayload.playbackEndSec||0); const current=Number(video.currentTime||start); videoTimeLabel.textContent=`video ${formatSeconds(current)} | segment ${formatSeconds(start)}-${formatSeconds(end)} | rate ${video.playbackRate.toFixed(2)}x`; videoProgress.value=String(clamp(current,start,end));}",
            "function nearestFrameIndex(time){let best=0, bestDelta=Infinity; frames.forEach((f,idx)=>{const delta=Math.abs(Number(f.timestamp_sec)-time); if(delta<bestDelta){best=idx; bestDelta=delta;}}); return best;}",
            "function heatmapForFrame(idx,layerName){const f=frames[idx]||{}; return heatmapLayers.find(row=>row.layer_name===layerName&&Number(row.frame_index)===Number(f.frame_index));}",
            "function drawHeatmapOverlay(idx){const layerName=heatmapLayerSelect.value; if(layerName==='none'||!heatmapLayers.length){heatmapOverlay.style.display='none'; heatmapOverlay.removeAttribute('src'); return;} const row=heatmapForFrame(idx,layerName); if(!row){heatmapOverlay.style.display='none'; heatmapOverlay.removeAttribute('src'); heatmapStatusLabel.textContent='heatmap unavailable for this sampled frame'; return;} heatmapOverlay.src=row.heatmap_path; heatmapOverlay.style.opacity=String(Number(heatmapOpacity.value)/100); heatmapOverlay.style.display='block'; heatmapStatusLabel.textContent=`image-space heatmap: ${row.layer_name} | no geolocation | no meters`;}",
            "function update(options={}){const idx=Number(slider.value); const f=frames[idx]||{}; img.src=f.path||''; label.textContent=`sample ${idx+1}/${frames.length} | frame ${f.frame_index ?? ''} @ ${Number(f.timestamp_sec ?? 0).toFixed(3)} sec`; const povStats=drawPovReplay(idx); drawGeometry(idx); drawDroneStateAnimation(idx); drawHeatmapOverlay(idx); setActiveReliabilityBar(idx); updateSixDofStatePanel(idx); updatePlayerDiagnosis(idx,povStats); if(options.seekVideo&&syncSliderVideo.checked)seekVideoToFrame(idx); updateVideoTimeLabel();}",
            "function stopAnimation(){if(animationTimer){clearInterval(animationTimer); animationTimer=null; animateDrone.textContent='Animate Drone';}}",
            "function stepAnimation(){let idx=Number(slider.value)+1; if(idx>=frames.length)idx=0; slider.value=String(idx); update({seekVideo:true});}",
            "slider.addEventListener('input',()=>update({seekVideo:true})); pointBudget.addEventListener('input',()=>update()); showPoints.addEventListener('change',()=>update()); showFrustums.addEventListener('change',()=>update()); useSmoothedPath.addEventListener('change',()=>update());",
            "presentationSmoothPath.addEventListener('change',()=>update()); viewTransformSelect.addEventListener('change',()=>update());",
            "pointColorMode.addEventListener('change',()=>update()); confidenceMin.addEventListener('input',()=>update()); relativeDepthMax.addEventListener('input',()=>update()); resetOrbit.addEventListener('click',resetOrbitState);",
            "fitScene.addEventListener('click',()=>setViewPreset(-0.65,-0.45,1)); viewTop.addEventListener('click',()=>setViewPreset(0,-1.2,1.05)); viewSide.addEventListener('click',()=>setViewPreset(-1.57,-0.18,1.1)); viewChase.addEventListener('click',chaseActiveView);",
            "povConventionSelect.addEventListener('change',()=>update()); povPointBudget.addEventListener('input',()=>update()); povFocal.addEventListener('input',()=>update());",
            "heatmapLayerSelect.addEventListener('change',()=>update()); heatmapOpacity.addEventListener('input',()=>drawHeatmapOverlay(Number(slider.value)));",
            "geometry.addEventListener('pointerdown',event=>{orbitState.dragging=true; orbitState.lastX=event.clientX; orbitState.lastY=event.clientY; geometry.setPointerCapture(event.pointerId);});",
            "geometry.addEventListener('pointermove',event=>{if(!orbitState.dragging)return; const dx=event.clientX-orbitState.lastX, dy=event.clientY-orbitState.lastY; orbitState.lastX=event.clientX; orbitState.lastY=event.clientY; orbitState.yaw+=dx*0.008; orbitState.pitch=clamp(orbitState.pitch+dy*0.008,-1.35,1.35); update();});",
            "geometry.addEventListener('pointerup',event=>{orbitState.dragging=false; try{geometry.releasePointerCapture(event.pointerId);}catch(_){}});",
            "geometry.addEventListener('wheel',event=>{event.preventDefault(); orbitState.zoom=clamp(orbitState.zoom*(event.deltaY<0?1.12:.89),.25,8); update();},{passive:false});",
            "prevSample.addEventListener('click',()=>{slider.value=String(Math.max(0,Number(slider.value)-1)); update({seekVideo:true});});",
            "nextSample.addEventListener('click',()=>{slider.value=String(Math.min(frames.length-1,Number(slider.value)+1)); update({seekVideo:true});});",
            "syncVideoFrame.addEventListener('click',()=>seekVideoToFrame(Number(slider.value)));",
            "segmentStart.addEventListener('click',()=>seekVideoTime(Number(reviewPayload.playbackStartSec||0)));",
            "segmentEnd.addEventListener('click',()=>seekVideoTime(Number(reviewPayload.playbackEndSec||0)));",
            "playbackRate.addEventListener('change',()=>{video.playbackRate=Number(playbackRate.value)||1; updateVideoTimeLabel();});",
            "videoProgress.addEventListener('input',()=>seekVideoTime(Number(videoProgress.value)));",
            "function setPlaybackBlockedMessage(){videoTimeLabel.textContent='Playback is ready. Use the native video controls if this browser blocks scripted play.';}",
            "async function toggleSegmentPlayback(){if(!video)return; const start=Number(reviewPayload.playbackStartSec||0), end=Number(reviewPayload.playbackEndSec||0); if(video.paused){if(video.currentTime<start||video.currentTime>end)seekVideoTime(start); try{await video.play(); playSegment.textContent='Pause Segment';}catch(_){setPlaybackBlockedMessage();}} else {video.pause(); playSegment.textContent='Play Segment';}}",
            "playSegment.addEventListener('click',toggleSegmentPlayback);",
            "video.addEventListener('loadedmetadata',updateVideoTimeLabel);",
            "video.addEventListener('play',()=>{playSegment.textContent='Pause Segment';});",
            "video.addEventListener('pause',()=>{playSegment.textContent='Play Segment';});",
            "video.addEventListener('timeupdate',()=>{updateVideoTimeLabel(); if(syncSliderVideo.checked){const idx=nearestFrameIndex(video.currentTime); if(String(idx)!==slider.value){slider.value=String(idx); update();}} if(video.currentTime>Number(reviewPayload.playbackEndSec||Infinity)){if(loopSegment.checked){seekVideoTime(Number(reviewPayload.playbackStartSec||0)); video.play();} else {video.pause();}}});",
            "animateDrone.addEventListener('click',()=>{if(animationTimer){stopAnimation(); return;} animateDrone.textContent='Stop Animation'; animationTimer=setInterval(stepAnimation,450);});",
            "if(reviewPayload.heatmapStatus!=='available'){heatmapLayerSelect.disabled=true; heatmapOpacity.disabled=true; heatmapStatusLabel.textContent=(reviewPayload.heatmapWarnings||[]).join(' | ')||'heatmaps unavailable';}",
            "renderReliabilityTimeline(); update();</script>",
            "</body></html>",
        ]
    )
