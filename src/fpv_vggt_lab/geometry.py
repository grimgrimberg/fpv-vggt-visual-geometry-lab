from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schemas import (
    PathDescriptors,
    ReconstructionSummary,
    ReliabilitySummary,
    model_to_dict,
)
from .vggt import VggtBundle, load_bundle
from .vggt import load_metadata_if_present, validate_bundle


def summarize_bundle(bundle_path: Path, output: Path | None = None) -> ReconstructionSummary:
    report = validate_bundle(bundle_path)
    if not report.valid:
        summary = failed_reconstruction_summary(bundle_path, report.errors)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(model_to_dict(summary), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return summary

    bundle = load_bundle(bundle_path)
    reliability = score_reconstruction(bundle)
    descriptors = extract_path_descriptors(bundle, reliability)

    summary = ReconstructionSummary(
        video_id=bundle.metadata.video_id,
        segment_id=bundle.metadata.segment_id,
        reliability=reliability,
        descriptors=descriptors,
        warnings=[
            "VGGT coordinates are relative, scale ambiguous, and not meters",
            *bundle.metadata.warnings,
        ],
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(model_to_dict(summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summary


def failed_reconstruction_summary(
    bundle_path: Path, errors: list[str]
) -> ReconstructionSummary:
    metadata = load_metadata_if_present(bundle_path)
    return ReconstructionSummary(
        video_id=metadata.video_id if metadata else "unknown-video",
        segment_id=metadata.segment_id if metadata else "unknown-segment",
        reliability=ReliabilitySummary(
            score=0.0,
            label="failed",
            failure_flags=["bundle_validation_failed", *errors],
            safe_to_use_for_descriptors=False,
        ),
        descriptors=PathDescriptors(
            sampled_frame_count=len(metadata.frame_indices) if metadata else 0,
            valid_pose_count=0,
            pose_jump_count=0,
        ),
        warnings=[
            "VGGT bundle validation failed; descriptors and smoothing are gated",
            *errors,
        ],
    )


def score_reconstruction(bundle: VggtBundle) -> ReliabilitySummary:
    n_frames = len(bundle.metadata.frame_indices)
    failure_flags: list[str] = []
    if n_frames == 0:
        return ReliabilitySummary(
            score=0,
            label="failed",
            failure_flags=["no_frames"],
            safe_to_use_for_descriptors=False,
        )

    valid_pose_mask = bundle.valid_pose_mask & np.isfinite(bundle.camera_centers).all(axis=1)
    valid_fraction = float(valid_pose_mask.mean())
    confidence = float(np.nanmean(bundle.pose_confidence)) if n_frames else 0.0
    jump_count = count_pose_jumps(bundle.camera_centers, valid_pose_mask)

    if valid_fraction < 1.0:
        failure_flags.append("missing_or_invalid_poses")
    if confidence < 0.5:
        failure_flags.append("low_pose_confidence")
    if jump_count:
        failure_flags.append("pose_jumps_detected")
    if bundle.points is None or len(bundle.points) == 0:
        failure_flags.append("missing_point_cloud")

    score = max(0.0, min(1.0, 0.65 * valid_fraction + 0.35 * confidence))
    if jump_count:
        score = max(0.0, score - min(0.3, 0.05 * jump_count))

    if score >= 0.8:
        label = "good"
    elif score >= 0.55:
        label = "mixed"
    elif score > 0:
        label = "poor"
    else:
        label = "failed"

    return ReliabilitySummary(
        score=round(score, 4),
        label=label,
        failure_flags=failure_flags,
        safe_to_use_for_descriptors=label in {"good", "mixed"} and not jump_count,
    )


def extract_path_descriptors(
    bundle: VggtBundle, reliability: ReliabilitySummary
) -> PathDescriptors:
    n_frames = len(bundle.metadata.frame_indices)
    valid_mask = bundle.valid_pose_mask & np.isfinite(bundle.camera_centers).all(axis=1)
    valid_centers = bundle.camera_centers[valid_mask]
    valid_count = int(len(valid_centers))
    jump_count = count_pose_jumps(bundle.camera_centers, valid_mask)

    if not reliability.safe_to_use_for_descriptors or valid_count < 2:
        return PathDescriptors(
            sampled_frame_count=n_frames,
            valid_pose_count=valid_count,
            pose_jump_count=jump_count,
        )

    steps = np.linalg.norm(np.diff(valid_centers, axis=0), axis=1)
    path_length = float(steps.sum())
    displacement = float(np.linalg.norm(valid_centers[-1] - valid_centers[0]))
    bbox_extent = np.ptp(valid_centers, axis=0)
    bbox_diag = float(np.linalg.norm(bbox_extent))
    normalized_path_length = path_length / max(bbox_diag, 1e-8)
    displacement_ratio = displacement / max(path_length, 1e-8)
    turn_angles = _turn_angles(valid_centers)

    return PathDescriptors(
        sampled_frame_count=n_frames,
        valid_pose_count=valid_count,
        normalized_path_length=round(normalized_path_length, 6),
        displacement_ratio=round(displacement_ratio, 6),
        mean_turn_angle_rad=round(float(turn_angles.mean()), 6)
        if len(turn_angles)
        else 0.0,
        max_turn_angle_rad=round(float(turn_angles.max()), 6) if len(turn_angles) else 0.0,
        pose_jump_count=jump_count,
    )


def count_pose_jumps(camera_centers: np.ndarray, valid_pose_mask: np.ndarray) -> int:
    valid_centers = camera_centers[valid_pose_mask]
    if len(valid_centers) < 3:
        return 0
    steps = np.linalg.norm(np.diff(valid_centers, axis=0), axis=1)
    median_step = float(np.median(steps))
    threshold = max(median_step * 5.0, 1e-6)
    return int(np.sum(steps > threshold))


def _turn_angles(centers: np.ndarray) -> np.ndarray:
    if len(centers) < 3:
        return np.array([], dtype=np.float32)
    vectors = np.diff(centers, axis=0)
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 1e-8
    vectors = vectors[valid]
    norms = norms[valid]
    if len(vectors) < 2:
        return np.array([], dtype=np.float32)
    unit = vectors / norms[:, None]
    dots = np.sum(unit[:-1] * unit[1:], axis=1)
    return np.arccos(np.clip(dots, -1.0, 1.0))
