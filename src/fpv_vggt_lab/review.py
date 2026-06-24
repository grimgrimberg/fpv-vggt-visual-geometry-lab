from __future__ import annotations

import json
from pathlib import Path

from .export_video import export_side_by_side_mp4
from .frames import read_frame_manifest
from .geometry import summarize_bundle
from .heatmaps import generate_heatmaps_from_manifest
from .pipeline import SAFETY_WARNINGS
from .smoothing import smooth_relative_poses
from .vggt import load_bundle, validate_bundle
from .viz import render_comparison_html, render_review_html, render_run_landing_html


def run_three_clip_review(
    frame_manifests: list[Path],
    bundles: list[Path],
    output_dir: Path,
    smooth: bool = False,
    export_video: bool = False,
    generate_heatmaps: bool = False,
) -> Path:
    if len(frame_manifests) != len(bundles):
        raise ValueError("frame-manifest and bundle counts must match")
    if len(frame_manifests) < 3:
        raise ValueError("three-clip review requires at least three clips")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[Path] = []
    review_paths: list[Path] = []
    clips: list[dict[str, str | None]] = []

    for index, (frame_manifest, bundle_path) in enumerate(zip(frame_manifests, bundles)):
        report = validate_bundle(bundle_path)
        if not report.valid:
            raise ValueError(f"invalid bundle {bundle_path}: {'; '.join(report.errors)}")
        alignment_errors = bundle_frame_alignment_errors(frame_manifest, bundle_path)
        if alignment_errors:
            raise ValueError(
                f"bundle {bundle_path} does not match frame manifest {frame_manifest}: "
                + "; ".join(alignment_errors)
            )
        bundle = load_bundle(bundle_path)
        stem = f"{index + 1:02d}_{bundle.metadata.video_id}_{bundle.metadata.segment_id}"
        clip_dir = output_dir / stem
        clip_dir.mkdir(parents=True, exist_ok=True)

        summary_path = clip_dir / "summary.json"
        review_path = clip_dir / "review.html"
        summarize_bundle(bundle_path, summary_path)

        smoothing_path: Path | None = None
        smoothing_status: str | None = None
        smoothing_error: str | None = None
        if smooth:
            smoothing_path = clip_dir / "smoothed_poses.npz"
            try:
                smooth_relative_poses(
                    bundle_path=bundle_path,
                    summary_path=summary_path,
                    output=smoothing_path,
                    metadata_output=clip_dir / "smoothed_poses.json",
                )
                smoothing_status = "done"
            except Exception as exc:
                smoothing_path = None
                smoothing_status = "skipped"
                smoothing_error = str(exc)

        heatmap_manifest_path: Path | None = None
        heatmap_status: str | None = None
        heatmap_error: str | None = None
        if generate_heatmaps:
            try:
                heatmap_manifest_path = generate_heatmaps_from_manifest(
                    frame_manifest_path=frame_manifest,
                    output_dir=clip_dir / "heatmaps",
                )
                heatmap_status = "done"
            except Exception as exc:
                heatmap_manifest_path = None
                heatmap_status = "failed_soft"
                heatmap_error = str(exc)

        render_review_html(
            frame_manifest,
            bundle_path,
            summary_path,
            review_path,
            smoothing_path if smoothing_status == "done" else None,
            heatmap_manifest_path if heatmap_status == "done" else None,
        )
        summaries.append(summary_path)
        review_paths.append(review_path)

        side_by_side_video_path: Path | None = None
        side_by_side_video_status: str | None = None
        side_by_side_video_error: str | None = None
        if export_video:
            side_by_side_video_path = clip_dir / "side_by_side.mp4"
            try:
                export_side_by_side_mp4(
                    frame_manifest_path=frame_manifest,
                    bundle_path=bundle_path,
                    summary_path=summary_path,
                    output=side_by_side_video_path,
                )
                side_by_side_video_status = "done"
            except Exception as exc:
                side_by_side_video_path = None
                side_by_side_video_status = "failed_soft"
                side_by_side_video_error = str(exc)

        clips.append(
            {
                "video_id": bundle.metadata.video_id,
                "segment_id": bundle.metadata.segment_id,
                "frame_manifest": str(frame_manifest.resolve()),
                "bundle": str(bundle_path.resolve()),
                "summary": str(summary_path),
                "review_html": str(review_path),
                "side_by_side_video": str(side_by_side_video_path)
                if side_by_side_video_path
                else None,
                "side_by_side_video_status": side_by_side_video_status,
                "side_by_side_video_error": side_by_side_video_error,
                "smoothing_output": str(smoothing_path) if smoothing_path else None,
                "smoothing_status": smoothing_status,
                "smoothing_error": smoothing_error,
                "heatmap_manifest": str(heatmap_manifest_path)
                if heatmap_manifest_path
                else None,
                "heatmap_status": heatmap_status,
                "heatmap_error": heatmap_error,
            }
        )

    comparison_path = output_dir / "comparison.html"
    render_comparison_html(summaries, comparison_path, review_paths=review_paths)
    landing_path = output_dir / "index.html"
    report = {
        "status": "done",
        "warnings": SAFETY_WARNINGS,
        "comparison_html": str(comparison_path),
        "run_landing_html": str(landing_path),
        "clips": clips,
    }
    render_run_landing_html(report, landing_path)
    report_path = output_dir / "run_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report_path


def audit_three_clip_inputs(
    frame_manifests: list[Path],
    bundles: list[Path],
    output: Path,
) -> dict:
    if len(frame_manifests) != len(bundles):
        raise ValueError("frame-manifest and bundle counts must match")
    clips = []
    ready = len(frame_manifests) >= 3
    for frame_manifest, bundle_path in zip(frame_manifests, bundles):
        frame_exists = frame_manifest.exists()
        bundle_exists = bundle_path.exists()
        bundle_valid = False
        errors: list[str] = []
        metadata = None
        if not frame_exists:
            errors.append(f"frame manifest path does not exist: {frame_manifest}")
        if bundle_exists:
            report = validate_bundle(bundle_path)
            bundle_valid = report.valid
            errors.extend(report.errors)
            metadata = report.metadata
            if bundle_valid and frame_exists:
                alignment_errors = bundle_frame_alignment_errors(frame_manifest, bundle_path)
                if alignment_errors:
                    bundle_valid = False
                    errors.extend(alignment_errors)
        else:
            errors.append(f"bundle path does not exist: {bundle_path}")
        if not frame_exists or not bundle_exists or not bundle_valid:
            ready = False
        clips.append(
            {
                "frame_manifest": str(frame_manifest),
                "bundle": str(bundle_path),
                "frame_manifest_exists": frame_exists,
                "bundle_exists": bundle_exists,
                "bundle_valid": bundle_valid,
                "errors": errors,
                "metadata": metadata,
            }
        )
    report_data = {
        "ready": ready,
        "clip_count": len(frame_manifests),
        "warnings": SAFETY_WARNINGS,
        "clips": clips,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report_data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report_data


def bundle_frame_alignment_errors(frame_manifest: Path, bundle_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        manifest = read_frame_manifest(frame_manifest)
    except Exception as exc:
        return [f"could not read frame manifest: {exc}"]
    try:
        metadata = load_bundle(bundle_path).metadata
    except Exception as exc:
        return [f"could not read bundle metadata for frame alignment: {exc}"]

    if metadata.video_id != manifest.video_id:
        errors.append(
            f"bundle video_id {metadata.video_id!r} does not match frame manifest {manifest.video_id!r}"
        )
    if metadata.segment_id != manifest.segment_id:
        errors.append(
            f"bundle segment_id {metadata.segment_id!r} does not match frame manifest {manifest.segment_id!r}"
        )

    expected_indices = [frame.frame_index for frame in manifest.frames]
    if metadata.frame_indices != expected_indices:
        errors.append("frame_indices do not match frame manifest")

    expected_timestamps = [frame.timestamp_sec for frame in manifest.frames]
    if len(metadata.frame_timestamps_sec) != len(expected_timestamps) or any(
        abs(actual - expected) > 1e-6
        for actual, expected in zip(metadata.frame_timestamps_sec, expected_timestamps)
    ):
        errors.append("frame_timestamps_sec do not match frame manifest")
    return errors
