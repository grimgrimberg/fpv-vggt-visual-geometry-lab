from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .frames import deterministic_indices, measure_frame_quality, write_frame_manifest
from .image_ops import apply_overlay_masks
from .media import find_media_record
from .pipeline import environment_snapshot
from .schemas import FrameManifest, FrameRecord, model_to_dict
from .segments import read_annotations


WINDOW_WARNINGS = [
    "no geolocation",
    "no meters",
    "relative VGGT frame",
    "local-only media",
    "window scores are reconstruction-input diagnostics, not truth claims",
]


def propose_stable_windows(
    *,
    media_inventory: Path,
    annotations: Path,
    output_dir: Path,
    frames_root: Path,
    video_ids: list[str] | None = None,
    segment_id: str | None = "segment-001",
    window_sec: float = 6.0,
    stride_sec: float = 3.0,
    candidate_limit: int = 3,
    frame_count: int = 64,
    resized_long_edge: int | None = 1024,
    eval_samples: int = 12,
    neighbor_radius: int = 2,
    mask_static_overlays: bool = True,
) -> dict[str, Any]:
    if window_sec <= 0:
        raise ValueError("window_sec must be positive")
    if stride_sec <= 0:
        raise ValueError("stride_sec must be positive")
    if candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if eval_samples <= 1:
        raise ValueError("eval_samples must be greater than 1")
    if neighbor_radius < 0:
        raise ValueError("neighbor_radius must be non-negative")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    _write_text(log_path, "")

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")

    selected_videos = set(video_ids or [])
    accepted = [row for row in read_annotations(annotations) if row.status == "accepted"]
    if selected_videos:
        accepted = [row for row in accepted if row.video_id in selected_videos]
    if segment_id:
        accepted = [row for row in accepted if row.segment_id == segment_id]

    stage_status = {
        "accepted_segments": "done" if accepted else "needs_human_review",
        "window_proposal": "pending",
        "frame_sampling": "pending",
    }
    all_candidates: list[dict[str, Any]] = []
    selected_windows: list[dict[str, Any]] = []
    failures: list[str] = []

    log("start stable-window proposal")
    if not accepted:
        summary = _proposal_summary(
            output_dir=output_dir,
            log_path=log_path,
            status="needs_human_review",
            stage_status=stage_status | {"window_proposal": "blocked", "frame_sampling": "blocked"},
            selected_windows=[],
            failures=["no accepted segments matched the requested inputs"],
            inputs=_proposal_inputs(
                media_inventory,
                annotations,
                frames_root,
                video_ids,
                segment_id,
                window_sec,
                stride_sec,
                candidate_limit,
                frame_count,
                resized_long_edge,
                eval_samples,
                neighbor_radius,
                mask_static_overlays,
            ),
        )
        _write_proposal_artifacts(output_dir, [], [], summary)
        return summary

    for annotation in accepted:
        try:
            media = find_media_record(media_inventory, annotation.video_id)
            candidates = _scan_segment_windows(
                video_path=media.local_path,
                video_id=annotation.video_id,
                source_segment_id=annotation.segment_id,
                segment_start_sec=annotation.start_sec,
                segment_end_sec=annotation.end_sec,
                fps=float(media.fps),
                total_frames=int(media.frame_count),
                window_sec=window_sec,
                stride_sec=stride_sec,
                eval_samples=eval_samples,
            )
            mask_regions = _window_mask_regions(candidates) if mask_static_overlays else []
            for candidate in candidates:
                candidate["mask_regions_norm_xyxy"] = mask_regions
                candidate["mask_static_overlays"] = bool(mask_regions)
            all_candidates.extend(candidates)
            chosen = sorted(candidates, key=lambda row: row["score"], reverse=True)[
                :candidate_limit
            ]
            for rank, candidate in enumerate(chosen, start=1):
                window_segment_id = f"{annotation.segment_id}__window-{rank:03d}"
                frame_output = (
                    frames_root
                    / "windows"
                    / output_dir.name
                    / annotation.video_id
                    / window_segment_id
                )
                manifest = sample_quality_aware_window_frames(
                    video_path=media.local_path,
                    output=frame_output,
                    video_id=annotation.video_id,
                    segment_id=window_segment_id,
                    start_frame=int(candidate["start_frame"]),
                    end_frame=int(candidate["end_frame"]),
                    count=frame_count,
                    resized_long_edge=resized_long_edge,
                    neighbor_radius=neighbor_radius,
                    mask_regions=mask_regions,
                )
                row = {
                    **candidate,
                    "rank": rank,
                    "window_segment_id": window_segment_id,
                    "frame_manifest": str(frame_output / "frames.json"),
                    "sampled_frame_count": len(manifest.frames),
                    "resized_long_edge": resized_long_edge,
                    "mask_regions_norm_xyxy": mask_regions,
                    "mask_static_overlays": bool(mask_regions),
                }
                selected_windows.append(row)
                log(
                    "sampled window "
                    f"{annotation.video_id}/{window_segment_id} "
                    f"score={candidate['score']:.3f} frames={len(manifest.frames)}"
                )
        except Exception as exc:
            failures.append(f"{annotation.video_id}/{annotation.segment_id}: {exc}")
            log(f"failed_soft {annotation.video_id}/{annotation.segment_id}: {exc}")

    stage_status["window_proposal"] = "done" if all_candidates else "failed_soft"
    stage_status["frame_sampling"] = "done" if selected_windows else "failed_soft"
    status = "done" if selected_windows else "failed_soft"
    summary = _proposal_summary(
        output_dir=output_dir,
        log_path=log_path,
        status=status,
        stage_status=stage_status,
        selected_windows=selected_windows,
        failures=failures,
        inputs=_proposal_inputs(
            media_inventory,
            annotations,
            frames_root,
            video_ids,
            segment_id,
            window_sec,
            stride_sec,
            candidate_limit,
            frame_count,
            resized_long_edge,
            eval_samples,
            neighbor_radius,
            mask_static_overlays,
        ),
    )
    _write_proposal_artifacts(output_dir, all_candidates, selected_windows, summary)
    log(f"finished stable-window proposal status={status}")
    return summary


def _window_mask_regions(candidates: list[dict[str, Any]]) -> list[list[float]]:
    if not candidates:
        return []
    ratios = [float(row.get("bottom_left_overlay_ratio_mean", 0.0) or 0.0) for row in candidates]
    if ratios and float(np.median(ratios)) > 0.006:
        return [[0.0, 0.54, 0.22, 1.0]]
    return []


def _bottom_left_overlay_ratio(frame: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = hsv.shape[:2]
    roi = hsv[int(height * 0.48) : int(height * 0.92), : int(width * 0.24)]
    if roi.size == 0:
        return 0.0
    hue = roi[:, :, 0]
    sat = roi[:, :, 1]
    val = roi[:, :, 2]
    yellow = (hue >= 16) & (hue <= 42) & (sat >= 80) & (val >= 105)
    red = ((hue <= 8) | (hue >= 170)) & (sat >= 85) & (val >= 95)
    return float(np.mean(yellow | red))


def sample_quality_aware_window_frames(
    *,
    video_path: Path,
    output: Path,
    video_id: str,
    segment_id: str,
    start_frame: int,
    end_frame: int,
    count: int,
    resized_long_edge: int | None,
    neighbor_radius: int = 2,
    mask_regions: list[list[float]] | None = None,
) -> FrameManifest:
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if end_frame < start_frame:
        raise ValueError("end_frame must be greater than or equal to start_frame")
    if count <= 0:
        raise ValueError("count must be positive")

    video_path = video_path.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 1.0
        start_frame = max(0, min(start_frame, max(total_frames - 1, 0)))
        end_frame = max(start_frame, min(end_frame, max(total_frames - 1, 0)))
        selected_indices = _quality_aware_indices(
            capture=capture,
            start_frame=start_frame,
            end_frame=end_frame,
            count=count,
            neighbor_radius=neighbor_radius,
        )
        records: list[FrameRecord] = []
        for ordinal, frame_index in enumerate(selected_indices):
            frame = _read_frame(capture, frame_index)
            frame = apply_overlay_masks(frame, mask_regions)
            frame = _resize_if_needed(frame, resized_long_edge)
            height, width = frame.shape[:2]
            frame_path = output / f"frame_{ordinal:04d}_{frame_index:06d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"could not write frame: {frame_path}")
            records.append(
                FrameRecord(
                    video_id=video_id,
                    segment_id=segment_id,
                    frame_index=int(frame_index),
                    timestamp_sec=float(frame_index / fps),
                    path=frame_path.resolve(),
                    width=width,
                    height=height,
                    resized_long_edge=resized_long_edge,
                    quality=measure_frame_quality(frame),
                )
            )
    finally:
        capture.release()

    manifest = FrameManifest(
        video_id=video_id,
        segment_id=segment_id,
        source_video=video_path,
        frame_count=total_frames,
        source_fps=fps,
        frames=records,
    )
    write_frame_manifest(manifest, output / "frames.json")
    return manifest


def selected_window_frame_manifests(proposal_run: Path) -> list[Path]:
    rows = _read_selected_windows(proposal_run)
    return [Path(row["frame_manifest"]) for row in rows]


def rank_window_bundles(
    *,
    proposal_run: Path,
    vggt_root: Path,
    output_dir: Path | None = None,
    render_review: bool = True,
    stitched_review: bool = False,
) -> dict[str, Any]:
    from .geometry import summarize_bundle
    from .viz import render_review_html

    proposal_run = proposal_run.resolve()
    output_dir = (output_dir or (proposal_run / "ranked_review")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_selected_windows(proposal_run)
    rank_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for row in rows:
        video_id = str(row["video_id"])
        window_segment_id = str(row["window_segment_id"])
        frame_manifest = Path(str(row["frame_manifest"]))
        bundle = vggt_root / video_id / window_segment_id
        summary_path = (
            output_dir
            / "summaries"
            / _safe_name(video_id)
            / _safe_name(window_segment_id)
            / "reconstruction_summary.json"
        )
        review_html = None
        reconstruction: dict[str, Any] | None = None
        status = "needs_vggt_bundle"
        error = ""
        try:
            if not bundle.exists():
                raise FileNotFoundError(f"missing VGGT bundle: {bundle}")
            summary = summarize_bundle(bundle, summary_path)
            reconstruction = model_to_dict(summary)
            status = "done"
            if render_review:
                review_html_path = (
                    output_dir
                    / "reviews"
                    / _safe_name(video_id)
                    / _safe_name(window_segment_id)
                    / "review.html"
                )
                render_review_html(frame_manifest, bundle, summary_path, review_html_path)
                review_html = str(review_html_path)
        except Exception as exc:
            error = str(exc)
            failures.append(f"{video_id}/{window_segment_id}: {exc}")

        reliability = reconstruction.get("reliability", {}) if reconstruction else {}
        descriptors = reconstruction.get("descriptors", {}) if reconstruction else {}
        reliability_score = float(reliability.get("score", 0.0) or 0.0)
        candidate_score = float(row.get("score", 0.0) or 0.0)
        combined_score = (0.65 * reliability_score) + (0.35 * candidate_score)
        safe_to_use = bool(reliability.get("safe_to_use_for_descriptors", False))
        rank_rows.append(
            {
                "video_id": video_id,
                "source_segment_id": row.get("source_segment_id"),
                "window_segment_id": window_segment_id,
                "window_id": row.get("window_id"),
                "start_sec": row.get("start_sec"),
                "end_sec": row.get("end_sec"),
                "candidate_score": candidate_score,
                "reliability_score": reliability_score,
                "combined_score": combined_score,
                "reliability_label": reliability.get("label", "missing"),
                "safe_to_use_for_descriptors": safe_to_use,
                "pose_jump_count": descriptors.get("pose_jump_count"),
                "bundle": str(bundle),
                "frame_manifest": str(frame_manifest),
                "summary": str(summary_path) if reconstruction else None,
                "review_html": review_html,
                "status": status,
                "error": error,
            }
        )

    best_rows = _best_windows(rank_rows)
    artifacts = _write_ranking_artifacts(
        output_dir=output_dir,
        rank_rows=rank_rows,
        best_rows=best_rows,
        proposal_run=proposal_run,
        stitched_review=stitched_review,
    )
    status = "done" if best_rows else "needs_vggt_bundle" if rank_rows else "failed_soft"
    report = {
        "schema_version": "window-ranking-v1",
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "proposal_run": str(proposal_run),
        "vggt_root": str(vggt_root),
        "window_count": len(rank_rows),
        "best_window_count": len(best_rows),
        "warnings": WINDOW_WARNINGS,
        "artifacts": artifacts,
        "failures": failures,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _scan_segment_windows(
    *,
    video_path: Path,
    video_id: str,
    source_segment_id: str,
    segment_start_sec: float,
    segment_end_sec: float,
    fps: float,
    total_frames: int,
    window_sec: float,
    stride_sec: float,
    eval_samples: int,
) -> list[dict[str, Any]]:
    if fps <= 0:
        fps = 1.0
    segment_start_frame = max(0, int(round(segment_start_sec * fps)))
    segment_end_frame = min(total_frames - 1, max(segment_start_frame, int(round(segment_end_sec * fps)) - 1))
    window_frames = max(2, int(round(window_sec * fps)))
    stride_frames = max(1, int(round(stride_sec * fps)))
    if window_frames > (segment_end_frame - segment_start_frame + 1):
        window_frames = segment_end_frame - segment_start_frame + 1
    starts = list(range(segment_start_frame, segment_end_frame - window_frames + 2, stride_frames))
    if not starts:
        starts = [segment_start_frame]
    final_start = max(segment_start_frame, segment_end_frame - window_frames + 1)
    if starts[-1] != final_start:
        starts.append(final_start)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    candidates: list[dict[str, Any]] = []
    try:
        for ordinal, start_frame in enumerate(starts, start=1):
            end_frame = min(segment_end_frame, start_frame + window_frames - 1)
            metrics = _window_metrics(capture, start_frame, end_frame, eval_samples)
            score = _window_score(metrics)
            candidates.append(
                {
                    "video_id": video_id,
                    "source_segment_id": source_segment_id,
                    "window_id": f"window-candidate-{ordinal:03d}",
                    "start_sec": float(start_frame / fps),
                    "end_sec": float((end_frame + 1) / fps),
                    "duration_sec": float((end_frame - start_frame + 1) / fps),
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "score": score,
                    **metrics,
                }
            )
    finally:
        capture.release()
    return candidates


def _window_metrics(
    capture: cv2.VideoCapture,
    start_frame: int,
    end_frame: int,
    eval_samples: int,
) -> dict[str, Any]:
    total = end_frame - start_frame + 1
    indices = [start_frame + index for index in deterministic_indices(total, min(eval_samples, total))]
    qualities = []
    grays = []
    overlay_ratios = []
    read_failures = 0
    for frame_index in indices:
        try:
            frame = _read_frame(capture, frame_index)
        except RuntimeError:
            read_failures += 1
            continue
        small = _resize_if_needed(frame, 360)
        qualities.append(measure_frame_quality(small))
        overlay_ratios.append(_bottom_left_overlay_ratio(small))
        grays.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

    absdiff_values = []
    flow_means = []
    flow_p95s = []
    for previous, current in zip(grays, grays[1:]):
        absdiff_values.append(float(np.mean(cv2.absdiff(previous, current))))
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            0.5,
            2,
            15,
            2,
            5,
            1.1,
            0,
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        flow_means.append(float(np.mean(magnitude)))
        flow_p95s.append(float(np.percentile(magnitude, 95)))

    blur_values = [quality.blur_score for quality in qualities]
    brightness_values = [quality.brightness_mean for quality in qualities]
    contrast_values = [quality.contrast_std for quality in qualities]
    motion_mean = _mean(absdiff_values)
    motion_std = _std(absdiff_values)
    flow_mean = _mean(flow_means)
    flow_p95 = _mean(flow_p95s)
    cut_like_pairs = sum(1 for value in absdiff_values if value > max(38.0, motion_mean + 2.5 * motion_std))
    return {
        "eval_sample_count": len(qualities),
        "read_failures": read_failures,
        "blur_mean": _mean(blur_values),
        "blur_min": min(blur_values) if blur_values else 0.0,
        "brightness_mean": _mean(brightness_values),
        "contrast_mean": _mean(contrast_values),
        "bottom_left_overlay_ratio_mean": _mean(overlay_ratios),
        "motion_mean": motion_mean,
        "motion_std": motion_std,
        "flow_mean": flow_mean,
        "flow_p95": flow_p95,
        "cut_like_pairs": int(cut_like_pairs),
    }


def _window_score(metrics: dict[str, Any]) -> float:
    blur_score = _clip01(math.log1p(float(metrics["blur_mean"])) / math.log1p(600.0))
    contrast_score = _clip01(float(metrics["contrast_mean"]) / 55.0)
    brightness = float(metrics["brightness_mean"])
    brightness_score = _clip01(1.0 - (abs(brightness - 118.0) / 118.0))
    motion = float(metrics["motion_mean"])
    motion_score = _clip01(1.0 - abs(motion - 12.0) / 38.0)
    continuity_score = _clip01(1.0 - (float(metrics["motion_std"]) / max(motion, 1.0)) / 2.5)
    flow_score = _clip01(1.0 - max(0.0, float(metrics["flow_p95"]) - 18.0) / 42.0)
    read_score = 1.0 if int(metrics["read_failures"]) == 0 else 0.65
    cut_penalty = min(0.35, int(metrics["cut_like_pairs"]) * 0.12)
    score = (
        0.22 * blur_score
        + 0.18 * contrast_score
        + 0.12 * brightness_score
        + 0.18 * motion_score
        + 0.17 * continuity_score
        + 0.08 * flow_score
        + 0.05 * read_score
        - cut_penalty
    )
    return round(_clip01(score), 6)


def _quality_aware_indices(
    *,
    capture: cv2.VideoCapture,
    start_frame: int,
    end_frame: int,
    count: int,
    neighbor_radius: int,
) -> list[int]:
    total = end_frame - start_frame + 1
    anchors = [start_frame + index for index in deterministic_indices(total, min(count, total))]
    selected: list[int] = []
    used: set[int] = set()
    for anchor in anchors:
        candidates = [
            index
            for index in range(anchor - neighbor_radius, anchor + neighbor_radius + 1)
            if start_frame <= index <= end_frame and index not in used
        ]
        if not candidates:
            continue
        best_index = max(candidates, key=lambda index: _frame_selection_score(capture, index))
        selected.append(best_index)
        used.add(best_index)
    for fallback in anchors:
        if len(selected) >= min(count, total):
            break
        if fallback not in used:
            selected.append(fallback)
            used.add(fallback)
    return sorted(selected)


def _frame_selection_score(capture: cv2.VideoCapture, frame_index: int) -> float:
    try:
        frame = _read_frame(capture, frame_index)
    except RuntimeError:
        return -1.0
    quality = measure_frame_quality(_resize_if_needed(frame, 360))
    blur = math.log1p(float(quality.blur_score))
    contrast = float(quality.contrast_std) / 45.0
    brightness_penalty = abs(float(quality.brightness_mean) - 118.0) / 118.0
    return blur + contrast - brightness_penalty


def _read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"could not read frame {frame_index}")
    return frame


def _resize_if_needed(frame: np.ndarray, resized_long_edge: int | None) -> np.ndarray:
    if resized_long_edge is None:
        return frame
    height, width = frame.shape[:2]
    long_edge = max(width, height)
    if long_edge <= resized_long_edge:
        return frame
    scale = resized_long_edge / long_edge
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _proposal_summary(
    *,
    output_dir: Path,
    log_path: Path,
    status: str,
    stage_status: dict[str, str],
    selected_windows: list[dict[str, Any]],
    failures: list[str],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "stable-window-proposal-v1",
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "stage_status": stage_status,
        "warnings": WINDOW_WARNINGS,
        "window_count": len(selected_windows),
        "frame_manifests": [row["frame_manifest"] for row in selected_windows],
        "artifacts": {
            "run_log": str(log_path),
            "summary_json": str(output_dir / "summary.json"),
            "next_steps": str(output_dir / "NEXT_STEPS.md"),
            "dashboard": str(output_dir / "index.html"),
            "candidate_table": str(output_dir / "window_candidates.parquet"),
            "selected_table": str(output_dir / "selected_windows.parquet"),
            "frame_manifest_list": str(output_dir / "frame_manifests.txt"),
        },
        "environment": environment_snapshot(),
        "failures": failures,
    }


def _proposal_inputs(
    media_inventory: Path,
    annotations: Path,
    frames_root: Path,
    video_ids: list[str] | None,
    segment_id: str | None,
    window_sec: float,
    stride_sec: float,
    candidate_limit: int,
    frame_count: int,
    resized_long_edge: int | None,
    eval_samples: int,
    neighbor_radius: int,
    mask_static_overlays: bool,
) -> dict[str, Any]:
    return {
        "media_inventory": str(media_inventory),
        "annotations": str(annotations),
        "frames_root": str(frames_root),
        "video_ids": video_ids or [],
        "segment_id": segment_id,
        "window_sec": window_sec,
        "stride_sec": stride_sec,
        "candidate_limit": candidate_limit,
        "frame_count": frame_count,
        "resized_long_edge": resized_long_edge,
        "eval_samples": eval_samples,
        "neighbor_radius": neighbor_radius,
        "mask_static_overlays": mask_static_overlays,
    }


def _write_proposal_artifacts(
    output_dir: Path,
    candidates: list[dict[str, Any]],
    selected_windows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    _write_table(output_dir / "window_candidates.parquet", candidates)
    _write_table(output_dir / "selected_windows.parquet", selected_windows)
    _write_json(output_dir / "window_candidates.json", candidates)
    _write_json(output_dir / "selected_windows.json", selected_windows)
    (output_dir / "frame_manifests.txt").write_text(
        "\n".join(row["frame_manifest"] for row in selected_windows) + ("\n" if selected_windows else ""),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "NEXT_STEPS.md").write_text(_proposal_next_steps(summary), encoding="utf-8")
    (output_dir / "index.html").write_text(
        _proposal_dashboard(summary, selected_windows),
        encoding="utf-8",
    )


def _proposal_next_steps(summary: dict[str, Any]) -> str:
    if summary["status"] != "done":
        return "\n".join(
            [
                "# Stable-Window Next Steps",
                "",
                f"Status: `{summary['status']}`",
                "",
                "No VGGT window package is ready yet. Fix accepted segment/media inputs, then rerun.",
                "",
                "Safety reminders:",
                "- no geolocation",
                "- no meters",
                "- relative VGGT frame",
                "- local-only media",
                "",
            ]
        )
    return "\n".join(
        [
            "# Stable-Window Next Steps",
            "",
            "Status: `done`",
            "",
            "This run produced quality-aware frame manifests for candidate windows.",
            "",
            "Package them for a GPU run:",
            "",
            "```powershell",
            "fpv windows h100-package \\",
            f"  --proposal-run {summary['artifacts']['summary_json']} \\",
            "  --workdir outputs/h100/window_run",
            "```",
            "",
            "After importing returned VGGT bundles, rank windows and render best reviews:",
            "",
            "```powershell",
            "fpv windows rank \\",
            f"  --proposal-run {summary['artifacts']['summary_json']} \\",
            "  --vggt-root data/vggt \\",
            "  --output-dir outputs/reviews/window_ranked \\",
            "  --stitched-review",
            "```",
            "",
            "Safety reminders:",
            "- no geolocation",
            "- no meters",
            "- relative VGGT frame",
            "- local-only media",
            "",
        ]
    )


def _proposal_dashboard(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    body_rows = "\n".join(
        "<tr>"
        f"<td>{_html(row.get('video_id'))}</td>"
        f"<td>{_html(row.get('window_segment_id'))}</td>"
        f"<td>{float(row.get('score', 0.0)):.3f}</td>"
        f"<td>{float(row.get('start_sec', 0.0)):.2f}-{float(row.get('end_sec', 0.0)):.2f}</td>"
        f"<td>{float(row.get('blur_mean', 0.0)):.1f}</td>"
        f"<td>{float(row.get('motion_mean', 0.0)):.1f}</td>"
        f"<td>{float(row.get('flow_p95', 0.0)):.1f}</td>"
        f"<td>{_html(row.get('frame_manifest'))}</td>"
        "</tr>"
        for row in rows
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stable Window Proposal</title>
<style>
:root{{color-scheme:dark;--bg:#101214;--panel:#181b1f;--line:#30363d;--text:#eef0f2;--muted:#aeb6bf;--accent:#74c7ec;--warn:#ffd166}}
body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,Segoe UI,Arial,sans-serif}}
header,main{{max-width:1180px;margin:0 auto;padding:22px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:14px 0}}
.card{{border:1px solid var(--line);background:var(--panel);border-radius:8px;padding:12px}}
.card strong{{display:block;font-size:24px}}
.warn{{border-left:4px solid var(--warn);padding:10px 12px;background:#211d12;color:#fff4cc}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line)}}
th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{color:var(--accent);font-size:12px;text-transform:uppercase;letter-spacing:.03em}}
td{{word-break:break-word}} code{{color:var(--accent)}}
</style>
</head>
<body>
<header>
<h1>Stable Window Proposal</h1>
<p>Quality-aware local frame windows for VGGT. These are reconstruction-input diagnostics, not operational analysis.</p>
<div class="warn">No geolocation. No meters. Relative VGGT frame. Local-only media-derived artifacts.</div>
<div class="cards">
<div class="card"><span>Status</span><strong>{_html(summary["status"])}</strong></div>
<div class="card"><span>Selected windows</span><strong>{len(rows)}</strong></div>
<div class="card"><span>Frame manifests</span><strong>{len(summary.get("frame_manifests", []))}</strong></div>
</div>
</header>
<main>
<h2>Selected Windows</h2>
<table>
<thead><tr><th>Video</th><th>Window segment</th><th>Score</th><th>Time</th><th>Blur</th><th>Motion</th><th>Flow p95</th><th>Frame manifest</th></tr></thead>
<tbody>{body_rows}</tbody>
</table>
</main>
</body>
</html>
"""


def _write_ranking_artifacts(
    *,
    output_dir: Path,
    rank_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    proposal_run: Path,
    stitched_review: bool,
) -> dict[str, str]:
    _write_table(output_dir / "window_rankings.parquet", rank_rows)
    _write_table(output_dir / "best_windows.parquet", best_rows)
    _write_json(output_dir / "window_rankings.json", rank_rows)
    _write_json(output_dir / "best_windows.json", best_rows)
    dashboard = output_dir / "index.html"
    dashboard.write_text(_ranking_dashboard(rank_rows, best_rows), encoding="utf-8")
    artifacts = {
        "dashboard": str(dashboard),
        "window_rankings": str(output_dir / "window_rankings.parquet"),
        "best_windows": str(output_dir / "best_windows.parquet"),
        "proposal_run": str(proposal_run),
    }
    if stitched_review:
        stitched_path = output_dir / "stitched_review.html"
        stitched_path.write_text(_stitched_review_html(best_rows), encoding="utf-8")
        artifacts["stitched_review"] = str(stitched_path)
    return artifacts


def _ranking_dashboard(rank_rows: list[dict[str, Any]], best_rows: list[dict[str, Any]]) -> str:
    sorted_rows = sorted(rank_rows, key=lambda row: float(row.get("combined_score", 0.0)), reverse=True)
    body = "\n".join(
        "<tr>"
        f"<td>{_html(row.get('video_id'))}</td>"
        f"<td>{_html(row.get('window_segment_id'))}</td>"
        f"<td>{float(row.get('combined_score', 0.0)):.3f}</td>"
        f"<td>{float(row.get('candidate_score', 0.0)):.3f}</td>"
        f"<td>{float(row.get('reliability_score', 0.0)):.3f}</td>"
        f"<td>{_html(row.get('reliability_label'))}</td>"
        f"<td>{_html(row.get('status'))}</td>"
        f"<td>{_review_link(row)}</td>"
        f"<td>{_html(row.get('error'))}</td>"
        "</tr>"
        for row in sorted_rows
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VGGT Window Ranking</title>
<style>
:root{{color-scheme:dark;--bg:#0f1115;--panel:#171a20;--line:#303846;--text:#eef1f5;--muted:#abb4c0;--accent:#8bd5ca;--warn:#ffd166}}
body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,Segoe UI,Arial,sans-serif}}
header,main{{max-width:1200px;margin:0 auto;padding:22px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:14px 0}}
.card{{border:1px solid var(--line);background:var(--panel);border-radius:8px;padding:12px}}
.card strong{{display:block;font-size:24px}}
.warn{{border-left:4px solid var(--warn);padding:10px 12px;background:#211d12;color:#fff4cc}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line)}}
th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{color:var(--accent);font-size:12px;text-transform:uppercase;letter-spacing:.03em}}
a{{color:var(--accent)}} td{{word-break:break-word}}
</style>
</head>
<body>
<header>
<h1>VGGT Window Ranking</h1>
<p>Ranks returned VGGT bundles by reconstruction reliability plus the original stable-window score.</p>
<div class="warn">No geolocation. No meters. Relative VGGT frame. Local-only media-derived artifacts.</div>
<div class="cards">
<div class="card"><span>Windows</span><strong>{len(rank_rows)}</strong></div>
<div class="card"><span>Best windows</span><strong>{len(best_rows)}</strong></div>
<div class="card"><span>Rendered reviews</span><strong>{sum(1 for row in rank_rows if row.get("review_html"))}</strong></div>
</div>
</header>
<main>
<h2>Ranked Windows</h2>
<table>
<thead><tr><th>Video</th><th>Window</th><th>Combined</th><th>Input</th><th>VGGT</th><th>Label</th><th>Status</th><th>Review</th><th>Issue</th></tr></thead>
<tbody>{body}</tbody>
</table>
</main>
</body>
</html>
"""


def _stitched_review_html(best_rows: list[dict[str, Any]]) -> str:
    rows = sorted(best_rows, key=lambda row: (str(row.get("video_id")), float(row.get("start_sec") or 0.0)))
    body = "\n".join(
        "<section>"
        f"<h2>{_html(row.get('video_id'))}</h2>"
        f"<p>{_html(row.get('window_segment_id'))} | "
        f"{float(row.get('start_sec') or 0.0):.2f}-{float(row.get('end_sec') or 0.0):.2f}s | "
        f"combined {float(row.get('combined_score') or 0.0):.3f}</p>"
        f"{_review_link(row)}"
        "</section>"
        for row in rows
    )
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Best Window Review</title>
<style>
body{{margin:0;background:#101214;color:#eef0f2;font:14px/1.45 system-ui,Segoe UI,Arial,sans-serif}}
header,main{{max-width:960px;margin:0 auto;padding:22px}}
section{{border:1px solid #30363d;background:#181b1f;border-radius:8px;padding:14px;margin:12px 0}}
a{{color:#74c7ec}} .warn{{border-left:4px solid #ffd166;padding:10px 12px;background:#211d12;color:#fff4cc}}
</style></head>
<body>
<header>
<h1>Best Window Review</h1>
<div class="warn">This is a local review index across selected windows, not a coordinate-aligned trajectory stitch. No geolocation, no meters, relative VGGT frame.</div>
</header>
<main>{body}</main>
</body></html>
"""


def _best_windows(rank_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rank_rows:
        if row.get("status") != "done":
            continue
        key = (str(row.get("video_id")), str(row.get("source_segment_id")))
        current = best.get(key)
        if current is None or float(row["combined_score"]) > float(current["combined_score"]):
            best[key] = row
    return sorted(best.values(), key=lambda row: (str(row.get("video_id")), str(row.get("source_segment_id"))))


def _read_selected_windows(proposal_run: Path) -> list[dict[str, Any]]:
    proposal_run = proposal_run.resolve()
    if proposal_run.is_file():
        proposal_run = proposal_run.parent
    json_path = proposal_run / "selected_windows.json"
    if json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))
    parquet_path = proposal_run / "selected_windows.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path).to_dict("records")
    raise FileNotFoundError(f"missing selected_windows.json/parquet under {proposal_run}")


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "unnamed"


def _html(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _review_link(row: dict[str, Any]) -> str:
    review = row.get("review_html")
    if not review:
        return ""
    return f'<a href="{_html(review)}">open</a>'
