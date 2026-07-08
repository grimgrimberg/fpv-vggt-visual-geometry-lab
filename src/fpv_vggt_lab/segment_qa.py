from __future__ import annotations

import html
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .frames import deterministic_indices, read_frame_manifest
from .media import read_media_inventory
from .pipeline import SAFETY_WARNINGS, environment_snapshot
from .schemas import MediaInventoryRecord, SegmentAnnotation, model_to_dict
from .segments import get_annotation, upsert_annotation


SEGMENT_QA_WARNINGS = [
    *SAFETY_WARNINGS,
    "segment QA is a workflow gate, not a reconstruction truth claim",
    "proposed bounds require human accept/edit/reject before expensive VGGT",
]


def run_segment_qa(
    *,
    media_inventory: Path,
    annotations: Path,
    output_dir: Path,
    frames_root: Path | None = None,
    video_ids: list[str] | None = None,
    segment_id: str = "segment-001",
    probe_samples: int = 64,
    contact_samples: int = 16,
    boundary_samples: int = 12,
    write_proposals: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    if probe_samples < 6:
        raise ValueError("probe_samples must be at least 6")
    if contact_samples < 4:
        raise ValueError("contact_samples must be at least 4")
    if boundary_samples < 4:
        raise ValueError("boundary_samples must be at least 4")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    if not resume or not log_path.exists():
        log_path.write_text("", encoding="utf-8")

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")

    inventory = read_media_inventory(media_inventory)
    if video_ids:
        wanted = set(video_ids)
        selected = inventory[inventory["video_id"].isin(wanted)]
        missing = sorted(wanted - set(selected["video_id"].astype(str)))
    else:
        selected = inventory
        missing = []

    clips: list[dict[str, Any]] = []
    failures: list[str] = [f"{video_id}: missing from media inventory" for video_id in missing]
    manifest_index = _build_frame_manifest_index(frames_root, segment_id) if frames_root else {}
    log(f"start segment QA clips={len(selected)} missing={len(missing)} frame_manifests={len(manifest_index)}")

    for _index, row in selected.iterrows():
        record = MediaInventoryRecord.model_validate(row.to_dict())
        clip_dir = output_dir / _safe_name(record.video_id)
        try:
            clip_dir = output_dir / _safe_name(record.video_id)
            cached_report = clip_dir / "segment_qa.json"
            if resume and cached_report.exists():
                report = json.loads(cached_report.read_text(encoding="utf-8"))
                clips.append(report)
                log(f"{record.video_id}: cached {report.get('status', 'unknown')}")
                continue
            current = get_annotation(annotations, record.video_id, segment_id)
            report = analyze_segment_candidate(
                media=record,
                current_annotation=current,
                clip_dir=clip_dir,
                segment_id=segment_id,
                sampled_manifest=manifest_index.get((record.video_id, segment_id)),
                probe_samples=probe_samples,
                contact_samples=contact_samples,
                boundary_samples=boundary_samples,
            )
            if write_proposals:
                _write_annotation_proposal(
                    annotations=annotations,
                    media=record,
                    segment_id=segment_id,
                    current=current,
                    report=report,
                )
            report_path = clip_dir / "segment_qa.json"
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            clips.append(report)
            log(f"{record.video_id}: {report['status']} {report['proposed_segment']['start_sec']:.2f}-{report['proposed_segment']['end_sec']:.2f}")
        except Exception as exc:
            failures.append(f"{record.video_id}: {exc}")
            log(f"{record.video_id}: failed_soft {exc}")
            clips.append(
                {
                    "schema_version": "segment-qa-v1",
                    "video_id": record.video_id,
                    "segment_id": segment_id,
                    "status": "failed_soft",
                    "media_path": str(record.local_path),
                    "warnings": [str(exc)],
                    "artifacts": {},
                }
            )

    ready_count = sum(1 for clip in clips if clip.get("status") == "ready_for_vggt")
    needs_review_count = sum(1 for clip in clips if clip.get("status") == "needs_human_review")
    failed_count = sum(1 for clip in clips if clip.get("status") == "failed_soft") + len(missing)
    status = "ready_for_vggt" if clips and ready_count == len(clips) and not failures else "needs_human_review"
    if failed_count and not ready_count and not needs_review_count:
        status = "failed_soft"
    summary = {
        "schema_version": "segment-qa-run-v1",
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "media_inventory": str(media_inventory),
            "annotations": str(annotations),
            "frames_root": str(frames_root) if frames_root else None,
            "video_ids": video_ids or [],
            "segment_id": segment_id,
            "probe_samples": probe_samples,
            "contact_samples": contact_samples,
            "boundary_samples": boundary_samples,
            "write_proposals": write_proposals,
            "resume": resume,
        },
        "clip_count": len(clips),
        "ready_count": ready_count,
        "needs_review_count": needs_review_count,
        "failed_count": failed_count,
        "warnings": SEGMENT_QA_WARNINGS,
        "artifacts": {
            "run_log": str(log_path),
            "summary_json": str(output_dir / "summary.json"),
            "clip_table": str(output_dir / "segment_qa.parquet"),
            "clip_table_json": str(output_dir / "segment_qa.json"),
            "dashboard": str(output_dir / "index.html"),
            "next_steps": str(output_dir / "NEXT_STEPS.md"),
        },
        "environment": environment_snapshot(),
        "failures": failures,
        "clips": clips,
    }
    _write_run_artifacts(output_dir, summary, clips)
    log(f"finished segment QA status={status}")
    return summary


def analyze_segment_candidate(
    *,
    media: MediaInventoryRecord,
    current_annotation: SegmentAnnotation | None,
    clip_dir: Path,
    segment_id: str,
    sampled_manifest: Path | None,
    probe_samples: int,
    contact_samples: int,
    boundary_samples: int,
) -> dict[str, Any]:
    clip_dir.mkdir(parents=True, exist_ok=True)
    frames = _sample_video_metrics(media.local_path, media.frame_count, media.fps, probe_samples)
    if not frames:
        raise ValueError("could not decode probe frames")
    proposed = _propose_bounds(frames, duration_sec=media.duration_sec)
    overlay = _overlay_mask_candidates(frames)
    current = model_to_dict(current_annotation) if current_annotation else None
    warnings = _qa_warnings(proposed, current_annotation, media.duration_sec, overlay)
    status = "needs_human_review" if warnings else "ready_for_vggt"

    source_sheet = clip_dir / "source_contact_sheet.jpg"
    start_sheet = clip_dir / "start_boundary_contact_sheet.jpg"
    end_sheet = clip_dir / "end_boundary_contact_sheet.jpg"
    _render_contact_sheet(
        media.local_path,
        _linspace_times(0.0, media.duration_sec, contact_samples),
        source_sheet,
        thumb_width=320,
    )
    start_window = max(0.0, float(proposed["start_sec"]) - 2.0)
    _render_contact_sheet(
        media.local_path,
        _linspace_times(start_window, min(media.duration_sec, start_window + 6.0), boundary_samples),
        start_sheet,
        thumb_width=320,
    )
    end_window = max(0.0, float(proposed["end_sec"]) - 4.5)
    _render_contact_sheet(
        media.local_path,
        _linspace_times(end_window, min(media.duration_sec, end_window + 6.0), boundary_samples),
        end_sheet,
        thumb_width=320,
    )

    sampled_sheet = None
    sampled_warnings: list[str] = []
    if sampled_manifest is not None:
        sampled_sheet = clip_dir / "sampled_frames_contact_sheet.jpg"
        sampled_warnings = _render_sampled_frame_sheet(sampled_manifest, sampled_sheet)

    artifacts = {
        "source_contact_sheet": str(source_sheet),
        "start_boundary_contact_sheet": str(start_sheet),
        "end_boundary_contact_sheet": str(end_sheet),
        "sampled_frames_contact_sheet": str(sampled_sheet) if sampled_sheet else None,
        "sampled_frame_manifest": str(sampled_manifest) if sampled_manifest else None,
        "segment_qa_json": str(clip_dir / "segment_qa.json"),
    }
    next_command = (
        "fpv segment edit "
        f"--annotations {Path('data/annotations/segments.jsonl')} "
        f"--video-id {media.video_id} --segment-id {segment_id} "
        f"--start {float(proposed['start_sec']):.3f} --end {float(proposed['end_sec']):.3f} "
        '--notes "Accepted after segment QA contact-sheet review." --confidence high'
    )
    return {
        "schema_version": "segment-qa-v1",
        "video_id": media.video_id,
        "segment_id": segment_id,
        "status": status,
        "media_path": str(media.local_path),
        "duration_sec": float(media.duration_sec),
        "fps": float(media.fps),
        "frame_count": int(media.frame_count),
        "current_annotation": current,
        "proposed_segment": proposed,
        "overlay_mask": overlay,
        "sampled_frame_warnings": sampled_warnings,
        "warnings": warnings + sampled_warnings,
        "qa_decision_required": status != "ready_for_vggt",
        "next_command": next_command,
        "artifacts": artifacts,
        "probe_frames": frames,
    }


def frame_manifest_cut_warnings(frame_manifest_path: Path) -> dict[str, Any]:
    manifest = read_frame_manifest(frame_manifest_path)
    rows: list[dict[str, Any]] = []
    for frame in manifest.frames:
        row = {
            "frame_index": frame.frame_index,
            "timestamp_sec": frame.timestamp_sec,
            "brightness_mean": frame.quality.brightness_mean,
            "contrast_std": frame.quality.contrast_std,
            "blur_score": frame.quality.blur_score,
        }
        try:
            image = cv2.imread(str(frame.path))
            if image is not None:
                row.update(_artifact_metrics(image))
        except Exception:
            pass
        rows.append(row)
    warnings: list[str] = []
    dark = [row for row in rows if row.get("dark_card_like") or row.get("low_info_like")]
    low_contrast = [row for row in rows if float(row.get("contrast_std", 999.0) or 999.0) < 12.0]
    glitch = [row for row in rows if row.get("horizontal_glitch_like")]
    if dark:
        warnings.append(f"sampled frames include dark/title/outro-like frames: {len(dark)}")
    if low_contrast:
        warnings.append(f"sampled frames include very low-contrast frames: {len(low_contrast)}")
    if glitch:
        warnings.append(f"sampled frames include possible horizontal static/glitch frames: {len(glitch)}")
    return {
        "status": "needs_human_review" if warnings else "no_sampled_cut_warning",
        "warnings": warnings,
        "checked_frame_count": len(rows),
        "flagged_frame_count": len(dark) + len(low_contrast) + len(glitch),
        "flagged_samples": [
            {
                "frame_index": int(row["frame_index"]),
                "timestamp_sec": float(row["timestamp_sec"]),
                "flags": [
                    name
                    for name in ["dark_card_like", "low_info_like", "horizontal_glitch_like"]
                    if row.get(name)
                ],
            }
            for row in rows
            if row.get("dark_card_like") or row.get("low_info_like") or row.get("horizontal_glitch_like")
        ],
    }


def _sample_video_metrics(
    video_path: Path,
    frame_count: int,
    fps: float,
    samples: int,
) -> list[dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or frame_count)
    effective_fps = float(capture.get(cv2.CAP_PROP_FPS) or fps or 1.0)
    indices = deterministic_indices(max(total, 1), min(samples, max(total, 1)))
    rows: list[dict[str, Any]] = []
    previous_gray: np.ndarray | None = None
    try:
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            metrics = _artifact_metrics(frame)
            gray = cv2.cvtColor(_resize_long_edge(frame, 360), cv2.COLOR_BGR2GRAY)
            if previous_gray is not None:
                metrics["sample_absdiff_from_previous"] = float(np.mean(cv2.absdiff(previous_gray, gray)))
            else:
                metrics["sample_absdiff_from_previous"] = None
            previous_gray = gray
            rows.append(
                {
                    "frame_index": int(frame_index),
                    "timestamp_sec": float(frame_index / effective_fps),
                    **metrics,
                }
            )
    finally:
        capture.release()
    _add_cut_flags(rows)
    return rows


def _artifact_metrics(frame: np.ndarray) -> dict[str, Any]:
    small = _resize_long_edge(frame, 360)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 60, 120)
    edge_density = float(np.mean(edges > 0))
    saturation = float(np.mean(hsv[:, :, 1]))
    height, width = hsv.shape[:2]
    roi = hsv[int(height * 0.48) : int(height * 0.92), : int(width * 0.24)]
    if roi.size:
        hue = roi[:, :, 0]
        sat = roi[:, :, 1]
        val = roi[:, :, 2]
        yellow = (hue >= 16) & (hue <= 42) & (sat >= 80) & (val >= 105)
        red = ((hue <= 8) | (hue >= 170)) & (sat >= 85) & (val >= 95)
        bottom_left_overlay_ratio = float(np.mean(yellow | red))
    else:
        bottom_left_overlay_ratio = 0.0
    row_means = gray.mean(axis=1)
    row_diff = np.abs(np.diff(row_means))
    horizontal_score = float(np.percentile(row_diff, 99) / (np.mean(row_diff) + 1e-6)) if len(row_diff) else 0.0
    dark_card_like = brightness < 46.0 and contrast < 62.0 and edge_density < 0.16
    low_info_like = brightness < 10.0 or contrast < 9.0
    fog_or_smoke_like = brightness < 92.0 and contrast < 19.0
    horizontal_glitch_like = horizontal_score > 7.5 and contrast > 18.0
    flight_like = (
        brightness > 35.0
        and contrast > 13.0
        and not dark_card_like
        and not low_info_like
        and not fog_or_smoke_like
    )
    return {
        "brightness_mean": brightness,
        "contrast_std": contrast,
        "blur_score": blur,
        "edge_density": edge_density,
        "saturation_mean": saturation,
        "bottom_left_overlay_ratio": bottom_left_overlay_ratio,
        "horizontal_band_score": horizontal_score,
        "dark_card_like": bool(dark_card_like),
        "low_info_like": bool(low_info_like),
        "fog_or_smoke_like": bool(fog_or_smoke_like),
        "horizontal_glitch_like": bool(horizontal_glitch_like),
        "flight_like": bool(flight_like),
    }


def _add_cut_flags(rows: list[dict[str, Any]]) -> None:
    diffs = [
        float(row["sample_absdiff_from_previous"])
        for row in rows
        if row.get("sample_absdiff_from_previous") is not None
    ]
    if not diffs:
        return
    median = float(np.median(diffs))
    mad = float(np.median(np.abs(np.asarray(diffs) - median)))
    threshold = max(46.0, median + 5.0 * max(mad, 1.0))
    for row in rows:
        value = row.get("sample_absdiff_from_previous")
        row["hard_cut_like"] = bool(value is not None and float(value) >= threshold)


def _propose_bounds(rows: list[dict[str, Any]], duration_sec: float) -> dict[str, Any]:
    flight_indices = [index for index, row in enumerate(rows) if row.get("flight_like")]
    if not flight_indices:
        return {
            "start_sec": 0.0,
            "end_sec": float(duration_sec),
            "confidence": "low",
            "reason": "no stable flight-like run found; manual review required",
            "excluded_ranges_sec": [],
            "detected_events": _detected_events(rows),
        }
    run = max(2, min(3, len(rows) // 8 or 2))
    start_index = _first_run(rows, run)
    end_index = _last_run(rows, run)
    if start_index is None:
        start_index = flight_indices[0]
    if end_index is None:
        end_index = flight_indices[-1]
    step = _median_time_step(rows)
    start_sec = float(rows[start_index]["timestamp_sec"])
    end_sec = float(rows[end_index]["timestamp_sec"] + max(0.2, step * 0.25))
    if end_index + 1 < len(rows):
        end_sec = min(end_sec, float(rows[end_index + 1]["timestamp_sec"]))
    end_sec = max(start_sec + 0.5, min(float(duration_sec), end_sec))
    internal_events = [
        row
        for row in rows
        if start_sec <= float(row["timestamp_sec"]) <= end_sec
        and (row.get("hard_cut_like") or row.get("horizontal_glitch_like"))
    ]
    excluded = [
        [
            max(start_sec, float(row["timestamp_sec"]) - step * 0.5),
            min(end_sec, float(row["timestamp_sec"]) + step * 0.5),
        ]
        for row in internal_events
    ]
    return {
        "start_sec": round(start_sec, 3),
        "end_sec": round(end_sec, 3),
        "confidence": "medium" if start_sec > 0.75 or end_sec < float(duration_sec) - 0.75 else "low",
        "reason": "first and last stable flight-like frame runs after edit-artifact filtering",
        "excluded_ranges_sec": [[round(a, 3), round(b, 3)] for a, b in excluded],
        "detected_events": _detected_events(rows),
    }


def _first_run(rows: list[dict[str, Any]], run: int) -> int | None:
    for index in range(0, max(1, len(rows) - run + 1)):
        if all(rows[index + offset].get("flight_like") for offset in range(run)):
            return index
    return None


def _last_run(rows: list[dict[str, Any]], run: int) -> int | None:
    for index in range(max(0, len(rows) - run), -1, -1):
        if all(rows[index + offset].get("flight_like") for offset in range(run)):
            return index + run - 1
    return None


def _detected_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for row in rows:
        flags = [
            name
            for name in [
                "dark_card_like",
                "low_info_like",
                "fog_or_smoke_like",
                "horizontal_glitch_like",
                "hard_cut_like",
            ]
            if row.get(name)
        ]
        if flags:
            events.append(
                {
                    "timestamp_sec": round(float(row["timestamp_sec"]), 3),
                    "frame_index": int(row["frame_index"]),
                    "flags": flags,
                }
            )
    return events


def _overlay_mask_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    yellow_hits = sum(1 for row in rows if _bottom_left_yellow_like(row))
    hit_ratio = yellow_hits / max(len(rows), 1)
    regions = [[0.0, 0.54, 0.22, 1.0]] if hit_ratio >= 0.45 else []
    return {
        "status": "mask_recommended" if regions else "not_detected",
        "reason": "static lower-left saturated logo/blur region detected" if regions else "",
        "regions_norm_xyxy": regions,
        "hit_ratio": round(hit_ratio, 3),
    }


def _bottom_left_yellow_like(row: dict[str, Any]) -> bool:
    return bool(float(row.get("bottom_left_overlay_ratio", 0.0) or 0.0) > 0.006)


def _qa_warnings(
    proposed: dict[str, Any],
    current: SegmentAnnotation | None,
    duration_sec: float,
    overlay: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    start = float(proposed["start_sec"])
    end = float(proposed["end_sec"])
    if start > 0.75:
        warnings.append(f"leading edit/title material likely before {start:.2f}s")
    if end < float(duration_sec) - 0.75:
        warnings.append(f"trailing outro/smoke/edit material likely after {end:.2f}s")
    if proposed.get("excluded_ranges_sec"):
        warnings.append("internal hard cut/static-like region detected inside proposed segment")
    if overlay.get("status") == "mask_recommended":
        warnings.append("static overlay/censor/logo mask recommended before VGGT frame export")
    if current is None:
        warnings.append("no existing segment annotation; human accept/edit/reject required")
    elif current.status != "accepted":
        warnings.append(f"current segment status is {current.status}; human accept/edit/reject required")
    else:
        if abs(float(current.start_sec) - start) > 0.75 or abs(float(current.end_sec) - end) > 0.75:
            warnings.append(
                "accepted segment differs from QA proposal; review current bounds before spending GPU time"
            )
    return warnings


def _write_annotation_proposal(
    *,
    annotations: Path,
    media: MediaInventoryRecord,
    segment_id: str,
    current: SegmentAnnotation | None,
    report: dict[str, Any],
) -> None:
    if current is not None and current.status == "accepted":
        return
    proposed = report["proposed_segment"]
    annotation = SegmentAnnotation(
        video_id=media.video_id,
        segment_id=segment_id,
        start_sec=float(proposed["start_sec"]),
        end_sec=float(proposed["end_sec"]),
        status="proposed",
        excluded_ranges_sec=proposed.get("excluded_ranges_sec", []),
        annotation_confidence=proposed.get("confidence", "medium"),
        annotation_notes="Auto-proposed by strict pre-VGGT segment QA. Human review required.",
    )
    upsert_annotation(annotations, annotation)


def _render_contact_sheet(video_path: Path, times: list[float], output: Path, thumb_width: int) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 1.0)
    thumbs: list[np.ndarray] = []
    try:
        for timestamp in times:
            frame_index = max(0, int(round(timestamp * fps)))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            thumbs.append(_thumbnail(frame, f"{timestamp:.2f}s", thumb_width))
    finally:
        capture.release()
    if not thumbs:
        raise ValueError("could not render contact sheet")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet = _tile_images(thumbs, columns=4)
    if not cv2.imwrite(str(output), sheet):
        raise ValueError(f"could not write contact sheet: {output}")


def _render_sampled_frame_sheet(manifest_path: Path, output: Path) -> list[str]:
    manifest = read_frame_manifest(manifest_path)
    frames = manifest.frames[:4] + manifest.frames[-4:] if len(manifest.frames) > 8 else manifest.frames
    thumbs: list[np.ndarray] = []
    warnings = frame_manifest_cut_warnings(manifest_path)["warnings"]
    for frame in frames:
        image = cv2.imread(str(frame.path))
        if image is None:
            continue
        thumbs.append(_thumbnail(image, f"{frame.frame_index} / {frame.timestamp_sec:.2f}s", 240))
    if thumbs:
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), _tile_images(thumbs, columns=4))
    return warnings


def _build_frame_manifest_index(frames_root: Path | None, segment_id: str) -> dict[tuple[str, str], Path]:
    if frames_root is None or not frames_root.exists():
        return {}
    index: dict[tuple[str, str], Path] = {}
    direct_parent_depth_bonus = 0
    for path in frames_root.rglob("frames.json"):
        if path.parent.name != segment_id:
            continue
        video_id = path.parent.parent.name
        key = (video_id, segment_id)
        current = index.get(key)
        if current is None or len(path.parts) < len(current.parts) + direct_parent_depth_bonus:
            index[key] = path
    return index


def _find_frame_manifest(frames_root: Path | None, video_id: str, segment_id: str) -> Path | None:
    if frames_root is None or not frames_root.exists():
        return None
    direct = frames_root / video_id / segment_id / "frames.json"
    if direct.exists():
        return direct
    matches = []
    for path in frames_root.rglob("frames.json"):
        if path.parent.name == segment_id and path.parent.parent.name == video_id:
            matches.append(path)
    if not matches:
        return None
    return sorted(matches, key=lambda path: len(path.parts))[0]


def _write_run_artifacts(output_dir: Path, summary: dict[str, Any], clips: list[dict[str, Any]]) -> None:
    rows = [_clip_table_row(clip) for clip in clips]
    pd.DataFrame(rows).to_parquet(output_dir / "segment_qa.parquet", index=False)
    (output_dir / "segment_qa.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "NEXT_STEPS.md").write_text(_next_steps(summary), encoding="utf-8")
    (output_dir / "index.html").write_text(_dashboard(summary, rows), encoding="utf-8")


def _clip_table_row(clip: dict[str, Any]) -> dict[str, Any]:
    proposed = clip.get("proposed_segment", {})
    return {
        "video_id": clip.get("video_id"),
        "segment_id": clip.get("segment_id"),
        "status": clip.get("status"),
        "proposed_start_sec": proposed.get("start_sec"),
        "proposed_end_sec": proposed.get("end_sec"),
        "current_start_sec": (clip.get("current_annotation") or {}).get("start_sec"),
        "current_end_sec": (clip.get("current_annotation") or {}).get("end_sec"),
        "warning_count": len(clip.get("warnings") or []),
        "warnings": " | ".join(clip.get("warnings") or []),
        "source_contact_sheet": (clip.get("artifacts") or {}).get("source_contact_sheet"),
        "start_boundary_contact_sheet": (clip.get("artifacts") or {}).get("start_boundary_contact_sheet"),
        "end_boundary_contact_sheet": (clip.get("artifacts") or {}).get("end_boundary_contact_sheet"),
        "sampled_frames_contact_sheet": (clip.get("artifacts") or {}).get("sampled_frames_contact_sheet"),
        "next_command": clip.get("next_command"),
    }


def _next_steps(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Segment QA Next Steps",
            "",
            f"Status: `{summary['status']}`",
            "",
            "Open `index.html`, review each source/start/end/sampled-frame sheet, then accept or edit bounds.",
            "Do not send frames to VGGT/COLMAP until clips show `ready_for_vggt` or have a human accepted override.",
            "",
            "Useful command pattern:",
            "",
            "```powershell",
            "fpv segment edit --video-id <video_id> --segment-id segment-001 --start <sec> --end <sec> --confidence high",
            "fpv windows propose --media-inventory data/media/media_inventory.parquet --annotations data/annotations/segments.jsonl --output-dir outputs/windows/stable_window_run --frames-root data/frames --frames 128 --resized-long-edge 1536",
            "```",
            "",
            "Warnings:",
            *[f"- {warning}" for warning in SEGMENT_QA_WARNINGS],
            "",
        ]
    )


def _dashboard(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    body = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('video_id') or ''))}</td>"
        f"<td><span class='status {html.escape(str(row.get('status') or ''))}'>{html.escape(str(row.get('status') or ''))}</span></td>"
        f"<td>{_fmt(row.get('proposed_start_sec'))}-{_fmt(row.get('proposed_end_sec'))}</td>"
        f"<td>{_fmt(row.get('current_start_sec'))}-{_fmt(row.get('current_end_sec'))}</td>"
        f"<td>{html.escape(str(row.get('warning_count') or 0))}</td>"
        f"<td>{_sheet_link(row.get('source_contact_sheet'), 'source')}</td>"
        f"<td>{_sheet_link(row.get('start_boundary_contact_sheet'), 'start')}</td>"
        f"<td>{_sheet_link(row.get('end_boundary_contact_sheet'), 'end')}</td>"
        f"<td>{_sheet_link(row.get('sampled_frames_contact_sheet'), 'sampled')}</td>"
        f"<td><code>{html.escape(str(row.get('next_command') or ''))}</code></td>"
        "</tr>"
        for row in rows
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Segment QA</title>
<style>
:root{{color-scheme:dark;--bg:#101214;--panel:#181b1f;--line:#30363d;--text:#eef0f2;--muted:#aeb6bf;--accent:#8bd5ca;--warn:#ffd166;--bad:#ff8a80}}
body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,Segoe UI,Arial,sans-serif}}
header,main{{max-width:1320px;margin:0 auto;padding:22px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:14px 0}}
.card{{border:1px solid var(--line);background:var(--panel);border-radius:8px;padding:12px}}
.card strong{{display:block;font-size:24px}}
.warn{{border-left:4px solid var(--warn);padding:10px 12px;background:#211d12;color:#fff4cc}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line)}}
th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{color:var(--accent);font-size:12px;text-transform:uppercase}}
a{{color:var(--accent)}} code{{white-space:pre-wrap;color:#d4e7ff;font-size:12px}}
.status{{font-weight:800}} .needs_human_review,.failed_soft{{color:var(--bad)}} .ready_for_vggt{{color:#98e6a8}}
</style>
</head>
<body>
<header>
<h1>Pre-VGGT Segment QA</h1>
<p>Workflow gate for edited videos: source sheets, proposed bounds, sampled-frame checks, and local-only warnings.</p>
<div class="warn">No geolocation. No meters. Relative VGGT frame. Human review required before expensive VGGT/COLMAP.</div>
<div class="cards">
<div class="card"><span>Status</span><strong>{html.escape(str(summary["status"]))}</strong></div>
<div class="card"><span>Clips</span><strong>{summary["clip_count"]}</strong></div>
<div class="card"><span>Needs review</span><strong>{summary["needs_review_count"]}</strong></div>
<div class="card"><span>Ready</span><strong>{summary["ready_count"]}</strong></div>
</div>
</header>
<main>
<table>
<thead><tr><th>Video</th><th>Status</th><th>Proposed</th><th>Current</th><th>Warnings</th><th>Source</th><th>Start</th><th>End</th><th>Sampled</th><th>Next command</th></tr></thead>
<tbody>{body}</tbody>
</table>
</main>
</body>
</html>
"""


def _sheet_link(path: object, label: str) -> str:
    if not path:
        return ""
    return f"<a href='{html.escape(str(path), quote=True)}'>{html.escape(label)}</a>"


def _fmt(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def _linspace_times(start: float, end: float, count: int) -> list[float]:
    if end <= start:
        end = start + 0.001
    return [float(value) for value in np.linspace(start, end, count)]


def _median_time_step(rows: list[dict[str, Any]]) -> float:
    times = [float(row["timestamp_sec"]) for row in rows]
    if len(times) < 2:
        return 1.0
    return float(np.median(np.diff(times)))


def _thumbnail(frame: np.ndarray, label: str, thumb_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = thumb_width / max(width, 1)
    thumb = cv2.resize(frame, (thumb_width, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
    cv2.rectangle(thumb, (0, 0), (thumb.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(thumb, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return thumb


def _tile_images(images: list[np.ndarray], columns: int) -> np.ndarray:
    columns = max(1, columns)
    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)
    rows = int(math.ceil(len(images) / columns))
    sheet = np.zeros((rows * max_height, columns * max_width, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row = index // columns
        column = index % columns
        y = row * max_height
        x = column * max_width
        sheet[y : y + image.shape[0], x : x + image.shape[1]] = image
    return sheet


def _resize_long_edge(frame: np.ndarray, long_edge: int) -> np.ndarray:
    height, width = frame.shape[:2]
    current = max(width, height)
    if current <= long_edge:
        return frame
    scale = long_edge / current
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "unnamed"
