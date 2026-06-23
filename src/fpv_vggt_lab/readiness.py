from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from .frames import read_frame_manifest
from .media import find_media_record
from .pipeline import SAFETY_WARNINGS
from .segments import get_accepted_segment


CheckStatus = Literal["pass", "warn", "fail"]


def audit_data_readiness(
    media_inventory: Path,
    annotations: Path,
    frames_root: Path,
    video_ids: list[str],
    segment_id: str = "segment-001",
    min_expected_frames: int = 1,
    dark_brightness_threshold: float = 5.0,
    low_blur_threshold: float = 10.0,
    low_contrast_threshold: float = 5.0,
    color_dominance_threshold: float = 0.48,
    color_saturation_threshold: float = 120.0,
) -> dict[str, Any]:
    clips = [
        _audit_clip(
            media_inventory=media_inventory,
            annotations=annotations,
            frames_root=frames_root,
            video_id=video_id,
            segment_id=segment_id,
            min_expected_frames=min_expected_frames,
            dark_brightness_threshold=dark_brightness_threshold,
            low_blur_threshold=low_blur_threshold,
            low_contrast_threshold=low_contrast_threshold,
            color_dominance_threshold=color_dominance_threshold,
            color_saturation_threshold=color_saturation_threshold,
        )
        for video_id in video_ids
    ]
    ready_count = sum(1 for clip in clips if clip["ready_for_cloud"])
    status = "ready_for_cloud" if ready_count == len(clips) and clips else "needs_review"
    return {
        "status": status,
        "clip_count": len(clips),
        "ready_count": ready_count,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "warnings": SAFETY_WARNINGS,
        "interpretation": (
            "Local workflow readiness only. This is not a claim about location, "
            "scale, physical speed, target identity, or operational usefulness."
        ),
        "clips": clips,
    }


def write_readiness_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _audit_clip(
    media_inventory: Path,
    annotations: Path,
    frames_root: Path,
    video_id: str,
    segment_id: str,
    min_expected_frames: int,
    dark_brightness_threshold: float,
    low_blur_threshold: float,
    low_contrast_threshold: float,
    color_dominance_threshold: float,
    color_saturation_threshold: float,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    media = None
    segment = None

    try:
        media = find_media_record(media_inventory, video_id)
        media_messages = []
        if not media.local_path.exists():
            media_messages.append(f"local media path missing: {media.local_path}")
        if media.frame_count <= 0:
            media_messages.append("media inventory frame_count is zero")
        if media.fps <= 0:
            media_messages.append("media inventory fps is not positive")
        checks["media_present"] = _check("fail" if media_messages else "pass", media_messages)
    except Exception as exc:
        checks["media_present"] = _check("fail", [str(exc)])

    try:
        segment = get_accepted_segment(annotations, video_id, segment_id)
        segment_messages = []
        if media is not None and segment.end_sec > media.duration_sec + 0.5:
            segment_messages.append("accepted segment exceeds media duration")
        checks["accepted_segment"] = _check("fail" if segment_messages else "pass", segment_messages)
    except Exception as exc:
        checks["accepted_segment"] = _check("fail", [str(exc)])

    if media is not None and media.local_path.exists() and segment is not None:
        probe = _decode_probe(
            media.local_path,
            frame_count=media.frame_count,
            fps=media.fps,
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
        )
        probe_status: CheckStatus = "pass" if probe["failed_reads"] == 0 else "warn"
        checks["decode_probe"] = _check(
            probe_status,
            [f"decode probe failures: {probe['failed_reads']}/3"] if probe["failed_reads"] else [],
            probe,
        )
    elif media is not None and media.local_path.exists():
        checks["decode_probe"] = _check("fail", ["accepted segment is unavailable for decode probe"])
    else:
        checks["decode_probe"] = _check("fail", ["media is unavailable for decode probe"])

    manifest_path = frames_root / video_id / segment_id / "frames.json"
    manifest = None
    try:
        manifest = read_frame_manifest(manifest_path)
        manifest_messages = []
        if manifest.video_id != video_id:
            manifest_messages.append(f"manifest video_id mismatch: {manifest.video_id}")
        if manifest.segment_id != segment_id:
            manifest_messages.append(f"manifest segment_id mismatch: {manifest.segment_id}")
        if len(manifest.frames) < min_expected_frames:
            manifest_messages.append(
                f"sampled frame count below minimum: {len(manifest.frames)} < {min_expected_frames}"
            )
        missing_frames = [frame for frame in manifest.frames if not frame.path.exists()]
        if missing_frames:
            manifest_messages.append(f"missing frame files: {len(missing_frames)}")
        checks["frame_manifest"] = _check(
            "fail" if manifest_messages else "pass",
            manifest_messages,
            {"path": str(manifest_path), "sampled_frame_count": len(manifest.frames)},
        )
    except Exception as exc:
        checks["frame_manifest"] = _check("fail", [f"{manifest_path}: {exc}"])

    if manifest is None:
        checks["frame_quality"] = _check("fail", ["frame manifest is unavailable"])
    else:
        dark_frames = [
            frame for frame in manifest.frames if frame.quality.brightness_mean < dark_brightness_threshold
        ]
        low_blur_frames = [
            frame for frame in manifest.frames if frame.quality.blur_score < low_blur_threshold
        ]
        low_contrast_frames = [
            frame for frame in manifest.frames if frame.quality.contrast_std < low_contrast_threshold
        ]
        color_dominant_frames = [
            frame
            for frame in manifest.frames
            if frame.path.exists()
            and _is_color_dominant_frame(
                frame.path,
                dominance_threshold=color_dominance_threshold,
                saturation_threshold=color_saturation_threshold,
            )
        ]
        quality_messages = []
        if dark_frames:
            quality_messages.append(f"very dark sampled frames: {len(dark_frames)}")
        if low_blur_frames:
            quality_messages.append(f"low-blur sampled frames: {len(low_blur_frames)}")
        if low_contrast_frames:
            quality_messages.append(f"low-contrast sampled frames: {len(low_contrast_frames)}")
        if color_dominant_frames:
            quality_messages.append(f"color-dominant sampled frames: {len(color_dominant_frames)}")
        checks["frame_quality"] = _check(
            "warn" if quality_messages else "pass",
            quality_messages,
            {
                "min_brightness": min(
                    (frame.quality.brightness_mean for frame in manifest.frames), default=None
                ),
                "min_blur_score": min((frame.quality.blur_score for frame in manifest.frames), default=None),
                "min_contrast_std": min((frame.quality.contrast_std for frame in manifest.frames), default=None),
                "color_dominant_frame_count": len(color_dominant_frames),
                "blocking_warning": bool(color_dominant_frames),
            },
        )

    ready = all(_check_allows_cloud(check) for check in checks.values())
    return {
        "video_id": video_id,
        "segment_id": segment_id,
        "ready_for_cloud": ready,
        "status": "ready_for_cloud" if ready else "needs_review",
        "media_path": str(media.local_path) if media is not None else None,
        "segment": {
            "start_sec": segment.start_sec,
            "end_sec": segment.end_sec,
            "status": segment.status,
        }
        if segment is not None
        else None,
        "checks": checks,
    }


def _decode_probe(
    video_path: Path,
    frame_count: int,
    fps: float,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if start_sec is not None and end_sec is not None:
        start_frame = max(0, int(round(start_sec * fps)))
        end_frame = max(start_frame, int(round(end_sec * fps)) - 1)
    else:
        start_frame = 0
        end_frame = max(frame_count - 1, 0)
    midpoint = start_frame + max((end_frame - start_frame) // 2, 0)
    positions = [start_frame, midpoint, end_frame]
    reads = []
    try:
        opened = capture.isOpened()
        for position in positions:
            ok = False
            if opened and frame_count > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, position)
                ok, _frame = capture.read()
            reads.append({"frame_index": int(position), "ok": bool(ok)})
    finally:
        capture.release()
    return {
        "path": str(video_path),
        "opened": bool(reads and any(read["ok"] for read in reads)),
        "reads": reads,
        "failed_reads": sum(1 for read in reads if not read["ok"]),
    }


def _is_color_dominant_frame(
    frame_path: Path,
    dominance_threshold: float,
    saturation_threshold: float,
) -> bool:
    image = cv2.imread(str(frame_path))
    if image is None:
        return False
    channel_means = image.reshape(-1, 3).mean(axis=0)
    total_mean = float(channel_means.sum())
    if total_mean <= 0:
        return False
    dominance_ratio = float(channel_means.max() / total_mean)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_mean = float(np.mean(hsv[:, :, 1]))
    return dominance_ratio >= dominance_threshold and saturation_mean >= saturation_threshold


def _check(status: CheckStatus, messages: list[str], details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"status": status, "messages": messages, "details": details or {}}


def _check_allows_cloud(check: dict[str, Any]) -> bool:
    if check["status"] == "fail":
        return False
    if check["status"] == "warn" and check.get("details", {}).get("blocking_warning"):
        return False
    return True
