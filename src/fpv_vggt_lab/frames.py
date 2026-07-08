from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .image_ops import apply_overlay_masks
from .schemas import FrameManifest, FrameQuality, FrameRecord, model_to_dict


def sample_video_frames(
    video: Path,
    output: Path,
    count: int,
    video_id: str,
    segment_id: str,
    resized_long_edge: int | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    mask_regions: list[list[float]] | None = None,
) -> FrameManifest:
    if count <= 0:
        raise ValueError("count must be positive")

    video = video.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    if total_frames <= 0:
        capture.release()
        raise RuntimeError(f"video has no readable frames: {video}")
    if fps <= 0:
        fps = 1.0

    start_frame = int(round((start_sec or 0.0) * fps))
    end_frame = int(round(end_sec * fps)) - 1 if end_sec is not None else total_frames - 1
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame, min(end_frame, total_frames - 1))
    segment_frame_count = end_frame - start_frame + 1
    indices = [
        start_frame + index
        for index in deterministic_indices(total_frames=segment_frame_count, count=count)
    ]
    records: list[FrameRecord] = []

    try:
        for ordinal, frame_index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"could not read frame {frame_index} from {video}")

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
        source_video=video,
        frame_count=total_frames,
        source_fps=fps,
        frames=records,
    )
    write_frame_manifest(manifest, output / "frames.json")
    return manifest


def deterministic_indices(total_frames: int, count: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    if count >= total_frames:
        return list(range(total_frames))
    return [int(round(value)) for value in np.linspace(0, total_frames - 1, count)]


def measure_frame_quality(frame: np.ndarray) -> FrameQuality:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return FrameQuality(
        blur_score=blur_score,
        brightness_mean=float(gray.mean()),
        contrast_std=float(gray.std()),
    )


def write_frame_manifest(manifest: FrameManifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(model_to_dict(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def read_frame_manifest(path: Path) -> FrameManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    return FrameManifest.model_validate(data)


def sample_accepted_segment_frames(
    media_inventory: Path,
    annotations: Path,
    video_id: str,
    segment_id: str,
    output: Path,
    count: int,
    resized_long_edge: int | None = None,
    mask_regions: list[list[float]] | None = None,
) -> FrameManifest:
    from .media import find_media_record
    from .segments import get_accepted_segment

    media = find_media_record(media_inventory, video_id)
    segment = get_accepted_segment(annotations, video_id, segment_id)
    return sample_video_frames(
        video=media.local_path,
        output=output,
        count=count,
        video_id=video_id,
        segment_id=segment_id,
        resized_long_edge=resized_long_edge,
        start_sec=segment.start_sec,
        end_sec=segment.end_sec,
        mask_regions=mask_regions,
    )


def _resize_if_needed(frame: np.ndarray, resized_long_edge: int | None) -> np.ndarray:
    if resized_long_edge is None:
        return frame
    if resized_long_edge <= 0:
        raise ValueError("resized_long_edge must be positive")

    height, width = frame.shape[:2]
    long_edge = max(width, height)
    if long_edge <= resized_long_edge:
        return frame

    scale = resized_long_edge / long_edge
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
