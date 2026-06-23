from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .frames import measure_frame_quality
from .media import find_media_record
from .schemas import SegmentAnnotation, SegmentDiagnostics, model_to_dict


def propose_segment(
    inventory_path: Path,
    annotations_path: Path,
    video_id: str,
    segment_id: str = "segment-001",
) -> SegmentAnnotation:
    media = find_media_record(inventory_path, video_id)
    diagnostics = compute_segment_diagnostics(media.local_path)
    annotation = SegmentAnnotation(
        video_id=video_id,
        segment_id=segment_id,
        start_sec=0.0,
        end_sec=media.duration_sec,
        status="proposed",
        annotation_confidence="medium",
        annotation_notes="Automatically proposed using non-operational quality diagnostics.",
        diagnostics=diagnostics,
    )
    upsert_annotation(annotations_path, annotation)
    return annotation


def accept_segment(
    annotations_path: Path,
    video_id: str,
    segment_id: str,
    start_sec: float,
    end_sec: float,
    notes: str = "",
) -> SegmentAnnotation:
    existing = get_annotation(annotations_path, video_id, segment_id)
    annotation = SegmentAnnotation(
        video_id=video_id,
        segment_id=segment_id,
        start_sec=start_sec,
        end_sec=end_sec,
        status="accepted",
        include_terminal_event=existing.include_terminal_event if existing else False,
        excluded_ranges_sec=existing.excluded_ranges_sec if existing else [],
        annotation_confidence="high",
        annotation_notes=notes or "Human accepted segment bounds.",
        diagnostics=existing.diagnostics if existing else None,
    )
    upsert_annotation(annotations_path, annotation)
    return annotation


def edit_segment(
    annotations_path: Path,
    video_id: str,
    segment_id: str,
    start_sec: float | None = None,
    end_sec: float | None = None,
    notes: str | None = None,
    confidence: str | None = None,
) -> SegmentAnnotation:
    existing = get_annotation(annotations_path, video_id, segment_id)
    if existing is None:
        raise ValueError(f"segment not found: {video_id}/{segment_id}")
    data = model_to_dict(existing)
    if start_sec is not None:
        data["start_sec"] = start_sec
    if end_sec is not None:
        data["end_sec"] = end_sec
    if notes is not None:
        data["annotation_notes"] = notes
    if confidence is not None:
        data["annotation_confidence"] = confidence
    annotation = SegmentAnnotation.model_validate(data)
    upsert_annotation(annotations_path, annotation)
    return annotation


def reject_segment(annotations_path: Path, video_id: str, segment_id: str) -> SegmentAnnotation:
    existing = get_annotation(annotations_path, video_id, segment_id)
    if existing is None:
        raise ValueError(f"segment not found: {video_id}/{segment_id}")
    annotation = existing.model_copy(update={"status": "rejected"})
    upsert_annotation(annotations_path, annotation)
    return annotation


def render_segment_contact_sheet(
    inventory_path: Path,
    annotations_path: Path,
    video_id: str,
    segment_id: str,
    output: Path,
    samples: int = 12,
    thumb_width: int = 180,
) -> Path:
    media = find_media_record(inventory_path, video_id)
    annotation = get_annotation(annotations_path, video_id, segment_id)
    if annotation is None:
        raise ValueError(f"segment not found: {video_id}/{segment_id}")

    capture = cv2.VideoCapture(str(media.local_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {media.local_path}")
    fps = media.fps or 1.0
    start_frame = int(round(annotation.start_sec * fps))
    end_frame = max(start_frame, int(round(annotation.end_sec * fps)) - 1)
    frame_indices = np.linspace(start_frame, end_frame, samples)
    thumbs = []
    try:
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(float(frame_index))))
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            scale = thumb_width / max(width, 1)
            thumb = cv2.resize(
                frame,
                (thumb_width, max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
            label = f"{int(round(float(frame_index)))} / {frame_index / fps:.2f}s"
            cv2.rectangle(thumb, (0, 0), (thumb.shape[1], 20), (0, 0, 0), -1)
            cv2.putText(
                thumb,
                label,
                (4, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            thumbs.append(thumb)
    finally:
        capture.release()

    if not thumbs:
        raise ValueError("could not render any frames for contact sheet")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet = _tile_images(thumbs, columns=4)
    if not cv2.imwrite(str(output), sheet):
        raise ValueError(f"could not write contact sheet: {output}")
    return output


def get_accepted_segment(
    annotations_path: Path, video_id: str, segment_id: str
) -> SegmentAnnotation:
    annotation = get_annotation(annotations_path, video_id, segment_id)
    if annotation is None or annotation.status != "accepted":
        raise ValueError(f"segment must be accepted before downstream use: {video_id}/{segment_id}")
    return annotation


def get_annotation(
    annotations_path: Path, video_id: str, segment_id: str
) -> SegmentAnnotation | None:
    for annotation in read_annotations(annotations_path):
        if annotation.video_id == video_id and annotation.segment_id == segment_id:
            return annotation
    return None


def read_annotations(path: Path) -> list[SegmentAnnotation]:
    if not path.exists():
        return []
    annotations: list[SegmentAnnotation] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            annotations.append(SegmentAnnotation.model_validate(json.loads(line)))
    return annotations


def upsert_annotation(path: Path, annotation: SegmentAnnotation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    annotations = [
        item
        for item in read_annotations(path)
        if not (item.video_id == annotation.video_id and item.segment_id == annotation.segment_id)
    ]
    annotations.append(annotation)
    path.write_text(
        "\n".join(json.dumps(model_to_dict(item), sort_keys=True) for item in annotations)
        + "\n",
        encoding="utf-8",
    )


def compute_segment_diagnostics(video_path: Path, samples: int = 8) -> SegmentDiagnostics:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, max(frame_count - 1, 0), min(samples, max(frame_count, 1)))
        qualities = []
        grays = []
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(float(frame_index))))
            ok, frame = capture.read()
            if not ok:
                continue
            qualities.append(measure_frame_quality(frame))
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()

    motions = []
    for previous, current in zip(grays, grays[1:]):
        motions.append(float(np.mean(cv2.absdiff(previous, current))))
    return SegmentDiagnostics(
        sampled_frames=len(qualities),
        brightness_mean=float(np.mean([quality.brightness_mean for quality in qualities]))
        if qualities
        else None,
        blur_mean=float(np.mean([quality.blur_score for quality in qualities])) if qualities else None,
        motion_mean=float(np.mean(motions)) if motions else 0.0,
    )


def _tile_images(images: list[np.ndarray], columns: int) -> np.ndarray:
    columns = max(1, columns)
    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)
    rows = int(np.ceil(len(images) / columns))
    sheet = np.zeros((rows * max_height, columns * max_width, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row = index // columns
        column = index % columns
        y = row * max_height
        x = column * max_width
        sheet[y : y + image.shape[0], x : x + image.shape[1]] = image
    return sheet
