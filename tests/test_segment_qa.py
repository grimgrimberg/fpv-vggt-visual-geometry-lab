import json
from pathlib import Path

import cv2
import numpy as np
from typer.testing import CliRunner

from fpv_vggt_lab.cli import app
from fpv_vggt_lab.media import probe_video, sha256_file, upsert_inventory
from fpv_vggt_lab.schemas import MediaInventoryRecord
from fpv_vggt_lab.segments import accept_segment


runner = CliRunner()


def _write_edited_video(path: Path, fps: float = 10.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (192, 108))
    assert writer.isOpened()
    try:
        for index in range(120):
            if index < 22:
                frame = np.zeros((108, 192, 3), dtype=np.uint8)
                cv2.putText(frame, "TITLE CARD", (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 210, 60), 2)
            elif index >= 94:
                frame = np.zeros((108, 192, 3), dtype=np.uint8)
                noise = np.random.default_rng(index).integers(0, 35, frame.shape, dtype=np.uint8)
                frame = cv2.add(frame, noise)
                cv2.putText(frame, "OUTRO", (52, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)
            else:
                frame = np.zeros((108, 192, 3), dtype=np.uint8)
                frame[:] = (80, 140, 70)
                cv2.rectangle(frame, (0, 70), (192, 108), (80, 105, 70), -1)
                cv2.circle(frame, (30 + (index % 40) * 3, 54), 15, (185, 190, 180), -1)
                cv2.line(frame, (0, 90 - index % 20), (191, 75 - index % 20), (210, 210, 210), 2)
                cv2.rectangle(frame, (12, 68), (31, 95), (0, 220, 235), -1)
                cv2.circle(frame, (24, 72), 4, (0, 0, 220), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def _inventory_and_annotations(tmp_path: Path) -> tuple[Path, Path, str]:
    video_id = "edited-segment-video"
    video = _write_edited_video(tmp_path / "media" / f"{video_id}.mp4")
    inventory = tmp_path / "media_inventory.parquet"
    upsert_inventory(
        inventory,
        MediaInventoryRecord(
            video_id=video_id,
            source_url=str(video),
            local_path=video,
            sha256=sha256_file(video),
            bytes=video.stat().st_size,
            **probe_video(video),
        ),
    )
    annotations = tmp_path / "segments.jsonl"
    accept_segment(
        annotations,
        video_id=video_id,
        segment_id="segment-001",
        start_sec=0.0,
        end_sec=12.0,
        notes="Deliberately broad edited clip for QA test.",
    )
    return inventory, annotations, video_id


def test_segment_qa_flags_edited_intro_and_outro_without_overwriting_acceptance(tmp_path: Path):
    inventory, annotations, video_id = _inventory_and_annotations(tmp_path)
    output_dir = tmp_path / "segment_qa"

    result = runner.invoke(
        app,
        [
            "segment",
            "qa",
            "--media-inventory",
            str(inventory),
            "--annotations",
            str(annotations),
            "--output-dir",
            str(output_dir),
            "--frames-root",
            str(tmp_path / "frames"),
            "--video-id",
            video_id,
            "--probe-samples",
            "32",
            "--contact-samples",
            "8",
            "--boundary-samples",
            "8",
            "--write-proposals",
        ],
    )

    assert result.exit_code == 1, result.output
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "needs_human_review"
    clip = summary["clips"][0]
    assert clip["status"] == "needs_human_review"
    assert clip["proposed_segment"]["start_sec"] > 1.5
    assert clip["proposed_segment"]["end_sec"] < 10.5
    assert any("leading edit" in warning for warning in clip["warnings"])
    assert any("trailing" in warning for warning in clip["warnings"])
    assert any("accepted segment differs" in warning for warning in clip["warnings"])
    assert clip["overlay_mask"]["status"] == "mask_recommended"
    assert Path(clip["artifacts"]["source_contact_sheet"]).exists()
    assert Path(clip["artifacts"]["start_boundary_contact_sheet"]).exists()
    assert Path(clip["artifacts"]["end_boundary_contact_sheet"]).exists()

    annotations_text = annotations.read_text(encoding="utf-8")
    assert '"status": "accepted"' in annotations_text
    assert '"start_sec": 0.0' in annotations_text
