import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from typer.testing import CliRunner

from fpv_vggt_lab.cli import app
from tests.test_run_video_ids import write_three_clip_catalog


runner = CliRunner()


def prepare_single_ready_clip(tmp_path: Path) -> tuple[Path, Path, Path, Path, str]:
    readme, manifest, video_ids = write_three_clip_catalog(tmp_path)
    video_id = video_ids[0]
    catalog_dir = tmp_path / "catalog"
    annotations = tmp_path / "segments.jsonl"
    media_dir = tmp_path / "media"
    media_inventory = media_dir / "media_inventory.parquet"
    frames_root = tmp_path / "frames"

    assert runner.invoke(
        app,
        [
            "catalog",
            "sync",
            "--readme",
            str(readme),
            "--manifest",
            str(manifest),
            "--output",
            str(catalog_dir),
        ],
    ).exit_code == 0
    assert runner.invoke(
        app,
        [
            "media",
            "fetch",
            "--catalog",
            str(catalog_dir / "catalog.parquet"),
            "--media-dir",
            str(media_dir),
            "--inventory",
            str(media_inventory),
            "--video-id",
            video_id,
        ],
    ).exit_code == 0
    assert runner.invoke(
        app,
        [
            "segment",
            "accept",
            "--annotations",
            str(annotations),
            "--video-id",
            video_id,
            "--segment-id",
            "segment-001",
            "--start",
            "0",
            "--end",
            "1.5",
        ],
    ).exit_code == 0
    assert runner.invoke(
        app,
        [
            "frames",
            "sample-accepted",
            "--media-inventory",
            str(media_inventory),
            "--annotations",
            str(annotations),
            "--video-id",
            video_id,
            "--segment-id",
            "segment-001",
            "--output",
            str(frames_root / video_id / "segment-001"),
            "--count",
            "6",
        ],
    ).exit_code == 0
    return media_inventory, annotations, frames_root, catalog_dir, video_id


def test_review_audit_readiness_reports_ready_clip(tmp_path: Path):
    media_inventory, annotations, frames_root, _catalog_dir, video_id = prepare_single_ready_clip(tmp_path)
    report = tmp_path / "readiness.json"

    result = runner.invoke(
        app,
        [
            "review",
            "audit-readiness",
            "--media-inventory",
            str(media_inventory),
            "--annotations",
            str(annotations),
            "--frames-root",
            str(frames_root),
            "--video-id",
            video_id,
            "--segment-id",
            "segment-001",
            "--report",
            str(report),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "ready_for_cloud" in result.output
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["status"] == "ready_for_cloud"
    assert data["ready_count"] == 1
    assert data["clip_count"] == 1
    clip = data["clips"][0]
    assert clip["video_id"] == video_id
    assert clip["ready_for_cloud"] is True
    assert clip["checks"]["media_present"]["status"] == "pass"
    assert clip["checks"]["accepted_segment"]["status"] == "pass"
    assert clip["checks"]["frame_manifest"]["status"] == "pass"
    assert clip["checks"]["frame_quality"]["status"] == "pass"
    assert "no geolocation" in data["warnings"]


def test_review_audit_readiness_flags_bad_frame_manifest(tmp_path: Path):
    media_inventory, annotations, frames_root, _catalog_dir, video_id = prepare_single_ready_clip(tmp_path)
    manifest = frames_root / video_id / "segment-001" / "frames.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    Path(data["frames"][0]["path"]).unlink()
    data["frames"][1]["quality"]["brightness_mean"] = 1.0
    data["frames"][2]["quality"]["contrast_std"] = 1.0
    blue_frame = np.full((32, 32, 3), (255, 0, 0), dtype=np.uint8)
    assert cv2.imwrite(data["frames"][2]["path"], blue_frame)
    manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    report = tmp_path / "readiness.json"

    result = runner.invoke(
        app,
        [
            "review",
            "audit-readiness",
            "--media-inventory",
            str(media_inventory),
            "--annotations",
            str(annotations),
            "--frames-root",
            str(frames_root),
            "--video-id",
            video_id,
            "--segment-id",
            "segment-001",
            "--report",
            str(report),
        ],
    )

    assert result.exit_code != 0
    assert "needs_review" in result.output
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["status"] == "needs_review"
    clip = data["clips"][0]
    assert clip["ready_for_cloud"] is False
    assert clip["checks"]["frame_manifest"]["status"] == "fail"
    assert "missing frame files: 1" in clip["checks"]["frame_manifest"]["messages"]
    assert clip["checks"]["frame_quality"]["status"] == "warn"
    assert "very dark sampled frames: 1" in clip["checks"]["frame_quality"]["messages"]
    assert "low-contrast sampled frames: 1" in clip["checks"]["frame_quality"]["messages"]
    assert "color-dominant sampled frames: 1" in clip["checks"]["frame_quality"]["messages"]


def test_review_audit_readiness_allows_nonblocking_quality_warning(tmp_path: Path):
    media_inventory, annotations, frames_root, _catalog_dir, video_id = prepare_single_ready_clip(tmp_path)
    manifest = frames_root / video_id / "segment-001" / "frames.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["frames"][0]["quality"]["brightness_mean"] = 1.0
    data["frames"][0]["quality"]["blur_score"] = 1.0
    manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    report = tmp_path / "readiness.json"

    result = runner.invoke(
        app,
        [
            "review",
            "audit-readiness",
            "--media-inventory",
            str(media_inventory),
            "--annotations",
            str(annotations),
            "--frames-root",
            str(frames_root),
            "--video-id",
            video_id,
            "--segment-id",
            "segment-001",
            "--report",
            str(report),
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["status"] == "ready_for_cloud"
    clip = data["clips"][0]
    assert clip["ready_for_cloud"] is True
    assert clip["checks"]["frame_quality"]["status"] == "warn"
    assert clip["checks"]["frame_quality"]["details"]["blocking_warning"] is False
    assert "very dark sampled frames: 1" in clip["checks"]["frame_quality"]["messages"]
    assert "low-blur sampled frames: 1" in clip["checks"]["frame_quality"]["messages"]


def test_review_audit_readiness_probes_accepted_segment_not_media_tail(tmp_path: Path):
    media_inventory, annotations, frames_root, _catalog_dir, video_id = prepare_single_ready_clip(tmp_path)
    inventory = pd.read_parquet(media_inventory)
    inventory.loc[inventory["video_id"] == video_id, "frame_count"] = 100_000
    inventory.to_parquet(media_inventory, index=False)
    report = tmp_path / "readiness.json"

    result = runner.invoke(
        app,
        [
            "review",
            "audit-readiness",
            "--media-inventory",
            str(media_inventory),
            "--annotations",
            str(annotations),
            "--frames-root",
            str(frames_root),
            "--video-id",
            video_id,
            "--segment-id",
            "segment-001",
            "--report",
            str(report),
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(report.read_text(encoding="utf-8"))
    clip = data["clips"][0]
    assert clip["ready_for_cloud"] is True
    assert clip["checks"]["decode_probe"]["status"] == "pass"
