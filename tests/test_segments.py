import json
from pathlib import Path

from typer.testing import CliRunner

from fpv_vggt_lab.cli import app
from tests.test_catalog_media import write_fixture_catalog


runner = CliRunner()


def prepare_fetched_video(tmp_path: Path) -> tuple[str, Path, Path]:
    readme, manifest, video_id = write_fixture_catalog(tmp_path)
    catalog_dir = tmp_path / "catalog"
    inventory_path = tmp_path / "media_inventory.parquet"
    media_dir = tmp_path / "media"

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
            str(inventory_path),
            "--video-id",
            video_id,
        ],
    ).exit_code == 0
    return video_id, inventory_path, tmp_path / "segments.jsonl"


def test_segment_proposal_acceptance_and_accepted_sampling_gate(tmp_path: Path):
    video_id, inventory_path, annotations = prepare_fetched_video(tmp_path)
    frames_dir = tmp_path / "accepted_frames"

    proposed = runner.invoke(
        app,
        [
            "segment",
            "propose",
            "--media-inventory",
            str(inventory_path),
            "--annotations",
            str(annotations),
            "--video-id",
            video_id,
        ],
    )
    assert proposed.exit_code == 0, proposed.output
    assert annotations.exists()
    assert "proposed" in annotations.read_text(encoding="utf-8")

    contact_sheet = tmp_path / "contact_sheet.jpg"
    sheet_result = runner.invoke(
        app,
        [
            "segment",
            "contact-sheet",
            "--media-inventory",
            str(inventory_path),
            "--annotations",
            str(annotations),
            "--video-id",
            video_id,
            "--segment-id",
            "segment-001",
            "--output",
            str(contact_sheet),
        ],
    )
    assert sheet_result.exit_code == 0, sheet_result.output
    assert contact_sheet.exists()

    blocked = runner.invoke(
        app,
        [
            "frames",
            "sample-accepted",
            "--media-inventory",
            str(inventory_path),
            "--annotations",
            str(annotations),
            "--video-id",
            video_id,
            "--segment-id",
            "segment-001",
            "--output",
            str(frames_dir),
            "--count",
            "4",
        ],
    )
    assert blocked.exit_code != 0
    assert "accepted" in blocked.output.lower()

    accepted = runner.invoke(
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
    )
    assert accepted.exit_code == 0, accepted.output

    listed = runner.invoke(app, ["segment", "list", "--annotations", str(annotations)])
    assert listed.exit_code == 0, listed.output
    assert "accepted" in listed.output

    sampled = runner.invoke(
        app,
        [
            "frames",
            "sample-accepted",
            "--media-inventory",
            str(inventory_path),
            "--annotations",
            str(annotations),
            "--video-id",
            video_id,
            "--segment-id",
            "segment-001",
            "--output",
            str(frames_dir),
            "--count",
            "4",
        ],
    )
    assert sampled.exit_code == 0, sampled.output
    assert (frames_dir / "frames.json").exists()


def test_segment_edit_updates_existing_annotation_without_breaking_sampling_gate(tmp_path: Path):
    video_id, inventory_path, annotations = prepare_fetched_video(tmp_path)
    frames_dir = tmp_path / "edited_frames"

    assert runner.invoke(
        app,
        [
            "segment",
            "propose",
            "--media-inventory",
            str(inventory_path),
            "--annotations",
            str(annotations),
            "--video-id",
            video_id,
        ],
    ).exit_code == 0
    accepted = runner.invoke(
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
    )
    assert accepted.exit_code == 0, accepted.output

    edited = runner.invoke(
        app,
        [
            "segment",
            "edit",
            "--annotations",
            str(annotations),
            "--video-id",
            video_id,
            "--segment-id",
            "segment-001",
            "--start",
            "0.25",
            "--end",
            "1.25",
            "--notes",
            "Tightened after contact sheet review.",
            "--confidence",
            "medium",
        ],
    )

    assert edited.exit_code == 0, edited.output
    record = json.loads(annotations.read_text(encoding="utf-8").splitlines()[0])
    assert record["status"] == "accepted"
    assert record["start_sec"] == 0.25
    assert record["end_sec"] == 1.25
    assert record["annotation_notes"] == "Tightened after contact sheet review."
    assert record["annotation_confidence"] == "medium"
    assert record["diagnostics"]["sampled_frames"] > 0

    sampled = runner.invoke(
        app,
        [
            "frames",
            "sample-accepted",
            "--media-inventory",
            str(inventory_path),
            "--annotations",
            str(annotations),
            "--video-id",
            video_id,
            "--segment-id",
            "segment-001",
            "--output",
            str(frames_dir),
            "--count",
            "4",
        ],
    )
    assert sampled.exit_code == 0, sampled.output
    manifest = json.loads((frames_dir / "frames.json").read_text(encoding="utf-8"))
    assert manifest["frames"][0]["timestamp_sec"] >= 0.25
