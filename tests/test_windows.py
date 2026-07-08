import json
import zipfile
from pathlib import Path

from typer.testing import CliRunner

from fpv_vggt_lab.cli import app
from fpv_vggt_lab.media import probe_video, sha256_file, upsert_inventory
from fpv_vggt_lab.schemas import MediaInventoryRecord
from fpv_vggt_lab.segments import accept_segment
from fpv_vggt_lab.synthetic import create_synthetic_video
from fpv_vggt_lab.vggt import create_mock_bundle


runner = CliRunner()


def _prepare_window_inputs(tmp_path: Path) -> tuple[Path, Path, str]:
    video_id = "window-synthetic-video"
    video = create_synthetic_video(
        output=tmp_path / "media" / f"{video_id}.mp4",
        frames=96,
        width=180,
        height=120,
        fps=12,
    )
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
        end_sec=7.0,
        notes="Synthetic accepted segment.",
    )
    return inventory, annotations, video_id


def test_windows_propose_writes_quality_aware_frame_manifests(tmp_path: Path):
    inventory, annotations, video_id = _prepare_window_inputs(tmp_path)
    output_dir = tmp_path / "window_run"
    frames_root = tmp_path / "frames"

    result = runner.invoke(
        app,
        [
            "windows",
            "propose",
            "--media-inventory",
            str(inventory),
            "--annotations",
            str(annotations),
            "--output-dir",
            str(output_dir),
            "--frames-root",
            str(frames_root),
            "--video-id",
            video_id,
            "--window-sec",
            "2.0",
            "--stride-sec",
            "1.0",
            "--candidate-limit",
            "2",
            "--frames",
            "8",
            "--resized-long-edge",
            "128",
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "done"
    assert summary["window_count"] == 2
    assert (output_dir / "index.html").exists()
    assert (output_dir / "selected_windows.parquet").exists()
    assert (output_dir / "frame_manifests.txt").exists()

    selected = json.loads((output_dir / "selected_windows.json").read_text(encoding="utf-8"))
    assert len(selected) == 2
    assert all(row["window_segment_id"].startswith("segment-001__window-") for row in selected)
    assert all(row["score"] >= 0 for row in selected)

    for row in selected:
        manifest = json.loads(Path(row["frame_manifest"]).read_text(encoding="utf-8"))
        assert manifest["video_id"] == video_id
        assert manifest["segment_id"] == row["window_segment_id"]
        assert len(manifest["frames"]) == 8
        frame_indices = [frame["frame_index"] for frame in manifest["frames"]]
        assert frame_indices == sorted(frame_indices)
        assert all(Path(frame["path"]).exists() for frame in manifest["frames"])


def test_windows_rank_renders_best_review_from_mocked_bundles(tmp_path: Path):
    inventory, annotations, video_id = _prepare_window_inputs(tmp_path)
    proposal_dir = tmp_path / "window_run"
    frames_root = tmp_path / "frames"

    proposed = runner.invoke(
        app,
        [
            "windows",
            "propose",
            "--media-inventory",
            str(inventory),
            "--annotations",
            str(annotations),
            "--output-dir",
            str(proposal_dir),
            "--frames-root",
            str(frames_root),
            "--video-id",
            video_id,
            "--window-sec",
            "2.0",
            "--stride-sec",
            "1.0",
            "--candidate-limit",
            "2",
            "--frames",
            "8",
        ],
    )
    assert proposed.exit_code == 0, proposed.output

    selected = json.loads((proposal_dir / "selected_windows.json").read_text(encoding="utf-8"))
    vggt_root = tmp_path / "vggt"
    for row in selected:
        create_mock_bundle(
            Path(row["frame_manifest"]),
            vggt_root / video_id / row["window_segment_id"],
        )

    output_dir = tmp_path / "ranked"
    ranked = runner.invoke(
        app,
        [
            "windows",
            "rank",
            "--proposal-run",
            str(proposal_dir),
            "--vggt-root",
            str(vggt_root),
            "--output-dir",
            str(output_dir),
            "--stitched-review",
        ],
    )

    assert ranked.exit_code == 0, ranked.output
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "done"
    assert summary["best_window_count"] == 1
    assert (output_dir / "index.html").exists()
    assert (output_dir / "stitched_review.html").exists()

    best = json.loads((output_dir / "best_windows.json").read_text(encoding="utf-8"))
    assert best[0]["video_id"] == video_id
    assert best[0]["review_html"]
    assert Path(best[0]["review_html"]).exists()


def test_windows_h100_package_uses_selected_window_manifests(tmp_path: Path):
    inventory, annotations, video_id = _prepare_window_inputs(tmp_path)
    proposal_dir = tmp_path / "window_run"
    frames_root = tmp_path / "frames"
    proposed = runner.invoke(
        app,
        [
            "windows",
            "propose",
            "--media-inventory",
            str(inventory),
            "--annotations",
            str(annotations),
            "--output-dir",
            str(proposal_dir),
            "--frames-root",
            str(frames_root),
            "--video-id",
            video_id,
            "--candidate-limit",
            "1",
            "--frames",
            "6",
        ],
    )
    assert proposed.exit_code == 0, proposed.output

    workdir = tmp_path / "h100_windows"
    packaged = runner.invoke(
        app,
        [
            "windows",
            "h100-package",
            "--proposal-run",
            str(proposal_dir),
            "--workdir",
            str(workdir),
            "--media-inventory",
            str(inventory),
            "--annotations",
            str(annotations),
            "--frames-root",
            str(frames_root),
        ],
    )

    assert packaged.exit_code == 0, packaged.output
    summary = json.loads((workdir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "done"
    assert summary["clip_count"] == 1
    assert (workdir / "runpod_job.zip").exists()



def test_windows_colab_t4_creates_2fps_window_cloud_package(tmp_path: Path):
    inventory, annotations, video_id = _prepare_window_inputs(tmp_path)
    output_dir = tmp_path / "colab_t4"
    frames_root = tmp_path / "frames"

    result = runner.invoke(
        app,
        [
            "windows",
            "colab-t4",
            "--media-inventory",
            str(inventory),
            "--annotations",
            str(annotations),
            "--frames-root",
            str(frames_root),
            "--output-dir",
            str(output_dir),
            "--video-id",
            video_id,
            "--window-sec",
            "4.0",
            "--stride-sec",
            "2.0",
            "--target-fps",
            "2.0",
            "--max-frames",
            "10",
            "--candidate-limit",
            "1",
            "--resized-long-edge",
            "160",
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "done"
    assert summary["inputs"]["actual_frames_per_window"] == 8
    assert summary["window_count"] == 1
    assert (output_dir / "NEXT_STEPS.md").exists()

    zip_path = output_dir / "cloud_vggt_job" / "cloud_vggt_job.zip"
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert "run_vggt_job.py" in names
    assert "job_manifest.json" in names
    assert not any(name.lower().endswith(".mp4") for name in names)
