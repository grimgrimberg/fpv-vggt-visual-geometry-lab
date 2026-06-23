import json
from pathlib import Path

from typer.testing import CliRunner

from fpv_vggt_lab.cli import app


runner = CliRunner()


def test_cli_help_displays_commands():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "synthetic-video" in result.output
    assert "frames" in result.output
    assert "vggt" in result.output
    assert "reconstruct" in result.output
    assert "run" in result.output


def test_synthetic_video_frame_sampling_mock_bundle_and_summary(tmp_path: Path):
    video_path = tmp_path / "synthetic.mp4"
    frames_dir = tmp_path / "frames"
    bundle_dir = tmp_path / "vggt_bundle"
    summary_path = tmp_path / "summary.json"

    created = runner.invoke(
        app,
        [
            "synthetic-video",
            "create",
            "--output",
            str(video_path),
            "--frames",
            "24",
            "--width",
            "160",
            "--height",
            "120",
            "--fps",
            "12",
        ],
    )
    assert created.exit_code == 0, created.output
    assert video_path.exists()

    sampled = runner.invoke(
        app,
        [
            "frames",
            "sample",
            "--video",
            str(video_path),
            "--output",
            str(frames_dir),
            "--count",
            "6",
            "--video-id",
            "synthetic-video",
            "--segment-id",
            "segment-001",
        ],
    )
    assert sampled.exit_code == 0, sampled.output

    frame_manifest_path = frames_dir / "frames.json"
    frame_manifest = json.loads(frame_manifest_path.read_text(encoding="utf-8"))
    assert len(frame_manifest["frames"]) == 6
    assert [row["frame_index"] for row in frame_manifest["frames"]] == [
        0,
        5,
        9,
        14,
        18,
        23,
    ]
    assert all(Path(row["path"]).exists() for row in frame_manifest["frames"])
    assert all(row["quality"]["blur_score"] >= 0 for row in frame_manifest["frames"])

    mocked = runner.invoke(
        app,
        [
            "vggt",
            "mock",
            "--frame-manifest",
            str(frame_manifest_path),
            "--output",
            str(bundle_dir),
        ],
    )
    assert mocked.exit_code == 0, mocked.output
    assert (bundle_dir / "metadata.json").exists()
    assert (bundle_dir / "cameras.npz").exists()
    assert (bundle_dir / "points.npz").exists()

    validated = runner.invoke(app, ["vggt", "validate", "--bundle", str(bundle_dir)])
    assert validated.exit_code == 0, validated.output
    assert "valid" in validated.output.lower()

    summarized = runner.invoke(
        app,
        [
            "reconstruct",
            "summarize",
            "--bundle",
            str(bundle_dir),
            "--output",
            str(summary_path),
        ],
    )
    assert summarized.exit_code == 0, summarized.output

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["video_id"] == "synthetic-video"
    assert summary["segment_id"] == "segment-001"
    assert summary["reliability"]["label"] == "good"
    assert summary["reliability"]["safe_to_use_for_descriptors"] is True
    assert summary["descriptors"]["sampled_frame_count"] == 6
    assert summary["descriptors"]["valid_pose_count"] == 6
    assert "normalized_path_length" in summary["descriptors"]


def test_vggt_validate_fails_clearly_for_malformed_bundle(tmp_path: Path):
    bad_bundle = tmp_path / "bad_bundle"
    bad_bundle.mkdir()
    (bad_bundle / "metadata.json").write_text("{}", encoding="utf-8")

    result = runner.invoke(app, ["vggt", "validate", "--bundle", str(bad_bundle)])

    assert result.exit_code != 0
    assert "cameras.npz" in result.output


def test_synthetic_run_writes_debuggable_run_folder(tmp_path: Path):
    run_dir = tmp_path / "run-demo"

    result = runner.invoke(
        app,
        ["run", "--synthetic", "--workdir", str(run_dir), "--frames", "8"],
    )

    assert result.exit_code == 0, result.output
    assert (run_dir / "run.log").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "NEXT_STEPS.md").exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "done"
    assert summary["stage_status"]["synthetic_video"] == "done"
    assert summary["stage_status"]["frame_sampling"] == "done"
    assert summary["stage_status"]["mock_vggt_bundle"] == "done"
    assert summary["stage_status"]["reconstruction_summary"] == "done"
    assert summary["warnings"] == [
        "no geolocation",
        "no meters",
        "relative VGGT frame",
        "local-only media",
    ]
    assert "python" in summary["environment"]
    assert Path(summary["artifacts"]["run_log"]).exists()
    assert Path(summary["artifacts"]["summary_json"]).exists()
    assert Path(summary["artifacts"]["next_steps"]).exists()
    assert Path(summary["artifacts"]["reconstruction_summary"]).exists()
