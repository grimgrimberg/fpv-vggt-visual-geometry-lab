import json
import zipfile
from pathlib import Path

from typer.testing import CliRunner

from fpv_vggt_lab.cli import app
from tests.test_viz_compare_smoothing import create_mocked_summary


runner = CliRunner()


def test_h100_prepare_packages_frame_manifests_without_raw_video(tmp_path: Path):
    manifests = []
    for index in range(2):
        frame_manifest, _bundle, _summary, _video = create_mocked_summary(
            tmp_path, f"h100-package-video-{index}"
        )
        manifests.append(frame_manifest)

    workdir = tmp_path / "h100-run"
    result = runner.invoke(
        app,
        [
            "h100",
            "prepare",
            "--dataset",
            "none",
            "--workdir",
            str(workdir),
            "--frame-manifest",
            str(manifests[0]),
            "--frame-manifest",
            str(manifests[1]),
            "--metadata-policy",
            "provenance-only",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "runpod_job.zip" in result.output
    assert (workdir / "run.log").exists()
    assert (workdir / "summary.json").exists()
    assert (workdir / "NEXT_STEPS.md").exists()
    assert (workdir / "frame_pack_manifest.parquet").exists()
    assert (workdir / "runpod_job" / "run_all.sh").exists()
    assert (workdir / "runpod_job" / "run_vggt_job.py").exists()
    assert (workdir / "runpod_job" / "scripts" / "00_env_check.py").exists()
    assert (workdir / "runpod_job" / "scripts" / "70_package_return.py").exists()
    assert (workdir / "runpod_job.zip").exists()

    summary = json.loads((workdir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "done"
    assert summary["stage_status"]["package"] == "done"
    assert summary["warnings"] == [
        "no geolocation",
        "no meters",
        "relative VGGT frame",
        "local-only media",
    ]

    manifest = json.loads(
        (workdir / "runpod_job" / "job_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "h100-job-v1"
    assert manifest["metadata_policy"] == "provenance-only"
    assert len(manifest["clips"]) == 2
    assert {clip["tier"] for clip in manifest["clips"]} == {"main"}
    assert all(clip["bundle_output"].startswith("bundles/main/") for clip in manifest["clips"])
    assert all(clip["frame_manifest"].startswith("frame_packs/main/") for clip in manifest["clips"])
    assert "D:\\" not in json.dumps(manifest)
    assert "C:\\" not in json.dumps(manifest)

    packaged_frame_manifest = (
        workdir / "runpod_job" / manifest["clips"][0]["frame_manifest"]
    )
    frame_data = json.loads(packaged_frame_manifest.read_text(encoding="utf-8"))
    assert frame_data["source_video"] == "source_video_not_packaged"
    assert all(str(row["path"]).startswith("frames/") for row in frame_data["frames"])

    with zipfile.ZipFile(workdir / "runpod_job.zip") as archive:
        names = set(archive.namelist())
    assert "run_all.sh" in names
    assert "run_vggt_job.py" in names
    assert "scripts/70_package_return.py" in names
    assert "job_manifest.json" in names
    assert not any(name.lower().endswith(".mp4") for name in names)
    assert not any(":\\" in name for name in names)
    assert not any(name.startswith("/") for name in names)


def test_h100_prepare_without_inputs_writes_actionable_partial_run(tmp_path: Path):
    workdir = tmp_path / "empty-h100-run"
    result = runner.invoke(
        app,
        [
            "h100",
            "prepare",
            "--dataset",
            "none",
            "--workdir",
            str(workdir),
            "--annotations",
            str(tmp_path / "empty_segments.jsonl"),
            "--media-inventory",
            str(tmp_path / "missing_media_inventory.parquet"),
        ],
    )

    assert result.exit_code != 0
    assert "needs_human_review" in result.output
    summary = json.loads((workdir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "needs_human_review"
    assert summary["stage_status"]["package"] == "blocked"
    next_steps = (workdir / "NEXT_STEPS.md").read_text(encoding="utf-8")
    assert "fpv h100 prepare" in next_steps
    assert "--frame-manifest" in next_steps
    assert "No geolocation" in next_steps

