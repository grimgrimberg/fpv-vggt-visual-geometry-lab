import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from fpv_vggt_lab.cli import app
from tests.test_viz_compare_smoothing import create_mocked_summary


runner = CliRunner()


def test_vggt_export_request_and_import_bundle_normalization(tmp_path: Path):
    frame_manifest, source_bundle, _summary, _video = create_mocked_summary(
        tmp_path, "handoff-video"
    )
    request_dir = tmp_path / "request"
    imported_bundle = tmp_path / "imported_bundle"

    exported = runner.invoke(
        app,
        [
            "vggt",
            "export-request",
            "--frame-manifest",
            str(frame_manifest),
            "--output",
            str(request_dir),
        ],
    )
    assert exported.exit_code == 0, exported.output
    assert (request_dir / "frames").exists()
    assert (request_dir / "frames.zip").exists()
    assert (request_dir / "request_manifest.json").exists()
    assert "Hugging Face VGGT" in (request_dir / "README.md").read_text(encoding="utf-8")

    imported = runner.invoke(
        app,
        [
            "vggt",
            "import-bundle",
            "--source",
            str(source_bundle),
            "--frame-manifest",
            str(frame_manifest),
            "--output",
            str(imported_bundle),
            "--source-tool",
            "huggingface-space",
            "--source-url-or-repo",
            "https://huggingface.co/spaces/facebook/vggt",
            "--source-commit-or-version",
            "manual-export",
            "--export-notes",
            "Converted from external VGGT outputs.",
        ],
    )
    assert imported.exit_code == 0, imported.output
    metadata = json.loads((imported_bundle / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_tool"] == "huggingface-space"
    assert metadata["source_url_or_repo"] == "https://huggingface.co/spaces/facebook/vggt"
    assert metadata["frame_indices"] == [0, 5, 9, 14, 18, 23]
    assert (imported_bundle / "cameras.npz").exists()
    assert (imported_bundle / "points.npz").exists()

    validated = runner.invoke(app, ["vggt", "validate", "--bundle", str(imported_bundle)])
    assert validated.exit_code == 0, validated.output


def test_vggt_import_predictions_npz_from_official_space_shape(tmp_path: Path):
    frame_manifest, _source_bundle, _summary, _video = create_mocked_summary(
        tmp_path, "predictions-video"
    )
    predictions_path = tmp_path / "predictions.npz"
    output = tmp_path / "predictions_bundle"

    extrinsic = np.repeat(np.eye(4, dtype=np.float32)[None, :3, :], 6, axis=0)
    extrinsic[:, 0, 3] = -np.linspace(0.0, 1.0, 6, dtype=np.float32)
    world_points = np.zeros((6, 2, 2, 3), dtype=np.float32)
    for frame_index in range(6):
        world_points[frame_index, :, :, 0] = frame_index
        world_points[frame_index, :, :, 1] = np.array([[0.0, 0.1], [0.2, 0.3]])
        world_points[frame_index, :, :, 2] = np.array([[0.5, 0.6], [0.7, 0.8]])
    depth_conf = np.full((6, 2, 2), 0.75, dtype=np.float32)
    np.savez_compressed(
        predictions_path,
        extrinsic=extrinsic,
        world_points_from_depth=world_points,
        depth_conf=depth_conf,
    )

    imported = runner.invoke(
        app,
        [
            "vggt",
            "import-predictions",
            "--predictions",
            str(predictions_path),
            "--frame-manifest",
            str(frame_manifest),
            "--output",
            str(output),
            "--source-url-or-repo",
            "https://huggingface.co/spaces/facebook/vggt",
        ],
    )

    assert imported.exit_code == 0, imported.output
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_tool"] == "huggingface-space"
    assert metadata["video_id"] == "predictions-video"
    cameras = np.load(output / "cameras.npz")
    assert cameras["camera_centers"].shape == (6, 3)
    assert cameras["quaternions_xyzw"].shape == (6, 4)
    assert np.allclose(cameras["pose_confidence"], 0.75)
    points = np.load(output / "points.npz")["points"]
    assert points.shape == (24, 3)

    validated = runner.invoke(app, ["vggt", "validate", "--bundle", str(output)])
    assert validated.exit_code == 0, validated.output


def test_vggt_run_installed_fails_clearly_without_optional_vggt(tmp_path: Path):
    frame_manifest, _source_bundle, _summary, _video = create_mocked_summary(
        tmp_path, "installed-run-video"
    )

    result = runner.invoke(
        app,
        [
            "vggt",
            "run-installed",
            "--frame-manifest",
            str(frame_manifest),
            "--predictions-output",
            str(tmp_path / "predictions.npz"),
            "--bundle-output",
            str(tmp_path / "bundle"),
        ],
    )

    assert result.exit_code != 0
    assert "optional VGGT dependency is not installed" in result.output


def test_review_run_three_uses_existing_imported_bundles(tmp_path: Path):
    frame_manifests: list[Path] = []
    bundles: list[Path] = []
    for index in range(3):
        frame_manifest, bundle, _summary, _video = create_mocked_summary(
            tmp_path, f"runner-video-{index}"
        )
        frame_manifests.append(frame_manifest)
        bundles.append(bundle)

    output_dir = tmp_path / "review-run"
    result = runner.invoke(
        app,
        [
            "review",
            "run-three",
            "--frame-manifest",
            str(frame_manifests[0]),
            "--bundle",
            str(bundles[0]),
            "--frame-manifest",
            str(frame_manifests[1]),
            "--bundle",
            str(bundles[1]),
            "--frame-manifest",
            str(frame_manifests[2]),
            "--bundle",
            str(bundles[2]),
            "--output-dir",
            str(output_dir),
            "--smooth",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (output_dir / "comparison.html").exists()
    assert (output_dir / "run_report.json").exists()
    report = json.loads((output_dir / "run_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "done"
    assert len(report["clips"]) == 3
    assert all(Path(clip["review_html"]).exists() for clip in report["clips"])
    assert all(Path(clip["summary"]).exists() for clip in report["clips"])
    assert all(Path(clip["smoothing_output"]).exists() for clip in report["clips"])
    comparison_html = (output_dir / "comparison.html").read_text(encoding="utf-8")
    for clip in report["clips"]:
        relative_review = Path(clip["review_html"]).relative_to(output_dir).as_posix()
        assert f'href="{relative_review}"' in comparison_html
    assert "Open review" in comparison_html
    assert "local-only media" in comparison_html
    assert report["warnings"] == [
        "no geolocation",
        "no meters",
        "relative VGGT frame",
        "local-only media",
    ]


def test_review_audit_three_reports_missing_and_ready_bundles(tmp_path: Path):
    frame_manifests: list[Path] = []
    bundles: list[Path] = []
    for index in range(3):
        frame_manifest, bundle, _summary, _video = create_mocked_summary(
            tmp_path, f"audit-video-{index}"
        )
        frame_manifests.append(frame_manifest)
        bundles.append(bundle)

    missing = runner.invoke(
        app,
        [
            "review",
            "audit-three",
            "--frame-manifest",
            str(frame_manifests[0]),
            "--bundle",
            str(bundles[0]),
            "--frame-manifest",
            str(frame_manifests[1]),
            "--bundle",
            str(bundles[1]),
            "--frame-manifest",
            str(frame_manifests[2]),
            "--bundle",
            str(tmp_path / "missing-bundle"),
            "--output",
            str(tmp_path / "missing_audit.json"),
        ],
    )
    assert missing.exit_code != 0
    missing_report = json.loads((tmp_path / "missing_audit.json").read_text(encoding="utf-8"))
    assert missing_report["ready"] is False
    assert any(not clip["bundle_exists"] for clip in missing_report["clips"])

    ready = runner.invoke(
        app,
        [
            "review",
            "audit-three",
            "--frame-manifest",
            str(frame_manifests[0]),
            "--bundle",
            str(bundles[0]),
            "--frame-manifest",
            str(frame_manifests[1]),
            "--bundle",
            str(bundles[1]),
            "--frame-manifest",
            str(frame_manifests[2]),
            "--bundle",
            str(bundles[2]),
            "--output",
            str(tmp_path / "ready_audit.json"),
        ],
    )
    assert ready.exit_code == 0, ready.output
    ready_report = json.loads((tmp_path / "ready_audit.json").read_text(encoding="utf-8"))
    assert ready_report["ready"] is True
    assert all(clip["bundle_valid"] for clip in ready_report["clips"])


def test_top_level_run_reports_needs_vggt_bundle_with_next_steps(tmp_path: Path):
    frame_manifests: list[Path] = []
    bundles: list[Path] = []
    for index in range(3):
        frame_manifest, bundle, _summary, _video = create_mocked_summary(
            tmp_path, f"top-run-missing-{index}"
        )
        frame_manifests.append(frame_manifest)
        bundles.append(bundle)

    workdir = tmp_path / "top-run-missing"
    result = runner.invoke(
        app,
        [
            "run",
            "--workdir",
            str(workdir),
            "--frame-manifest",
            str(frame_manifests[0]),
            "--bundle",
            str(bundles[0]),
            "--frame-manifest",
            str(frame_manifests[1]),
            "--bundle",
            str(bundles[1]),
            "--frame-manifest",
            str(frame_manifests[2]),
            "--bundle",
            str(tmp_path / "missing-bundle"),
        ],
    )

    assert result.exit_code != 0
    assert "needs_vggt_bundle" in result.output
    summary = json.loads((workdir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "needs_vggt_bundle"
    assert summary["stage_status"]["audit_three"] == "needs_vggt_bundle"
    assert Path(summary["artifacts"]["audit_report"]).exists()
    assert Path(summary["artifacts"]["run_log"]).exists()
    assert Path(summary["artifacts"]["summary_json"]).exists()
    assert Path(summary["artifacts"]["next_steps"]).exists()
    next_steps = (workdir / "NEXT_STEPS.md").read_text(encoding="utf-8")
    assert "fpv vggt cloud-job" in next_steps
    assert "cloud_bundle_import_dry_run.json" in next_steps
    assert "--expected-from-run outputs/reviews/three_clip_run/summary.json" in next_steps
    assert "--dry-run" in next_steps
    assert "no geolocation" in next_steps


def test_top_level_run_executes_three_clip_review_when_ready(tmp_path: Path):
    frame_manifests: list[Path] = []
    bundles: list[Path] = []
    for index in range(3):
        frame_manifest, bundle, _summary, _video = create_mocked_summary(
            tmp_path, f"top-run-ready-{index}"
        )
        frame_manifests.append(frame_manifest)
        bundles.append(bundle)

    workdir = tmp_path / "top-run-ready"
    result = runner.invoke(
        app,
        [
            "run",
            "--workdir",
            str(workdir),
            "--frame-manifest",
            str(frame_manifests[0]),
            "--bundle",
            str(bundles[0]),
            "--frame-manifest",
            str(frame_manifests[1]),
            "--bundle",
            str(bundles[1]),
            "--frame-manifest",
            str(frame_manifests[2]),
            "--bundle",
            str(bundles[2]),
            "--smooth",
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads((workdir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "done"
    assert summary["stage_status"]["audit_three"] == "done"
    assert summary["stage_status"]["review_three"] == "done"
    assert Path(summary["artifacts"]["review_report"]).exists()
    assert Path(summary["artifacts"]["comparison_html"]).exists()
    assert (workdir / "run.log").exists()
