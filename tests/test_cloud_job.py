import json
import py_compile
import shutil
import subprocess
import sys
import zipfile
import importlib.util
import hashlib
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from fpv_vggt_lab.cli import app
from tests.test_viz_compare_smoothing import create_mocked_summary


runner = CliRunner()


def test_cloud_job_package_contains_manifests_frames_and_runner(tmp_path: Path):
    manifests: list[Path] = []
    for index in range(3):
        frame_manifest, _bundle, _summary, _video = create_mocked_summary(
            tmp_path, f"cloud-video-{index}"
        )
        manifests.append(frame_manifest)

    output_dir = tmp_path / "cloud_job"
    result = runner.invoke(
        app,
        [
            "vggt",
            "cloud-job",
            "--frame-manifest",
            str(manifests[0]),
            "--frame-manifest",
            str(manifests[1]),
            "--frame-manifest",
            str(manifests[2]),
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "cloud_vggt_job.zip" in result.output
    assert "python -m zipfile -e cloud_vggt_job.zip ." in result.output
    assert "python run_vggt_job.py" in result.output
    assert "bundles.zip" in result.output
    assert (output_dir / "run_vggt_job.py").exists()
    assert (output_dir / "README.md").exists()
    assert (output_dir / "job_manifest.json").exists()
    assert (output_dir / "cloud_vggt_job.zip").exists()
    manifest = json.loads((output_dir / "job_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["clips"]) == 3
    assert manifest["warnings"] == [
        "local-only media-derived frames",
        "no geolocation",
        "no meters",
        "relative VGGT frame",
    ]
    for clip in manifest["clips"]:
        assert clip["bundle_output"] == f"bundles/{clip['video_id']}/{clip['segment_id']}"
        assert clip["predictions_output"] == (
            f"predictions/{clip['video_id']}/{clip['segment_id']}/predictions.npz"
        )
        clip_dir = output_dir / "clips" / clip["clip_id"]
        assert (clip_dir / "frames.json").exists()
        assert (clip_dir / "frames").exists()
        assert len(list((clip_dir / "frames").glob("*.jpg"))) == 6
        frame_manifest_text = (clip_dir / "frames.json").read_text(encoding="utf-8")
        frame_manifest = json.loads(frame_manifest_text)
        assert clip["frame_count"] == 6
        assert clip["frame_manifest_sha256"] == hashlib.sha256(
            (clip_dir / "frames.json").read_bytes()
        ).hexdigest()
        assert frame_manifest["source_video"] == "source_video_not_packaged"
        assert all(str(row["path"]).startswith("frames/") for row in frame_manifest["frames"])
        assert all("\\" not in str(row["path"]) for row in frame_manifest["frames"])
    runner_text = (output_dir / "run_vggt_job.py").read_text(encoding="utf-8")
    assert "VGGT.from_pretrained" in runner_text
    assert "pip\", \"install\", \"-e\"" not in runner_text
    assert "cloud_run.log" in runner_text
    assert "cloud_summary.json" in runner_text
    assert "bundles.zip" in runner_text
    assert "validate_normalized_bundle" in runner_text
    assert "bundle_valid" in runner_text
    assert 'summary["status"] = "done_partial"' in runner_text
    assert 'if summary["status"] == "failed"' in runner_text
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "cloud_summary.json" in readme
    assert "bundles.zip" in readme
    assert "frame_manifest_sha256" in readme
    py_compile.compile(str(output_dir / "run_vggt_job.py"), doraise=True)


def test_cloud_job_runner_marks_mixed_clip_results_done_partial(tmp_path: Path):
    good_manifest, good_bundle, _summary, _video = create_mocked_summary(
        tmp_path, "mixed-cloud-good"
    )
    bad_manifest, _bad_bundle, _bad_summary, _bad_video = create_mocked_summary(
        tmp_path, "mixed-cloud-bad"
    )

    output_dir = tmp_path / "cloud_job_mixed"
    result = runner.invoke(
        app,
        [
            "vggt",
            "cloud-job",
            "--frame-manifest",
            str(good_manifest),
            "--frame-manifest",
            str(bad_manifest),
            "--output",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    manifest_path = output_dir / "job_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    good_clip, bad_clip = manifest["clips"]
    shutil.copytree(good_bundle, output_dir / good_clip["bundle_output"])
    bad_clip["frame_manifest_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    runner_script = output_dir / "run_vggt_job.py"
    runner_script.write_text(
        runner_script.read_text(encoding="utf-8").replace(
            'if __name__ == "__main__":\n    main()',
            'def ensure_dependencies() -> None:\n    return\n\nif __name__ == "__main__":\n    main()',
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(runner_script)],
        cwd=output_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    cloud_summary = json.loads((output_dir / "cloud_summary.json").read_text(encoding="utf-8"))
    assert cloud_summary["status"] == "done_partial"
    assert [clip["status"] for clip in cloud_summary["clips"]] == ["done", "failed"]
    assert cloud_summary["clips"][0]["skipped_existing"] is True
    assert "frame_manifest_sha256 mismatch" in cloud_summary["clips"][1]["error"]
    assert (output_dir / "bundles.zip").exists()


def test_cloud_job_package_excludes_stale_local_artifacts(tmp_path: Path):
    frame_manifest, _bundle, _summary, _video = create_mocked_summary(
        tmp_path, "clean-package-video"
    )
    output_dir = tmp_path / "cloud_job"
    (output_dir / "__pycache__").mkdir(parents=True)
    (output_dir / "__pycache__" / "run_vggt_job.pyc").write_bytes(b"stale bytecode")
    (output_dir / "clips" / "old-clip" / "frames").mkdir(parents=True)
    (output_dir / "clips" / "old-clip" / "frames" / "old.jpg").write_bytes(b"stale frame")
    (output_dir / "bundles" / "old-video" / "segment-001").mkdir(parents=True)
    (output_dir / "bundles" / "old-video" / "segment-001" / "metadata.json").write_text(
        "{}", encoding="utf-8"
    )
    (output_dir / "cloud_run.log").write_text("old run\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "vggt",
            "cloud-job",
            "--frame-manifest",
            str(frame_manifest),
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    with zipfile.ZipFile(output_dir / "cloud_vggt_job.zip") as archive:
        names = set(archive.namelist())
    assert not any(name.startswith("__pycache__/") for name in names)
    assert not any(name.startswith("clips/old-clip/") for name in names)
    assert not any(name.startswith("bundles/") for name in names)
    assert "cloud_run.log" not in names
    assert "job_manifest.json" in names
    assert "run_vggt_job.py" in names


def test_import_cloud_job_bundles_installs_valid_returned_bundles(tmp_path: Path):
    frame_manifests: list[Path] = []
    returned_bundles = tmp_path / "returned" / "bundles"
    for index in range(3):
        frame_manifest, bundle, _summary, _video = create_mocked_summary(
            tmp_path, f"returned-video-{index}"
        )
        frame_manifests.append(frame_manifest)
        destination = returned_bundles / f"returned-video-{index}" / "segment-001"
        shutil.copytree(bundle, destination)

    output_root = tmp_path / "data" / "vggt"
    report_path = tmp_path / "import_report.json"
    result = runner.invoke(
        app,
        [
            "vggt",
            "import-cloud-job",
            "--source",
            str(returned_bundles),
            "--output-root",
            str(output_root),
            "--report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "done"
    assert report["imported_count"] == 3
    assert report["warnings"] == [
        "no geolocation",
        "no meters",
        "relative VGGT frame",
        "local-only media",
    ]

    installed_bundles = [
        output_root / f"returned-video-{index}" / "segment-001" for index in range(3)
    ]
    assert all((bundle / "metadata.json").exists() for bundle in installed_bundles)
    assert all((bundle / "cameras.npz").exists() for bundle in installed_bundles)

    review = runner.invoke(
        app,
        [
            "run",
            "--workdir",
            str(tmp_path / "review"),
            "--frame-manifest",
            str(frame_manifests[0]),
            "--bundle",
            str(installed_bundles[0]),
            "--frame-manifest",
            str(frame_manifests[1]),
            "--bundle",
            str(installed_bundles[1]),
            "--frame-manifest",
            str(frame_manifests[2]),
            "--bundle",
            str(installed_bundles[2]),
            "--smooth",
        ],
    )
    assert review.exit_code == 0, review.output


def test_import_cloud_job_dry_run_validates_without_installing(tmp_path: Path):
    returned_bundles = tmp_path / "returned" / "bundles"
    for index in range(3):
        _frame_manifest, bundle, _summary, _video = create_mocked_summary(
            tmp_path, f"dry-return-video-{index}"
        )
        shutil.copytree(bundle, returned_bundles / f"dry-return-video-{index}" / "segment-001")

    output_root = tmp_path / "data" / "vggt"
    report_path = tmp_path / "dry_import_report.json"
    result = runner.invoke(
        app,
        [
            "vggt",
            "import-cloud-job",
            "--source",
            str(returned_bundles),
            "--output-root",
            str(output_root),
            "--report",
            str(report_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "done"
    assert report["dry_run"] is True
    assert report["imported_count"] == 0
    assert report["would_import_count"] == 3
    assert all(not clip["copied"] for clip in report["clips"])
    assert all(clip["would_copy"] for clip in report["clips"])
    assert not output_root.exists()


def test_import_cloud_job_dry_run_filters_expected_bundles_from_run_summary(tmp_path: Path):
    expected_video_ids = [f"expected-return-video-{index}" for index in range(3)]
    returned_bundles = tmp_path / "returned" / "bundles"
    for video_id in expected_video_ids:
        _frame_manifest, bundle, _summary, _video = create_mocked_summary(tmp_path, video_id)
        shutil.copytree(bundle, returned_bundles / video_id / "segment-001")
    _frame_manifest, extra_bundle, _summary, _video = create_mocked_summary(
        tmp_path, "unexpected-return-video"
    )
    shutil.copytree(extra_bundle, returned_bundles / "unexpected-return-video" / "segment-001")

    run_summary = tmp_path / "run" / "summary.json"
    run_summary.parent.mkdir(parents=True)
    run_summary.write_text(
        json.dumps(
            {
                "inputs": {
                    "video_ids": expected_video_ids,
                    "segment_id": "segment-001",
                }
            }
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "data" / "vggt"
    report_path = tmp_path / "dry_import_report.json"
    result = runner.invoke(
        app,
        [
            "vggt",
            "import-cloud-job",
            "--source",
            str(returned_bundles),
            "--output-root",
            str(output_root),
            "--report",
            str(report_path),
            "--expected-from-run",
            str(run_summary),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "done"
    assert report["dry_run"] is True
    assert report["would_import_count"] == 3
    assert report["skipped_unexpected_count"] == 1
    assert report["missing_expected_bundles"] == []
    assert sorted(report["expected_bundles"], key=lambda row: row["video_id"]) == [
        {"video_id": video_id, "segment_id": "segment-001"}
        for video_id in expected_video_ids
    ]
    unexpected = next(
        clip for clip in report["clips"] if clip["metadata"]["video_id"] == "unexpected-return-video"
    )
    assert unexpected["copied"] is False
    assert unexpected["would_copy"] is False
    assert unexpected["skipped_reason"] == "unexpected bundle for this run"
    assert not output_root.exists()


def test_import_cloud_job_dry_run_flags_failed_expected_cloud_summary_clip(tmp_path: Path):
    expected_video_ids = [f"summary-return-video-{index}" for index in range(3)]
    returned_root = tmp_path / "returned"
    returned_bundles = returned_root / "bundles"
    for video_id in expected_video_ids:
        _frame_manifest, bundle, _summary, _video = create_mocked_summary(tmp_path, video_id)
        shutil.copytree(bundle, returned_bundles / video_id / "segment-001")

    bundle_zip = returned_root / "bundles.zip"
    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in returned_bundles.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(returned_root))

    (returned_root / "cloud_summary.json").write_text(
        json.dumps(
            {
                "status": "failed_soft",
                "clips": [
                    {
                        "video_id": expected_video_ids[0],
                        "segment_id": "segment-001",
                        "status": "done",
                        "bundle_valid": True,
                    },
                    {
                        "video_id": expected_video_ids[1],
                        "segment_id": "segment-001",
                        "status": "failed",
                        "bundle_valid": False,
                        "error": "CUDA out of memory",
                    },
                    {
                        "video_id": expected_video_ids[2],
                        "segment_id": "segment-001",
                        "status": "done",
                        "bundle_valid": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (returned_root / "cloud_run.log").write_text(
        "one expected clip failed\n", encoding="utf-8"
    )

    run_summary = tmp_path / "run" / "summary.json"
    run_summary.parent.mkdir(parents=True)
    run_summary.write_text(
        json.dumps(
            {
                "inputs": {
                    "video_ids": expected_video_ids,
                    "segment_id": "segment-001",
                }
            }
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "data" / "vggt"
    report_path = tmp_path / "dry_import_report.json"
    result = runner.invoke(
        app,
        [
            "vggt",
            "import-cloud-job",
            "--source",
            str(bundle_zip),
            "--output-root",
            str(output_root),
            "--report",
            str(report_path),
            "--expected-from-run",
            str(run_summary),
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed_soft"
    assert report["dry_run"] is True
    assert report["would_import_count"] == 3
    assert report["cloud_summary_audit"]["status"] == "needs_review"
    assert report["cloud_summary_audit"]["issues"] == [
        "cloud_summary status is failed_soft",
        (
            "expected clip summary-return-video-1/segment-001 status is failed "
            "with bundle_valid=False: CUDA out of memory"
        ),
    ]
    assert not output_root.exists()


def test_import_cloud_job_refuses_to_install_when_cloud_summary_needs_review(
    tmp_path: Path,
):
    expected_video_ids = [f"blocked-return-video-{index}" for index in range(3)]
    returned_root = tmp_path / "returned"
    returned_bundles = returned_root / "bundles"
    for video_id in expected_video_ids:
        _frame_manifest, bundle, _summary, _video = create_mocked_summary(tmp_path, video_id)
        shutil.copytree(bundle, returned_bundles / video_id / "segment-001")

    bundle_zip = returned_root / "bundles.zip"
    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in returned_bundles.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(returned_root))

    (returned_root / "cloud_summary.json").write_text(
        json.dumps(
            {
                "status": "failed_soft",
                "clips": [
                    {
                        "video_id": expected_video_ids[0],
                        "segment_id": "segment-001",
                        "status": "done",
                        "bundle_valid": True,
                    },
                    {
                        "video_id": expected_video_ids[1],
                        "segment_id": "segment-001",
                        "status": "failed",
                        "bundle_valid": False,
                        "error": "CUDA out of memory",
                    },
                    {
                        "video_id": expected_video_ids[2],
                        "segment_id": "segment-001",
                        "status": "done",
                        "bundle_valid": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    run_summary = tmp_path / "run" / "summary.json"
    run_summary.parent.mkdir(parents=True)
    run_summary.write_text(
        json.dumps(
            {
                "inputs": {
                    "video_ids": expected_video_ids,
                    "segment_id": "segment-001",
                }
            }
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "data" / "vggt"
    report_path = tmp_path / "import_report.json"
    result = runner.invoke(
        app,
        [
            "vggt",
            "import-cloud-job",
            "--source",
            str(bundle_zip),
            "--output-root",
            str(output_root),
            "--report",
            str(report_path),
            "--expected-from-run",
            str(run_summary),
        ],
    )

    assert result.exit_code != 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed_soft"
    assert report["imported_count"] == 0
    assert all(not clip["copied"] for clip in report["clips"])
    assert all(
        "cloud_summary_audit needs review; run dry-run first" in clip["errors"]
        for clip in report["clips"]
        if clip["metadata"] is not None
    )
    assert not output_root.exists()


def test_import_cloud_job_accepts_returned_bundles_zip_with_cloud_report(tmp_path: Path):
    returned_root = tmp_path / "returned"
    returned_bundles = returned_root / "bundles"
    for index in range(3):
        _frame_manifest, bundle, _summary, _video = create_mocked_summary(
            tmp_path, f"zip-return-video-{index}"
        )
        shutil.copytree(bundle, returned_bundles / f"zip-return-video-{index}" / "segment-001")

    bundle_zip = returned_root / "bundles.zip"
    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in returned_bundles.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(returned_root))
    (returned_root / "cloud_summary.json").write_text(
        json.dumps({"status": "done", "clips": [{"status": "done"} for _ in range(3)]}),
        encoding="utf-8",
    )
    (returned_root / "cloud_run.log").write_text("cloud run completed\n", encoding="utf-8")

    output_root = tmp_path / "data" / "vggt"
    report_path = tmp_path / "reports" / "cloud_bundle_import.json"
    result = runner.invoke(
        app,
        [
            "vggt",
            "import-cloud-job",
            "--source",
            str(bundle_zip),
            "--output-root",
            str(output_root),
            "--report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "done"
    assert report["source_was_zip"] is True
    assert Path(report["extracted_to"]).exists()
    assert report["cloud_summary"]["status"] == "done"
    assert report["cloud_log"] == str(returned_root / "cloud_run.log")
    assert report["imported_count"] == 3
    assert all(
        (output_root / f"zip-return-video-{index}" / "segment-001" / "metadata.json").exists()
        for index in range(3)
    )




def test_generated_cloud_runner_derives_point_colors_from_packaged_frames(tmp_path: Path):
    frame_manifest, _bundle, _summary, _video = create_mocked_summary(
        tmp_path, "rgb-cloud-video"
    )
    output_dir = tmp_path / "cloud_job"
    result = runner.invoke(
        app,
        [
            "vggt",
            "cloud-job",
            "--frame-manifest",
            str(frame_manifest),
            "--output",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    spec = importlib.util.spec_from_file_location(
        "generated_rgb_runner", output_dir / "run_vggt_job.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    manifest = json.loads((output_dir / "job_manifest.json").read_text(encoding="utf-8"))
    packaged_manifest = output_dir / manifest["clips"][0]["frame_manifest"]
    predictions_path = tmp_path / "predictions_rgb.npz"
    points = np.zeros((2, 3, 4, 3), dtype=np.float32)
    points[..., 2] = 1.0
    np.savez_compressed(predictions_path, world_points_from_depth=points)

    with np.load(predictions_path, allow_pickle=False) as pred:
        bundle = module.point_bundle(pred, packaged_manifest, max_points=24)

    assert bundle is not None
    assert bundle["points"].shape == (24, 3)
    assert bundle["point_colors_rgb"].shape == (24, 3)
    assert bundle["point_colors_rgb"].dtype == np.uint8
    assert int(bundle["point_colors_rgb"].sum()) > 0

def test_cloud_job_runner_skips_existing_valid_bundles_on_rerun(tmp_path: Path):
    frame_manifest, bundle, _summary, _video = create_mocked_summary(
        tmp_path, "resume-video"
    )
    output_dir = tmp_path / "cloud_job"
    result = runner.invoke(
        app,
        [
            "vggt",
            "cloud-job",
            "--frame-manifest",
            str(frame_manifest),
            "--output",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    returned_bundle = output_dir / "bundles" / "resume-video" / "segment-001"
    shutil.copytree(bundle, returned_bundle)

    spec = importlib.util.spec_from_file_location(
        "generated_cloud_runner", output_dir / "run_vggt_job.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.ensure_dependencies = lambda: None

    def fail_if_called(_frame_manifest: Path, _predictions_output: Path) -> None:
        raise AssertionError("VGGT should not rerun when a valid bundle already exists")

    module.run_vggt = fail_if_called
    module.main()

    cloud_summary = json.loads((output_dir / "cloud_summary.json").read_text(encoding="utf-8"))
    assert cloud_summary["status"] == "done"
    assert cloud_summary["clips"][0]["status"] == "done"
    assert cloud_summary["clips"][0]["bundle_valid"] is True
    assert cloud_summary["clips"][0]["skipped_existing"] is True


def test_cloud_job_runner_rejects_non_finite_normalized_arrays(tmp_path: Path):
    frame_manifest, bundle, _summary, _video = create_mocked_summary(
        tmp_path, "cloud-finite-video"
    )
    output_dir = tmp_path / "cloud_job"
    result = runner.invoke(
        app,
        [
            "vggt",
            "cloud-job",
            "--frame-manifest",
            str(frame_manifest),
            "--output",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    returned_bundle = output_dir / "bundles" / "cloud-finite-video" / "segment-001"
    shutil.copytree(bundle, returned_bundle)
    cameras_path = returned_bundle / "cameras.npz"
    cameras = np.load(cameras_path)
    quaternions = cameras["quaternions_xyzw"].copy()
    quaternions[0, 0] = np.inf
    pose_confidence = cameras["pose_confidence"].copy()
    pose_confidence[1] = np.nan
    np.savez_compressed(
        cameras_path,
        camera_centers=cameras["camera_centers"],
        quaternions_xyzw=quaternions,
        valid_pose_mask=cameras["valid_pose_mask"],
        pose_confidence=pose_confidence,
    )
    points_path = returned_bundle / "points.npz"
    points_data = np.load(points_path)
    points = points_data["points"].copy()
    points[0, 0] = np.inf
    np.savez_compressed(points_path, points=points)

    spec = importlib.util.spec_from_file_location(
        "generated_finite_runner", output_dir / "run_vggt_job.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    errors = module.validate_normalized_bundle(returned_bundle, frame_count=6)

    assert "quaternions_xyzw contains non-finite values" in errors
    assert "pose_confidence contains non-finite values" in errors
    assert "points contains non-finite values" in errors


def test_cloud_job_runner_rejects_ambiguous_frame_metadata(tmp_path: Path):
    frame_manifest, bundle, _summary, _video = create_mocked_summary(
        tmp_path, "cloud-metadata-video"
    )
    output_dir = tmp_path / "cloud_job"
    result = runner.invoke(
        app,
        [
            "vggt",
            "cloud-job",
            "--frame-manifest",
            str(frame_manifest),
            "--output",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    returned_bundle = output_dir / "bundles" / "cloud-metadata-video" / "segment-001"
    shutil.copytree(bundle, returned_bundle)
    metadata_path = returned_bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["frame_indices"] = [0, 5, 5, 14, 18, 23]
    metadata["frame_timestamps_sec"] = [0.0, 0.4, 0.3, 1.0, 1.5, 1.9]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    spec = importlib.util.spec_from_file_location(
        "generated_metadata_runner", output_dir / "run_vggt_job.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    errors = module.validate_normalized_bundle(returned_bundle, frame_count=6)

    assert "metadata frame_indices must be strictly increasing" in errors
    assert "metadata frame_timestamps_sec must be non-decreasing" in errors


def test_cloud_job_runner_checks_frame_manifest_provenance_before_vggt(tmp_path: Path):
    frame_manifest, _bundle, _summary, _video = create_mocked_summary(
        tmp_path, "tamper-video"
    )
    output_dir = tmp_path / "cloud_job"
    result = runner.invoke(
        app,
        [
            "vggt",
            "cloud-job",
            "--frame-manifest",
            str(frame_manifest),
            "--output",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    manifest = json.loads((output_dir / "job_manifest.json").read_text(encoding="utf-8"))
    packaged_manifest = output_dir / manifest["clips"][0]["frame_manifest"]
    packaged_data = json.loads(packaged_manifest.read_text(encoding="utf-8"))
    packaged_data["frames"] = packaged_data["frames"][:-1]
    packaged_manifest.write_text(json.dumps(packaged_data), encoding="utf-8")

    spec = importlib.util.spec_from_file_location(
        "generated_tamper_runner", output_dir / "run_vggt_job.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.ensure_dependencies = lambda: None
    called_vggt = {"value": False}

    def fail_if_called(_frame_manifest: Path, _predictions_output: Path) -> None:
        called_vggt["value"] = True
        raise AssertionError("VGGT should not run when packaged frame manifest changed")

    module.run_vggt = fail_if_called
    with pytest.raises(SystemExit):
        module.main()

    cloud_summary = json.loads((output_dir / "cloud_summary.json").read_text(encoding="utf-8"))
    assert cloud_summary["status"] == "failed"
    assert called_vggt["value"] is False
    assert cloud_summary["clips"][0]["status"] == "failed"
    assert "frame_count mismatch" in cloud_summary["clips"][0]["error"]
    assert "frame_manifest_sha256 mismatch" in cloud_summary["clips"][0]["error"]
