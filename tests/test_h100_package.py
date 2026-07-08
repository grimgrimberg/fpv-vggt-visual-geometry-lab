import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from fpv_vggt_lab.cli import app
from fpv_vggt_lab.h100_package import _emergency_return_packager_source, write_runpod_launch_manifest
from fpv_vggt_lab.method_contract import expanded_return_artifact_files, validate_return_layout
from tests.test_viz_compare_smoothing import create_mocked_summary


runner = CliRunner()


def test_emergency_return_packager_source_writes_minimal_return_zip(tmp_path: Path):
    (tmp_path / "cloud_run.log").write_text("cloud log\n", encoding="utf-8")
    (tmp_path / "environment.json").write_text(json.dumps({"python": "test"}), encoding="utf-8")
    failure_dir = tmp_path / "failure_packages"
    failure_dir.mkdir()
    (failure_dir / "expanded_method_stage_error.json").write_text(
        json.dumps({"status": "failed_soft", "error": "boom"}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-c", _emergency_return_packager_source()],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "emergency_return_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "h100-emergency-return-v1"
    assert manifest["status"] == "failed_soft"
    assert (tmp_path / "h100_return.zip").exists()
    with zipfile.ZipFile(tmp_path / "h100_return.zip") as archive:
        names = set(archive.namelist())
    assert "emergency_return_manifest.json" in names
    assert "cloud_run.log" in names
    assert "environment.json" in names
    assert "failure_packages/expanded_method_stage_error.json" in names


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
    assert (workdir / "runpod_job" / "scripts" / "05_expanded_runtime_check.py").exists()
    assert (workdir / "runpod_job" / "scripts" / "06_hf_dataset_preflight.py").exists()
    assert (workdir / "runpod_job" / "scripts" / "07_hf_dataset_frame_packs.py").exists()
    assert (workdir / "runpod_job" / "scripts" / "70_package_return.py").exists()
    assert (workdir / "runpod_job" / "method_matrix.json").exists()
    assert (workdir / "runpod_job" / "return_contract.json").exists()
    assert (workdir / "runpod_job" / "METHOD_CONTRACT.md").exists()
    assert (workdir / "runpod_job" / "method_stage_plan.json").exists()
    assert (workdir / "runpod_job" / "scripts" / "80_run_expanded_methods.py").exists()
    assert (workdir / "runpod_job.zip").exists()

    cloud_script_paths = [
        "run_all.sh",
        "run_vggt_job.py",
        "scripts/00_env_check.py",
        "scripts/05_expanded_runtime_check.py",
        "scripts/06_hf_dataset_preflight.py",
        "scripts/07_hf_dataset_frame_packs.py",
        "scripts/70_package_return.py",
        "scripts/80_run_expanded_methods.py",
    ]
    for rel_path in cloud_script_paths:
        script_bytes = (workdir / "runpod_job" / rel_path).read_bytes()
        assert bytes([13]) not in script_bytes, rel_path
    run_all_text = (workdir / "runpod_job" / "run_all.sh").read_text(encoding="utf-8")
    assert "emergency_return_manifest.json" in run_all_text
    assert "70_package_return.py failed or did not create h100_return.zip" in run_all_text
    assert "python scripts/06_hf_dataset_preflight.py" in run_all_text
    assert "HF_DATASET_STATUS" in run_all_text
    assert "HF_FRAME_PACK_STATUS" in run_all_text
    assert "cloud_run_status.json" in run_all_text
    assert "exit 0" in run_all_text
    assert "REQUIRE_HF_DATASET" in run_all_text
    assert "HF_DATASET_BUILD_FRAME_PACKS" not in run_all_text

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
    assert manifest["schema_version"] == "runpod-job-v1"
    assert manifest["profile"] == "runpod-gpu-inference"
    assert "RTX 4090" in manifest["recommended_gpu"]
    assert manifest["metadata_policy"] == "provenance-only"
    assert manifest["method_contract"] == {
        "method_matrix": "method_matrix.json",
        "return_contract": "return_contract.json",
        "method_contract": "METHOD_CONTRACT.md",
        "method_stage_plan": "method_stage_plan.json",
    }
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
    assert "scripts/05_expanded_runtime_check.py" in names
    assert "scripts/06_hf_dataset_preflight.py" in names
    assert "scripts/07_hf_dataset_frame_packs.py" in names
    assert "scripts/70_package_return.py" in names
    assert "job_manifest.json" in names
    assert "method_matrix.json" in names
    assert "return_contract.json" in names
    assert "METHOD_CONTRACT.md" in names
    assert "method_stage_plan.json" in names
    assert "scripts/80_run_expanded_methods.py" in names
    assert not any(name.lower().endswith(".mp4") for name in names)
    assert not any(":\\" in name for name in names)
    assert not any(name.startswith("/") for name in names)

    inspected_dir = runner.invoke(
        app,
        ["h100", "inspect-package", "--source", str(workdir / "runpod_job")],
    )
    assert inspected_dir.exit_code == 0, inspected_dir.output
    dir_payload = json.loads(inspected_dir.output)
    assert dir_payload["valid"] is True
    assert dir_payload["clip_count"] == 2
    assert dir_payload["has_expanded_runtime_check"] is True
    assert dir_payload["has_hf_dataset_preflight"] is True
    assert dir_payload["has_hf_dataset_frame_pack_builder"] is True
    assert dir_payload["has_expanded_methods"] is True
    assert "vggt_colmap_ba_windowed" in dir_payload["method_ids"]

    expanded_runner = workdir / "runpod_job" / "scripts" / "80_run_expanded_methods.py"
    expanded_runner_text = expanded_runner.read_text(encoding="utf-8")
    expanded_runner.unlink()
    inspected_missing_runner = runner.invoke(
        app,
        ["h100", "inspect-package", "--source", str(workdir / "runpod_job")],
    )
    assert inspected_missing_runner.exit_code != 0
    assert "missing required package file: scripts/80_run_expanded_methods.py" in inspected_missing_runner.output
    expanded_runner.write_text(expanded_runner_text, encoding="utf-8")

    inspected_zip = runner.invoke(
        app,
        ["h100", "inspect-package", "--source", str(workdir / "runpod_job.zip")],
    )
    assert inspected_zip.exit_code == 0, inspected_zip.output
    zip_payload = json.loads(inspected_zip.output)
    assert zip_payload["valid"] is True
    assert zip_payload["clip_count"] == 2
    assert zip_payload["zip_sha256"]
    assert zip_payload["zip_sha256_skipped"] is False
    assert zip_payload["zip_integrity_test_skipped"] is False

    inspected_zip_fast = runner.invoke(
        app,
        ["h100", "inspect-package", "--source", str(workdir / "runpod_job.zip"), "--fast"],
    )
    assert inspected_zip_fast.exit_code == 0, inspected_zip_fast.output
    fast_payload = json.loads(inspected_zip_fast.output)
    assert fast_payload["valid"] is True
    assert fast_payload["clip_count"] == 2
    assert fast_payload["zip_sha256"] is None
    assert fast_payload["zip_sha256_skipped"] is True
    assert fast_payload["zip_integrity_test_skipped"] is True



def test_runpod_smoke_package_checks_script_syntax_without_gpu(tmp_path: Path):
    frame_manifest, _bundle, _summary, _video = create_mocked_summary(tmp_path, "runpod-smoke-package-video")
    workdir = tmp_path / "runpod-smoke"
    prepared = runner.invoke(
        app,
        [
            "h100",
            "prepare",
            "--dataset",
            "none",
            "--workdir",
            str(workdir),
            "--frame-manifest",
            str(frame_manifest),
        ],
    )
    assert prepared.exit_code == 0, prepared.output

    result = runner.invoke(app, ["runpod", "smoke-package", "--source", str(workdir / "runpod_job.zip")])

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["schema_version"] == "runpod-package-smoke-v1"
    assert report["valid"] is True
    assert report["gpu_execution"] is False
    assert report["checks"]["package_contract"]["clip_count"] == 1
    assert all(row["status"] == "passed" for row in report["checks"]["python_syntax"])
    assert report["checks"]["shell_syntax"][0]["status"] in {"passed", "skipped_bash_not_found", "skipped_bash_unavailable"}
    assert "This smoke check does not run VGGT" in report["next_actions"][1]

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




def test_h100_return_packager_writes_compact_selected_zip(tmp_path: Path):
    _frame_manifest, bundle, _summary, _video = create_mocked_summary(
        tmp_path, "h100-compact-return-video"
    )
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
            str(_frame_manifest),
            "--metadata-policy",
            "provenance-only",
        ],
    )
    assert result.exit_code == 0, result.output

    runpod = workdir / "runpod_job"
    bundle_target = runpod / "bundles" / "high_detail" / "h100-compact-return-video" / "segment-001"
    shutil.copytree(bundle, bundle_target)
    (runpod / "cloud_summary.json").write_text(
        json.dumps({"status": "done", "clips": [{"status": "done"}]}),
        encoding="utf-8",
    )
    (runpod / "cloud_run.log").write_text("fake cloud log\n", encoding="utf-8")
    (runpod / "environment.json").write_text(json.dumps({"python": "test"}), encoding="utf-8")
    predictions = runpod / "predictions" / "high_detail" / "raw" / "predictions.npz"
    predictions.parent.mkdir(parents=True)
    predictions.write_bytes(b"raw prediction payload that must not be returned")
    job_manifest = json.loads((runpod / "job_manifest.json").read_text(encoding="utf-8"))
    clip = job_manifest["clips"][0]
    (runpod / "cloud_summary.json").write_text(
        json.dumps({"status": "done", "clips": [{"clip_id": clip["clip_id"], "status": "done"}]}),
        encoding="utf-8",
    )
    cloud_predictions = runpod / clip["predictions_output"]
    cloud_predictions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cloud_predictions,
        depth=np.linspace(0.1, 4.0, 4 * 12 * 10, dtype=np.float32).reshape(4, 12, 10, 1),
    )
    fake_vggt = runpod / "fake_vggt"
    fake_vggt.mkdir()
    (fake_vggt / "demo_colmap.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "scene = Path(sys.argv[sys.argv.index('--scene_dir') + 1])\n"
        "sparse = scene / 'sparse' / '0'\n"
        "sparse.mkdir(parents=True, exist_ok=True)\n"
        "(sparse / 'cameras.txt').write_text('# fake sparse model\\n', encoding='utf-8')\n"
        "(sparse / 'images.txt').write_text('# fake sparse images\\n', encoding='utf-8')\n"
        "(sparse / 'points3D.txt').write_text('# fake sparse points\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    extra_frame = runpod / "frame_packs" / "high_detail" / "frame.jpg"
    extra_frame.parent.mkdir(parents=True, exist_ok=True)
    extra_frame.write_bytes(b"frame payload that must not be returned")
    (runpod / "bundles.zip").write_bytes(b"all tier archive should not be nested")
    (runpod / "cloud_run_status.json").write_text(
        json.dumps({"status": "done_partial", "stage_exit_codes": {"job_status": 1}}),
        encoding="utf-8",
    )

    runtime = subprocess.run(
        [sys.executable, str(runpod / "scripts" / "05_expanded_runtime_check.py")],
        cwd=runpod,
        text=True,
        capture_output=True,
        check=False,
    )
    assert runtime.returncode == 0, runtime.stderr
    runtime_report = json.loads((runpod / "expanded_runtime_check.json").read_text(encoding="utf-8"))
    assert runtime_report["status"] == "done"
    assert runtime_report["readiness_status"] in {"ready_full_stack", "ready_4090_pilot", "blocked_primary_runtime"}
    assert runtime_report["vggt_inference_required_checks"] == ["vggt_feedforward_ready"]
    assert runtime_report["full_stack_required_checks"] == [
        "vggt_feedforward_ready",
        "vggt_colmap_ba_ready",
        "classical_colmap_ready",
    ]
    assert runtime_report["primary_required_checks"] == runtime_report["full_stack_required_checks"]
    assert "next_actions" in runtime_report
    assert "vggt_colmap_ba_ready" in runtime_report["checks"]
    assert set(runtime_report["missing_primary_checks"]).issubset(runtime_report["primary_required_checks"])
    assert set(runtime_report["missing_vggt_inference_checks"]).issubset(runtime_report["vggt_inference_required_checks"])
    assert set(runtime_report["missing_full_stack_checks"]).issubset(runtime_report["full_stack_required_checks"])
    assert "research_method_runners" in runtime_report
    assert "dust3r_builtin_runner_ready" in runtime_report["checks"]
    assert "mast3r_builtin_runner_ready" in runtime_report["checks"]
    assert "vggetr_template_ready" in runtime_report["checks"]
    assert "VG2GT_COMMAND_TEMPLATE" in runtime_report["research_method_runners"]["command_templates"]
    assert "dust3r_builtin_python_api" in runtime_report["research_method_runners"]
    assert "mast3r_builtin_python_api" in runtime_report["research_method_runners"]
    assert "command_templates" in runtime_report["research_method_runners"]

    hf_optional = subprocess.run(
        [sys.executable, str(runpod / "scripts" / "06_hf_dataset_preflight.py")],
        cwd=runpod,
        text=True,
        capture_output=True,
        check=False,
    )
    assert hf_optional.returncode == 0, hf_optional.stderr
    hf_report = json.loads((runpod / "hf_dataset_report.json").read_text(encoding="utf-8"))
    assert hf_report["status"] == "skipped_no_dataset_id"
    assert hf_report["token_present"] is False
    assert "no geolocation" in hf_report["safety_warnings"]

    required_env = os.environ.copy()
    required_env["REQUIRE_HF_DATASET"] = "1"
    hf_required = subprocess.run(
        [sys.executable, str(runpod / "scripts" / "06_hf_dataset_preflight.py")],
        cwd=runpod,
        env=required_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert hf_required.returncode == 2
    hf_required_report = json.loads((runpod / "hf_dataset_report.json").read_text(encoding="utf-8"))
    assert hf_required_report["status"] == "skipped_no_dataset_id"
    assert hf_required_report["required"] is True

    hf_frame_pack_disabled = subprocess.run(
        [sys.executable, str(runpod / "scripts" / "07_hf_dataset_frame_packs.py")],
        cwd=runpod,
        text=True,
        capture_output=True,
        check=False,
    )
    assert hf_frame_pack_disabled.returncode == 0, hf_frame_pack_disabled.stderr
    hf_frame_pack_disabled_report = json.loads((runpod / "hf_frame_pack_report.json").read_text(encoding="utf-8"))
    assert hf_frame_pack_disabled_report["status"] == "skipped_disabled"

    original_job_manifest_text = (runpod / "job_manifest.json").read_text(encoding="utf-8")
    snapshot = tmp_path / "hf_snapshot"
    image_group = snapshot / "image_clip_alpha"
    image_group.mkdir(parents=True)
    original_frame_dir = Path(json.loads(_frame_manifest.read_text(encoding="utf-8"))["frames"][0]["path"]).parent
    for index, source in enumerate(sorted(original_frame_dir.glob("*.jpg"))[:5]):
        shutil.copy2(source, image_group / f"frame_{index:03d}.jpg")
    hf_pack_env = os.environ.copy()
    hf_pack_env["HF_DATASET_BUILD_FRAME_PACKS"] = "1"
    hf_pack_env["HF_DATASET_LOCAL_DIR"] = str(snapshot)
    hf_pack_env["HF_DATASET_FRAME_PACK_TIERS"] = "scout"
    hf_pack_env["HF_DATASET_SCOUT_FRAMES"] = "3"
    hf_frame_pack = subprocess.run(
        [sys.executable, str(runpod / "scripts" / "07_hf_dataset_frame_packs.py")],
        cwd=runpod,
        env=hf_pack_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert hf_frame_pack.returncode == 0, hf_frame_pack.stderr
    hf_frame_pack_report = json.loads((runpod / "hf_frame_pack_report.json").read_text(encoding="utf-8"))
    assert hf_frame_pack_report["status"] == "done"
    assert hf_frame_pack_report["clip_count"] == 1
    hf_manifest = json.loads((runpod / "job_manifest.json").read_text(encoding="utf-8"))
    assert hf_manifest["hf_dataset_frame_pack_source"]["clip_count"] == 1
    assert hf_manifest["clips"][0]["tier"] == "scout"
    assert hf_manifest["clips"][0]["video_id"] == "image_clip_alpha"
    assert hf_manifest["clips"][0]["frame_count"] == 3
    hf_stage_plan = json.loads((runpod / "method_stage_plan.json").read_text(encoding="utf-8"))
    assert hf_stage_plan["clip_count"] == 1
    assert hf_stage_plan["hf_dataset_frame_pack_source"]["clip_count"] == 1
    hf_frames_path = runpod / hf_manifest["clips"][0]["frame_manifest"]
    hf_frames = json.loads(hf_frames_path.read_text(encoding="utf-8"))
    assert len(hf_frames["frames"]) == 3
    assert (hf_frames_path.parent / hf_frames["frames"][0]["path"]).exists()
    assert (runpod / "frame_packs.packaged_backup").exists()
    (runpod / "job_manifest.json").write_text(original_job_manifest_text, encoding="utf-8")
    for name in ["frame_packs", "bundles", "predictions"]:
        current = runpod / name
        backup = runpod / f"{name}.packaged_backup"
        if backup.exists():
            if current.exists():
                shutil.rmtree(current)
            backup.rename(current)

    expanded_env = os.environ.copy()
    expanded_env["VGGT_REPO_DIR"] = str(fake_vggt)
    expanded = subprocess.run(
        [sys.executable, str(runpod / "scripts" / "80_run_expanded_methods.py")],
        cwd=runpod,
        env=expanded_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert expanded.returncode == 0, expanded.stderr
    method_report = json.loads((runpod / "method_stage_report.json").read_text(encoding="utf-8"))
    assert method_report["status"] == "done_partial"
    assert any(stage["method_id"] == "trajectory_preprocess_evo_eval" for stage in method_report["stages"])
    assert any(stage["status"] == "skipped_missing_dependency" for stage in method_report["stages"])
    assert (runpod / "trajectories.zip").exists()
    assert (runpod / "evo_reports.zip").exists()
    assert (runpod / "odm_artifacts.zip").exists()
    assert (runpod / "odm_project_reports.zip").exists()
    assert (runpod / "odm_logs.zip").exists()
    assert (runpod / "odm_relative_artifacts_manifest.json").exists()
    assert (runpod / "openmvs_artifacts.zip").exists()
    assert (runpod / "openmvs_project_reports.zip").exists()
    assert (runpod / "openmvs_logs.zip").exists()
    assert (runpod / "openmvs_dense_mesh_manifest.json").exists()
    openmvs_manifest = json.loads((runpod / "openmvs_dense_mesh_manifest.json").read_text(encoding="utf-8"))
    assert openmvs_manifest["execution_enabled"] is False
    assert openmvs_manifest["generated_project_count"] == 1
    assert openmvs_manifest["projects"][0]["image_count"] > 0
    assert openmvs_manifest["projects"][0]["status"] in {
        "prepared_missing_dependency",
        "ready_pending_execution_disabled",
    }
    openmvs_project_dir = runpod / openmvs_manifest["projects"][0]["project_dir"]
    openmvs_report_dir = runpod / openmvs_manifest["projects"][0]["report_dir"]
    assert (openmvs_project_dir / "README.md").exists()
    assert (openmvs_report_dir / "run_openmvs.sh").exists()
    odm_manifest = json.loads((runpod / "odm_relative_artifacts_manifest.json").read_text(encoding="utf-8"))
    assert odm_manifest["execution_enabled"] is False
    assert odm_manifest["generated_project_count"] == 1
    assert odm_manifest["projects"][0]["image_count"] > 0
    assert odm_manifest["projects"][0]["status"] in {
        "prepared_missing_dependency",
        "ready_pending_execution_disabled",
    }
    odm_project_dir = runpod / odm_manifest["projects"][0]["project_dir"]
    odm_report_dir = runpod / odm_manifest["projects"][0]["report_dir"]
    assert (odm_project_dir / "images").exists()
    assert (odm_report_dir / "run_odm.sh").exists()
    assert (runpod / "nerfstudio_gsplat_showcase.zip").exists()
    assert (runpod / "nerfstudio_projects.zip").exists()
    assert (runpod / "gsplat_exports.zip").exists()
    assert (runpod / "showcase_renders.zip").exists()
    assert (runpod / "showcase_manifest.json").exists()
    showcase_manifest = json.loads((runpod / "showcase_manifest.json").read_text(encoding="utf-8"))
    assert showcase_manifest["execution_enabled"] is False
    assert showcase_manifest["generated_project_count"] == 1
    assert showcase_manifest["projects"][0]["image_count"] > 0
    assert showcase_manifest["projects"][0]["status"] in {
        "prepared_missing_dependency",
        "ready_pending_execution_disabled",
    }
    nerfstudio_project_dir = runpod / showcase_manifest["projects"][0]["project_dir"]
    assert (nerfstudio_project_dir / "raw_images").exists()
    assert (nerfstudio_project_dir / "processed").exists()
    assert (nerfstudio_project_dir / "runs").exists()
    assert (nerfstudio_project_dir / "run_nerfstudio.sh").exists()
    assert (nerfstudio_project_dir / "project_report.json").exists()
    assert (runpod / "relative_depth_overlays.zip").exists()
    assert (runpod / "depth_overlay_manifest.json").exists()
    depth_manifest = json.loads((runpod / "depth_overlay_manifest.json").read_text(encoding="utf-8"))
    assert depth_manifest["status"] == "done"
    assert depth_manifest["generated_overlay_count"] == 1
    assert depth_manifest["overlay_rows"][0]["output"].endswith("relative_depth_contact_sheet.svg")
    assert (runpod / "research_methods.zip").exists()
    assert (runpod / "research_methods_outputs.zip").exists()
    assert (runpod / "research_methods_manifest.json").exists()
    assert (runpod / "candidate_identity.md").exists()
    assert (runpod / "vggetr_feasibility_report.json").exists()
    research_methods_manifest = json.loads((runpod / "research_methods_manifest.json").read_text(encoding="utf-8"))
    assert research_methods_manifest["execution_enabled"] is True
    assert research_methods_manifest["generated_project_count"] == 1
    research_method_project = research_methods_manifest["projects"][0]
    assert research_method_project["image_count"] > 0
    assert research_method_project["status"] in {
        "prepared_missing_dependency",
        "prepared_missing_command_template",
        "ready_pending_execution_disabled",
    }
    research_method_project_dir = runpod / research_method_project["project_dir"]
    assert (research_method_project_dir / "raw_images").exists()
    assert (research_method_project_dir / "outputs").exists()
    assert (research_method_project_dir / "run_research_methods.sh").exists()
    assert (research_method_project_dir / "run_research_method.py").exists()
    runner_source = (research_method_project_dir / "run_research_method.py").read_text(encoding="utf-8")
    assert "run_dust3r" in runner_source
    assert "run_mast3r" in runner_source
    assert "DUST3R_MODEL_NAME" in runner_source
    assert "MAST3R_MODEL_NAME" in runner_source
    assert (research_method_project_dir / "method_input.json").exists()
    assert (research_method_project_dir / "project_report.json").exists()
    assert (runpod / research_method_project["contract"]).exists()
    assert len(research_method_project["methods"]) == 3
    methods_by_id = {row["method_id"]: row for row in research_method_project["methods"]}
    assert methods_by_id["dust3r_optional_baseline"]["builtin_runner"] == "dust3r"
    assert methods_by_id["mast3r_optional_baseline"]["builtin_runner"] == "mast3r"
    assert methods_by_id["vggetr_candidate_baseline"]["builtin_runner"] is None
    assert "VG2GT_COMMAND_TEMPLATE" in methods_by_id["vggetr_candidate_baseline"]["template_envs"]
    assert "vg2gt_module" in methods_by_id["vggetr_candidate_baseline"]["module_keys"]
    feasibility = json.loads((runpod / "vggetr_feasibility_report.json").read_text(encoding="utf-8"))
    assert feasibility["requires_command_template"] == "VGGETR_COMMAND_TEMPLATE, VG2GT_COMMAND_TEMPLATE, or RESEARCH_METHOD_COMMAND_TEMPLATE"
    assert "vg2gt_module" in feasibility["dependency_probe"]
    assert feasibility["candidate_project_count"] == 1
    assert (runpod / "points_ply.zip").exists()
    assert (runpod / "points_ply_report.json").exists()
    assert (runpod / "converted_colmap_bundles.zip").exists()
    assert (runpod / "quality_report.json").exists()
    assert (runpod / "trajectory_qc.json").exists()
    trajectory_qc = json.loads((runpod / "trajectory_qc.json").read_text(encoding="utf-8"))
    assert trajectory_qc["artifacts"]["trajectory_qc_csv"] == "evo_reports/trajectory_qc.csv"
    assert trajectory_qc["trajectories"][0]["relative_path_length"] >= 0.0
    assert trajectory_qc["warning"].startswith("relative VGGT-frame")
    points_report = json.loads((runpod / "points_ply_report.json").read_text(encoding="utf-8"))
    assert points_report["artifact"] == "points_ply.zip"
    with zipfile.ZipFile(runpod / "converted_colmap_bundles.zip") as archive:
        converted_names = set(archive.namelist())
    assert "converted_colmap_bundles/conversion_manifest.json" in converted_names
    assert "converted_colmap_bundles/README.md" in converted_names
    with zipfile.ZipFile(runpod / "evo_reports.zip") as archive:
        evo_names = set(archive.namelist())
    assert "evo_reports/trajectory_qc.csv" in evo_names
    with zipfile.ZipFile(runpod / "relative_depth_overlays.zip") as archive:
        depth_names = set(archive.namelist())
    assert any(name.endswith("relative_depth_contact_sheet.svg") for name in depth_names)
    assert "relative_depth_overlays/overlay_manifest.json" in depth_names
    with zipfile.ZipFile(runpod / "odm_project_reports.zip") as archive:
        odm_report_names = set(archive.namelist())
    assert any(name.endswith("run_odm.sh") for name in odm_report_names)
    assert any(name.endswith("project_report.json") for name in odm_report_names)
    with zipfile.ZipFile(runpod / "openmvs_project_reports.zip") as archive:
        openmvs_report_names = set(archive.namelist())
    assert any(name.endswith("run_openmvs.sh") for name in openmvs_report_names)
    assert any(name.endswith("project_report.json") for name in openmvs_report_names)
    with zipfile.ZipFile(runpod / "nerfstudio_projects.zip") as archive:
        nerfstudio_names = set(archive.namelist())
    assert any(name.endswith("run_nerfstudio.sh") for name in nerfstudio_names)
    assert any(name.endswith("project_report.json") for name in nerfstudio_names)
    assert any("raw_images/" in name for name in nerfstudio_names)
    with zipfile.ZipFile(runpod / "research_methods_outputs.zip") as archive:
        research_method_output_names = set(archive.namelist())
    assert any(name.endswith("run_research_methods.sh") for name in research_method_output_names)
    assert any(name.endswith("run_research_method.py") for name in research_method_output_names)
    assert any(name.endswith("project_report.json") for name in research_method_output_names)
    assert any(name.endswith("method_input.json") for name in research_method_output_names)
    assert any("raw_images/" in name for name in research_method_output_names)
    with zipfile.ZipFile(runpod / "research_methods.zip") as archive:
        research_method_names = set(archive.namelist())
    assert any(name.endswith("method_output_contract.json") for name in research_method_names)


    completed = subprocess.run(
        [sys.executable, str(runpod / "scripts" / "70_package_return.py")],
        cwd=runpod,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (runpod / "DOWNLOAD_ME.md").exists()
    download_me = (runpod / "DOWNLOAD_ME.md").read_text(encoding="utf-8")
    assert "h100_return.zip" in download_me
    assert "fpv h100 inspect-return" in download_me
    assert "fpv h100 postflight-return" in download_me
    assert "runpod_launch_4090_hf_manifest.json" in download_me
    assert "runpod_launch_manifest.json" not in download_me
    assert "fpv h100 optional-report" in download_me
    assert "no geolocation" in download_me

    with zipfile.ZipFile(runpod / "h100_return.zip") as archive:
        names = set(archive.namelist())
        return_manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    assert return_manifest["status"] == "done_partial"
    assert return_manifest["cloud_run_status"] == "cloud_run_status.json"
    assert "manifest.json" in names
    assert "method_matrix.json" in names
    assert "return_contract.json" in names
    assert "METHOD_CONTRACT.md" in names
    assert "method_stage_plan.json" in names
    assert "method_stage_report.json" in names
    assert "expanded_runtime_check.json" in names
    assert "hf_dataset_report.json" in names
    assert "hf_frame_pack_report.json" in names
    assert "cloud_run_status.json" in names
    assert "quality_report.json" in names
    assert "trajectory_qc.json" in names
    if (runpod / "trajectory_qc.parquet").exists():
        assert "trajectory_qc.parquet" in names
    assert "DOWNLOAD_ME.md" in names
    assert "trajectories.zip" in names
    assert "evo_reports.zip" in names
    assert "odm_artifacts.zip" in names
    assert "odm_project_reports.zip" in names
    assert "odm_logs.zip" in names
    assert "odm_relative_artifacts_manifest.json" in names
    assert "openmvs_artifacts.zip" in names
    assert "openmvs_project_reports.zip" in names
    assert "openmvs_logs.zip" in names
    assert "openmvs_dense_mesh_manifest.json" in names
    assert "nerfstudio_gsplat_showcase.zip" in names
    assert "nerfstudio_projects.zip" in names
    assert "gsplat_exports.zip" in names
    assert "showcase_renders.zip" in names
    assert "showcase_manifest.json" in names
    assert "relative_depth_overlays.zip" in names
    assert "depth_overlay_manifest.json" in names
    assert "research_methods.zip" in names
    assert "research_methods_outputs.zip" in names
    assert "research_methods_manifest.json" in names
    assert "candidate_identity.md" in names
    assert "vggetr_feasibility_report.json" in names
    assert "points_ply.zip" in names
    assert "points_ply_report.json" in names
    assert "converted_colmap_bundles.zip" in names
    assert any(name.startswith("selected/h100-compact-return-video/segment-001/") for name in names)
    assert not any(name.startswith("predictions/") for name in names)
    assert not any(name.startswith("frame_packs/") for name in names)
    assert not any(name.startswith("bundles/") for name in names)
    assert "bundles.zip" not in names
    contract_report = validate_return_layout(runpod / "h100_return.zip")
    assert contract_report["valid"], contract_report["issues"]



def test_return_packager_include_list_covers_contract_artifacts():
    from fpv_vggt_lab.h100_package import _return_packager_include_files

    include_files = set(_return_packager_include_files())
    assert set(expanded_return_artifact_files()).issubset(include_files)
    assert "method_stage_report.json" in include_files
    assert "hf_dataset_report.json" in include_files
    assert "hf_frame_pack_report.json" in include_files
    assert "cloud_run_status.json" in include_files
    assert "odm_artifacts.zip" in include_files
    assert "openmvs_artifacts.zip" in include_files
    assert "openmvs_dense_mesh_manifest.json" in include_files
    assert "nerfstudio_gsplat_showcase.zip" in include_files



def test_h100_launch_manifest_writes_operator_checklist(tmp_path: Path):
    frame_manifest, _bundle, _summary, _video = create_mocked_summary(
        tmp_path, "h100-launch-manifest-video"
    )
    workdir = tmp_path / "h100-launch-run"
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
            str(frame_manifest),
            "--metadata-policy",
            "provenance-only",
        ],
    )
    assert result.exit_code == 0, result.output

    output_dir = tmp_path / "launch"
    manifest = write_runpod_launch_manifest(workdir / "runpod_job.zip", output_dir)

    assert manifest["status"] == "ready_to_upload"
    assert manifest["package"]["sha256"]
    assert manifest["package"]["clip_count"] == 1
    assert "vggt_colmap_ba_windowed" in manifest["methods"]
    assert "python scripts/05_expanded_runtime_check.py" in manifest["remote"]["commands"]
    assert "python scripts/06_hf_dataset_preflight.py" in manifest["remote"]["commands"]
    assert "python scripts/07_hf_dataset_frame_packs.py" in manifest["remote"]["commands"]
    assert "cat hf_dataset_report.json" in manifest["remote"]["commands"]
    assert "cat hf_frame_pack_report.json" in manifest["remote"]["commands"]
    assert any("HF_DATASET_ID=Grimster/FPV_Hezbo" in item for item in manifest["remote"]["required_env"])
    assert any("HF_DATASET_BUILD_FRAME_PACKS=1" in item for item in manifest["remote"]["required_env"])
    assert any("ready_4090_pilot" in command and "STOP" in command for command in manifest["remote"]["commands"])
    assert "bash run_all.sh" in manifest["remote"]["commands"]
    assert "RUN_STATUS=$?" in manifest["remote"]["commands"]
    assert "h100_return.zip" == manifest["expected_return"]["primary_download"]
    assert any("fpv h100 postflight-return" in command for command in manifest["expected_return"]["local_validation_commands"])
    assert any("runpod_launch_manifest.json" in command for command in manifest["expected_return"]["local_validation_commands"])
    assert not any("runpod_launch_4090_hf_manifest.json" in command for command in manifest["expected_return"]["local_validation_commands"])
    assert any("fpv h100 optional-report" in command for command in manifest["expected_return"]["local_validation_commands"])
    assert "method_stage_report.json" in manifest["expected_return"]["required_files_inside_return"]
    assert "odm_artifacts.zip" in manifest["expected_return"]["expanded_artifacts_when_available"]
    assert "openmvs_artifacts.zip" in manifest["expected_return"]["expanded_artifacts_when_available"]
    assert "no geolocation" in manifest["safety_warnings"]
    assert (output_dir / "runpod_launch_manifest.json").exists()
    assert (output_dir / "RUNPOD_LAUNCH.md").exists()
    assert (output_dir / "VALIDATE_RETURN.ps1").exists()

    saved = json.loads((output_dir / "runpod_launch_manifest.json").read_text(encoding="utf-8"))
    assert saved["artifacts"]["markdown"].endswith("RUNPOD_LAUNCH.md")
    assert saved["artifacts"]["validate_return_powershell"].endswith("VALIDATE_RETURN.ps1")
    validate_helper = (output_dir / "VALIDATE_RETURN.ps1").read_text(encoding="utf-8")
    assert "inspect-return" in validate_helper
    assert "postflight-return" in validate_helper
    assert "optional-report" in validate_helper
    assert "import-return" in validate_helper
    assert "--dry-run" in validate_helper
    assert bytes([13]) not in (output_dir / "VALIDATE_RETURN.ps1").read_bytes()
    markdown = (output_dir / "RUNPOD_LAUNCH.md").read_text(encoding="utf-8")
    assert "RunPod Launch Checklist" in markdown
    assert "VALIDATE_RETURN.ps1" in markdown
    assert "python -m zipfile -e /workspace/runpod_job.zip ." in markdown
    assert "06_hf_dataset_preflight.py" in markdown
    assert "07_hf_dataset_frame_packs.py" in markdown
    assert "hf_dataset_report.json" in markdown
    assert "hf_frame_pack_report.json" in markdown
    assert "readiness_status" in markdown
    assert "STOP" in markdown
    assert "fpv h100 inspect-return" in markdown
    assert "fpv h100 postflight-return" in markdown
    assert "fpv h100 optional-report" in markdown
    assert "no geolocation" in markdown

    cli_result = runner.invoke(
        app,
        [
            "h100",
            "launch-manifest",
            "--source",
            str(workdir / "runpod_job.zip"),
            "--output-dir",
            str(tmp_path / "launch_cli"),
        ],
    )
    assert cli_result.exit_code == 0, cli_result.output
    assert "ready_to_upload" in cli_result.output

