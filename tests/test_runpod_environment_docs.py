import json
import shutil
import zipfile
from pathlib import Path

from typer.testing import CliRunner

from fpv_vggt_lab.cli import app
from fpv_vggt_lab.h100_package import audit_runpod_4090_hf_launch_kit


ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def test_runpod_expanded_environment_doc_points_to_current_package_and_runtime():
    doc = (ROOT / "docs" / "runpod_expanded_environment.md").read_text(encoding="utf-8")
    dockerfile = (ROOT / "containers" / "runpod-vggt-colmap.Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "runpod_job_expanded_methods_store.zip" in doc
    assert "runpod_job_expanded_methods.zip" in doc
    assert "stale" in doc
    assert "containers/runpod-vggt-colmap.Dockerfile" in doc
    assert "VGGT_REF" in doc
    assert "VGGT_REPO_DIR" in doc
    assert "demo_colmap.py" in doc
    assert "--use_ba" in doc
    assert "expanded_runtime_check.json" in doc
    assert "method_stage_report.json" in doc
    assert "no geolocation" in doc
    assert "no map projection" in doc

    assert "ARG VGGT_REF=" in dockerfile
    assert "ARG DUST3R_REF=" in dockerfile
    assert "ARG MAST3R_REF=" in dockerfile
    assert "ARG INSTALL_RESEARCH_METHODS=false" in dockerfile
    assert "ARG COMPILE_RESEARCH_CUDA_KERNELS=false" in dockerfile
    assert "ENV VGGT_REPO_DIR=/workspace/vggt" in dockerfile
    assert "ENV DUST3R_REPO_DIR=/workspace/dust3r" in dockerfile
    assert "ENV MAST3R_REPO_DIR=/workspace/mast3r" in dockerfile
    assert "git clone --recursive https://github.com/naver/dust3r.git" in dockerfile
    assert "git clone --recursive https://github.com/naver/mast3r.git" in dockerfile
    assert "colmap" in dockerfile
    assert "pycolmap" in dockerfile
    assert "ARG INSTALL_OPENMVS=" in dockerfile
    assert "InterfaceCOLMAP" in dockerfile
    assert "DensifyPointCloud" in dockerfile
    assert "trimesh" in dockerfile
    assert "evo" in dockerfile

    assert "INSTALL_RESEARCH_METHODS=true" in doc
    assert "DUST3R_REF" in doc
    assert "MAST3R_REF" in doc
    assert "CC BY-NC-SA 4.0" in doc


def test_expanded_runbooks_link_to_environment_recipe():
    paths = [
        ROOT / "outputs" / "h100" / "full_161_run" / "RUNPOD_EXPANDED_SETUP.md",
        ROOT / "outputs" / "h100" / "full_161_run" / "NEXT_STEPS_EXPANDED.md",
        ROOT
        / "outputs"
        / "inference_prep"
        / "full_dataset_vggt_colmap_sota_2026-06-29"
        / "NEXT_RUN_COMMANDS.md",
        ROOT
        / "outputs"
        / "inference_prep"
        / "full_dataset_vggt_colmap_sota_2026-06-29"
        / "RUN_NEXT_INFERENCE.md",
    ]

    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "docs/runpod_expanded_environment.md" in text
        assert "containers/runpod-vggt-colmap.Dockerfile" in text

def test_4090_runbook_records_hf_dataset_and_safe_scope():
    doc = (ROOT / "docs" / "runpod_4090_full_stack.md").read_text(encoding="utf-8")
    milestone = (ROOT / "docs" / "milestones" / "RUNPOD_4090_FULL_STACK.md").read_text(
        encoding="utf-8"
    )

    for text in [doc, milestone]:
        assert "Grimster/FPV_Hezbo" in text
        assert "RTX 4090" in text
        assert "HF_TOKEN" in text
        assert "06_hf_dataset_preflight.py" in text
        assert "07_hf_dataset_frame_packs.py" in text
        assert "hf_dataset_report.json" in text
        assert "hf_frame_pack_report.json" in text
        assert "runpod_job_4090_hf_full_stack_clean.zip" in text
        assert "legacy internal contract names kept for compatibility" in text
        assert "RUNPOD_4090_PILOT_PROFILE.json" in text
        assert "no geolocation" in text.lower()

    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in doc
    assert "RESEARCH_METHOD_MAX_IMAGES=32" in doc
    assert "OPTIONAL_SHOWCASE_MAX_CLIPS=3" in doc
    assert "PRELAUNCH_AUDIT_4090_HF.ps1" in doc
    assert "CONTROL_RUNPOD_4090_HF.ps1" in doc
    assert "ready_to_run_4090_pilot" in doc
    assert "fpv runpod status-4090-launch" in doc
    assert "fpv runpod audit-4090-launch" in doc
    assert "RUNPOD_LAUNCH.md" in doc
    assert "runpod_launch_manifest.json" in doc
    assert "RETURN_STATUS_AUDIT.md" in doc
    assert "older expanded-methods launch surfaces" in doc
    assert "-Foreground" in doc
    assert "-SkipPrelaunchAudit" in doc


def test_4090_hf_launch_checklist_has_valid_paths_and_reports():
    launch = ROOT / "outputs" / "h100" / "full_161_run" / "launch" / "RUNPOD_LAUNCH_4090_HF.md"
    text = launch.read_text(encoding="utf-8")
    assert "ready_to_run_4090_pilot" in text
    assert "Package status: `ready_to_upload`" in text
    assert "Operator status: `ready_to_run_4090_pilot`" in text
    assert "legacy internal contract names kept for compatibility" in text
    assert "fpv runpod status-4090-launch" in text
    assert "fpv runpod audit-4090-launch" in text
    assert "fpv runpod smoke-package --source" in text
    assert "RUNPOD_LAUNCH.md" in text
    assert "runpod_launch_manifest.json" in text
    assert "runpod_job_4090_hf_full_stack_clean.zip" in text
    assert "RUNPOD_RUN_4090_HF.sh" in text
    assert "UPLOAD_AND_OPTIONALLY_START_4090_HF.ps1" in text
    assert "RUN_FULL_4090_R3_LINGBOT.ps1" in text
    assert "PRELAUNCH_AUDIT_4090_HF.ps1" in text
    assert "-Foreground" in text
    assert "-SkipPrelaunchAudit" in text
    assert "MONITOR_RUNPOD_4090_HF.ps1" in text
    assert "CONTROL_RUNPOD_4090_HF.ps1" in text
    assert "-Action stop" in text
    assert "-ForceKill" in text
    assert "sha256sum -c" in text
    assert "python scripts/07_hf_dataset_frame_packs.py" in text
    assert "cloud_run_status.json" in text
    assert "1d36a3968b6dc64664265120e85489d120ede5b1e33032d7088047bbe97c232d" in text
    assert "No true speed/standoff/dive-angle claims" in text
    assert "No route, approach, launch, target-coordinate, guidance, or next-maneuver inference" in text
    assert chr(12) not in text


def test_4090_hf_pod_script_has_hash_token_and_modes():
    script = ROOT / "outputs" / "h100" / "full_161_run" / "launch" / "RUNPOD_RUN_4090_HF.sh"
    text = script.read_text(encoding="utf-8")
    assert "sha256sum -c" in text
    assert "HF_TOKEN is not set" in text
    assert "pilot_hf_scout3" in text
    assert "full_packaged" in text
    assert "full_hf_rebuild" in text
    assert "HF_DATASET_BUILD_FRAME_PACKS=1" in text
    assert "HF_DATASET_MAX_ITEMS" in text
    assert "ready_4090_pilot" in text
    assert "OPTIONAL_RUN_ODM" in text
    assert "OPTIONAL_RUN_NERFSTUDIO" in text
    assert "FPV_GPU_PROFILE" in text
    assert "HF_DATASET_HIGH_DETAIL_FRAMES" in text
    assert "80gb_quality" in text
    assert "48gb_quality" in text
    assert "24gb_quality_safe" in text
    assert "run_optimizer_status.json" in text
    assert "SAFETY_WARNING" in text
    assert "No true speed/standoff/dive-angle claims" in text
    assert "No route, approach, launch, target-coordinate, guidance, or next-maneuver inference" in text
    assert bytes([13]) not in script.read_bytes()



def test_4090_prelaunch_audit_helper_checks_manifest_package_and_zip_contract():
    script = ROOT / "outputs" / "h100" / "full_161_run" / "launch" / "PRELAUNCH_AUDIT_4090_HF.ps1"
    text = script.read_text(encoding="utf-8")

    assert "runpod_launch_4090_hf_manifest.json" in text
    assert "Get-FileHash" in text
    assert "Fast" in text
    assert "System.IO.Compression.ZipFile" in text
    assert "duplicate ZIP entries" in text
    assert "scripts/70_package_return.py" in text
    assert "scripts/80_run_expanded_methods.py" in text
    assert "DOWNLOAD_ME.md" in text
    assert "stale runpod_launch_manifest.json" in text
    assert "no geolocation" in text
    assert "no true speed/standoff/dive-angle claims" in text
    assert "no route/approach/launch/target-coordinate/guidance/next-maneuver inference" in text
    assert "validate_return_powershell" in text
    assert "control_powershell" in text
    assert bytes([13]) not in script.read_bytes()

def test_4090_validate_helper_uses_4090_hf_launch_manifest():
    script = ROOT / "outputs" / "h100" / "full_161_run" / "launch" / "VALIDATE_RETURN.ps1"
    text = script.read_text(encoding="utf-8")

    assert "runpod_launch_4090_hf_manifest.json" in text
    assert "runpod_launch_manifest.json" not in text
    assert "RunPod return validation" in text
    assert "no geolocation" in text.lower()
    assert "no true speed/standoff/dive-angle claims" in text.lower()
    assert bytes([13]) not in script.read_bytes()

def test_4090_return_downloader_uses_safe_ssh_config_and_validation():
    script = ROOT / "outputs" / "h100" / "full_161_run" / "launch" / "DOWNLOAD_AND_VALIDATE_RETURN.ps1"
    text = script.read_text(encoding="utf-8")
    assert "empty_ssh_config" in text
    assert "scp" in text
    assert "ssh" in text
    assert "h100_return.zip" in text
    assert "cloud_run_status.json" in text
    assert "VALIDATE_RETURN.ps1" in text
    assert "SkipValidation" in text
    assert "no geolocation" in text.lower()
    assert "no true speed/standoff/dive-angle claims" in text.lower()
    assert "${Remote}:$RemoteRoot" in text
    assert bytes([13]) not in script.read_bytes()


def test_4090_upload_helper_hashes_uploads_and_starts_explicitly():
    script = ROOT / "outputs" / "h100" / "full_161_run" / "launch" / "UPLOAD_AND_OPTIONALLY_START_4090_HF.ps1"
    text = script.read_text(encoding="utf-8")
    assert "Get-FileHash" in text
    assert "Local package SHA256 mismatch" in text
    assert "empty_ssh_config" in text
    assert "runpod_job_4090_hf_full_stack_clean.zip" in text
    assert "RUNPOD_RUN_4090_HF.sh" in text
    assert "UPLOAD_AND_OPTIONALLY_START_4090_HF.ps1" in text
    assert "PRELAUNCH_AUDIT_4090_HF.ps1" in text
    assert "PrelaunchAuditPath" in text
    assert "-Fast" in text
    assert "[switch]$SkipPrelaunchAudit" in text
    assert "[switch]$Foreground" in text
    assert "local prelaunch audit failed" in text
    assert "nohup bash -lc" in text
    assert "fpv_4090.pid" in text
    assert "fpv_4090_start.log" in text
    assert "`$(cat '$RemotePidFile')" in text
    assert "start RunPod FPV pipeline background" in text
    assert "start RunPod FPV pipeline foreground" in text
    assert "sha256sum -c" in text
    assert "[switch]$Start" in text
    assert "FPV_RUN_MODE=$RunMode" in text
    assert "[switch]$EnableFullOptionalMethods" in text
    assert "RemoteEnvPrefix" in text
    assert "OPTIONAL_RUN_ODM=1" in text
    assert "OPTIONAL_RUN_NERFSTUDIO=1" in text
    assert "SKIP_RESEARCH_METHODS=0" in text
    assert "pilot_hf_scout3" in text
    assert "full_packaged" in text
    assert "full_hf_rebuild" in text
    assert "[string]$HF_TOKEN" not in text
    assert "HF_TOKEN=" not in text
    assert bytes([13]) not in script.read_bytes()

def test_4090_control_helper_checks_and_stops_detached_run_explicitly():
    script = ROOT / "outputs" / "h100" / "full_161_run" / "launch" / "CONTROL_RUNPOD_4090_HF.ps1"
    text = script.read_text(encoding="utf-8")
    lower = text.lower()

    assert "ValidateSet(\"status\", \"stop\")" in text
    assert "Action = \"status\"" in text
    assert "ForceKill" in text
    assert "kill -0" in text
    assert "kill -9" in text
    assert "fpv_4090.pid" in text
    assert "fpv_4090_start.log" in text
    assert "`$(cat \"`$PID_FILE\"" in text
    assert "nothing to stop" in text
    assert "no geolocation" in lower
    assert "no true speed/standoff/dive-angle claims" in lower
    assert "no route, approach, launch, target-coordinate, guidance, or next-maneuver inference" in lower
    assert bytes([13]) not in script.read_bytes()

def test_4090_monitor_helper_summarizes_remote_status_without_operational_outputs():
    script = ROOT / "outputs" / "h100" / "full_161_run" / "launch" / "MONITOR_RUNPOD_4090_HF.ps1"
    text = script.read_text(encoding="utf-8")
    lower = text.lower()

    assert "empty_ssh_config" in text
    assert "cloud_run_status.json" in text
    assert "cloud_summary.json" in text
    assert "method_stage_report.json" in text
    assert "quality_report.json" in text
    assert "fpv_4090.pid" in text
    assert "fpv_4090_start.log" in text
    assert "background PID" in text
    assert "`$(cat fpv_4090.pid)" in text
    assert "startup log tail" in text
    assert "cloud_run.log" in text
    assert "TailLines" in text
    assert "Follow" in text
    assert "no geolocation" in lower
    assert "no meters" in lower
    assert "no true speed/standoff/dive-angle claims" in lower
    assert "no route, approach, launch, target-coordinate, guidance, or next-maneuver inference" in lower
    assert bytes([13]) not in script.read_bytes()


def test_4090_launch_manifest_references_all_operator_helpers():
    manifest = (
        ROOT
        / "outputs"
        / "h100"
        / "full_161_run"
        / "launch"
        / "runpod_launch_4090_hf_manifest.json"
    )
    data = json.loads(manifest.read_text(encoding="utf-8"))

    assert data["status"] == "ready_to_upload"
    assert data["pod_script"].endswith("RUNPOD_RUN_4090_HF.sh")
    assert data["prelaunch_audit_powershell"].endswith("PRELAUNCH_AUDIT_4090_HF.ps1")
    assert data["upload_and_start_powershell"].endswith(
        "UPLOAD_AND_OPTIONALLY_START_4090_HF.ps1"
    )
    assert data["monitor_powershell"].endswith("MONITOR_RUNPOD_4090_HF.ps1")
    assert data["control_powershell"].endswith("CONTROL_RUNPOD_4090_HF.ps1")
    assert data["download_and_validate_powershell"].endswith(
        "DOWNLOAD_AND_VALIDATE_RETURN.ps1"
    )
    assert data["validate_return_powershell"].endswith("VALIDATE_RETURN.ps1")
    assert data["start_here_markdown"].endswith("START_HERE_4090_HF.md")
    assert data["pilot_profile"].endswith("RUNPOD_4090_PILOT_PROFILE.json")
    assert "no true speed/standoff/dive-angle claims" in data["safety_warnings"]
    assert (
        "no route/approach/launch/target-coordinate/guidance/next-maneuver inference"
        in data["safety_warnings"]
    )
    assert (
        data["package"]["sha256"]
        == "1d36a3968b6dc64664265120e85489d120ede5b1e33032d7088047bbe97c232d"
    )
    package_path = ROOT / data["package"]["source"]
    assert package_path.exists()
    assert package_path.stat().st_size == data["package"]["size_bytes"]

    summary = json.loads(
        (
            ROOT
            / "outputs"
            / "h100"
            / "full_161_run"
            / "runpod_job_4090_hf_full_stack_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["sha256"] == data["package"]["sha256"]
    assert summary["size_bytes"] == data["package"]["size_bytes"]
    assert summary["postflight_launch_manifest"].endswith(
        "runpod_launch_4090_hf_manifest.json"
    )
    assert summary["prelaunch_audit_powershell"].endswith(
        "PRELAUNCH_AUDIT_4090_HF.ps1"
    )
    assert summary["control_powershell"].endswith("CONTROL_RUNPOD_4090_HF.ps1")
    assert summary["validate_return_powershell"].endswith("VALIDATE_RETURN.ps1")
    assert summary["start_here_markdown"].endswith("START_HERE_4090_HF.md")
    assert summary["pilot_profile"].endswith("RUNPOD_4090_PILOT_PROFILE.json")

    optional = data["optional_research_stages"]
    assert optional["mast3r_optional_baseline"]["repo"] == "https://github.com/naver/mast3r"
    assert optional["dust3r_optional_baseline"]["repo"] == "https://github.com/naver/dust3r"
    assert "VGGETR_COMMAND_TEMPLATE" in optional["vggetr_candidate_baseline"]["enabled_by"]
    assert "do not claim" in optional["vggetr_candidate_baseline"]["execution_policy"].lower()
    assert optional["hloc_learning_based_matching_optional"]["repo"] == "https://github.com/cvg/Hierarchical-Localization"
    assert optional["glomap_global_sfm_optional"]["repo"] == "https://github.com/colmap/glomap"
    assert "archived" in optional["glomap_global_sfm_optional"]["status_note"]
    assert optional["depth_anything_v2_prior_optional"]["repo"] == "https://github.com/DepthAnything/Depth-Anything-V2"
    assert optional["metric3d_prior_optional"]["repo"] == "https://github.com/YvanYin/Metric3D"
    assert optional["viser_local_viewer_optional"]["repo"] == "https://github.com/viser-project/viser"
    assert optional["supersplat_postprocess_optional"]["repo"] == "https://github.com/playcanvas/supersplat"
    assert "no geolocation" in optional["hloc_learning_based_matching_optional"]["execution_policy"]
    assert "do not treat as metric distance" in optional["depth_anything_v2_prior_optional"]["execution_policy"]


def test_4090_source_job_return_instructions_match_clean_zip_manifest():
    script = (
        ROOT
        / "outputs"
        / "h100"
        / "full_161_run"
        / "runpod_job"
        / "scripts"
        / "70_package_return.py"
    )
    text = script.read_text(encoding="utf-8")

    assert "runpod_launch_4090_hf_manifest.json" in text
    assert "runpod_launch_manifest.json" not in text
    assert bytes([13]) not in script.read_bytes()


def test_4090_clean_zip_embeds_current_return_instructions_without_duplicates():
    package = ROOT / "outputs" / "h100" / "full_161_run" / "runpod_job_4090_hf_full_stack_clean.zip"

    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert "scripts/70_package_return.py" in names
        assert "scripts/80_run_expanded_methods.py" in names
        run_all = archive.read("run_all.sh").decode("utf-8")
        expanded_runner = archive.read("scripts/80_run_expanded_methods.py").decode("utf-8")
        script = archive.read("scripts/70_package_return.py").decode("utf-8")

    assert "python scripts/80_run_expanded_methods.py" in run_all
    assert "method_stage_report.json" in expanded_runner

    assert "runpod_launch_4090_hf_manifest.json" in script
    assert "runpod_launch_manifest.json" not in script
    assert "DOWNLOAD_ME.md" in script
    assert "no geolocation" in script

def test_4090_launch_kit_source_audit_passes_fast():
    package = ROOT / "outputs" / "h100" / "full_161_run" / "runpod_job_4090_hf_full_stack_clean.zip"
    launch_dir = ROOT / "outputs" / "h100" / "full_161_run" / "launch"

    report = audit_runpod_4090_hf_launch_kit(package=package, launch_dir=launch_dir, fast=True)

    assert report["valid"], report["issues"]
    assert report["package_inspection"]["has_expanded_methods"] is True
    assert report["package_inspection"]["clip_count"] == 483
    assert report["manifest"]["schema_version"] == "runpod-4090-hf-launch-v1"
    assert report["manifest"]["status"] == "ready_to_upload"
    assert "no map projection" in report["manifest"]["safety_warnings"]
    assert "no route/approach/launch/target-coordinate/guidance/next-maneuver inference" in report["manifest"]["safety_warnings"]
    assert report["helpers"]["pod_script"]["exists"] is True
    assert report["helpers"]["validate_return_powershell"]["exists"] is True
    assert report["helpers"]["one_shot_full_powershell"]["exists"] is True
    assert report["helpers"]["r3_lingbot_runner"]["exists"] is True
    assert report["docs"]["start_here_markdown"]["exists"] is True
    assert report["docs"]["pilot_profile"]["exists"] is True


def test_4090_launch_kit_cli_audit_outputs_json():
    package = ROOT / "outputs" / "h100" / "full_161_run" / "runpod_job_4090_hf_full_stack_clean.zip"
    launch_dir = ROOT / "outputs" / "h100" / "full_161_run" / "launch"

    result = runner.invoke(
        app,
        [
            "runpod",
            "audit-4090-launch",
            "--package",
            str(package),
            "--launch-dir",
            str(launch_dir),
            "--fast",
        ],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["valid"] is True
    assert report["package_inspection"]["zip_sha256_skipped"] is True


def test_4090_launch_kit_audit_fails_closed_on_missing_safety_warning(tmp_path: Path):
    package = ROOT / "outputs" / "h100" / "full_161_run" / "runpod_job_4090_hf_full_stack_clean.zip"
    launch_dir = ROOT / "outputs" / "h100" / "full_161_run" / "launch"
    audit_dir = tmp_path / "launch"
    shutil.copytree(launch_dir, audit_dir)

    manifest_path = audit_dir / "runpod_launch_4090_hf_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["safety_warnings"] = [
        warning for warning in manifest["safety_warnings"] if warning != "no map projection"
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = audit_runpod_4090_hf_launch_kit(package=package, launch_dir=audit_dir, fast=True)

    assert report["valid"] is False
    assert "missing launch safety warning: no map projection" in report["issues"]

def test_4090_start_here_note_points_to_only_current_launch_surface():
    note = ROOT / "outputs" / "h100" / "full_161_run" / "launch" / "START_HERE_4090_HF.md"
    text = note.read_text(encoding="utf-8")

    assert "ready_to_run_4090_pilot" in text
    assert "Package status: `ready_to_upload`" in text
    assert "Operator status: `ready_to_run_4090_pilot`" in text
    assert "legacy internal contract names kept for compatibility" in text
    assert "runpod_job_4090_hf_full_stack_clean.zip" in text
    assert "fpv runpod audit-4090-launch" in text
    assert "fpv runpod smoke-package --source" in text
    assert "PRELAUNCH_AUDIT_4090_HF.ps1" in text
    assert ".\\outputs\\h100\\full_161_run\\launch\\PRELAUNCH_AUDIT_4090_HF.ps1" in text
    assert "UPLOAD_AND_OPTIONALLY_START_4090_HF.ps1" in text
    assert ".\\outputs\\h100\\full_161_run\\launch\\UPLOAD_AND_OPTIONALLY_START_4090_HF.ps1" in text
    assert "DOWNLOAD_AND_VALIDATE_RETURN.ps1" in text
    assert "final-audit-4090-return" in text
    assert "pilot_hf_scout3" in text
    assert "HF_TOKEN" in text
    assert "RUNPOD_LAUNCH.md" in text
    assert "runpod_launch_manifest.json" in text
    assert "No geolocation" in text
    assert "No map projection" in text
    assert "No true speed/standoff/dive-angle claims" in text
    assert "No route, approach, launch, target-coordinate, guidance, or next-maneuver inference" in text
    assert bytes([13]) not in note.read_bytes()

def test_4090_start_here_note_can_be_regenerated_from_cli(tmp_path: Path):
    package = ROOT / "outputs" / "h100" / "full_161_run" / "runpod_job_4090_hf_full_stack_clean.zip"
    source_launch_dir = ROOT / "outputs" / "h100" / "full_161_run" / "launch"
    launch_dir = tmp_path / "launch"
    shutil.copytree(source_launch_dir, launch_dir)
    (launch_dir / "START_HERE_4090_HF.md").unlink()
    manifest_path = launch_dir / "runpod_launch_4090_hf_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("start_here_markdown", None)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "runpod",
            "write-4090-start-here",
            "--package",
            str(package),
            "--launch-dir",
            str(launch_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "done"
    assert payload["summary_updated"] is False
    assert payload["audit"]["valid"] is True
    assert payload["audit"]["docs"]["start_here_markdown"]["exists"] is True
    note_text = (launch_dir / "START_HERE_4090_HF.md").read_text(encoding="utf-8")
    assert "ready_to_run_4090_pilot" in note_text
    assert "fpv runpod status-4090-launch" in note_text
    assert "fpv runpod audit-4090-launch" in note_text
    assert "pilot_hf_scout3" in note_text
    assert "legacy internal contract names kept for compatibility" in note_text
    regenerated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert regenerated_manifest["start_here_markdown"].endswith("START_HERE_4090_HF.md")


def test_legacy_launch_files_are_explicitly_superseded_for_4090_hf():
    launch_dir = ROOT / "outputs" / "h100" / "full_161_run" / "launch"
    legacy_markdown = (launch_dir / "RUNPOD_LAUNCH.md").read_text(encoding="utf-8")
    legacy_manifest = json.loads((launch_dir / "runpod_launch_manifest.json").read_text(encoding="utf-8"))

    assert "SUPERSEDED: Do Not Use For 4090/HF" in legacy_markdown
    assert "START_HERE_4090_HF.md" in legacy_markdown
    assert "RUNPOD_LAUNCH_4090_HF.md" in legacy_markdown
    assert legacy_manifest["status"] == "superseded_by_4090_hf_launch"
    assert legacy_manifest["superseded_by"].endswith("runpod_launch_4090_hf_manifest.json")

def test_4090_launch_status_cli_prints_next_commands():
    result = runner.invoke(app, ["runpod", "status-4090-launch"])

    assert result.exit_code == 0, result.output
    assert "RunPod 4090/HF launch status" in result.output
    assert "status: ready_to_run_4090_pilot" in result.output
    assert "clip jobs: 483" in result.output
    assert "start here:" in result.output
    assert "pod profile:" in result.output
    assert "fpv runpod audit-4090-launch --fast" in result.output
    assert "fpv runpod smoke-package --source" in result.output
    assert "UPLOAD_AND_OPTIONALLY_START_4090_HF.ps1" in result.output
    assert "RUN_FULL_4090_R3_LINGBOT.ps1" in result.output
    assert "one-shot full 4090/R3/LingBot command" in result.output
    assert "-RunMode pilot_hf_scout3" in result.output
    assert "DOWNLOAD_AND_VALIDATE_RETURN.ps1" in result.output
    assert "final audit after download" in result.output
    assert "final-audit-4090-return --source <local_return_zip>" in result.output
    assert "remaining proof: run on RunPod" in result.output
    assert "no geolocation" in result.output
    assert "no route/approach/launch/target-coordinate/guidance/next-maneuver inference" in result.output

def test_4090_pilot_profile_is_machine_readable_and_secret_free():
    profile_path = ROOT / "outputs" / "h100" / "full_161_run" / "launch" / "RUNPOD_4090_PILOT_PROFILE.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    assert profile["schema_version"] == "runpod-4090-hf-profile-v1"
    assert profile["gpu"]["primary"] == "RTX 4090 24 GB"
    assert "L40S" in profile["gpu"]["fallback_48gb"]
    assert "RTX 6000 Ada" in profile["gpu"]["fallback_48gb"]
    assert "A6000" in profile["gpu"]["fallback_48gb"]
    assert profile["resources"]["system_ram_gb_min"] >= 64
    assert profile["resources"]["container_disk_gb_min"] >= 40
    assert profile["resources"]["volume_disk_gb_min"] >= 150
    assert profile["container_image"]["dockerfile"] == "containers/runpod-vggt-colmap.Dockerfile"
    assert profile["container_image"]["default_base_image"] == "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
    assert profile["run_modes"]["default"] == "pilot_hf_scout3"
    assert profile["hugging_face"]["dataset_id"] == "Grimster/FPV_Hezbo"
    assert profile["hugging_face"]["requires_token_secret"] == "HF_TOKEN"
    assert profile["environment"]["required"]["HF_DATASET_ID"] == "Grimster/FPV_Hezbo"
    assert profile["environment"]["full_after_pilot"]["OPTIONAL_RUN_ODM"] == "1"
    assert profile["environment"]["full_after_pilot"]["OPTIONAL_RUN_OPENMVS"] == "1"
    assert profile["environment"]["full_after_pilot"]["OPTIONAL_RUN_NERFSTUDIO"] == "1"
    assert profile["environment"]["full_after_pilot"]["SKIP_RESEARCH_METHODS"] == "0"
    assert "HF_TOKEN" not in json.dumps(profile["environment"])
    assert "no geolocation" in profile["safety_warnings"]
    assert "no map projection" in profile["safety_warnings"]
    assert bytes([13]) not in profile_path.read_bytes()


def test_4090_profile_cli_writes_profile(tmp_path: Path):
    result = runner.invoke(app, ["runpod", "profile-4090-launch", "--launch-dir", str(tmp_path), "--write"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "done"
    profile_path = tmp_path / "RUNPOD_4090_PILOT_PROFILE.json"
    assert profile_path.exists()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["gpu"]["primary"] == "RTX 4090 24 GB"
    assert profile["run_modes"]["default"] == "pilot_hf_scout3"
    assert profile["environment"]["full_after_pilot"]["OPTIONAL_RUN_ODM"] == "1"
    assert profile["environment"]["full_after_pilot"]["OPTIONAL_RUN_OPENMVS"] == "1"
    assert profile["environment"]["full_after_pilot"]["OPTIONAL_RUN_NERFSTUDIO"] == "1"
    assert profile["environment"]["full_after_pilot"]["SKIP_RESEARCH_METHODS"] == "0"








def test_r3_lingbot_wrapper_reports_exit_codes_and_failed_soft_statuses():
    script = ROOT / "outputs" / "h100" / "full_161_run" / "launch" / "RUNPOD_INSTALL_TRAIN_R3_LINGBOT_4090.sh"
    text = script.read_text(encoding="utf-8")

    assert "R3_INFERENCE_EXIT" in text
    assert "R3_TRAIN_EXIT" in text
    assert "LINGBOT_EXIT" in text
    assert "ENABLE_R3" in text
    assert "RUN_R3_INFERENCE" in text
    assert "RUN_R3_TRAINING" in text
    assert "r3_reconstruction_manifest.json" in text
    assert "r3_dataset_prep_report.json" in text
    assert "r3_outputs.zip" in text
    assert "ENABLE_LINGBOT_MAP" in text
    assert "RUN_LINGBOT_MAP_INFERENCE" in text
    assert "LINGBOT_MODEL_PATH" in text
    assert "lingbot_map_manifest.json" in text
    assert "lingbot_map_outputs.zip" in text
    assert "\\\"exit_code\\\":" in text
    assert "failed_soft" in text
    assert "skipped_missing_or_empty_r3_data_root" in text
    assert "r3_data_root_ready" in text
    assert "KevinXu02/R3" in text
    assert "robbyant/lingbot-map" in text
    assert "zipfile.ZipFile" in text
    assert bytes([13]) not in script.read_bytes()
