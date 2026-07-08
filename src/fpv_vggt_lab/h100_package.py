from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .catalog import DEFAULT_MANIFEST_URL, DEFAULT_README_URL, sync_catalog
from .cloud_job import _runner_script
from .frames import read_frame_manifest, sample_accepted_segment_frames
from .method_contract import CORE_RETURN_FILES, expanded_return_artifact_files, write_method_contract_files
from .schemas import FrameManifest, model_to_dict
from .segments import read_annotations


H100_WARNINGS = [
    "no geolocation",
    "no meters",
    "relative VGGT frame",
    "local-only media",
]
RUNPOD_4090_HF_MANIFEST = "runpod_launch_4090_hf_manifest.json"
RUNPOD_4090_HF_SAFETY_WARNINGS = [
    "no geolocation",
    "no map projection",
    "no meters",
    "no true speed/standoff/dive-angle claims",
    "no route/approach/launch/target-coordinate/guidance/next-maneuver inference",
    "relative VGGT frame",
    "local-only media",
]
RUNPOD_4090_HF_HELPERS = {
    "pod_script": "RUNPOD_RUN_4090_HF.sh",
    "prelaunch_audit_powershell": "PRELAUNCH_AUDIT_4090_HF.ps1",
    "upload_and_start_powershell": "UPLOAD_AND_OPTIONALLY_START_4090_HF.ps1",
    "one_shot_full_powershell": "RUN_FULL_4090_R3_LINGBOT.ps1",
    "r3_lingbot_runner": "RUNPOD_INSTALL_TRAIN_R3_LINGBOT_4090.sh",
    "monitor_powershell": "MONITOR_RUNPOD_4090_HF.ps1",
    "control_powershell": "CONTROL_RUNPOD_4090_HF.ps1",
    "download_and_validate_powershell": "DOWNLOAD_AND_VALIDATE_RETURN.ps1",
    "validate_return_powershell": "VALIDATE_RETURN.ps1",
}
RUNPOD_4090_HF_DOCS = {
    "start_here_markdown": "START_HERE_4090_HF.md",
    "pilot_profile": "RUNPOD_4090_PILOT_PROFILE.json",
}

RAW_MEDIA_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SECRET_PATTERN = re.compile(r"(sk-[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{20,}|(?:api[_-]?key|token|secret)\s*[:=]\s*[\'\"][^\'\"\s]{16,}[\'\"])", re.IGNORECASE)
WINDOWS_ABSOLUTE_PATTERN = re.compile(r"\b[A-Za-z]:\\")


@dataclass(frozen=True)
class H100Tier:
    name: Literal["smoke", "scout", "main", "high_detail"]
    frame_count: int
    resized_long_edge: int | None


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {message}\n")


def _write_text_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def prepare_h100_run(
    *,
    dataset: str,
    workdir: Path,
    media_dir: Path,
    media_inventory: Path,
    annotations: Path,
    frames_root: Path,
    frame_manifests: list[Path] | None = None,
    frame_scout: int = 32,
    frame_main: int = 96,
    frame_high_detail: int = 128,
    resize_scout: int | None = 768,
    resize_main: int | None = 1024,
    resize_high_detail: int | None = 1024,
    auto_segment: str = "strict",
    metadata_policy: str = "provenance-only",
) -> dict[str, Any]:
    if metadata_policy != "provenance-only":
        raise ValueError("metadata_policy must be provenance-only for the H100 milestone")
    if auto_segment != "strict":
        raise ValueError("auto_segment must be strict for the H100 milestone")

    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(workdir / "run.log")
    logger.log("start H100 prepare")

    stage_status: dict[str, str] = {
        "preflight": "done",
        "dataset_snapshot": "skipped",
        "segment_selection": "pending",
        "frame_packs": "pending",
        "package": "pending",
    }
    summary: dict[str, Any] = {
        "schema_version": "h100-prepare-v1",
        "status": "failed_soft",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "dataset": dataset,
            "workdir": str(workdir),
            "media_dir": str(media_dir),
            "media_inventory": str(media_inventory),
            "annotations": str(annotations),
            "frames_root": str(frames_root),
            "frame_manifests": [str(path) for path in frame_manifests or []],
            "auto_segment": auto_segment,
            "metadata_policy": metadata_policy,
        },
        "warnings": H100_WARNINGS,
        "stage_status": stage_status,
        "artifacts": {
            "run_log": str(workdir / "run.log"),
            "summary": str(workdir / "summary.json"),
            "next_steps": str(workdir / "NEXT_STEPS.md"),
        },
        "clip_count": 0,
        "package_validation": None,
        "environment": environment_snapshot(),
    }

    (workdir / "environment_snapshot.json").write_text(
        json.dumps(summary["environment"], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    dataset_result = _write_dataset_snapshot(dataset, workdir, logger)
    summary["dataset_snapshot"] = dataset_result
    stage_status["dataset_snapshot"] = dataset_result["status"]

    tiers = [
        H100Tier("scout", frame_scout, resize_scout),
        H100Tier("main", frame_main, resize_main),
        H100Tier("high_detail", frame_high_detail, resize_high_detail),
    ]
    frame_sources, segment_rows = _collect_frame_sources(
        frame_manifests=frame_manifests or [],
        annotations=annotations,
        media_inventory=media_inventory,
        frames_root=frames_root,
        workdir=workdir,
        tiers=tiers,
        logger=logger,
    )
    stage_status["segment_selection"] = "done" if frame_sources else "needs_human_review"
    stage_status["frame_packs"] = "done" if frame_sources else "blocked"
    _write_table(workdir / "segment_decisions.parquet", segment_rows)
    summary["artifacts"]["segment_decisions"] = str(workdir / "segment_decisions.parquet")

    if not frame_sources:
        stage_status["package"] = "blocked"
        summary["status"] = "needs_human_review"
        _write_empty_frame_pack_manifest(workdir / "frame_pack_manifest.parquet")
        summary["artifacts"]["frame_pack_manifest"] = str(workdir / "frame_pack_manifest.parquet")
        write_h100_summary(summary, workdir)
        write_next_steps(workdir, status="needs_human_review")
        logger.log("no accepted frame manifests or accepted local segments were available")
        return summary

    runpod_job = workdir / "runpod_job"
    if runpod_job.exists():
        shutil.rmtree(runpod_job)
    runpod_job.mkdir(parents=True)

    job_manifest, pack_rows = _write_runpod_job(
        runpod_job=runpod_job,
        run_id=workdir.name,
        frame_sources=frame_sources,
        metadata_policy=metadata_policy,
        logger=logger,
    )
    frame_pack_manifest = workdir / "frame_pack_manifest.parquet"
    _write_table(frame_pack_manifest, pack_rows)
    zip_path = workdir / "runpod_job.zip"
    if zip_path.exists():
        zip_path.unlink()
    _zip_runpod_job(runpod_job, zip_path)
    validation = validate_runpod_job_package(runpod_job, zip_path)

    summary["clip_count"] = len(job_manifest["clips"])
    summary["package_validation"] = validation
    summary["artifacts"].update(
        {
            "frame_pack_manifest": str(frame_pack_manifest),
            "runpod_job": str(runpod_job),
            "runpod_job_zip": str(zip_path),
            "runpod_job_sha256": _sha256_file(zip_path),
        }
    )
    stage_status["package"] = "done" if validation["valid"] else "failed_soft"
    summary["status"] = "done" if validation["valid"] else "failed_soft"
    write_h100_summary(summary, workdir)
    write_next_steps(workdir, status=summary["status"])
    logger.log(f"finished H100 prepare with status {summary['status']}")
    return summary


def environment_snapshot() -> dict[str, Any]:
    packages = {}
    for package in ["numpy", "opencv-python", "pandas", "pyarrow", "pydantic", "typer"]:
        try:
            packages[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "ffmpeg": _ffmpeg_version(),
    }


def write_h100_summary(summary: dict[str, Any], workdir: Path) -> Path:
    path = workdir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_next_steps(workdir: Path, *, status: str) -> Path:
    if status == "done":
        text = f"""# H100 Next Steps

Status: `done`

Warning block:
- No geolocation
- No meters or true speed/standoff/dive-angle claims
- VGGT outputs are relative-frame diagnostics
- Real media-derived artifacts stay local-only

Upload:

```bash
/workspace/runpod_job.zip
```

Run on RunPod:

```bash
mkdir -p /workspace/fpv-h100
cd /workspace/fpv-h100
python -m zipfile -e /workspace/runpod_job.zip .
bash run_all.sh
```

Bring back `/workspace/fpv-h100/h100_return.zip`, then run locally:

```powershell
fpv h100 inspect-return --source outputs/h100_returns/h100_return.zip
fpv h100 import-return --source outputs/h100_returns/h100_return.zip --workdir {workdir} --vggt-root data/vggt --review-output outputs/reviews/h100_latest_full_run
```
"""
    else:
        text = f"""# H100 Next Steps

Status: `{status}`

Warning block:
- No geolocation
- No meters or true speed/standoff/dive-angle claims
- VGGT outputs are relative-frame diagnostics
- Real media-derived artifacts stay local-only

No RunPod package was launched. Add accepted frame manifests or accepted local
segments, then rerun:

```powershell
fpv h100 prepare --dataset none --workdir {workdir} --frame-manifest data/frames/<video_id>/segment-001/frames.json
```

For dataset media, first make sure the clip is local and accepted:

```powershell
fpv media audit --inventory data/media/media_inventory.parquet --report outputs/reviews/media_audit.json
fpv segment list --annotations data/annotations/segments.jsonl
```
"""
    path = workdir / "NEXT_STEPS.md"
    path.write_text(text, encoding="utf-8")
    return path


def inspect_runpod_job_package(source: Path, *, fast: bool = False, compute_sha256: bool = True) -> dict[str, Any]:
    source = source.resolve()
    issues: list[str] = []
    warnings: list[str] = []
    required = [
        "job_manifest.json",
        "run_all.sh",
        "run_vggt_job.py",
        "scripts/00_env_check.py",
        "scripts/05_expanded_runtime_check.py",
        "scripts/06_hf_dataset_preflight.py",
        "scripts/07_hf_dataset_frame_packs.py",
        "scripts/70_package_return.py",
        "scripts/80_run_expanded_methods.py",
        "method_matrix.json",
        "return_contract.json",
        "METHOD_CONTRACT.md",
        "method_stage_plan.json",
    ]
    names: set[str] = set()
    file_count = 0
    total_bytes = 0
    manifest: dict[str, Any] | None = None
    matrix: list[dict[str, Any]] | None = None
    stage_plan: dict[str, Any] | None = None

    def check_member_name(name: str) -> None:
        member = Path(name)
        if member.is_absolute() or ".." in member.parts:
            issues.append(f"unsafe package member path: {name}")
        if name.lower().endswith(tuple(RAW_MEDIA_SUFFIXES)):
            issues.append(f"raw media file is not allowed in package: {name}")

    if source.is_file() and source.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(source) as archive:
                if not fast:
                    bad = archive.testzip()
                    if bad is not None:
                        issues.append(f"bad zip member: {bad}")
                for info in archive.infolist():
                    name = info.filename
                    names.add(name)
                    check_member_name(name)
                    if not name.endswith("/"):
                        file_count += 1
                        total_bytes += int(info.file_size)
                manifest = _read_json_from_zip(archive, "job_manifest.json", issues)
                matrix = _read_json_from_zip(archive, "method_matrix.json", issues)
                stage_plan = _read_json_from_zip(archive, "method_stage_plan.json", issues)
        except Exception as exc:
            issues.append(f"could not inspect zip: {exc}")
    elif source.is_dir():
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(source).as_posix()
            names.add(rel)
            check_member_name(rel)
            file_count += 1
            total_bytes += path.stat().st_size
        manifest = _read_json_file(source / "job_manifest.json", issues)
        matrix = _read_json_file(source / "method_matrix.json", issues)
        stage_plan = _read_json_file(source / "method_stage_plan.json", issues)
    else:
        issues.append(f"package source is neither zip nor directory: {source}")

    missing = [name for name in required if name not in names]
    issues.extend(f"missing required package file: {name}" for name in missing)
    clips = manifest.get("clips", []) if isinstance(manifest, dict) else []
    method_ids = [row.get("method_id") for row in matrix] if isinstance(matrix, list) else []
    stage_ids = [row.get("stage") for row in stage_plan.get("stages", [])] if isinstance(stage_plan, dict) else []
    if isinstance(manifest, dict):
        contract = manifest.get("method_contract") or {}
        for key in ["method_matrix", "return_contract", "method_contract", "method_stage_plan"]:
            if not contract.get(key):
                issues.append(f"job_manifest method_contract missing {key}")
    if clips and not any(str(clip.get("tier")) == "high_detail" for clip in clips):
        warnings.append("package has no high_detail tier clips")
    if "vggt_feedforward_full" not in method_ids:
        issues.append("method_matrix missing vggt_feedforward_full")
    if "vggt_colmap_ba_windowed" not in method_ids:
        issues.append("method_matrix missing vggt_colmap_ba_windowed")
    if "80_vggt_colmap_ba" not in stage_ids:
        issues.append("method_stage_plan missing 80_vggt_colmap_ba stage")
    return {
        "valid": not issues,
        "issues": issues,
        "warnings": warnings,
        "source": str(source),
        "source_kind": "zip" if source.is_file() else "directory" if source.is_dir() else "missing",
        "file_count": file_count,
        "total_uncompressed_bytes": total_bytes,
        "zip_sha256": _sha256_file(source) if source.is_file() and compute_sha256 else None,
        "zip_sha256_skipped": bool(source.is_file() and not compute_sha256),
        "zip_integrity_test_skipped": bool(source.is_file() and fast),
        "clip_count": len(clips),
        "method_ids": method_ids,
        "stage_ids": stage_ids,
        "has_expanded_runtime_check": "scripts/05_expanded_runtime_check.py" in names,
        "has_hf_dataset_preflight": "scripts/06_hf_dataset_preflight.py" in names,
        "has_hf_dataset_frame_pack_builder": "scripts/07_hf_dataset_frame_packs.py" in names,
        "has_expanded_methods": "scripts/80_run_expanded_methods.py" in names,
        "has_method_stage_plan": "method_stage_plan.json" in names,
    }








def _bash_for_package_syntax_check() -> str | None:
    """Prefer Windows-native Git Bash over WSL bash for stdin-based syntax checks."""
    if platform.system().lower() == "windows":
        git_bash = Path(r"C:\Program Files\Git\usr\bin\bash.exe")
        if git_bash.exists():
            return str(git_bash)
    return shutil.which("bash")

def smoke_runpod_job_package(source: Path, *, fast: bool = True) -> dict[str, Any]:
    """Run non-GPU syntax and contract checks against a RunPod job package."""
    source = source.resolve()
    inspection = inspect_runpod_job_package(source, fast=fast, compute_sha256=not fast)
    issues = [f"package inspection: {issue}" for issue in inspection.get("issues", [])]
    warnings = list(inspection.get("warnings", []))
    python_scripts = [
        "run_vggt_job.py",
        "scripts/00_env_check.py",
        "scripts/05_expanded_runtime_check.py",
        "scripts/06_hf_dataset_preflight.py",
        "scripts/07_hf_dataset_frame_packs.py",
        "scripts/70_package_return.py",
        "scripts/80_run_expanded_methods.py",
    ]
    shell_scripts = ["run_all.sh"]
    python_results: list[dict[str, Any]] = []
    shell_results: list[dict[str, Any]] = []

    for rel_path in python_scripts:
        read_issues: list[str] = []
        text = _read_package_text(source, rel_path, read_issues)
        result: dict[str, Any] = {"path": rel_path, "status": "missing" if text is None else "passed"}
        if read_issues:
            result["issues"] = read_issues
            issues.extend(read_issues)
        if text is not None:
            if "\r" in text:
                issue = f"{rel_path} contains CR bytes"
                result.setdefault("issues", []).append(issue)
                issues.append(issue)
            try:
                compile(text, rel_path, "exec")
            except SyntaxError as exc:
                issue = f"{rel_path} syntax error: {exc}"
                result.setdefault("issues", []).append(issue)
                issues.append(issue)
                result["status"] = "failed"
        python_results.append(result)

    bash_exe = _bash_for_package_syntax_check()
    for rel_path in shell_scripts:
        read_issues = []
        text = _read_package_text(source, rel_path, read_issues)
        result = {"path": rel_path, "status": "missing" if text is None else "pending"}
        if read_issues:
            result["issues"] = read_issues
            issues.extend(read_issues)
        if text is not None:
            if "\r" in text:
                issue = f"{rel_path} contains CR bytes"
                result.setdefault("issues", []).append(issue)
                issues.append(issue)
            if bash_exe is None:
                warning = f"bash not found; skipped shell syntax check for {rel_path}"
                warnings.append(warning)
                result["status"] = "skipped_bash_not_found"
            else:
                completed = subprocess.run(
                    [bash_exe, "-n", "-s"],
                    input=text,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                output = completed.stderr.strip() or completed.stdout.strip()
                wsl_unavailable = "wsl:" in output.lower() or "failed to start the systemd user session" in output.lower()
                result["returncode"] = completed.returncode
                if completed.returncode == 0:
                    result["status"] = "passed"
                elif wsl_unavailable:
                    warning = f"bash unavailable for shell syntax check on this host: {output}"
                    warnings.append(warning)
                    result["status"] = "skipped_bash_unavailable"
                    result["warning"] = warning
                else:
                    issue = f"{rel_path} shell syntax failed: {output}"
                    result.setdefault("issues", []).append(issue)
                    issues.append(issue)
                    result["status"] = "failed"
        shell_results.append(result)

    status = "failed_soft" if issues else "done_with_warnings" if warnings else "done"
    return {
        "schema_version": "runpod-package-smoke-v1",
        "status": status,
        "valid": not issues,
        "source": str(source),
        "gpu_execution": False,
        "checks": {
            "package_contract": inspection,
            "python_syntax": python_results,
            "shell_syntax": shell_results,
        },
        "issues": issues,
        "warnings": warnings,
        "next_actions": [
            "Upload package and run the 4090/HF pilot only after this smoke check and prelaunch audit pass.",
            "This smoke check does not run VGGT, COLMAP, MASt3R/DUSt3R, ODM, Nerfstudio, or CUDA inference.",
        ],
    }

def runpod_4090_hf_profile() -> dict[str, Any]:
    return {
        "schema_version": "runpod-4090-hf-profile-v1",
        "status": "recommended_for_4090_pilot",
        "gpu": {
            "primary": "RTX 4090 24 GB",
            "fallback_48gb": ["L40S", "RTX 6000 Ada", "A6000"],
            "h100_policy": "Use H100 only after a documented cheaper-GPU pilot fails from memory or runtime constraints.",
        },
        "resources": {
            "system_ram_gb_min": 64,
            "container_disk_gb_min": 40,
            "volume_disk_gb_min": 150,
            "gpu_process_policy": "one GPU process at a time; sequential clip processing",
        },
        "quality_profiles": {
            "24gb_quality_first": {
                "intended_gpu": "RTX 4090 24 GB or unknown GPU",
                "high_detail_tier": "up to 180 auto-trimmed frames at 1280px long edge, sampled near 18-20 FPS",
                "research_method_max_images": 80,
            },
            "48gb_quality": {
                "intended_gpu": "RTX 6000 Ada / L40S class",
                "high_detail_tier": "up to 200 auto-trimmed frames at 1536px long edge, sampled near 20 FPS",
                "research_method_max_images": 96,
            },
            "80gb_quality": {
                "intended_gpu": "A100 80 GB / H100 class",
                "high_detail_tier": "up to 240 auto-trimmed frames at 1536px long edge, sampled near 20 FPS",
                "research_method_max_images": 128,
            },
        },
        "container_image": {
            "dockerfile": "containers/runpod-vggt-colmap.Dockerfile",
            "default_base_image": "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
            "paid_run_rule": "build once, push to a registry, run by immutable digest",
            "pin_refs_for_paid_run": ["VGGT_REF", "DUST3R_REF", "MAST3R_REF"],
        },
        "run_modes": {
            "default": "pilot_hf_scout3",
            "available": ["pilot_hf_scout3", "full_packaged", "full_hf_rebuild"],
            "first_run": "full_hf_rebuild",
        },
        "hugging_face": {
            "dataset_id": "Grimster/FPV_Hezbo",
            "requires_token_secret": "HF_TOKEN",
            "token_policy": "Set through RunPod secrets or interactively; never write it into repo files, command history snippets, logs, or returned artifacts.",
        },
        "environment": {
            "required": {
                "HF_DATASET_ID": "Grimster/FPV_Hezbo",
                "REQUIRE_HF_DATASET": "1",
                "HF_DATASET_DOWNLOAD": "1",
                "HF_DATASET_BUILD_FRAME_PACKS": "1",
                "HF_DATASET_FETCH_CATALOG_MEDIA": "1",
                "HF_DATASET_AUTO_TRIM": "1",
                "HF_DATASET_HIGH_DETAIL_TARGET_FPS": "20",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
            "pilot": {
                "HF_DATASET_MAX_ITEMS": "3",
                "HF_DATASET_FRAME_PACK_TIERS": "scout,main,high_detail",
                "OPTIONAL_SHOWCASE_MAX_CLIPS": "3",
                "RESEARCH_METHOD_MAX_IMAGES": "32",
                "RESEARCH_METHOD_MAX_POINTS": "200000",
                "DUST3R_BATCH_SIZE": "1",
                "MAST3R_BATCH_SIZE": "1",
                "OPTIONAL_RUN_ODM": "1",
                "OPTIONAL_RUN_OPENMVS": "1",
                "OPTIONAL_RUN_NERFSTUDIO": "1",
                "SKIP_RESEARCH_METHODS": "0",
                "FPV_METHOD_POLICY": "mandatory_attempt",
            },
            "full_after_pilot": {
                "HF_DATASET_FRAME_PACK_TIERS": "scout,main,high_detail",
                "HF_DATASET_MAX_ITEMS": "unset",
                "HF_DATASET_HIGH_DETAIL_TARGET_FPS": "20",
                "HF_DATASET_TRIM_WINDOW_SEC": "10",
                "OPTIONAL_RUN_ODM": "1",
                "OPTIONAL_RUN_OPENMVS": "1",
                "OPTIONAL_RUN_NERFSTUDIO": "1",
                "SKIP_RESEARCH_METHODS": "0",
                "FPV_METHOD_POLICY": "mandatory_attempt",
            },
        },
        "safety_warnings": RUNPOD_4090_HF_SAFETY_WARNINGS,
    }


def write_runpod_4090_hf_profile(launch_dir: Path) -> dict[str, Any]:
    launch_dir = launch_dir.resolve()
    launch_dir.mkdir(parents=True, exist_ok=True)
    path = launch_dir / RUNPOD_4090_HF_DOCS["pilot_profile"]
    profile = runpod_4090_hf_profile()
    path.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    return {"status": "done", "profile": profile, "path": str(path)}


def write_runpod_4090_hf_start_here(
    package: Path,
    launch_dir: Path,
    *,
    status: str = "ready_to_run_4090_pilot",
) -> dict[str, Any]:
    """Write the canonical human start-here note for the current 4090/HF kit."""
    package = package.resolve()
    launch_dir = launch_dir.resolve()
    launch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = launch_dir / RUNPOD_4090_HF_MANIFEST
    issues: list[str] = []
    manifest = _read_json_file(manifest_path, issues)
    if not isinstance(manifest, dict):
        raise ValueError(f"missing or invalid 4090/HF launch manifest: {manifest_path}")
    package_report = inspect_runpod_job_package(package, fast=True, compute_sha256=False)
    if package_report["issues"]:
        raise ValueError("package is not inspectable: " + "; ".join(package_report["issues"]))

    profile_report = write_runpod_4090_hf_profile(launch_dir)
    profile_path = Path(profile_report["path"])
    start_here = launch_dir / RUNPOD_4090_HF_DOCS["start_here_markdown"]
    paths = {
        "package": _display_path(package),
        "manifest": _display_path(manifest_path),
        "checklist": _display_path(launch_dir / "RUNPOD_LAUNCH_4090_HF.md"),
        "pod_profile": _display_path(profile_path),
        "pod_script": _display_path(launch_dir / RUNPOD_4090_HF_HELPERS["pod_script"]),
        "prelaunch": _display_path(launch_dir / RUNPOD_4090_HF_HELPERS["prelaunch_audit_powershell"]),
        "upload": _display_path(launch_dir / RUNPOD_4090_HF_HELPERS["upload_and_start_powershell"]),
        "one_shot": _display_path(launch_dir / RUNPOD_4090_HF_HELPERS["one_shot_full_powershell"]),
        "monitor": _display_path(launch_dir / RUNPOD_4090_HF_HELPERS["monitor_powershell"]),
        "download": _display_path(launch_dir / RUNPOD_4090_HF_HELPERS["download_and_validate_powershell"]),
        "legacy_markdown": _display_path(launch_dir / "RUNPOD_LAUNCH.md"),
        "legacy_manifest": _display_path(launch_dir / "runpod_launch_manifest.json"),
    }
    start_here.write_text(_runpod_4090_hf_start_here_markdown(status=status, paths=paths), encoding="utf-8", newline="\n")

    start_here_ref = _display_path(start_here)
    manifest["start_here_markdown"] = start_here_ref.replace("/", "\\")
    manifest["pilot_profile"] = _display_path(profile_path).replace("/", "\\")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    summary_path = package.parent / "runpod_job_4090_hf_full_stack_summary.json"
    summary_updated = False
    canonical_launch_dir = (package.parent / "launch").resolve()
    if launch_dir == canonical_launch_dir and summary_path.exists():
        summary = _read_json_file(summary_path, issues)
        if isinstance(summary, dict):
            summary["start_here_markdown"] = start_here_ref.replace("/", "\\")
            summary["pilot_profile"] = _display_path(profile_path).replace("/", "\\")
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            summary_updated = True

    audit = audit_runpod_4090_hf_launch_kit(package=package, launch_dir=launch_dir, fast=True)
    return {
        "status": "done" if audit["valid"] else "failed_soft",
        "start_here_markdown": str(start_here),
        "manifest": str(manifest_path),
        "pilot_profile": str(profile_path),
        "summary_updated": summary_updated,
        "audit": audit,
    }


def audit_runpod_4090_hf_launch_kit(
    package: Path,
    launch_dir: Path,
    *,
    fast: bool = False,
) -> dict[str, Any]:
    """Validate the repo-owned 4090/HF launch kit against its package."""
    package = package.resolve()
    launch_dir = launch_dir.resolve()
    issues: list[str] = []
    warnings: list[str] = []
    helpers: dict[str, dict[str, Any]] = {}
    docs: dict[str, dict[str, Any]] = {}

    package_report = inspect_runpod_job_package(package, fast=fast, compute_sha256=not fast)
    issues.extend(f"package: {issue}" for issue in package_report["issues"])

    manifest_path = launch_dir / RUNPOD_4090_HF_MANIFEST
    if not manifest_path.exists():
        manifest: dict[str, Any] | None = None
        issues.append(f"missing launch manifest: {manifest_path}")
    else:
        loaded = _read_json_file(manifest_path, issues)
        manifest = loaded if isinstance(loaded, dict) else None
        if manifest is None:
            issues.append(f"launch manifest is not a JSON object: {manifest_path}")

    if manifest is not None:
        if manifest.get("schema_version") != "runpod-4090-hf-launch-v1":
            issues.append("manifest schema_version must be runpod-4090-hf-launch-v1")
        if manifest.get("status") != "ready_to_upload":
            issues.append("manifest status must be ready_to_upload")

        manifest_package = manifest.get("package")
        if not isinstance(manifest_package, dict):
            issues.append("manifest package must be an object")
            manifest_package = {}
        _audit_manifest_package(
            package=package,
            package_report=package_report,
            manifest_package=manifest_package,
            fast=fast,
            issues=issues,
            warnings=warnings,
        )
        _audit_manifest_safety_warnings(manifest, issues)
        _audit_manifest_helpers(manifest, launch_dir, helpers, issues)
        _audit_manifest_docs(manifest, launch_dir, docs, issues)

    _audit_4090_package_runner(package, issues)
    _audit_4090_helper_contents(helpers, issues)
    _audit_4090_doc_contents(docs, issues)

    return {
        "valid": not issues,
        "issues": issues,
        "warnings": warnings,
        "package": str(package),
        "launch_dir": str(launch_dir),
        "manifest_path": str(manifest_path),
        "manifest": {
            "schema_version": manifest.get("schema_version") if manifest else None,
            "status": manifest.get("status") if manifest else None,
            "package_sha256": (manifest.get("package") or {}).get("sha256") if manifest else None,
            "package_size_bytes": (manifest.get("package") or {}).get("size_bytes") if manifest else None,
            "safety_warnings": manifest.get("safety_warnings", []) if manifest else [],
        },
        "helpers": helpers,
        "docs": docs,
        "package_inspection": package_report,
    }




def _runpod_4090_hf_start_here_markdown(*, status: str, paths: dict[str, str]) -> str:
    prelaunch = _powershell_path(paths["prelaunch"])
    upload = _powershell_path(paths["upload"])
    one_shot = _powershell_path(paths["one_shot"])
    monitor = _powershell_path(paths["monitor"])
    download = _powershell_path(paths["download"])
    return f"""# Start Here: 4090/HF RunPod Pilot

Current local status on 2026-07-07: `{status}`.
Package status: `ready_to_upload`.
Operator status: `{status}`.

Naming note: `h100_return.zip`, `fpv h100`, and `outputs/h100/...` are legacy internal contract names kept for compatibility. They do not mean the current run used an H100; this launch targets the 4090/HF pod profile unless you explicitly choose a different GPU.

Use this file when launching the current RTX 4090 or similar cheaper high-end GPU run.

## Use These Files

- Package: `{paths['package']}`
- Launch manifest: `{paths['manifest']}`
- Pod profile: `{paths['pod_profile']}`
- Full checklist: `{paths['checklist']}`
- Pod script: `{paths['pod_script']}`
- One-shot full launcher: `{paths['one_shot']}`

Do not use these legacy files for the current 4090/HF run:

- `{paths['legacy_markdown']}`
- `{paths['legacy_manifest']}`
- older sibling package ZIPs under `outputs/h100/full_161_run/`

## Local Gates Before Upload

Check plain-English status first:

```powershell
fpv runpod status-4090-launch
```

Run the full source-owned audit before paid upload:

```powershell
fpv runpod audit-4090-launch
```


Run the non-GPU package smoke check before paid upload:

```powershell
fpv runpod smoke-package --source outputs/h100/full_161_run/runpod_job_4090_hf_full_stack_clean.zip
```
Then run the operator prelaunch helper:

```powershell
{prelaunch}
```

For quick local iteration only, use `--fast` / `-Fast`.

## One-Shot Full Run: 4090 + VGGT/COLMAP + R3/LingBot

After the pod is ready and `HF_TOKEN` is available on the pod, this is the shortest full command. It uploads the base package, uploads the R3/LingBot wrapper, starts `full_hf_rebuild`, enables the optional R3/LingBot stages, and requests the optional OpenMVS/ODM/Nerfstudio stages when their dependencies are available:

```powershell
{one_shot} `
  -HostName <runpod_ip> `
  -Port <tcp_port> `
  -IdentityFile $env:USERPROFILE\\.ssh\\id_ed25519
```

Use this for the real full run after the cheap scout pilot is proven.

## First Run: Cheap 4090 Pilot

After creating a RunPod pod with direct TCP SSH, upload and start the bounded 3-item scout pilot:

```powershell
{upload} `
  -HostName <runpod_ip> `
  -Port <tcp_port> `
  -IdentityFile $env:USERPROFILE\\.ssh\\id_ed25519 `
  -Start `
  -RunMode pilot_hf_scout3
```

Set `HF_TOKEN` on the pod through RunPod secrets or interactively. Do not put it in repo files, command history snippets, or returned artifacts.

## Monitor

```powershell
{monitor} `
  -HostName <runpod_ip> `
  -Port <tcp_port> `
  -IdentityFile $env:USERPROFILE\\.ssh\\id_ed25519
```

## Download And Validate

```powershell
{download} `
  -HostName <runpod_ip> `
  -Port <tcp_port> `
  -IdentityFile $env:USERPROFILE\\.ssh\\id_ed25519
```

The goal is not complete until the fresh `h100_return.zip` is downloaded and validated locally.

Then write the 4090 final audit:

```powershell
fpv runpod final-audit-4090-return --source <local_return_zip>
```

## Safety Boundary

No geolocation. No map projection. No meters. No true speed/standoff/dive-angle claims. No route, approach, launch, target-coordinate, guidance, or next-maneuver inference. Media-derived artifacts remain local-only.
"""


def _audit_manifest_package(
    *,
    package: Path,
    package_report: dict[str, Any],
    manifest_package: dict[str, Any],
    fast: bool,
    issues: list[str],
    warnings: list[str],
) -> None:
    source_text = str(manifest_package.get("source") or "")
    if source_text:
        manifest_source = _resolve_launch_path(source_text, package.parent)
        if manifest_source.resolve() != package:
            issues.append(
                "manifest package source does not match audited package: "
                f"{manifest_source} != {package}"
            )
    else:
        issues.append("manifest package missing source")

    if package.is_file():
        size_bytes = package.stat().st_size
        if manifest_package.get("size_bytes") != size_bytes:
            issues.append(
                "manifest package size_bytes does not match audited package: "
                f"{manifest_package.get('size_bytes')} != {size_bytes}"
            )
        if not fast:
            sha256 = package_report.get("zip_sha256")
            if manifest_package.get("sha256") != sha256:
                issues.append(
                    "manifest package sha256 does not match audited package: "
                    f"{manifest_package.get('sha256')} != {sha256}"
                )
        elif manifest_package.get("sha256"):
            warnings.append("fast mode skipped audited package SHA256 comparison")

    if manifest_package.get("clip_count") != package_report.get("clip_count"):
        issues.append(
            "manifest package clip_count does not match package inspection: "
            f"{manifest_package.get('clip_count')} != {package_report.get('clip_count')}"
        )


def _audit_manifest_safety_warnings(manifest: dict[str, Any], issues: list[str]) -> None:
    safety_warnings = manifest.get("safety_warnings")
    if not isinstance(safety_warnings, list):
        issues.append("manifest safety_warnings must be a list")
        return
    present = {str(item).lower() for item in safety_warnings}
    for warning in RUNPOD_4090_HF_SAFETY_WARNINGS:
        if warning.lower() not in present:
            issues.append(f"missing launch safety warning: {warning}")


def _audit_manifest_helpers(
    manifest: dict[str, Any],
    launch_dir: Path,
    helpers: dict[str, dict[str, Any]],
    issues: list[str],
) -> None:
    for key, expected_name in RUNPOD_4090_HF_HELPERS.items():
        value = manifest.get(key)
        if not value:
            issues.append(f"manifest missing helper path: {key}")
            continue
        path = _resolve_launch_path(str(value), launch_dir)
        helper_report: dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
            "expected_name": expected_name,
            "has_cr_bytes": None,
        }
        helpers[key] = helper_report
        if path.name != expected_name:
            issues.append(f"manifest helper {key} should end with {expected_name}: {path}")
        if not path.exists():
            issues.append(f"missing launch helper {key}: {path}")
            continue
        data = path.read_bytes()
        helper_report["has_cr_bytes"] = b"\r" in data
        if b"\r" in data:
            issues.append(f"launch helper contains CR bytes: {path}")




def _audit_manifest_docs(
    manifest: dict[str, Any],
    launch_dir: Path,
    docs: dict[str, dict[str, Any]],
    issues: list[str],
) -> None:
    for key, expected_name in RUNPOD_4090_HF_DOCS.items():
        value = manifest.get(key)
        if not value:
            issues.append(f"manifest missing doc path: {key}")
            continue
        path = _resolve_launch_path(str(value), launch_dir)
        doc_report: dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
            "expected_name": expected_name,
            "has_cr_bytes": None,
        }
        docs[key] = doc_report
        if path.name != expected_name:
            issues.append(f"manifest doc {key} should end with {expected_name}: {path}")
        if not path.exists():
            issues.append(f"missing launch doc {key}: {path}")
            continue
        data = path.read_bytes()
        doc_report["has_cr_bytes"] = b"\r" in data
        if b"\r" in data:
            issues.append(f"launch doc contains CR bytes: {path}")


def _audit_4090_helper_contents(helpers: dict[str, dict[str, Any]], issues: list[str]) -> None:
    pod_script = _read_helper_text(helpers, "pod_script")
    if pod_script is not None:
        for token in [
            "FPV_RUN_MODE",
            "pilot_hf_scout3",
            "full_packaged",
            "full_hf_rebuild",
            "SAFETY_WARNING",
            "HF_TOKEN is not set",
            "python -m zipfile -e",
            "bash run_all.sh",
            "FPV_GPU_PROFILE",
            "HF_DATASET_HIGH_DETAIL_FRAMES",
            "run_optimizer_status.json",
        ]:
            if token not in pod_script:
                issues.append(f"RUNPOD_RUN_4090_HF.sh missing {token!r}")

    prelaunch = _read_helper_text(helpers, "prelaunch_audit_powershell")
    if prelaunch is not None:
        for token in [
            RUNPOD_4090_HF_MANIFEST,
            "scripts/80_run_expanded_methods.py",
            "validate_return_powershell",
            "control_powershell",
        ]:
            if token not in prelaunch:
                issues.append(f"PRELAUNCH_AUDIT_4090_HF.ps1 missing {token!r}")

    one_shot = _read_helper_text(helpers, "one_shot_full_powershell")
    if one_shot is not None:
        for token in [
            "UPLOAD_AND_OPTIONALLY_START_4090_HF.ps1",
            "full_hf_rebuild",
            "UseR3LingBotRunner",
            "EnableFullOptionalMethods",
            "No geolocation",
        ]:
            if token not in one_shot:
                issues.append(f"RUN_FULL_4090_R3_LINGBOT.ps1 missing {token!r}")

    upload = _read_helper_text(helpers, "upload_and_start_powershell")
    if upload is not None:
        for token in [
            "EnableFullOptionalMethods",
            "OPTIONAL_RUN_ODM=1",
            "OPTIONAL_RUN_NERFSTUDIO=1",
            "SKIP_RESEARCH_METHODS=0",
            "RemoteEnvPrefix",
        ]:
            if token not in upload:
                issues.append(f"UPLOAD_AND_OPTIONALLY_START_4090_HF.ps1 missing {token!r}")

    r3_lingbot = _read_helper_text(helpers, "r3_lingbot_runner")
    if r3_lingbot is not None:
        for token in [
            "KevinXu02/R3",
            "robbyant/lingbot-map",
            "resolve_image_sequence_dir",
            "r3_data_root_ready",
            "R3_INFERENCE_EXIT",
            "R3_TRAIN_EXIT",
            "LINGBOT_EXIT",
            "ENABLE_R3",
            "RUN_R3_INFERENCE",
            "RUN_R3_TRAINING",
            "ENABLE_LINGBOT_MAP",
            "RUN_LINGBOT_MAP_INFERENCE",
            "LINGBOT_MODEL_PATH",
            "r3_reconstruction_manifest.json",
            "r3_dataset_prep_report.json",
            "r3_outputs.zip",
            "lingbot_map_manifest.json",
            "lingbot_map_outputs.zip",
            "zipfile.ZipFile",
            "No geolocation",
        ]:
            if token not in r3_lingbot:
                issues.append(f"RUNPOD_INSTALL_TRAIN_R3_LINGBOT_4090.sh missing {token!r}")

    validate_return = _read_helper_text(helpers, "validate_return_powershell")
    if validate_return is not None:
        if RUNPOD_4090_HF_MANIFEST not in validate_return:
            issues.append("VALIDATE_RETURN.ps1 must validate against runpod_launch_4090_hf_manifest.json")
        if "runpod_launch_manifest.json" in validate_return:
            issues.append("VALIDATE_RETURN.ps1 references stale runpod_launch_manifest.json")


def _audit_4090_package_runner(package: Path, issues: list[str]) -> None:
    run_all = _read_package_text(package, "run_all.sh", issues)
    expanded_runner = _read_package_text(package, "scripts/80_run_expanded_methods.py", issues)
    if run_all is not None and "python scripts/80_run_expanded_methods.py" not in run_all:
        issues.append("run_all.sh does not invoke scripts/80_run_expanded_methods.py")
    if expanded_runner is not None and "method_stage_report.json" not in expanded_runner:
        issues.append("scripts/80_run_expanded_methods.py does not write method_stage_report.json")




def _audit_4090_doc_contents(docs: dict[str, dict[str, Any]], issues: list[str]) -> None:
    start_here = _read_doc_text(docs, "start_here_markdown")
    if start_here is None:
        return
    profile = _read_doc_json(docs, "pilot_profile", issues)
    if isinstance(profile, dict):
        if profile.get("schema_version") != "runpod-4090-hf-profile-v1":
            issues.append("RUNPOD_4090_PILOT_PROFILE.json has wrong schema_version")
        if profile.get("gpu", {}).get("primary") != "RTX 4090 24 GB":
            issues.append("RUNPOD_4090_PILOT_PROFILE.json primary GPU must be RTX 4090 24 GB")
        if profile.get("resources", {}).get("volume_disk_gb_min", 0) < 150:
            issues.append("RUNPOD_4090_PILOT_PROFILE.json volume disk minimum must be at least 150 GB")
        if profile.get("run_modes", {}).get("default") != "pilot_hf_scout3":
            issues.append("RUNPOD_4090_PILOT_PROFILE.json default run mode must be pilot_hf_scout3")
        required_env = profile.get("environment", {}).get("required", {})
        if required_env.get("HF_DATASET_ID") != "Grimster/FPV_Hezbo":
            issues.append("RUNPOD_4090_PILOT_PROFILE.json must set HF_DATASET_ID to Grimster/FPV_Hezbo")
        if "HF_TOKEN" in json.dumps(required_env):
            issues.append("RUNPOD_4090_PILOT_PROFILE.json must not store HF_TOKEN as an environment value")
        full_env = profile.get("environment", {}).get("full_after_pilot", {})
        expected_full_env = {
            "OPTIONAL_RUN_ODM": "1",
            "OPTIONAL_RUN_OPENMVS": "1",
            "OPTIONAL_RUN_NERFSTUDIO": "1",
            "SKIP_RESEARCH_METHODS": "0",
        }
        for key, expected in expected_full_env.items():
            if full_env.get(key) != expected:
                issues.append(f"RUNPOD_4090_PILOT_PROFILE.json full_after_pilot must set {key}={expected}")
        for warning in RUNPOD_4090_HF_SAFETY_WARNINGS:
            if warning not in profile.get("safety_warnings", []):
                issues.append(f"RUNPOD_4090_PILOT_PROFILE.json missing safety warning: {warning}")
    for token in [
        "ready_to_run_4090_pilot",
        "runpod_job_4090_hf_full_stack_clean.zip",
        "RUNPOD_4090_PILOT_PROFILE.json",
        "legacy internal contract names kept for compatibility",
        "fpv runpod status-4090-launch",
        "fpv runpod audit-4090-launch",
        "fpv runpod smoke-package --source",
        "UPLOAD_AND_OPTIONALLY_START_4090_HF.ps1",
        "RUN_FULL_4090_R3_LINGBOT.ps1",
        "One-Shot Full Run",
        "full_hf_rebuild",
        "R3/LingBot",
        "optional OpenMVS/ODM/Nerfstudio",
        "DOWNLOAD_AND_VALIDATE_RETURN.ps1",
        "pilot_hf_scout3",
        "RUNPOD_LAUNCH.md",
        "runpod_launch_manifest.json",
        "No geolocation",
        "No map projection",
        "No true speed/standoff/dive-angle claims",
        "No route, approach, launch, target-coordinate, guidance, or next-maneuver inference",
    ]:
        if token not in start_here:
            issues.append(f"START_HERE_4090_HF.md missing {token!r}")


def _read_helper_text(helpers: dict[str, dict[str, Any]], key: str) -> str | None:
    path_text = helpers.get(key, {}).get("path")
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="ignore")






def _read_doc_json(docs: dict[str, dict[str, Any]], key: str, issues: list[str]) -> Any:
    text = _read_doc_text(docs, key)
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception as exc:
        issues.append(f"{RUNPOD_4090_HF_DOCS.get(key, key)} is invalid JSON: {exc}")
        return None


def _read_doc_text(docs: dict[str, dict[str, Any]], key: str) -> str | None:
    path_text = docs.get(key, {}).get("path")
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_package_text(package: Path, name: str, issues: list[str]) -> str | None:
    try:
        if package.is_file() and package.suffix.lower() == ".zip":
            with zipfile.ZipFile(package) as archive:
                if name not in archive.namelist():
                    issues.append(f"package missing {name}")
                    return None
                return archive.read(name).decode("utf-8", errors="ignore")
        if package.is_dir():
            path = package / name
            if not path.exists():
                issues.append(f"package missing {name}")
                return None
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        issues.append(f"could not read package member {name}: {exc}")
        return None
    issues.append(f"package source is neither zip nor directory: {package}")
    return None






def _powershell_path(path_text: str) -> str:
    normalized = path_text.replace("/", "\\")
    if normalized.startswith(".") or Path(normalized).is_absolute():
        return normalized
    return ".\\" + normalized


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_launch_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    base_candidate = (base_dir / path).resolve()
    if base_candidate.exists():
        return base_candidate
    return (base_dir / path.name).resolve()


def write_runpod_launch_manifest(source: Path, output_dir: Path) -> dict[str, Any]:
    """Write a deterministic RunPod launch checklist for a validated package."""
    source = source.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inspection = inspect_runpod_job_package(source)
    package_name = source.name if source.is_file() else "runpod_job.zip"
    remote_root = "/workspace/fpv-expanded"
    remote_zip = f"/workspace/{package_name}"
    preflight_gate_command = "\n".join(
        [
            "python - <<'PY'",
            "import json",
            "from pathlib import Path",
            "report = json.loads(Path('expanded_runtime_check.json').read_text(encoding='utf-8'))",
            "status = report.get('readiness_status')",
            "allowed = {'ready_full_stack', 'ready_4090_pilot', 'ready_full'}",
            "if status not in allowed:",
            "    print(f'STOP: readiness_status={status}')",
            "    for action in report.get('next_actions', []):",
            "        print(f'- {action}')",
            "    raise SystemExit(2)",
            "print(f'preflight gate: {status}')",
            "PY",
        ]
    )
    if not inspection["valid"]:
        launch_status = "blocked_invalid_package"
    elif inspection.get("zip_sha256"):
        launch_status = "ready_to_upload"
    else:
        launch_status = "ready_unhashed"
    launch = {
        "schema_version": "runpod-launch-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": launch_status,
        "package": {
            "source": str(source),
            "kind": inspection["source_kind"],
            "upload_name": package_name,
            "size_bytes": source.stat().st_size if source.is_file() else None,
            "sha256": inspection.get("zip_sha256"),
            "file_count": inspection["file_count"],
            "total_uncompressed_bytes": inspection["total_uncompressed_bytes"],
            "clip_count": inspection["clip_count"],
        },
        "inspection": inspection,
        "methods": inspection.get("method_ids", []),
        "stages": inspection.get("stage_ids", []),
        "remote": {
            "root": remote_root,
            "zip_path": remote_zip,
            "required_env": [
                "VGGT_REPO_DIR=/workspace/vggt or equivalent checkout containing demo_colmap.py",
                "Optional VGGT_COLMAP_EXTRA_ARGS for demo_colmap.py tuning",
                "Optional HF_DATASET_ID=Grimster/FPV_Hezbo with HF_TOKEN for cloud-side dataset access checks/download",
                "For a bounded 4090 HF pilot: HF_DATASET_BUILD_FRAME_PACKS=1, HF_DATASET_MAX_ITEMS=3, HF_DATASET_FRAME_PACK_TIERS=scout",
                "For the packaged full manifest: leave HF_DATASET_BUILD_FRAME_PACKS=0; for a full HF rebuild use HF_DATASET_BUILD_FRAME_PACKS=1 with no max-items limit",
            ],
            "commands": [
                "set -e",
                f"mkdir -p {remote_root}",
                f"cd {remote_root}",
                f"python -m zipfile -e {remote_zip} .",
                "python scripts/05_expanded_runtime_check.py",
                "python scripts/06_hf_dataset_preflight.py",
                "python scripts/07_hf_dataset_frame_packs.py",
                "cat expanded_runtime_check.json",
                "cat hf_dataset_report.json",
                "cat hf_frame_pack_report.json",
                preflight_gate_command,
                "set +e",
                "bash run_all.sh",
                "RUN_STATUS=$?",
                "ls -lh h100_return.zip cloud_summary.json cloud_run.log environment.json hf_dataset_report.json hf_frame_pack_report.json run_optimizer_status.json cloud_run_status.json method_stage_report.json quality_report.json || true",
                "exit $RUN_STATUS",
            ],
        },
        "expected_return": {
            "primary_download": "h100_return.zip",
            "required_files_inside_return": list(CORE_RETURN_FILES),
            "expanded_artifacts_when_available": expanded_return_artifact_files(),
            "local_validation_commands": [
                "fpv h100 inspect-return --source <local_return_zip>",
                "fpv h100 inspect-return --source <local_return_zip> --json",
                "fpv h100 postflight-return --source <local_return_zip> --launch-manifest outputs/h100/full_161_run/launch/runpod_launch_manifest.json --output-dir outputs/reviews/h100_expanded_postflight",
                "fpv h100 optional-report --source <local_return_zip> --output-dir outputs/reviews/h100_optional_methods",
                "fpv h100 import-return --source <local_return_zip> --workdir outputs/h100/full_161_run --review-output outputs/reviews/h100_expanded_import --dry-run",
            ],
        },
        "safety_warnings": H100_WARNINGS,
        "issues": inspection.get("issues", []),
        "warnings": inspection.get("warnings", []),
    }
    json_path = output_dir / "runpod_launch_manifest.json"
    md_path = output_dir / "RUNPOD_LAUNCH.md"
    validate_ps1_path = output_dir / "VALIDATE_RETURN.ps1"
    json_path.write_text(json.dumps(launch, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_runpod_launch_markdown(launch), encoding="utf-8")
    _write_text_lf(validate_ps1_path, _runpod_return_validation_powershell())
    launch["artifacts"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "validate_return_powershell": str(validate_ps1_path),
    }
    json_path.write_text(json.dumps(launch, indent=2, sort_keys=True), encoding="utf-8")
    return launch


def _runpod_return_validation_powershell() -> str:
    return r'''param(
    [Parameter(Mandatory = $true)]
    [string]$ReturnZip,
    [string]$PostflightOutput = "outputs/reviews/h100_expanded_postflight",
    [string]$OptionalOutput = "outputs/reviews/h100_optional_methods",
    [string]$ImportOutput = "outputs/reviews/h100_expanded_import"
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path -LiteralPath (Join-Path $ScriptRoot "..\..\..\..")
$ReturnPath = (Resolve-Path -LiteralPath $ReturnZip).Path
$Failures = @()

function Invoke-FpvStep {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    Write-Host ""
    Write-Host "== $Name =="
    & fpv @Arguments
    if ($LASTEXITCODE -ne 0) {
        $script:Failures += "$Name exited $LASTEXITCODE"
    }
}

Push-Location $RepoRoot
try {
    Write-Host "H100 return validation"
    Write-Host "Return ZIP: $ReturnPath"
    Write-Host "Warning: no geolocation, no meters, relative VGGT frame, local-only media"

    Invoke-FpvStep "inspect-return" @("h100", "inspect-return", "--source", $ReturnPath)
    Invoke-FpvStep "inspect-return-json" @("h100", "inspect-return", "--source", $ReturnPath, "--json")
    Invoke-FpvStep "postflight-return" @(
        "h100", "postflight-return",
        "--source", $ReturnPath,
        "--launch-manifest", "outputs/h100/full_161_run/launch/runpod_launch_manifest.json",
        "--output-dir", $PostflightOutput
    )
    Invoke-FpvStep "optional-report" @(
        "h100", "optional-report",
        "--source", $ReturnPath,
        "--output-dir", $OptionalOutput
    )
    Invoke-FpvStep "import-return-dry-run" @(
        "h100", "import-return",
        "--source", $ReturnPath,
        "--workdir", "outputs/h100/full_161_run",
        "--review-output", $ImportOutput,
        "--dry-run"
    )

    if ($Failures.Count -gt 0) {
        Write-Host ""
        Write-Host "Validation completed with issues:"
        foreach ($Failure in $Failures) {
            Write-Host "- $Failure"
        }
        exit 1
    }

    Write-Host ""
    Write-Host "Validation completed successfully. Review outputs under outputs/reviews/."
}
finally {
    Pop-Location
}
'''


def _runpod_launch_markdown(launch: dict[str, Any]) -> str:
    package = launch["package"]
    remote = launch["remote"]
    expected = launch["expected_return"]
    commands = "\n".join(remote["commands"])
    validation = "\n".join(expected["local_validation_commands"])
    methods = "\n".join(f"- `{method}`" for method in launch.get("methods", []))
    warnings = "\n".join(f"- {warning}" for warning in launch.get("safety_warnings", []))
    issues = "\n".join(f"- {issue}" for issue in launch.get("issues", [])) or "- none"
    return f"""# RunPod Launch Checklist

Status: `{launch['status']}`

## Upload

- Package: `{package['source']}`
- Upload name: `{package['upload_name']}`
- Size bytes: `{package['size_bytes']}`
- SHA256: `{package['sha256']}`
- Clip jobs: `{package['clip_count']}`

## Methods

{methods}

## Run On Pod

```bash
{commands}
```

## Bring Back

Download `{expected['primary_download']}` from `{remote['root']}` and keep it local-only.

## Validate Locally

Use the generated helper after downloading `h100_return.zip`:

```powershell
.\\outputs\\h100\\full_161_run\\launch\\VALIDATE_RETURN.ps1 -ReturnZip <local_return_zip>
```

Equivalent manual commands:

```powershell
{validation}
```

## Safety

{warnings}

## Issues

{issues}
"""


def _read_json_file(path: Path, issues: list[str]) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.append(f"invalid JSON file {path.name}: {exc}")
        return None


def _read_json_from_zip(archive: zipfile.ZipFile, name: str, issues: list[str]) -> Any:
    if name not in archive.namelist():
        return None
    try:
        return json.loads(archive.read(name).decode("utf-8"))
    except Exception as exc:
        issues.append(f"invalid JSON member {name}: {exc}")
        return None


def validate_runpod_job_package(runpod_job: Path, zip_path: Path) -> dict[str, Any]:
    issues: list[str] = []
    checked_files = 0

    for path in runpod_job.rglob("*"):
        if not path.is_file():
            continue
        checked_files += 1
        relative = path.relative_to(runpod_job)
        if path.suffix.lower() in RAW_MEDIA_SUFFIXES:
            issues.append(f"raw media file is not allowed in package: {relative.as_posix()}")
        if path.suffix.lower() in {".json", ".md", ".py", ".sh", ".yaml", ".yml"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if SECRET_PATTERN.search(text):
                issues.append(f"secret-looking text in package file: {relative.as_posix()}")
            if WINDOWS_ABSOLUTE_PATTERN.search(text):
                issues.append(f"absolute Windows path in package file: {relative.as_posix()}")

    if not zip_path.exists():
        issues.append("missing runpod_job.zip")
    else:
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                member = Path(name)
                if member.is_absolute() or ".." in member.parts:
                    issues.append(f"unsafe zip member path: {name}")
                if name.lower().endswith(tuple(RAW_MEDIA_SUFFIXES)):
                    issues.append(f"raw media file is not allowed in zip: {name}")

    return {
        "valid": not issues,
        "issues": issues,
        "checked_files": checked_files,
        "zip_path": str(zip_path),
    }


def _write_dataset_snapshot(dataset: str, workdir: Path, logger: RunLogger) -> dict[str, Any]:
    if dataset == "none":
        return {"status": "skipped", "reason": "dataset=none"}
    if dataset != "latest":
        return {"status": "skipped", "reason": f"unsupported dataset mode {dataset!r}"}
    try:
        catalog_path, snapshot_path = sync_catalog(
            readme_source=DEFAULT_README_URL,
            manifest_source=DEFAULT_MANIFEST_URL,
            output=workdir / "dataset_snapshot_source",
        )
        shutil.copyfile(catalog_path, workdir / "dataset_snapshot.parquet")
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        (workdir / "dataset_snapshot.json").write_text(
            json.dumps(snapshot, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.log(f"dataset snapshot row_count={snapshot.get('row_count')}")
        return {
            "status": "done",
            "row_count": snapshot.get("row_count"),
            "catalog": str(workdir / "dataset_snapshot.parquet"),
            "snapshot": str(workdir / "dataset_snapshot.json"),
        }
    except Exception as exc:
        logger.log(f"dataset snapshot failed_soft: {exc}")
        return {"status": "failed_soft", "reason": str(exc)}


def _collect_frame_sources(
    *,
    frame_manifests: list[Path],
    annotations: Path,
    media_inventory: Path,
    frames_root: Path,
    workdir: Path,
    tiers: list[H100Tier],
    logger: RunLogger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if frame_manifests:
        rows = []
        sources = []
        for manifest_path in frame_manifests:
            manifest = read_frame_manifest(manifest_path)
            sources.append(
                {
                    "tier": "main",
                    "frame_manifest": manifest_path,
                    "manifest": manifest,
                    "requested_frame_count": len(manifest.frames),
                    "resized_long_edge": manifest.frames[0].resized_long_edge
                    if manifest.frames
                    else None,
                }
            )
            rows.append(
                {
                    "video_id": manifest.video_id,
                    "segment_id": manifest.segment_id,
                    "decision": "accepted_external_manifest",
                    "accepted_by": "frame_manifest",
                    "reason": "explicit --frame-manifest input",
                }
            )
        logger.log(f"using {len(sources)} explicit frame manifests")
        return sources, rows

    segment_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    accepted = [annotation for annotation in read_annotations(annotations) if annotation.status == "accepted"]
    if not accepted:
        return sources, [
            {
                "video_id": None,
                "segment_id": None,
                "decision": "needs_human_review",
                "accepted_by": None,
                "reason": "no accepted segment annotations available",
            }
        ]

    for annotation in accepted:
        segment_rows.append(
            {
                "video_id": annotation.video_id,
                "segment_id": annotation.segment_id,
                "decision": "accepted_human",
                "accepted_by": "human",
                "reason": "existing accepted annotation",
            }
        )
        for tier in tiers:
            output = (
                frames_root
                / "h100"
                / workdir.name
                / tier.name
                / annotation.video_id
                / annotation.segment_id
            )
            manifest_path = output / "frames.json"
            manifest = _read_ready_existing_frame_manifest(manifest_path)
            if manifest is not None:
                logger.log(
                    f"reuse {tier.name} frame pack for "
                    f"{annotation.video_id}/{annotation.segment_id}"
                )
            else:
                logger.log(
                    f"sampling {tier.name} frame pack for "
                    f"{annotation.video_id}/{annotation.segment_id}"
                )
                try:
                    manifest = sample_accepted_segment_frames(
                        media_inventory=media_inventory,
                        annotations=annotations,
                        video_id=annotation.video_id,
                        segment_id=annotation.segment_id,
                        output=output,
                        count=tier.frame_count,
                        resized_long_edge=tier.resized_long_edge,
                    )
                except Exception as exc:
                    logger.log(
                        f"failed to sample {tier.name} for "
                        f"{annotation.video_id}/{annotation.segment_id}: {exc}"
                    )
                    continue
            sources.append(
                {
                    "tier": tier.name,
                    "frame_manifest": output / "frames.json",
                    "manifest": manifest,
                    "requested_frame_count": tier.frame_count,
                    "resized_long_edge": tier.resized_long_edge,
                }
            )
    return sources, segment_rows



def _read_ready_existing_frame_manifest(path: Path) -> FrameManifest | None:
    if not path.exists():
        return None
    try:
        manifest = read_frame_manifest(path)
    except Exception:
        return None
    for frame in manifest.frames:
        if not Path(frame.path).exists():
            return None
    return manifest
def _write_runpod_job(
    *,
    runpod_job: Path,
    run_id: str,
    frame_sources: list[dict[str, Any]],
    metadata_policy: str,
    logger: RunLogger,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frame_pack_rows: list[dict[str, Any]] = []
    clips = []
    for source in frame_sources:
        manifest: FrameManifest = source["manifest"]
        tier = str(source["tier"])
        safe_clip = _safe_name(f"{tier}__{manifest.video_id}__{manifest.segment_id}")
        clip_dir = runpod_job / "frame_packs" / tier / safe_clip
        packaged_manifest_path = _copy_frame_manifest_for_package(
            manifest=manifest,
            destination=clip_dir,
        )
        rel_manifest = packaged_manifest_path.relative_to(runpod_job).as_posix()
        clip = {
            "clip_id": safe_clip,
            "tier": tier,
            "video_id": manifest.video_id,
            "segment_id": manifest.segment_id,
            "frame_manifest": rel_manifest,
            "frame_count": len(manifest.frames),
            "requested_frame_count": int(source["requested_frame_count"]),
            "resized_long_edge": source["resized_long_edge"],
            "frame_manifest_sha256": _sha256_file(packaged_manifest_path),
            "bundle_output": (
                f"bundles/{tier}/{_safe_name(manifest.video_id)}/"
                f"{_safe_name(manifest.segment_id)}"
            ),
            "predictions_output": (
                f"predictions/{tier}/{_safe_name(manifest.video_id)}/"
                f"{_safe_name(manifest.segment_id)}/predictions.npz"
            ),
        }
        clips.append(clip)
        frame_pack_rows.append(
            {
                "run_id": run_id,
                "tier": tier,
                "video_id": manifest.video_id,
                "segment_id": manifest.segment_id,
                "frame_count": len(manifest.frames),
                "requested_frame_count": int(source["requested_frame_count"]),
                "resized_long_edge": source["resized_long_edge"],
                "frame_manifest": str(source["frame_manifest"]),
                "package_frame_manifest": rel_manifest,
                "frame_manifest_sha256": clip["frame_manifest_sha256"],
            }
        )

    job_manifest = {
        "schema_version": "runpod-job-v1",
        "profile": "runpod-gpu-inference",
        "recommended_gpu": "RTX 4090 24GB pilot; 48GB GPU or H100 fallback for full/high-detail runs",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata_policy": metadata_policy,
        "warnings": [
            "local-only media-derived frames",
            "no geolocation",
            "no meters",
            "relative VGGT frame",
        ],
        "tier_order": ["smoke", "scout", "main", "high_detail"],
        "method_contract": {
            "method_matrix": "method_matrix.json",
            "return_contract": "return_contract.json",
            "method_contract": "METHOD_CONTRACT.md",
            "method_stage_plan": "method_stage_plan.json",
        },
        "clips": clips,
    }
    write_method_contract_files(runpod_job, clips=clips)
    _write_text_lf(
        runpod_job / "job_manifest.json",
        json.dumps(job_manifest, indent=2, sort_keys=True),
    )
    _write_text_lf(runpod_job / "run_vggt_job.py", _h100_vggt_runner_script())
    _write_text_lf(runpod_job / "run_all.sh", _run_all_script())
    scripts = runpod_job / "scripts"
    scripts.mkdir()
    _write_text_lf(scripts / "00_env_check.py", _env_check_script())
    _write_text_lf(scripts / "05_expanded_runtime_check.py", _expanded_runtime_check_script())
    _write_text_lf(scripts / "06_hf_dataset_preflight.py", _hf_dataset_preflight_script())
    _write_text_lf(scripts / "07_hf_dataset_frame_packs.py", _hf_dataset_frame_pack_script())
    _write_text_lf(scripts / "10_validate_inputs.py", _stage_placeholder("validate inputs"))
    _write_text_lf(scripts / "20_run_vggt_tier.py", _stage_placeholder("run VGGT tiers"))
    _write_text_lf(scripts / "30_validate_bundle.py", _stage_placeholder("validate bundles"))
    _write_text_lf(scripts / "40_select_best_tier.py", _stage_placeholder("select best tier"))
    _write_text_lf(scripts / "50_extract_features.py", _stage_placeholder("extract features"))
    _write_text_lf(scripts / "60_train_diagnostics.py", _stage_placeholder("train diagnostics"))
    _write_text_lf(scripts / "70_package_return.py", _package_return_script())
    _write_text_lf(scripts / "80_run_expanded_methods.py", _expanded_methods_script())
    _write_text_lf(runpod_job / "README.md", _runpod_readme())
    _write_text_lf(runpod_job / "job.yaml", _job_yaml(run_id))
    logger.log(f"wrote runpod job with {len(clips)} tiered clip entries")
    return job_manifest, frame_pack_rows


def _copy_frame_manifest_for_package(manifest: FrameManifest, destination: Path) -> Path:
    frames_dir = destination / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    updated_frames = []
    for ordinal, frame in enumerate(manifest.frames):
        source = Path(frame.path)
        suffix = source.suffix or ".jpg"
        target = frames_dir / f"{ordinal:04d}_frame_{frame.frame_index:06d}{suffix}"
        shutil.copyfile(source, target)
        updated_frames.append(frame.model_copy(update={"path": Path("frames") / target.name}))
    packaged = manifest.model_copy(update={"source_video": Path("source_video_not_packaged"), "frames": updated_frames})
    data = model_to_dict(packaged)
    data["source_video"] = "source_video_not_packaged"
    for frame in data["frames"]:
        frame["path"] = Path(frame["path"]).as_posix()
    manifest_path = destination / "frames.json"
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _zip_runpod_job(runpod_job: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(runpod_job.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(runpod_job))


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_empty_frame_pack_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "run_id",
        "tier",
        "video_id",
        "segment_id",
        "frame_count",
        "requested_frame_count",
        "resized_long_edge",
        "frame_manifest",
        "package_frame_manifest",
        "frame_manifest_sha256",
    ]
    pd.DataFrame(columns=columns).to_parquet(path, index=False)


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "unnamed"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ffmpeg_version() -> str:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        return f"unavailable: {exc}"
    first_line = result.stdout.splitlines()[0] if result.stdout else ""
    return first_line or f"unavailable: exit {result.returncode}"


def _h100_vggt_runner_script() -> str:
    script = _runner_script()
    install_block = (
        "    except ModuleNotFoundError:\n"
        "        run([sys.executable, \"-m\", \"pip\", \"install\", "
        "\"git+https://github.com/facebookresearch/vggt.git\"])\n"
    )
    fail_block = (
        "    except ModuleNotFoundError as exc:\n"
        "        raise RuntimeError(\n"
        "            \"VGGT is not installed in this pinned image. \"\n"
        "            \"Use a prepared RunPod image; do not install dependencies during the expensive run.\"\n"
        "        ) from exc\n"
    )
    return script.replace(install_block, fail_block)


def _emergency_return_packager_source() -> str:
    return r"""from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

root = Path.cwd()
manifest = {
    "schema_version": "h100-emergency-return-v1",
    "status": "failed_soft",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "reason": "70_package_return.py failed or did not create h100_return.zip",
    "warnings": ["no geolocation", "no meters", "relative VGGT frame", "local-only media"],
}
(root / "emergency_return_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True),
    encoding="utf-8",
)
include_files = [
    "emergency_return_manifest.json",
    "cloud_run.log",
    "run.log",
    "environment.json",
    "expanded_runtime_check.json",
    "cloud_summary.json",
    "quality_report.json",
    "method_stage_report.json",
    "method_matrix.json",
    "return_contract.json",
    "method_stage_plan.json",
    "METHOD_CONTRACT.md",
    "job_manifest.json",
    "hf_dataset_report.json",
    "hf_frame_pack_report.json",
    "cloud_run_status.json",
    "run_optimizer_status.json",
    "openmvs_dense_mesh_manifest.json",
]
include_dirs = ["selected", "features", "failure_packages"]
with zipfile.ZipFile(root / "h100_return.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for name in include_files:
        path = root / name
        if path.is_file():
            archive.write(path, arcname=name)
    for dirname in include_dirs:
        folder = root / dirname
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(root).as_posix())
"""

def _gpu_quality_profile_shell() -> str:
    return r"""detect_fpv_gpu_quality_profile() {
  GPU_MEM_MB=0
  GPU_NAME="unknown"
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_MEM_MB_RAW=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -dc '0-9' || true)
    GPU_NAME_RAW=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)
    GPU_MEM_MB=${GPU_MEM_MB_RAW:-0}
    GPU_NAME=${GPU_NAME_RAW:-unknown}
  fi

  CPU_THREADS=${FPV_CPU_THREADS:-$(python - <<'PY'
import os
print(min(max(os.cpu_count() or 8, 1), 16))
PY
)}
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$CPU_THREADS}
  export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$CPU_THREADS}
  export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$CPU_THREADS}
  export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-$CPU_THREADS}
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
  export CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-LAZY}
  if [ -z "${HF_HUB_ENABLE_HF_TRANSFER:-}" ]; then
    if python -c "import hf_transfer" >/dev/null 2>&1; then
      export HF_HUB_ENABLE_HF_TRANSFER=1
    fi
  fi

  if [ "$GPU_MEM_MB" -ge 70000 ]; then
    export FPV_GPU_PROFILE=${FPV_GPU_PROFILE:-80gb_quality}
    export HF_DATASET_SCOUT_FRAMES=${HF_DATASET_SCOUT_FRAMES:-48}
    export HF_DATASET_SCOUT_RESIZE=${HF_DATASET_SCOUT_RESIZE:-768}
    export HF_DATASET_MAIN_FRAMES=${HF_DATASET_MAIN_FRAMES:-128}
    export HF_DATASET_MAIN_RESIZE=${HF_DATASET_MAIN_RESIZE:-1280}
    export HF_DATASET_HIGH_DETAIL_FRAMES=${HF_DATASET_HIGH_DETAIL_FRAMES:-240}
    export HF_DATASET_HIGH_DETAIL_RESIZE=${HF_DATASET_HIGH_DETAIL_RESIZE:-1536}
    export OPTIONAL_SHOWCASE_MAX_CLIPS=${OPTIONAL_SHOWCASE_MAX_CLIPS:-12}
    export RESEARCH_METHOD_MAX_IMAGES=${RESEARCH_METHOD_MAX_IMAGES:-128}
    export DUST3R_BATCH_SIZE=${DUST3R_BATCH_SIZE:-2}
    export MAST3R_BATCH_SIZE=${MAST3R_BATCH_SIZE:-2}
    export R3_PREPARED_FRAME_LIMIT=${R3_PREPARED_FRAME_LIMIT:-160}
  elif [ "$GPU_MEM_MB" -ge 43000 ]; then
    export FPV_GPU_PROFILE=${FPV_GPU_PROFILE:-48gb_quality}
    export HF_DATASET_SCOUT_FRAMES=${HF_DATASET_SCOUT_FRAMES:-40}
    export HF_DATASET_SCOUT_RESIZE=${HF_DATASET_SCOUT_RESIZE:-768}
    export HF_DATASET_MAIN_FRAMES=${HF_DATASET_MAIN_FRAMES:-96}
    export HF_DATASET_MAIN_RESIZE=${HF_DATASET_MAIN_RESIZE:-1280}
    export HF_DATASET_HIGH_DETAIL_FRAMES=${HF_DATASET_HIGH_DETAIL_FRAMES:-200}
    export HF_DATASET_HIGH_DETAIL_RESIZE=${HF_DATASET_HIGH_DETAIL_RESIZE:-1536}
    export OPTIONAL_SHOWCASE_MAX_CLIPS=${OPTIONAL_SHOWCASE_MAX_CLIPS:-8}
    export RESEARCH_METHOD_MAX_IMAGES=${RESEARCH_METHOD_MAX_IMAGES:-96}
    export DUST3R_BATCH_SIZE=${DUST3R_BATCH_SIZE:-1}
    export MAST3R_BATCH_SIZE=${MAST3R_BATCH_SIZE:-1}
    export R3_PREPARED_FRAME_LIMIT=${R3_PREPARED_FRAME_LIMIT:-128}
  else
    export FPV_GPU_PROFILE=${FPV_GPU_PROFILE:-24gb_quality_first}
    export HF_DATASET_SCOUT_FRAMES=${HF_DATASET_SCOUT_FRAMES:-32}
    export HF_DATASET_SCOUT_RESIZE=${HF_DATASET_SCOUT_RESIZE:-768}
    export HF_DATASET_MAIN_FRAMES=${HF_DATASET_MAIN_FRAMES:-64}
    export HF_DATASET_MAIN_RESIZE=${HF_DATASET_MAIN_RESIZE:-1024}
    export HF_DATASET_HIGH_DETAIL_FRAMES=${HF_DATASET_HIGH_DETAIL_FRAMES:-180}
    export HF_DATASET_HIGH_DETAIL_RESIZE=${HF_DATASET_HIGH_DETAIL_RESIZE:-1280}
    export OPTIONAL_SHOWCASE_MAX_CLIPS=${OPTIONAL_SHOWCASE_MAX_CLIPS:-6}
    export RESEARCH_METHOD_MAX_IMAGES=${RESEARCH_METHOD_MAX_IMAGES:-80}
    export DUST3R_BATCH_SIZE=${DUST3R_BATCH_SIZE:-1}
    export MAST3R_BATCH_SIZE=${MAST3R_BATCH_SIZE:-1}
    export R3_PREPARED_FRAME_LIMIT=${R3_PREPARED_FRAME_LIMIT:-96}
  fi

  export HF_DATASET_FETCH_CATALOG_MEDIA=${HF_DATASET_FETCH_CATALOG_MEDIA:-1}
  export HF_DATASET_AUTO_TRIM=${HF_DATASET_AUTO_TRIM:-1}
  export HF_DATASET_TRIM_WINDOW_SEC=${HF_DATASET_TRIM_WINDOW_SEC:-10}
  export HF_DATASET_TRIM_STRIDE_SEC=${HF_DATASET_TRIM_STRIDE_SEC:-2}
  export HF_DATASET_TRIM_MIN_START_SEC=${HF_DATASET_TRIM_MIN_START_SEC:-2}
  export HF_DATASET_TRIM_TAIL_MARGIN_SEC=${HF_DATASET_TRIM_TAIL_MARGIN_SEC:-0.5}
  export HF_DATASET_SCOUT_TARGET_FPS=${HF_DATASET_SCOUT_TARGET_FPS:-2}
  export HF_DATASET_MAIN_TARGET_FPS=${HF_DATASET_MAIN_TARGET_FPS:-12}
  export HF_DATASET_HIGH_DETAIL_TARGET_FPS=${HF_DATASET_HIGH_DETAIL_TARGET_FPS:-20}
  export OPTIONAL_RUN_ODM=${OPTIONAL_RUN_ODM:-1}
  export OPTIONAL_RUN_OPENMVS=${OPTIONAL_RUN_OPENMVS:-1}
  export OPTIONAL_RUN_NERFSTUDIO=${OPTIONAL_RUN_NERFSTUDIO:-1}
  export SKIP_RESEARCH_METHODS=${SKIP_RESEARCH_METHODS:-0}
  export FPV_METHOD_POLICY=${FPV_METHOD_POLICY:-mandatory_attempt}
  export RESEARCH_METHOD_MAX_POINTS=${RESEARCH_METHOD_MAX_POINTS:-200000}
  export GPU_MEM_MB GPU_NAME
  echo "GPU quality profile: $FPV_GPU_PROFILE on $GPU_NAME (${GPU_MEM_MB} MiB), CPU threads=$CPU_THREADS"
  echo "Frame tiers: scout=${HF_DATASET_SCOUT_FRAMES}@${HF_DATASET_SCOUT_RESIZE}/${HF_DATASET_SCOUT_TARGET_FPS}fps, main=${HF_DATASET_MAIN_FRAMES}@${HF_DATASET_MAIN_RESIZE}/${HF_DATASET_MAIN_TARGET_FPS}fps, high_detail=${HF_DATASET_HIGH_DETAIL_FRAMES}@${HF_DATASET_HIGH_DETAIL_RESIZE}/${HF_DATASET_HIGH_DETAIL_TARGET_FPS}fps, auto_trim=${HF_DATASET_AUTO_TRIM}"
}

write_fpv_gpu_profile_status() {
  python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "schema_version": "fpv-run-optimizer-status-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "gpu": {
        "name": os.environ.get("GPU_NAME", "unknown"),
        "memory_mib": int(os.environ.get("GPU_MEM_MB") or 0),
        "profile": os.environ.get("FPV_GPU_PROFILE", "unknown"),
    },
    "frame_tiers": {
        "scout": {
            "frames": int(os.environ.get("HF_DATASET_SCOUT_FRAMES", "32")),
            "resize_long_edge": int(os.environ.get("HF_DATASET_SCOUT_RESIZE", "768")),
            "target_fps": float(os.environ.get("HF_DATASET_SCOUT_TARGET_FPS", "2")),
        },
        "main": {
            "frames": int(os.environ.get("HF_DATASET_MAIN_FRAMES", "64")),
            "resize_long_edge": int(os.environ.get("HF_DATASET_MAIN_RESIZE", "1024")),
            "target_fps": float(os.environ.get("HF_DATASET_MAIN_TARGET_FPS", "12")),
        },
        "high_detail": {
            "frames": int(os.environ.get("HF_DATASET_HIGH_DETAIL_FRAMES", "96")),
            "resize_long_edge": int(os.environ.get("HF_DATASET_HIGH_DETAIL_RESIZE", "1024")),
            "target_fps": float(os.environ.get("HF_DATASET_HIGH_DETAIL_TARGET_FPS", "20")),
        },
    },
    "method_limits": {
        "research_method_max_images": int(os.environ.get("RESEARCH_METHOD_MAX_IMAGES", "80")),
        "research_method_max_points": int(os.environ.get("RESEARCH_METHOD_MAX_POINTS", "200000")),
        "dust3r_batch_size": int(os.environ.get("DUST3R_BATCH_SIZE", "1")),
        "mast3r_batch_size": int(os.environ.get("MAST3R_BATCH_SIZE", "1")),
        "optional_showcase_max_clips": int(os.environ.get("OPTIONAL_SHOWCASE_MAX_CLIPS", "6")),
        "r3_prepared_frame_limit": int(os.environ.get("R3_PREPARED_FRAME_LIMIT", "96")),
    },
    "notes": [
        "Manual environment variables override these defaults.",
        "Quality-first defaults prefer dense auto-trimmed frame packs over runtime cost.",
        "Expanded reconstruction methods are mandatory attempts by default; missing dependencies are reported, not hidden.",
        "Higher frame count and resize improve review detail but do not guarantee reconstruction correctness.",
    ],
    "safety_warnings": ["no geolocation", "no meters", "relative VGGT/COLMAP frame", "local-only media"],
}
Path("run_optimizer_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
}

detect_fpv_gpu_quality_profile
"""

def _run_all_script() -> str:
    return """#!/usr/bin/env bash
set +e
""" + _gpu_quality_profile_shell() + """write_fpv_gpu_profile_status
python scripts/05_expanded_runtime_check.py
RUNTIME_STATUS=$?
python scripts/06_hf_dataset_preflight.py
HF_DATASET_STATUS=$?
python scripts/07_hf_dataset_frame_packs.py
HF_FRAME_PACK_STATUS=$?
python scripts/00_env_check.py
ENV_STATUS=$?
METHOD_STATUS=0
if [ "${REQUIRE_HF_DATASET:-0}" = "1" ] && [ "$HF_DATASET_STATUS" -ne 0 ]; then
  ENV_STATUS=$HF_DATASET_STATUS
fi
if [ "$HF_FRAME_PACK_STATUS" -ne 0 ]; then
  ENV_STATUS=$HF_FRAME_PACK_STATUS
fi
if [ "$ENV_STATUS" -eq 0 ]; then
  python run_vggt_job.py
  JOB_STATUS=$?
  python scripts/80_run_expanded_methods.py
  METHOD_STATUS=$?
else
  JOB_STATUS=$ENV_STATUS
fi
python - "$RUNTIME_STATUS" "$HF_DATASET_STATUS" "$HF_FRAME_PACK_STATUS" "$ENV_STATUS" "$JOB_STATUS" "$METHOD_STATUS" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
names = ["runtime_status", "hf_dataset_status", "hf_frame_pack_status", "env_status", "job_status", "method_status"]
values = {name: int(value) for name, value in zip(names, sys.argv[1:])}
nonzero = {name: value for name, value in values.items() if value != 0}
status = "done" if not nonzero else "done_partial"
payload = {
    "schema_version": "cloud-run-status-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "status": status,
    "stage_exit_codes": values,
    "nonzero_stage_exit_codes": nonzero,
    "optimizer": {
        "gpu_name": os.environ.get("GPU_NAME"),
        "gpu_mem_mb": os.environ.get("GPU_MEM_MB"),
        "gpu_profile": os.environ.get("FPV_GPU_PROFILE"),
        "research_method_max_images": os.environ.get("RESEARCH_METHOD_MAX_IMAGES"),
        "main_frames": os.environ.get("HF_DATASET_MAIN_FRAMES"),
        "main_resize": os.environ.get("HF_DATASET_MAIN_RESIZE"),
        "high_detail_frames": os.environ.get("HF_DATASET_HIGH_DETAIL_FRAMES"),
        "high_detail_resize": os.environ.get("HF_DATASET_HIGH_DETAIL_RESIZE"),
        "high_detail_target_fps": os.environ.get("HF_DATASET_HIGH_DETAIL_TARGET_FPS"),
        "auto_trim": os.environ.get("HF_DATASET_AUTO_TRIM"),
        "method_policy": os.environ.get("FPV_METHOD_POLICY"),
        "optional_run_odm": os.environ.get("OPTIONAL_RUN_ODM"),
        "optional_run_openmvs": os.environ.get("OPTIONAL_RUN_OPENMVS"),
        "optional_run_nerfstudio": os.environ.get("OPTIONAL_RUN_NERFSTUDIO"),
        "skip_research_methods": os.environ.get("SKIP_RESEARCH_METHODS"),
        "trim_window_sec": os.environ.get("HF_DATASET_TRIM_WINDOW_SEC"),
    },
    "warning": "A nonzero stage can still produce a useful h100_return.zip; inspect returned reports before rerunning.",
    "safety_warnings": ["no geolocation", "no meters", "relative VGGT/COLMAP frame", "local-only media"],
}
Path("cloud_run_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
python scripts/70_package_return.py
PACK_STATUS=$?
if [ "$PACK_STATUS" -ne 0 ] || [ ! -f h100_return.zip ]; then
  python - <<'PY'
""" + _emergency_return_packager_source() + """PY
  EMERGENCY_PACK_STATUS=$?
fi
python - "$PACK_STATUS" "${EMERGENCY_PACK_STATUS:-0}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
path = Path("cloud_run_status.json")
payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"schema_version": "cloud-run-status-v1", "stage_exit_codes": {}}
payload["updated_at"] = datetime.now(timezone.utc).isoformat()
payload["stage_exit_codes"]["pack_status"] = int(sys.argv[1])
payload["stage_exit_codes"]["emergency_pack_status"] = int(sys.argv[2])
payload["return_zip_exists"] = Path("h100_return.zip").exists()
if payload["stage_exit_codes"]["pack_status"] != 0 and not payload["return_zip_exists"]:
    payload["status"] = "failed_packaging"
elif any(value != 0 for value in payload["stage_exit_codes"].values()):
    payload["status"] = "done_partial"
else:
    payload["status"] = "done"
path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
if [ -f h100_return.zip ]; then
  exit 0
fi
exit "$PACK_STATUS"
"""


def _hf_dataset_preflight_script() -> str:
    return r'''from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "hf_dataset_report.json"


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def write_report(payload: dict) -> None:
    payload.setdefault("schema_version", "hf-dataset-preflight-v1")
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    payload.setdefault(
        "safety_warnings",
        [
            "dataset media and media-derived artifacts are local-only by default",
            "no geolocation",
            "no map projection",
            "no route, approach, launch, target-coordinate, guidance, or next-maneuver inference",
        ],
    )
    REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sanitize_error(text: str, token: str | None) -> str:
    clean = text
    if token:
        clean = clean.replace(token, "<redacted>")
    clean = re.sub(r"hf_[A-Za-z0-9_\-]{20,}", "<redacted>", clean)
    return clean


def finish(payload: dict, *, code: int) -> None:
    write_report(payload)
    raise SystemExit(code)


def main() -> None:
    dataset_id = os.environ.get("HF_DATASET_ID", "").strip()
    require_dataset = env_flag("REQUIRE_HF_DATASET", False)
    download_dataset = env_flag("HF_DATASET_DOWNLOAD", False)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    token_present = bool(token)
    base_payload = {
        "dataset_id": dataset_id or None,
        "required": require_dataset,
        "download_requested": download_dataset,
        "token_present": token_present,
        "status": "unknown",
    }

    if not dataset_id:
        base_payload["status"] = "skipped_no_dataset_id"
        base_payload["next_action"] = (
            "Set HF_DATASET_ID and HF_TOKEN to verify or download a Hugging Face dataset, "
            "or leave unset to use packaged local frame packs."
        )
        finish(base_payload, code=2 if require_dataset else 0)

    if not token_present:
        base_payload["status"] = "blocked_missing_token"
        base_payload["next_action"] = "Set HF_TOKEN on the pod; do not write it into repo files or returned artifacts."
        finish(base_payload, code=2 if require_dataset else 0)

    try:
        from huggingface_hub import HfApi, snapshot_download
    except Exception as exc:
        base_payload["status"] = "blocked_missing_huggingface_hub"
        base_payload["error"] = sanitize_error(str(exc), token)
        base_payload["next_action"] = "Install huggingface_hub in the pinned cloud image before launching the run."
        finish(base_payload, code=2 if require_dataset else 0)

    try:
        api = HfApi(token=token)
        info = api.repo_info(dataset_id, repo_type="dataset")
        siblings = getattr(info, "siblings", None) or []
        base_payload.update(
            {
                "status": "accessible",
                "repo_id": getattr(info, "id", dataset_id),
                "private": bool(getattr(info, "private", False)),
                "sha": getattr(info, "sha", None),
                "siblings_count": len(siblings),
            }
        )
        if download_dataset:
            safe_name = dataset_id.replace("/", "__")
            local_dir = Path(os.environ.get("HF_DATASET_LOCAL_DIR", f"/workspace/hf_datasets/{safe_name}"))
            snapshot_path = Path(
                snapshot_download(
                    repo_id=dataset_id,
                    repo_type="dataset",
                    token=token,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                )
            )
            files = [path for path in snapshot_path.rglob("*") if path.is_file()]
            base_payload.update(
                {
                    "status": "downloaded",
                    "snapshot_path": str(snapshot_path),
                    "file_count": len(files),
                    "total_bytes": sum(path.stat().st_size for path in files),
                }
            )
        finish(base_payload, code=0)
    except Exception as exc:
        base_payload["status"] = "blocked_hf_access_failed"
        base_payload["error"] = sanitize_error(str(exc), token)
        base_payload["next_action"] = "Check HF_DATASET_ID, token access, network, and rate limits before rerunning."
        finish(base_payload, code=2 if require_dataset else 0)


if __name__ == "__main__":
    main()
'''


def _hf_dataset_frame_pack_script() -> str:
    return r'''from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "hf_frame_pack_report.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
DEFAULT_TIERS = [
    ("scout", 32, 768, 2.0),
    ("main", 64, 1024, 8.0),
    ("high_detail", 192, 1280, 20.0),
]
SAFETY_WARNINGS = [
    "dataset media and media-derived artifacts are local-only by default",
    "catalog metadata may contain source locations; keep those fields provenance-only unless explicitly reviewed",
    "offline-only retrospective reconstruction; no live guidance or operational recommendations",
    "no precise geolocation, launch-point, target-coordinate, or map-based approach-corridor output",
]


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_report(payload: dict, *, code: int = 0) -> None:
    payload.setdefault("schema_version", "hf-frame-pack-report-v2")
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    payload.setdefault("safety_warnings", SAFETY_WARNINGS)
    write_json(REPORT, payload)
    raise SystemExit(code)


def load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def manifest_sha(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")).hexdigest()


def sanitize_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("._-")
    return value[:140] or "hf_item"


def tiers() -> list[tuple[str, int, int, float]]:
    selected = os.environ.get("HF_DATASET_FRAME_PACK_TIERS", "scout,main,high_detail")
    wanted = {item.strip() for item in selected.split(",") if item.strip()}
    rows = []
    for name, count, resize, target_fps in DEFAULT_TIERS:
        if name not in wanted:
            continue
        prefix = name.upper()
        rows.append(
            (
                name,
                env_int(f"HF_DATASET_{prefix}_FRAMES", count),
                env_int(f"HF_DATASET_{prefix}_RESIZE", resize),
                env_float(f"HF_DATASET_{prefix}_TARGET_FPS", target_fps),
            )
        )
    return rows or [("scout", env_int("HF_DATASET_SCOUT_FRAMES", 32), env_int("HF_DATASET_SCOUT_RESIZE", 768), env_float("HF_DATASET_SCOUT_TARGET_FPS", 2.0))]


def progress(iterable, *, desc: str, total: int | None = None):
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total)
    except Exception:
        return iterable


def snapshot_root() -> Path | None:
    explicit = os.environ.get("HF_DATASET_LOCAL_DIR")
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    report = load_json(ROOT / "hf_dataset_report.json", {})
    snapshot = report.get("snapshot_path")
    if snapshot:
        path = Path(snapshot)
        if path.exists():
            return path
    embedded = ROOT / "dataset_catalog" / "latest"
    if embedded.exists():
        return embedded
    return None


def image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
        with Image.open(path) as image:
            return int(image.size[0]), int(image.size[1])
    except Exception:
        try:
            import cv2
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                return int(image.shape[1]), int(image.shape[0])
        except Exception:
            pass
    return 1, 1


def image_quality(path: Path) -> dict:
    try:
        import cv2
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            return {
                "blur_score": float(cv2.Laplacian(image, cv2.CV_64F).var()),
                "brightness_mean": float(image.mean()),
                "contrast_std": float(image.std()),
            }
    except Exception:
        pass
    return {"blur_score": 0.0, "brightness_mean": 0.0, "contrast_std": 0.0}


def deterministic_sample(paths: list[Path], count: int) -> list[Path]:
    if not paths:
        return []
    if len(paths) <= count:
        return paths
    if count <= 1:
        return [paths[0]]
    indexes = [round(i * (len(paths) - 1) / (count - 1)) for i in range(count)]
    return [paths[int(index)] for index in indexes]


def discover_image_groups(root: Path) -> list[dict]:
    groups = []
    folders = sorted({path.parent for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES})
    for folder in folders:
        images = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if len(images) < env_int("HF_DATASET_MIN_IMAGES_PER_GROUP", 4):
            continue
        groups.append({"kind": "image_group", "source": folder, "images": images, "video_id": sanitize_id(folder.name)})
    return groups


def discover_videos(root: Path) -> list[dict]:
    return [{"kind": "video", "source": path, "video_id": sanitize_id(path.stem)} for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES]


def _catalog_row_video_url(row: dict) -> str | None:
    for key in ["video_url", "download_url", "media_url", "url", "source_url"]:
        value = str(row.get(key) or "").strip()
        if value.startswith(("http://", "https://")) and any(suffix in value.lower() for suffix in VIDEO_SUFFIXES):
            return value
    return None


def _catalog_row_video_id(row: dict, url: str) -> str:
    for key in ["video_id", "target_stem", "current_stem", "id", "name"]:
        value = str(row.get(key) or "").strip()
        if value:
            return sanitize_id(Path(value).stem)
    parsed = urllib.parse.urlparse(url)
    return sanitize_id(Path(urllib.parse.unquote(parsed.path)).stem)


def _read_json_or_jsonl(path: Path) -> list[dict]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return [row for row in rows if isinstance(row, dict)]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ["rows", "items", "videos", "catalog"]:
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_parquet(path: Path) -> list[dict]:
    try:
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient="records")
    except Exception:
        return []


def discover_catalog_videos(root: Path) -> list[dict]:
    candidates = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name not in {"catalog.json", "catalog.jsonl", "catalog.ndjson", "catalog.csv", "catalog.parquet"}:
            continue
        rows = []
        if path.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
            rows = _read_json_or_jsonl(path)
        elif path.suffix.lower() == ".csv":
            rows = _read_csv(path)
        elif path.suffix.lower() == ".parquet":
            rows = _read_parquet(path)
        for row in rows:
            url = _catalog_row_video_url(row)
            if not url:
                continue
            video_id = _catalog_row_video_id(row, url)
            candidates.append({"kind": "catalog_video", "source": path, "video_id": video_id, "url": url, "metadata": row})
    deduped = {}
    for item in candidates:
        deduped.setdefault(item["video_id"], item)
    return list(deduped.values())


def _download_with_progress(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "fpv-vggt-runpod/1.0"})
    started = time.time()
    with urllib.request.urlopen(request, timeout=env_int("HF_DATASET_MEDIA_TIMEOUT_SEC", 120)) as response:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header and total_header.isdigit() else None
        chunk_size = 1024 * 1024
        try:
            from tqdm import tqdm
            bar = tqdm(total=total, unit="B", unit_scale=True, desc=target.name)
        except Exception:
            bar = None
        downloaded = 0
        with tmp.open("wb") as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if bar is not None:
                    bar.update(len(chunk))
                elif downloaded and downloaded % (50 * chunk_size) == 0:
                    elapsed = max(time.time() - started, 0.001)
                    print(f"download {target.name}: {downloaded / 1024 / 1024:.1f} MiB at {downloaded / elapsed / 1024 / 1024:.1f} MiB/s", flush=True)
        if bar is not None:
            bar.close()
    tmp.replace(target)


def ensure_catalog_video(item: dict) -> Path:
    parsed = urllib.parse.urlparse(item["url"])
    suffix = Path(urllib.parse.unquote(parsed.path)).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        suffix = ".mp4"
    media_dir = ROOT / os.environ.get("HF_DATASET_MEDIA_CACHE_DIR", "hf_media_cache/videos")
    target = media_dir / f"{item['video_id']}{suffix}"
    if target.exists() and target.stat().st_size > 0:
        return target
    _download_with_progress(item["url"], target)
    return target


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _window_score(metrics: dict) -> float:
    blur_score = _clip01(math.log1p(metrics["blur_mean"]) / math.log1p(700.0))
    contrast_score = _clip01(metrics["contrast_mean"] / 60.0)
    brightness_score = _clip01(1.0 - abs(metrics["brightness_mean"] - 118.0) / 118.0)
    motion = metrics["motion_mean"]
    motion_score = _clip01(1.0 - abs(motion - 12.0) / 42.0)
    continuity = _clip01(1.0 - metrics["motion_std"] / max(motion * 2.5, 1.0))
    cut_penalty = min(0.35, metrics["cut_like_pairs"] * 0.12)
    return round(_clip01(0.26 * blur_score + 0.20 * contrast_score + 0.14 * brightness_score + 0.22 * motion_score + 0.18 * continuity - cut_penalty), 6)


def _read_frame(capture, frame_index: int):
    import cv2
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"could not read frame {frame_index}")
    return frame


def _resize_frame(frame, long_edge: int):
    import cv2
    height, width = frame.shape[:2]
    scale = long_edge / max(width, height)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


def _video_window_metrics(capture, start_frame: int, end_frame: int, samples: int) -> dict:
    import cv2
    import numpy as np
    total = end_frame - start_frame + 1
    if total <= 0:
        return {"blur_mean": 0.0, "brightness_mean": 0.0, "contrast_mean": 0.0, "motion_mean": 0.0, "motion_std": 0.0, "cut_like_pairs": 0}
    indexes = [start_frame + round(i * (total - 1) / max(samples - 1, 1)) for i in range(min(samples, total))]
    grays = []
    blur = []
    brightness = []
    contrast = []
    for index in indexes:
        frame = _resize_frame(_read_frame(capture, index), 360)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grays.append(gray)
        blur.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        brightness.append(float(gray.mean()))
        contrast.append(float(gray.std()))
    diffs = [float(np.mean(cv2.absdiff(a, b))) for a, b in zip(grays, grays[1:])]
    motion_mean = float(np.mean(diffs)) if diffs else 0.0
    motion_std = float(np.std(diffs)) if diffs else 0.0
    cut_like_pairs = sum(1 for value in diffs if value > max(38.0, motion_mean + 2.5 * motion_std))
    return {
        "blur_mean": float(np.mean(blur)) if blur else 0.0,
        "brightness_mean": float(np.mean(brightness)) if brightness else 0.0,
        "contrast_mean": float(np.mean(contrast)) if contrast else 0.0,
        "motion_mean": motion_mean,
        "motion_std": motion_std,
        "cut_like_pairs": int(cut_like_pairs),
    }


def select_video_trim(video: Path) -> dict:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for video trimming") from exc
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 1.0) or 1.0
        if total <= 0:
            raise RuntimeError(f"video has no readable frames: {video}")
        duration = total / fps
        if not env_flag("HF_DATASET_AUTO_TRIM", True):
            return {"status": "full_video", "start_frame": 0, "end_frame": total - 1, "start_sec": 0.0, "end_sec": duration, "duration_sec": duration, "score": None}
        window_sec = max(1.0, env_float("HF_DATASET_TRIM_WINDOW_SEC", 10.0))
        stride_sec = max(0.5, env_float("HF_DATASET_TRIM_STRIDE_SEC", 2.0))
        min_start_sec = max(0.0, env_float("HF_DATASET_TRIM_MIN_START_SEC", 2.0))
        tail_margin_sec = max(0.0, env_float("HF_DATASET_TRIM_TAIL_MARGIN_SEC", 0.5))
        eval_samples = max(3, env_int("HF_DATASET_TRIM_EVAL_SAMPLES", 10))
        start_min = min(total - 1, int(round(min_start_sec * fps)))
        end_max = max(start_min, min(total - 1, int(round((duration - tail_margin_sec) * fps)) - 1))
        window_frames = min(max(2, int(round(window_sec * fps))), end_max - start_min + 1)
        stride_frames = max(1, int(round(stride_sec * fps)))
        starts = list(range(start_min, end_max - window_frames + 2, stride_frames))
        if not starts:
            starts = [start_min]
        final_start = max(start_min, end_max - window_frames + 1)
        if starts[-1] != final_start:
            starts.append(final_start)
        best = None
        for start_frame in starts:
            end_frame = min(end_max, start_frame + window_frames - 1)
            metrics = _video_window_metrics(capture, start_frame, end_frame, eval_samples)
            score = _window_score(metrics)
            candidate = {
                "status": "auto_trimmed",
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "start_sec": float(start_frame / fps),
                "end_sec": float((end_frame + 1) / fps),
                "duration_sec": float((end_frame - start_frame + 1) / fps),
                "score": score,
                **metrics,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
        return best or {"status": "full_video_fallback", "start_frame": 0, "end_frame": total - 1, "start_sec": 0.0, "end_sec": duration, "duration_sec": duration, "score": None}
    finally:
        capture.release()


def extract_video_frames(video: Path, destination: Path, count: int, target_fps: float = 0.0) -> tuple[list[dict], float, dict]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required for HF video frame extraction") from exc
    trim = select_video_trim(video)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 1.0) or 1.0
    if total <= 0:
        raise RuntimeError(f"video has no readable frames: {video}")
    start_frame = max(0, min(int(trim["start_frame"]), total - 1))
    end_frame = max(start_frame, min(int(trim["end_frame"]), total - 1))
    trim_total = end_frame - start_frame + 1
    if target_fps > 0:
        dynamic_count = max(2, int(round((trim_total / fps) * target_fps)))
        count = min(count, dynamic_count)
    indexes = [start_frame + round(i * (trim_total - 1) / max(count - 1, 1)) for i in range(min(count, trim_total))]
    rows = []
    destination.mkdir(parents=True, exist_ok=True)
    for ordinal, frame_index in enumerate(progress(indexes, desc=f"frames {video.stem}", total=len(indexes))):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        out = destination / f"{ordinal:04d}_frame_{int(frame_index):06d}.jpg"
        cv2.imwrite(str(out), frame)
        rows.append({"path": out, "frame_index": int(frame_index), "timestamp_sec": float(frame_index) / fps, "width": int(frame.shape[1]), "height": int(frame.shape[0])})
    capture.release()
    return rows, fps, trim


def copy_image_frames(images: list[Path], destination: Path, count: int) -> tuple[list[dict], float, dict | None]:
    sampled = deterministic_sample(images, count)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for ordinal, source in enumerate(sampled):
        target = destination / f"{ordinal:04d}_frame_{ordinal:06d}{source.suffix.lower() or '.jpg'}"
        shutil.copy2(source, target)
        width, height = image_size(target)
        rows.append({"path": target, "frame_index": ordinal, "timestamp_sec": float(ordinal), "width": int(width), "height": int(height)})
    return rows, 1.0, None


def write_frame_manifest(*, video_id: str, segment_id: str, source: Path, frame_rows: list[dict], resized_long_edge: int, target_dir: Path, source_fps: float, trim: dict | None = None) -> str:
    manifest_rows = []
    for row in frame_rows:
        manifest_rows.append(
            {
                "video_id": video_id,
                "segment_id": segment_id,
                "frame_index": int(row["frame_index"]),
                "timestamp_sec": float(row["timestamp_sec"]),
                "path": row["path"].relative_to(target_dir).as_posix(),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "resized_long_edge": resized_long_edge,
                "quality": image_quality(row["path"]),
            }
        )
    payload = {"video_id": video_id, "segment_id": segment_id, "source_video": str(source), "frame_count": len(manifest_rows), "source_fps": source_fps, "source_trim": trim, "frames": manifest_rows}
    manifest_path = target_dir / "frames.json"
    write_json(manifest_path, payload)
    return manifest_sha(payload)


def reset_generated_inputs() -> None:
    for name in ["frame_packs", "bundles", "predictions"]:
        target = ROOT / name
        if target.exists():
            backup = ROOT / f"{name}.packaged_backup"
            if backup.exists():
                shutil.rmtree(backup)
            target.rename(backup)


def main() -> None:
    if not env_flag("HF_DATASET_BUILD_FRAME_PACKS", False):
        write_report({"status": "skipped_disabled", "reason": "set HF_DATASET_BUILD_FRAME_PACKS=1 to build job frame packs from a downloaded/local HF snapshot"})
    root = snapshot_root()
    if root is None:
        write_report({"status": "blocked_missing_snapshot", "reason": "HF_DATASET_LOCAL_DIR or hf_dataset_report.json snapshot_path does not exist"}, code=2)
    catalog_items = discover_catalog_videos(root) if env_flag("HF_DATASET_FETCH_CATALOG_MEDIA", True) else []
    local_items = discover_image_groups(root) + discover_videos(root)
    catalog_ids = {item["video_id"] for item in catalog_items}
    items = catalog_items + [item for item in local_items if item["video_id"] not in catalog_ids]
    max_items = env_int("HF_DATASET_MAX_ITEMS", 0)
    if max_items > 0:
        items = items[:max_items]
    if not items:
        write_report({"status": "blocked_no_supported_media", "snapshot_path": str(root), "supported_suffixes": sorted(IMAGE_SUFFIXES | VIDEO_SUFFIXES), "catalog_file_names": ["catalog.jsonl", "catalog.json", "catalog.csv", "catalog.parquet"]}, code=2)
    reset_generated_inputs()
    clip_rows = []
    built_items = []
    segment_id = os.environ.get("HF_DATASET_SEGMENT_ID", "segment-001")
    for item in progress(items, desc="frame-pack items", total=len(items)):
        item_report = {"kind": item["kind"], "source": str(item["source"]), "video_id": item["video_id"], "tiers": []}
        if item["kind"] == "catalog_video":
            item_report["source_url"] = item.get("url")
            try:
                item["local_video"] = ensure_catalog_video(item)
                item_report["local_video"] = str(item["local_video"])
            except Exception as exc:
                item_report["status"] = "failed_media_download"
                item_report["error"] = str(exc)
                built_items.append(item_report)
                continue
        for tier, count, resized_long_edge, target_fps in tiers():
            pack_id = f"{tier}__hf__{item['video_id']}__{segment_id}"
            target_dir = ROOT / "frame_packs" / tier / pack_id
            frames_dir = target_dir / "frames"
            try:
                if item["kind"] == "image_group":
                    frame_rows, source_fps, trim = copy_image_frames(item["images"], frames_dir, count)
                    source_path = item["source"]
                elif item["kind"] == "catalog_video":
                    source_path = item["local_video"]
                    frame_rows, source_fps, trim = extract_video_frames(source_path, frames_dir, count, target_fps)
                else:
                    source_path = item["source"]
                    frame_rows, source_fps, trim = extract_video_frames(source_path, frames_dir, count, target_fps)
            except Exception as exc:
                item_report["tiers"].append({"tier": tier, "status": "failed_soft", "error": str(exc)})
                continue
            if not frame_rows:
                item_report["tiers"].append({"tier": tier, "status": "failed_no_frames"})
                continue
            manifest_sha256 = write_frame_manifest(video_id=item["video_id"], segment_id=segment_id, source=source_path, frame_rows=frame_rows, resized_long_edge=resized_long_edge, target_dir=target_dir, source_fps=source_fps, trim=trim)
            clip_rows.append({"tier": tier, "video_id": item["video_id"], "segment_id": segment_id, "clip_id": pack_id, "frame_manifest": (target_dir / "frames.json").relative_to(ROOT).as_posix(), "frame_manifest_sha256": manifest_sha256, "frame_count": len(frame_rows), "requested_frame_count": count, "target_fps": target_fps, "trim": trim, "resized_long_edge": resized_long_edge, "bundle_output": f"bundles/{tier}/{item['video_id']}/{segment_id}", "predictions_output": f"predictions/{tier}/{item['video_id']}/{segment_id}/predictions.npz"})
            item_report["tiers"].append({"tier": tier, "status": "done", "frame_count": len(frame_rows), "requested_frame_count": count, "target_fps": target_fps, "trim": trim})
        built_items.append(item_report)
    if not clip_rows:
        write_report({"status": "blocked_no_frame_packs_built", "snapshot_path": str(root), "items": built_items}, code=2)
    manifest = load_json(ROOT / "job_manifest.json", {})
    manifest["clips"] = clip_rows
    manifest.setdefault("warnings", []).extend(SAFETY_WARNINGS)
    source_payload = {"snapshot_path": str(root), "built_at": datetime.now(timezone.utc).isoformat(), "item_count": len(built_items), "clip_count": len(clip_rows), "catalog_item_count": len(catalog_items), "local_item_count": len(local_items), "auto_trim": env_flag("HF_DATASET_AUTO_TRIM", True), "note": "Replaced packaged frame_packs/job_manifest for this cloud run only; media-derived artifacts remain local-only."}
    manifest["hf_dataset_frame_pack_source"] = source_payload
    write_json(ROOT / "job_manifest.json", manifest)
    stage_plan_path = ROOT / "method_stage_plan.json"
    stage_plan = load_json(stage_plan_path, {})
    if stage_plan:
        stage_plan["clip_count"] = len(clip_rows)
        stage_plan["hf_dataset_frame_pack_source"] = source_payload
        write_json(stage_plan_path, stage_plan)
    write_report({"status": "done", "snapshot_path": str(root), "item_count": len(built_items), "clip_count": len(clip_rows), "catalog_item_count": len(catalog_items), "local_item_count": len(local_items), "items": built_items})


if __name__ == "__main__":
    main()
'''


def _expanded_runtime_check_script() -> str:
    return r"""from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Optional research repos are often cloned beside the job rather than installed.
def _add_optional_repo_paths() -> None:
    candidates = [
        "/workspace/mast3r",
        "/workspace/mast3r/dust3r",
        "/workspace/dust3r",
        "/workspace/r3",
        "/workspace/lingbot",
        "/workspace/lingbot-map",
    ]
    active = []
    for candidate in reversed(candidates):
        if os.path.isdir(candidate):
            active.insert(0, candidate)
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    merged = []
    for part in [*active, *parts]:
        if part and part not in merged:
            merged.append(part)
    if merged:
        os.environ["PYTHONPATH"] = os.pathsep.join(merged)


_add_optional_repo_paths()

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "expanded_runtime_check.json"

PYTHON_MODULES = [
    "torch",
    "vggt",
    "pycolmap",
    "lightglue",
    "trimesh",
    "numpy",
    "cv2",
    "PIL",
    "evo",
    "nerfstudio",
    "gsplat",
    "mast3r",
    "dust3r",
    "vggetr",
    "vg2gt",
    "vg_2gt",
]
EXECUTABLES = ["colmap", "opensfm", "odm", "odm_run", "InterfaceCOLMAP", "CreateStructure", "DensifyPointCloud", "ReconstructMesh", "RefineMesh", "TextureMesh", "TransformScene", "Viewer", "MvgMvsPipeline.py", "ns-process-data", "ns-train", "ns-viewer"]


def module_available(name: str) -> dict:
    spec = importlib.util.find_spec(name)
    return {"available": spec is not None, "origin": getattr(spec, "origin", None) if spec else None}


def executable_available(name: str) -> dict:
    path = shutil.which(name)
    return {"available": path is not None, "path": path}


def vggt_demo_candidates() -> list[str]:
    candidates = []
    if os.environ.get("VGGT_REPO_DIR"):
        candidates.append(str(Path(os.environ["VGGT_REPO_DIR"]) / "demo_colmap.py"))
    candidates.extend([
        str(ROOT / "vggt" / "demo_colmap.py"),
        "/workspace/vggt/demo_colmap.py",
        "/workspace/VGGT/demo_colmap.py",
    ])
    return candidates


def first_existing(paths: list[str]) -> str | None:
    for path in paths:
        if Path(path).exists():
            return path
    return None


def nvidia_smi() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        completed = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=False)
    except Exception as exc:
        return f"unavailable: {exc}"
    return completed.stdout.strip() or completed.stderr.strip() or f"exit {completed.returncode}"


def python_import_probe(label: str, code: str, timeout: int = 30) -> dict:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return {
            "label": label,
            "available": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "label": label,
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def research_method_runner_probe() -> dict:
    dust3r_probe = python_import_probe(
        "dust3r_builtin_python_api",
        "import dust3r.inference; import dust3r.model; import dust3r.cloud_opt; import dust3r.image_pairs; import dust3r.utils.image; print('ok')",
    )
    mast3r_probe = python_import_probe(
        "mast3r_builtin_python_api",
        "import mast3r.utils.path_to_dust3r; import mast3r.model; import dust3r.inference; import dust3r.cloud_opt; import dust3r.image_pairs; import dust3r.utils.image; print('ok')",
    )
    templates = {
        "MAST3R_COMMAND_TEMPLATE": bool(os.environ.get("MAST3R_COMMAND_TEMPLATE")),
        "DUST3R_COMMAND_TEMPLATE": bool(os.environ.get("DUST3R_COMMAND_TEMPLATE")),
        "VGGETR_COMMAND_TEMPLATE": bool(os.environ.get("VGGETR_COMMAND_TEMPLATE")),
        "VG2GT_COMMAND_TEMPLATE": bool(os.environ.get("VG2GT_COMMAND_TEMPLATE")),
        "RESEARCH_METHOD_COMMAND_TEMPLATE": bool(os.environ.get("RESEARCH_METHOD_COMMAND_TEMPLATE")),
        "RESEARCH_ADAPTER_COMMAND_TEMPLATE": bool(os.environ.get("RESEARCH_ADAPTER_COMMAND_TEMPLATE")),
    }
    runners = {
        "dust3r_builtin_python_api": dust3r_probe,
        "mast3r_builtin_python_api": mast3r_probe,
        "vggetr_template_ready": templates["VGGETR_COMMAND_TEMPLATE"] or templates["VG2GT_COMMAND_TEMPLATE"] or templates["RESEARCH_METHOD_COMMAND_TEMPLATE"] or templates["RESEARCH_ADAPTER_COMMAND_TEMPLATE"],
        "command_templates": templates,
        "notes": [
            "DUSt3R and MASt3R probes import runner APIs only; they do not download checkpoints or run inference.",
            "VGGeTR/VG2GT still requires an explicit command template until the exact public runner/API is confirmed.",
        ],
    }
    runners["any_ready"] = (
        dust3r_probe["available"]
        or mast3r_probe["available"]
        or runners["vggetr_template_ready"]
        or templates["MAST3R_COMMAND_TEMPLATE"]
        or templates["DUST3R_COMMAND_TEMPLATE"]
        or templates["VG2GT_COMMAND_TEMPLATE"]
        or templates["RESEARCH_METHOD_COMMAND_TEMPLATE"]
        or templates["RESEARCH_ADAPTER_COMMAND_TEMPLATE"]
    )
    return runners


def main() -> None:
    modules = {name: module_available(name) for name in PYTHON_MODULES}
    executables = {name: executable_available(name) for name in EXECUTABLES}
    demo_candidates = vggt_demo_candidates()
    demo_colmap = first_existing(demo_candidates)
    research_runners = research_method_runner_probe()
    checks = {
        "vggt_feedforward_ready": modules["torch"]["available"] and modules["vggt"]["available"],
        "vggt_colmap_ba_ready": demo_colmap is not None and modules["lightglue"]["available"] and (modules["pycolmap"]["available"] or executables["colmap"]["available"]),
        "classical_colmap_ready": executables["colmap"]["available"],
        "trajectory_evo_ready": modules["evo"]["available"],
        "odm_ready": executables["odm"]["available"] or executables["odm_run"]["available"],
        "openmvs_ready": executables["DensifyPointCloud"]["available"] and executables["ReconstructMesh"]["available"] and ((executables["InterfaceCOLMAP"]["available"] and executables["colmap"]["available"]) or executables["CreateStructure"]["available"]),
        "nerfstudio_gsplat_ready": modules["nerfstudio"]["available"] or executables["ns-train"]["available"] or modules["gsplat"]["available"],
        "research_methods_ready": bool(research_runners["any_ready"]),
        "dust3r_builtin_runner_ready": bool(research_runners["dust3r_builtin_python_api"]["available"]),
        "mast3r_builtin_runner_ready": bool(research_runners["mast3r_builtin_python_api"]["available"]),
        "vggetr_template_ready": bool(research_runners["vggetr_template_ready"]),
    }
    vggt_inference_required_checks = ["vggt_feedforward_ready"]
    full_stack_required_checks = ["vggt_feedforward_ready", "vggt_colmap_ba_ready", "classical_colmap_ready"]
    primary_required_checks = full_stack_required_checks
    optional_checks = [
        "trajectory_evo_ready",
        "odm_ready",
        "openmvs_ready",
        "nerfstudio_gsplat_ready",
        "research_methods_ready",
        "dust3r_builtin_runner_ready",
        "mast3r_builtin_runner_ready",
        "vggetr_template_ready",
    ]
    missing_vggt_inference_checks = [name for name in vggt_inference_required_checks if not checks[name]]
    missing_full_stack_checks = [name for name in full_stack_required_checks if not checks[name]]
    missing_primary_checks = missing_full_stack_checks
    missing_optional_checks = [name for name in optional_checks if not checks[name]]
    if not missing_full_stack_checks:
        readiness_status = "ready_full_stack"
    elif not missing_vggt_inference_checks:
        readiness_status = "ready_4090_pilot"
    else:
        readiness_status = "blocked_primary_runtime"
    next_actions = []
    if "vggt_feedforward_ready" in missing_vggt_inference_checks:
        next_actions.append("Use the pinned VGGT image or install/check torch and vggt before running run_all.sh.")
    if "vggt_colmap_ba_ready" in missing_full_stack_checks:
        next_actions.append("For full-stack VGGT+COLMAP BA, set VGGT_REPO_DIR to a VGGT checkout containing demo_colmap.py and ensure LightGlue plus pycolmap or colmap are installed. A 4090 VGGT feed-forward pilot can continue without this.")
    if "classical_colmap_ready" in missing_full_stack_checks:
        next_actions.append("For the classical COLMAP baseline, install COLMAP or use the prepared COLMAP-enabled image. A 4090 VGGT feed-forward pilot can continue without this.")
    if "openmvs_ready" in missing_optional_checks:
        next_actions.append("For the recommended ODM replacement, install OpenMVS binaries with InterfaceCOLMAP, DensifyPointCloud, and ReconstructMesh; otherwise it will be reported as skipped_missing_dependency.")
    if missing_optional_checks:
        next_actions.append("Optional missing methods will be recorded as skipped_missing_dependency unless you install their runtimes or set command templates.")
    if readiness_status == "ready_4090_pilot":
        next_actions.append("Runtime is ready for a bounded 4090 VGGT inference pilot; install missing full-stack checks before claiming the full stack ran.")
    if not next_actions:
        next_actions.append("Runtime is ready for the full configured run; run bash run_all.sh.")
    report = {
        "schema_version": "expanded-runtime-check-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "done",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "nvidia_smi": nvidia_smi(),
        "environment": {
            "VGGT_REPO_DIR": os.environ.get("VGGT_REPO_DIR"),
            "VGGT_COLMAP_EXTRA_ARGS": os.environ.get("VGGT_COLMAP_EXTRA_ARGS"),
        },
        "modules": modules,
        "executables": executables,
        "research_method_runners": research_runners,
        "vggt_demo_colmap": {
            "selected": demo_colmap,
            "candidates": demo_candidates,
        },
        "checks": checks,
        "vggt_inference_required_checks": vggt_inference_required_checks,
        "full_stack_required_checks": full_stack_required_checks,
        "primary_required_checks": primary_required_checks,
        "optional_checks": optional_checks,
        "missing_vggt_inference_checks": missing_vggt_inference_checks,
        "missing_full_stack_checks": missing_full_stack_checks,
        "missing_primary_checks": missing_primary_checks,
        "missing_optional_checks": missing_optional_checks,
        "readiness_status": readiness_status,
        "next_actions": next_actions,
        "warnings": [
            "no geolocation",
            "no meters or real-world speed/standoff/dive-angle claims",
            "runtime availability only; missing optional methods should return skipped_missing_dependency",
        ],
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "readiness_status": readiness_status,
                "missing_vggt_inference_checks": missing_vggt_inference_checks,
                "missing_full_stack_checks": missing_full_stack_checks,
                "missing_primary_checks": missing_primary_checks,
                "missing_optional_checks": missing_optional_checks,
                "next_actions": next_actions,
                "checks": checks,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
"""


def _env_check_script() -> str:
    return r'''from __future__ import annotations

import importlib.util
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    report = {
        "schema_version": "h100-env-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "status": "blocked_environment",
        "checks": {},
        "warnings": [
            "no geolocation",
            "no meters",
            "relative VGGT frame",
            "pinned image expected; live installs are not part of production mode",
        ],
    }
    try:
        import torch

        report["checks"]["torch"] = getattr(torch, "__version__", "unknown")
        report["checks"]["cuda_available"] = bool(torch.cuda.is_available())
        report["checks"]["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        report["checks"]["torch_error"] = str(exc)
    report["checks"]["vggt_installed"] = importlib.util.find_spec("vggt") is not None
    if report["checks"].get("cuda_available") and report["checks"].get("vggt_installed"):
        report["status"] = "done"
    (ROOT / "environment.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if report["status"] != "done":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
'''


def _stage_placeholder(label: str) -> str:
    return f'''from __future__ import annotations

print("Stage placeholder: {label}. The production run is orchestrated by run_vggt_job.py and 70_package_return.py.")
'''


def _expanded_methods_script() -> str:
    return r"""from __future__ import annotations

import csv
import importlib.util

import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# Optional research repos are often cloned beside the job rather than installed.
def _add_optional_repo_paths() -> None:
    candidates = [
        "/workspace/mast3r",
        "/workspace/mast3r/dust3r",
        "/workspace/dust3r",
        "/workspace/r3",
        "/workspace/lingbot",
        "/workspace/lingbot-map",
    ]
    active = []
    for candidate in reversed(candidates):
        if os.path.isdir(candidate):
            active.insert(0, candidate)
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    merged = []
    for part in [*active, *parts]:
        if part and part not in merged:
            merged.append(part)
    if merged:
        os.environ["PYTHONPATH"] = os.pathsep.join(merged)


_add_optional_repo_paths()

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_LOG = ROOT / "cloud_run.log"
REPORT_PATH = ROOT / "method_stage_report.json"
TUM_ROOT = ROOT / "trajectories"
EVO_ROOT = ROOT / "evo_reports"
FAILURE_ROOT = ROOT / "failure_packages"
VGGT_COLMAP_ROOT = ROOT / "method_runs" / "vggt_colmap_ba_windowed"
CLASSICAL_COLMAP_ROOT = ROOT / "method_runs" / "colmap_classical_sift_sequential"
POINTS_PLY_ROOT = ROOT / "points_ply"
CONVERTED_COLMAP_ROOT = ROOT / "converted_colmap_bundles"
ODM_ROOT = ROOT / "odm_artifacts"
ODM_REPORTS_ROOT = ROOT / "odm_project_reports"
ODM_LOGS_ROOT = ROOT / "odm_logs"
OPENMVS_ROOT = ROOT / "openmvs_artifacts"
OPENMVS_REPORTS_ROOT = ROOT / "openmvs_project_reports"
OPENMVS_LOGS_ROOT = ROOT / "openmvs_logs"
NERFSTUDIO_ROOT = ROOT / "nerfstudio_gsplat_showcase"
NERFSTUDIO_PROJECTS_ROOT = ROOT / "nerfstudio_projects"
GSPLAT_EXPORTS_ROOT = ROOT / "gsplat_exports"
SHOWCASE_RENDERS_ROOT = ROOT / "showcase_renders"
RELATIVE_DEPTH_ROOT = ROOT / "relative_depth_overlays"
RESEARCH_METHODS_ROOT = ROOT / "research_methods"
RESEARCH_METHOD_OUTPUTS_ROOT = ROOT / "research_methods_outputs"
SAFETY_WARNINGS = [
    "offline historical-media reconstruction QA only",
    "no geolocation",
    "no meters or true speed/standoff/dive-angle claims",
    "relative method-consistency diagnostics only",
]


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"{timestamp} {message}"
    print(line, flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def python_module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False

def zip_dir(source: Path, target: Path) -> str | None:
    if not source.exists() or not any(path.is_file() for path in source.rglob("*")):
        return None
    if target.exists():
        target.unlink()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(source.parent))
    return target.name


COLMAP_MODEL_TRIPLETS = (
    ("cameras.bin", "images.bin", "points3D.bin"),
    ("cameras.txt", "images.txt", "points3D.txt"),
)


def is_colmap_sparse_model(path: Path) -> bool:
    return any(all((path / marker).is_file() for marker in triplet) for triplet in COLMAP_MODEL_TRIPLETS)


def zip_colmap_sparse_models(source: Path, target: Path) -> str | None:
    if not source.exists():
        return None
    models = []
    for scene_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        model = find_colmap_sparse_model(scene_dir)
        if model is not None:
            models.append(model)
    if not models:
        return None
    if target.exists():
        target.unlink()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for model in models:
            for path in sorted(model.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=path.relative_to(source.parent))
    return target.name




def collect_points_ply_artifacts() -> dict:
    if POINTS_PLY_ROOT.exists():
        shutil.rmtree(POINTS_PLY_ROOT)
    POINTS_PLY_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in sorted(VGGT_COLMAP_ROOT.rglob("*.ply")) if VGGT_COLMAP_ROOT.exists() else []:
        if not source.is_file():
            continue
        relative = source.relative_to(VGGT_COLMAP_ROOT)
        destination = POINTS_PLY_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        rows.append(
            {
                "source": source.relative_to(ROOT).as_posix(),
                "artifact": destination.relative_to(ROOT).as_posix(),
                "size_bytes": source.stat().st_size,
            }
        )
    if not rows:
        (POINTS_PLY_ROOT / "README.md").write_text(
            "# points_ply\n\nNo .ply point-cloud exports were found in the VGGT COLMAP BA output. "
            "This is an artifact-collection report, not a reconstruction failure by itself.\n",
            encoding="utf-8",
        )
    report = {
        "method_id": "vggt_colmap_ba_windowed",
        "artifact": "points_ply.zip",
        "status": "done" if rows else "not_found",
        "warning": "relative point-cloud artifact only; no meters, no geolocation",
        "files": rows,
    }
    write_json(ROOT / "points_ply_report.json", report)
    return report


def write_colmap_conversion_manifest(vggt_stage: dict) -> dict:
    if CONVERTED_COLMAP_ROOT.exists():
        shutil.rmtree(CONVERTED_COLMAP_ROOT)
    CONVERTED_COLMAP_ROOT.mkdir(parents=True, exist_ok=True)
    scene_rows = []
    for row in vggt_stage.get("clips") or []:
        scene_rel = row.get("scene_dir")
        scene_dir = ROOT / scene_rel if scene_rel else None
        sparse_paths = []
        if scene_dir is not None and scene_dir.exists():
            for candidate in ["sparse", "sparse_txt", "sparse_bin"]:
                path = scene_dir / candidate
                if path.exists():
                    sparse_paths.append(path.relative_to(ROOT).as_posix())
        scene_rows.append(
            {
                "clip_id": row.get("clip_id"),
                "source_status": row.get("status"),
                "scene_dir": scene_rel,
                "sparse_paths": sparse_paths,
                "conversion_status": "pending_adapter",
                "reason": "COLMAP sparse output is preserved; conversion to repo-owned VGGT bundle is not implemented in the cloud runner yet.",
            }
        )
    payload = {
        "schema_version": "colmap-conversion-manifest-v1",
        "method_id": "vggt_colmap_ba_windowed",
        "artifact": "converted_colmap_bundles.zip",
        "status": "pending_adapter",
        "warning": "No physical scale, geolocation, route, or guidance claims. Sparse outputs remain relative diagnostics.",
        "scenes": scene_rows,
    }
    write_json(CONVERTED_COLMAP_ROOT / "conversion_manifest.json", payload)
    (CONVERTED_COLMAP_ROOT / "README.md").write_text(
        "# Converted COLMAP Bundles\n\n"
        "This archive preserves a conversion manifest for VGGT COLMAP BA outputs. "
        "The actual COLMAP-to-repo-bundle adapter is pending; use `colmap_sparse.zip` "
        "as the authoritative sparse reconstruction artifact for now.\n",
        encoding="utf-8",
    )
    return payload

def run_logged(command: list[str], *, cwd: Path, log_path: Path, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("+ " + " ".join(command) + "\n")
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env,
        )
        handle.write(f"exit_code={completed.returncode}\n")
    return int(completed.returncode)


def colmap_headless_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    return env


def frame_paths(frame_manifest_path: Path) -> list[Path]:
    manifest = load_json(frame_manifest_path, {"frames": []})
    paths = []
    for frame in manifest.get("frames", []):
        path = Path(frame["path"])
        if not path.is_absolute():
            path = frame_manifest_path.parent / path
        paths.append(path)
    return paths


def prepare_scene_images(scene_dir: Path, frame_manifest_path: Path) -> int:
    images = scene_dir / "images"
    if images.exists():
        shutil.rmtree(images)
    images.mkdir(parents=True, exist_ok=True)
    count = 0
    for index, source in enumerate(frame_paths(frame_manifest_path)):
        suffix = source.suffix or ".jpg"
        target = images / f"{index:04d}{suffix}"
        shutil.copyfile(source, target)
        count += 1
    return count


def find_vggt_demo_colmap() -> Path | None:
    candidates = []
    if os.environ.get("VGGT_REPO_DIR"):
        candidates.append(Path(os.environ["VGGT_REPO_DIR"]) / "demo_colmap.py")
    candidates.extend(
        [
            ROOT / "vggt" / "demo_colmap.py",
            Path("/workspace/vggt/demo_colmap.py"),
            Path("/workspace/VGGT/demo_colmap.py"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def done_clip_ids(cloud_summary: dict, manifest: dict) -> set[str]:
    clips = cloud_summary.get("clips", [])
    if not clips:
        return {clip.get("clip_id") for clip in manifest.get("clips", []) if clip.get("clip_id")}
    return {clip.get("clip_id") for clip in clips if clip.get("status") == "done"}


def run_vggt_colmap_ba(manifest: dict, cloud_summary: dict) -> dict:
    demo = find_vggt_demo_colmap()
    if demo is None:
        return {
            "method_id": "vggt_colmap_ba_windowed",
            "status": "skipped_missing_dependency",
            "reason": "VGGT demo_colmap.py was not found. Set VGGT_REPO_DIR or place the VGGT repo under /workspace/vggt.",
        }
    valid_ids = done_clip_ids(cloud_summary, manifest)
    rows = []
    for clip in manifest.get("clips", []):
        clip_id = clip.get("clip_id")
        if clip_id not in valid_ids:
            rows.append({"clip_id": clip_id, "status": "skipped_after_vggt_failure"})
            continue
        scene_dir = VGGT_COLMAP_ROOT / str(clip_id)
        log_path = ROOT / "ba_logs" / f"{clip_id}.log"
        try:
            image_count = prepare_scene_images(scene_dir, ROOT / clip["frame_manifest"])
            command = [sys.executable, str(demo), "--scene_dir", str(scene_dir), "--use_ba"]
            command.extend(os.environ.get("VGGT_COLMAP_EXTRA_ARGS", "").split())
            exit_code = run_logged(command, cwd=demo.parent, log_path=log_path, env=colmap_headless_env())
            sparse_model = find_colmap_sparse_model(scene_dir)
            sparse_exists = sparse_model is not None
            rows.append(
                {
                    "clip_id": clip_id,
                    "method_id": "vggt_colmap_ba_windowed",
                    "status": "done" if exit_code == 0 and sparse_exists else "failed_soft",
                    "image_count": image_count,
                    "scene_dir": scene_dir.relative_to(ROOT).as_posix(),
                    "log": log_path.relative_to(ROOT).as_posix(),
                    "exit_code": exit_code,
                    "sparse_exists": sparse_exists,
                    "sparse_model": sparse_model.relative_to(ROOT).as_posix() if sparse_model else None,
                    "command": " ".join(command),
                }
            )
        except Exception as exc:
            rows.append({"clip_id": clip_id, "method_id": "vggt_colmap_ba_windowed", "status": "failed_soft", "error": str(exc)})
    return {
        "method_id": "vggt_colmap_ba_windowed",
        "status": "done" if rows and any(row.get("status") == "done" for row in rows) else "failed_soft",
        "clips": rows,
    }


def run_classical_colmap(manifest: dict, cloud_summary: dict) -> dict:
    if not command_exists("colmap"):
        return {
            "method_id": "colmap_classical_sift_sequential",
            "status": "skipped_missing_dependency",
            "reason": "colmap executable was not found on PATH",
        }
    valid_ids = done_clip_ids(cloud_summary, manifest)
    rows = []
    for clip in manifest.get("clips", []):
        clip_id = clip.get("clip_id")
        if clip_id not in valid_ids:
            rows.append({"clip_id": clip_id, "status": "skipped_after_vggt_failure"})
            continue
        scene_dir = CLASSICAL_COLMAP_ROOT / str(clip_id)
        log_path = ROOT / "classical_colmap_logs" / f"{clip_id}.log"
        try:
            image_count = prepare_scene_images(scene_dir, ROOT / clip["frame_manifest"])
            database = scene_dir / "database.db"
            sparse = scene_dir / "sparse"
            sparse.mkdir(parents=True, exist_ok=True)
            commands = [
                ["colmap", "feature_extractor", "--database_path", str(database), "--image_path", str(scene_dir / "images"), "--ImageReader.single_camera", "1"],
                ["colmap", "sequential_matcher", "--database_path", str(database)],
                ["colmap", "mapper", "--database_path", str(database), "--image_path", str(scene_dir / "images"), "--output_path", str(sparse)],
            ]
            exit_codes = [run_logged(command, cwd=scene_dir, log_path=log_path, env=colmap_headless_env()) for command in commands]
            sparse_model = find_colmap_sparse_model(scene_dir)
            sparse_exists = sparse_model is not None
            rows.append(
                {
                    "clip_id": clip_id,
                    "method_id": "colmap_classical_sift_sequential",
                    "status": "done" if all(code == 0 for code in exit_codes) and sparse_exists else "failed_soft",
                    "image_count": image_count,
                    "scene_dir": scene_dir.relative_to(ROOT).as_posix(),
                    "log": log_path.relative_to(ROOT).as_posix(),
                    "exit_codes": exit_codes,
                    "sparse_exists": sparse_exists,
                    "sparse_model": sparse_model.relative_to(ROOT).as_posix() if sparse_model else None,
                }
            )
        except Exception as exc:
            rows.append({"clip_id": clip_id, "method_id": "colmap_classical_sift_sequential", "status": "failed_soft", "error": str(exc)})
    write_json(ROOT / "registration_report.json", {"method_id": "colmap_classical_sift_sequential", "clips": rows})
    return {
        "method_id": "colmap_classical_sift_sequential",
        "status": "done" if rows and any(row.get("status") == "done" for row in rows) else "failed_soft",
        "clips": rows,
    }


def finite_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return number if np.isfinite(number) else default


def trajectory_qc_metrics(centers: np.ndarray, quats: np.ndarray, timestamps: list) -> dict:
    times = np.asarray([finite_float(value, float(index)) for index, value in enumerate(timestamps)], dtype=float)
    deltas = np.linalg.norm(np.diff(centers, axis=0), axis=1) if len(centers) > 1 else np.asarray([], dtype=float)
    dts = np.diff(times) if len(times) > 1 else np.asarray([], dtype=float)
    quat_norms = np.linalg.norm(quats, axis=1) if len(quats) else np.asarray([], dtype=float)
    finite_steps = deltas[np.isfinite(deltas)]
    finite_dt = dts[np.isfinite(dts)]
    finite_quat_norms = quat_norms[np.isfinite(quat_norms)]
    median_step = float(np.median(finite_steps)) if len(finite_steps) else 0.0
    mad_step = float(np.median(np.abs(finite_steps - median_step))) if len(finite_steps) else 0.0
    jump_threshold = max(median_step * 5.0, median_step + 6.0 * mad_step, 1e-9)
    jump_count = int(np.sum(finite_steps > jump_threshold)) if len(finite_steps) else 0
    return {
        "pose_count": int(len(centers)),
        "duration_sec": float(times[-1] - times[0]) if len(times) > 1 else 0.0,
        "relative_path_length": float(np.sum(finite_steps)) if len(finite_steps) else 0.0,
        "median_relative_step": median_step,
        "max_relative_step": float(np.max(finite_steps)) if len(finite_steps) else 0.0,
        "relative_step_mad": mad_step,
        "relative_pose_jump_threshold": jump_threshold,
        "relative_pose_jump_count": jump_count,
        "relative_pose_jump_ratio": float(jump_count / len(finite_steps)) if len(finite_steps) else 0.0,
        "median_dt_sec": float(np.median(finite_dt)) if len(finite_dt) else 0.0,
        "max_dt_sec": float(np.max(finite_dt)) if len(finite_dt) else 0.0,
        "quat_norm_median": float(np.median(finite_quat_norms)) if len(finite_quat_norms) else 0.0,
        "quat_norm_max_abs_error": float(np.max(np.abs(finite_quat_norms - 1.0))) if len(finite_quat_norms) else 0.0,
        "warning": "relative VGGT-frame diagnostics only; no meters, no geolocation, no ground truth",
    }


def write_trajectory_qc_tables(rows: list[dict]) -> dict:
    csv_path = EVO_ROOT / "trajectory_qc.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["status"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    parquet_status = "skipped_missing_dependency"
    parquet_error = None
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(ROOT / "trajectory_qc.parquet", index=False)
        parquet_status = "done"
    except Exception as exc:
        parquet_error = str(exc)
        write_json(
            EVO_ROOT / "trajectory_qc_parquet_skipped.json",
            {
                "status": parquet_status,
                "reason": parquet_error,
                "fallback": csv_path.relative_to(ROOT).as_posix(),
            },
        )
    return {
        "trajectory_qc_csv": csv_path.relative_to(ROOT).as_posix(),
        "trajectory_qc_parquet_status": parquet_status,
        "trajectory_qc_parquet": "trajectory_qc.parquet" if parquet_status == "done" else None,
        "trajectory_qc_parquet_error": parquet_error,
    }


def export_trajectories() -> dict:
    TUM_ROOT.mkdir(parents=True, exist_ok=True)
    EVO_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    bundles_root = ROOT / "bundles"
    for metadata_path in sorted(bundles_root.rglob("metadata.json")) if bundles_root.exists() else []:
        bundle = metadata_path.parent
        cameras_path = bundle / "cameras.npz"
        if not cameras_path.exists():
            continue
        metadata = load_json(metadata_path, {})
        cameras = np.load(cameras_path, allow_pickle=False)
        centers = np.asarray(cameras["camera_centers"], dtype=float)
        quats = np.asarray(cameras["quaternions_xyzw"], dtype=float)
        timestamps = metadata.get("frame_timestamps_sec") or list(range(len(centers)))
        if len(timestamps) != len(centers) or quats.shape != (len(centers), 4):
            rows.append({"bundle": bundle.relative_to(ROOT).as_posix(), "status": "failed_soft", "reason": "trajectory arrays do not align"})
            continue
        out_dir = TUM_ROOT / str(metadata.get("video_id", bundle.parent.name))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{metadata.get('segment_id', bundle.name)}.tum.txt"
        with out_path.open("w", encoding="utf-8") as handle:
            handle.write("# relative VGGT-frame trajectory; no meters, no geolocation, no ground truth\n")
            for timestamp, center, quat in zip(timestamps, centers, quats):
                handle.write(
                    f"{float(timestamp):.6f} "
                    f"{center[0]:.9g} {center[1]:.9g} {center[2]:.9g} "
                    f"{quat[0]:.9g} {quat[1]:.9g} {quat[2]:.9g} {quat[3]:.9g}\n"
                )
        metrics = trajectory_qc_metrics(centers, quats, timestamps)
        rows.append(
            {
                "method_id": "trajectory_preprocess_evo_eval",
                "status": "done",
                "video_id": metadata.get("video_id"),
                "segment_id": metadata.get("segment_id"),
                "trajectory": out_path.relative_to(ROOT).as_posix(),
                **metrics,
            }
        )
    table_artifacts = write_trajectory_qc_tables(rows)
    qc = {
        "method_id": "trajectory_preprocess_evo_eval",
        "status": "done" if rows else "skipped_no_bundles",
        "warning": "relative VGGT-frame trajectory diagnostics only; no meters, no geolocation, no true ground truth metrics",
        "trajectories": rows,
        "artifacts": table_artifacts,
    }
    write_json(ROOT / "trajectory_qc.json", qc)
    (EVO_ROOT / "README.md").write_text(
        "# EVO Reports\n\nTUM-style trajectories were exported for relative comparison only. "
        "`trajectory_qc.csv` is always written here. `trajectory_qc.parquet` is written at the return root when pandas/parquet support exists. "
        "Run EVO only as a method-consistency diagnostic unless calibrated ground truth is supplied.\n",
        encoding="utf-8",
    )
    return qc

def optional_clip_limit() -> int:
    raw = os.environ.get("OPTIONAL_SHOWCASE_MAX_CLIPS", "10")
    try:
        return max(0, int(raw))
    except ValueError:
        return 10


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_stem(value: object) -> str:
    text = str(value or "unknown")
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)[:160] or "unknown"


def write_dir_readme(path: Path, title: str, body: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "README.md").write_text(f"# {title}\n\n{body}\n", encoding="utf-8")


def successful_method_scenes(*stages: dict) -> list[dict]:
    rows = []
    seen = set()
    for stage in stages:
        method_id = stage.get("method_id", "unknown")
        for row in stage.get("clips") or []:
            if row.get("status") != "done":
                continue
            scene_rel = row.get("scene_dir")
            if not scene_rel:
                continue
            scene_dir = ROOT / scene_rel
            images_dir = scene_dir / "images"
            if not images_dir.exists():
                continue
            key = (row.get("clip_id"), method_id)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "clip_id": row.get("clip_id"),
                    "source_method": method_id,
                    "scene_dir": scene_dir.relative_to(ROOT).as_posix(),
                    "images_dir": images_dir.relative_to(ROOT).as_posix(),
                    "image_count": row.get("image_count"),
                    "log": row.get("log"),
                }
            )
    return rows


def optional_dependency_probe() -> dict:
    return {
        "odm": {
            "odm": command_exists("odm"),
            "odm_run": command_exists("odm_run"),
        },
        "openmvs": {
            "InterfaceCOLMAP": command_exists("InterfaceCOLMAP"),
            "CreateStructure": command_exists("CreateStructure"),
            "DensifyPointCloud": command_exists("DensifyPointCloud"),
            "ReconstructMesh": command_exists("ReconstructMesh"),
            "RefineMesh": command_exists("RefineMesh"),
            "TextureMesh": command_exists("TextureMesh"),
            "TransformScene": command_exists("TransformScene"),
            "Viewer": command_exists("Viewer"),
            "MvgMvsPipeline.py": command_exists("MvgMvsPipeline.py"),
            "colmap": command_exists("colmap"),
        },
        "nerfstudio_gsplat": {
            "nerfstudio_module": python_module_exists("nerfstudio"),
            "gsplat_module": python_module_exists("gsplat"),
            "ns_process_data": command_exists("ns-process-data"),
            "ns_train": command_exists("ns-train"),
            "ns_export": command_exists("ns-export"),
            "ns_render": command_exists("ns-render"),
        },
        "relative_depth": {
            "depth_anything_v2_module": python_module_exists("depth_anything_v2"),
            "transformers_module": python_module_exists("transformers"),
            "torch_module": python_module_exists("torch"),
        },
        "research_methods": {
            "mast3r_module": python_module_exists("mast3r"),
            "dust3r_module": python_module_exists("dust3r"),
            "vggetr_module": python_module_exists("vggetr"),
            "vg2gt_module": python_module_exists("vg2gt"),
            "vg_2gt_module": python_module_exists("vg_2gt"),
        },
    }


def write_scene_selection(root: Path, scenes: list[dict], note: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "selected_scenes.json", {"note": note, "scenes": scenes})
    for scene in scenes:
        scene_dir = root / safe_stem(scene.get("clip_id"))
        scene_dir.mkdir(parents=True, exist_ok=True)
        write_json(scene_dir / "input_scene.json", scene)



def env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def svg_escape(value: object) -> str:
    text = str(value)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def load_depth_frames(predictions_path: Path) -> np.ndarray | None:
    if not predictions_path.exists():
        return None
    try:
        with np.load(predictions_path, allow_pickle=False) as predictions:
            if "depth" not in predictions:
                return None
            depth = np.asarray(predictions["depth"], dtype=np.float32)
    except Exception:
        return None
    while depth.ndim > 2 and depth.shape[-1] == 1:
        depth = np.squeeze(depth, axis=-1)
    if depth.ndim == 2:
        depth = depth[None, :, :]
    if depth.ndim != 3:
        return None
    if depth.shape[0] <= 0 or depth.shape[1] <= 0 or depth.shape[2] <= 0:
        return None
    return depth


def depth_grid(frame: np.ndarray, grid_size: int) -> np.ndarray:
    values = np.asarray(frame, dtype=np.float32)
    height, width = values.shape
    out_h = max(1, min(grid_size, height))
    out_w = max(1, min(grid_size, width))
    ys = np.linspace(0, height - 1, out_h).round().astype(int)
    xs = np.linspace(0, width - 1, out_w).round().astype(int)
    return values[np.ix_(ys, xs)]


def normalized_depth(grid: np.ndarray) -> np.ndarray:
    values = np.asarray(grid, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    samples = values[finite]
    lo = float(np.percentile(samples, 2.0))
    hi = float(np.percentile(samples, 98.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(samples))
        hi = float(np.max(samples))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    normalized = (values - lo) / (hi - lo)
    normalized[~finite] = 0.0
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def depth_color(value: float) -> str:
    v = max(0.0, min(1.0, float(value)))
    r = int(round(35 + 210 * v))
    g = int(round(190 - 125 * v))
    b = int(round(210 - 155 * v))
    return f"#{r:02x}{g:02x}{b:02x}"


def write_depth_contact_sheet_svg(
    *,
    output_path: Path,
    frames: np.ndarray,
    clip_id: str,
    max_frames: int,
    grid_size: int,
) -> dict:
    count = min(max_frames, int(frames.shape[0]))
    if count <= 0:
        raise ValueError("no depth frames available")
    indices = np.linspace(0, frames.shape[0] - 1, count).round().astype(int)
    cell = 4
    gap = 18
    label_h = 18
    grids = [normalized_depth(depth_grid(frames[index], grid_size)) for index in indices]
    tile_h = max(grid.shape[0] for grid in grids) * cell
    tile_w = max(grid.shape[1] for grid in grids) * cell
    cols = min(4, count)
    rows = int(np.ceil(count / cols))
    width = cols * tile_w + (cols + 1) * gap
    height = rows * (tile_h + label_h) + (rows + 1) * gap + 54
    rects = []
    for tile_index, (frame_index, grid) in enumerate(zip(indices, grids)):
        col = tile_index % cols
        row = tile_index // cols
        x0 = gap + col * (tile_w + gap)
        y0 = 42 + gap + row * (tile_h + label_h + gap)
        rects.append(f'<text x="{x0}" y="{y0 - 6}" font-size="11" fill="#d7ddd8">frame {int(frame_index)}</text>')
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                rects.append(
                    f'<rect x="{x0 + x * cell}" y="{y0 + y * cell}" width="{cell}" height="{cell}" fill="{depth_color(float(grid[y, x]))}" />'
                )
    title = svg_escape(clip_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="#101817" />',
                f'<text x="18" y="24" font-size="16" fill="#f2efe6" font-family="Arial">Relative VGGT depth contact sheet: {title}</text>',
                '<text x="18" y="42" font-size="11" fill="#c6c9c3" font-family="Arial">Qualitative only. No meters, no geolocation, no real standoff or route claims.</text>',
                *rects,
                '<text x="18" y="{}" font-size="11" fill="#c6c9c3" font-family="Arial">Color is percentile-normalized per frame; warmer means relatively farther in VGGT depth for review only.</text>'.format(height - 18),
                '</svg>',
            ]
        ),
        encoding="utf-8",
    )
    return {
        "output": output_path.relative_to(ROOT).as_posix(),
        "sampled_depth_frames": [int(index) for index in indices],
        "grid_size": int(grid_size),
    }


def generate_relative_depth_overlays(manifest: dict, selected: list[dict]) -> dict:
    max_frames = env_int("RELATIVE_DEPTH_MAX_FRAMES", 8, 1, 32)
    grid_size = env_int("RELATIVE_DEPTH_GRID_SIZE", 64, 8, 160)
    clips_by_id = {str(clip.get("clip_id")): clip for clip in manifest.get("clips", []) if clip.get("clip_id")}
    rows = []
    for scene in selected:
        clip_id = str(scene.get("clip_id"))
        clip = clips_by_id.get(clip_id)
        if not clip:
            rows.append({"clip_id": clip_id, "status": "skipped_missing_manifest_clip"})
            continue
        predictions_path = ROOT / str(clip.get("predictions_output", ""))
        frames = load_depth_frames(predictions_path)
        if frames is None:
            rows.append(
                {
                    "clip_id": clip_id,
                    "method_id": "relative_depth_overlay_optional",
                    "status": "skipped_no_dense_vggt_depth",
                    "predictions": predictions_path.relative_to(ROOT).as_posix() if predictions_path.exists() else str(clip.get("predictions_output", "")),
                }
            )
            continue
        out_dir = RELATIVE_DEPTH_ROOT / safe_stem(clip_id)
        svg_info = write_depth_contact_sheet_svg(
            output_path=out_dir / "relative_depth_contact_sheet.svg",
            frames=frames,
            clip_id=clip_id,
            max_frames=max_frames,
            grid_size=grid_size,
        )
        row = {
            "clip_id": clip_id,
            "method_id": "relative_depth_overlay_optional",
            "status": "done",
            "source_method": scene.get("source_method"),
            "predictions": predictions_path.relative_to(ROOT).as_posix(),
            "depth_shape": [int(value) for value in frames.shape],
            "warning": "relative depth visualization only; no meters, no geolocation",
            **svg_info,
        }
        rows.append(row)
        write_json(out_dir / "metadata.json", row)
    write_json(
        RELATIVE_DEPTH_ROOT / "overlay_manifest.json",
        {
            "method_id": "relative_depth_overlay_optional",
            "status": "done" if any(row.get("status") == "done" for row in rows) else "skipped_no_dense_vggt_depth" if rows else "skipped_no_successful_colmap_scene",
            "rows": rows,
            "generated_overlay_count": sum(1 for row in rows if row.get("status") == "done"),
            "max_frames": max_frames,
            "grid_size": grid_size,
            "safety": [*SAFETY_WARNINGS, "relative depth only; no meters"],
        },
    )
    return {
        "rows": rows,
        "generated_overlay_count": sum(1 for row in rows if row.get("status") == "done"),
    }


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def shell_quote(value: object) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def copy_scene_images(images_dir: Path, target_images: Path) -> int:
    if target_images.exists():
        shutil.rmtree(target_images)
    target_images.mkdir(parents=True, exist_ok=True)
    count = 0
    for source in sorted(images_dir.iterdir()):
        if not source.is_file() or source.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue
        shutil.copy2(source, target_images / source.name)
        count += 1
    return count



def openmvs_ready_for_execution(deps: dict) -> bool:
    has_dense_backend = bool(deps.get("DensifyPointCloud") and deps.get("ReconstructMesh"))
    has_colmap_bridge = bool(deps.get("InterfaceCOLMAP") and deps.get("colmap"))
    has_native_sfm = bool(deps.get("CreateStructure"))
    return has_dense_backend and (has_colmap_bridge or has_native_sfm)


def find_colmap_sparse_model(scene_dir: Path) -> Path | None:
    candidates = [
        scene_dir / "sparse" / "0",
        scene_dir / "sparse",
        scene_dir / "sparse_bin" / "0",
        scene_dir / "sparse_bin",
        scene_dir / "sparse_txt" / "0",
        scene_dir / "sparse_txt",
    ]
    for candidate in candidates:
        if candidate.exists() and is_colmap_sparse_model(candidate):
            return candidate
    return None


def openmvs_dense_commands(project_dir: Path, images_dir: Path, sparse_model: Path | None, deps: dict) -> tuple[list[list[str]] | None, str | None]:
    template = os.environ.get("OPENMVS_COMMAND_TEMPLATE")
    if template:
        rendered = template.format(project_dir=str(project_dir), images_dir=str(images_dir), sparse_model=str(sparse_model or ""))
        return [rendered.split()], "template"

    scene_mvs = project_dir / "scene.mvs"
    commands: list[list[str]] = []
    if deps.get("InterfaceCOLMAP") and deps.get("colmap") and sparse_model is not None:
        dense_dir = project_dir / "colmap_dense"
        commands.append([
            "colmap",
            "image_undistorter",
            "--image_path",
            str(images_dir),
            "--input_path",
            str(sparse_model),
            "--output_path",
            str(dense_dir),
            "--output_type",
            "COLMAP",
        ])
        commands.append(["InterfaceCOLMAP", "-i", str(dense_dir), "-o", str(scene_mvs), "--image-folder", str(dense_dir / "images")])
        source_mode = "colmap_image_undistorter_interfacecolmap"
    elif deps.get("CreateStructure"):
        raw_images = project_dir / "raw_images"
        copy_scene_images(images_dir, raw_images)
        scene_sfm = project_dir / "scene.sfm"
        commands.append(["CreateStructure", "-s", str(raw_images), "-o", str(scene_sfm), "--export-mvs", str(scene_mvs), "--extract-colors", "1"])
        source_mode = "openmvs_native_createstructure"
    else:
        return None, "missing_colmap_bridge_or_createstructure"

    scene_dense = project_dir / "scene_dense.mvs"
    scene_dense_ply = project_dir / "scene_dense.ply"
    scene_mesh_ply = project_dir / "scene_dense_mesh.ply"
    scene_mesh_refined = project_dir / "scene_dense_mesh_refine.mvs"
    scene_mesh_refined_ply = project_dir / "scene_dense_mesh_refine.ply"
    scene_textured = project_dir / "scene_dense_mesh_refine_texture.mvs"
    commands.append(["DensifyPointCloud", str(scene_mvs)])
    commands.append(["ReconstructMesh", str(scene_dense), "-p", str(scene_dense_ply)])
    if deps.get("RefineMesh") and env_flag("OPENMVS_RUN_REFINE", True):
        commands.append(["RefineMesh", str(scene_dense), "-m", str(scene_mesh_ply), "-o", str(scene_mesh_refined), "--scales", "1", "--max-face-area", "16"])
    if deps.get("TextureMesh") and env_flag("OPENMVS_RUN_TEXTURE", True):
        texture_input = scene_mesh_refined if deps.get("RefineMesh") and env_flag("OPENMVS_RUN_REFINE", True) else scene_dense
        texture_mesh = scene_mesh_refined_ply if texture_input == scene_mesh_refined else scene_mesh_ply
        commands.append(["TextureMesh", str(texture_input), "-m", str(texture_mesh), "-o", str(scene_textured)])
    if deps.get("TransformScene") and env_flag("OPENMVS_EXPORT_GLB", True):
        export_input = scene_textured if deps.get("TextureMesh") and env_flag("OPENMVS_RUN_TEXTURE", True) else scene_dense
        commands.append(["TransformScene", str(export_input), "--convert", "1", "--export-type", "glb"])
    extra = os.environ.get("OPENMVS_EXTRA_ARGS", "").split()
    if extra:
        commands = [command + extra if command and command[0] in {"DensifyPointCloud", "ReconstructMesh", "RefineMesh", "TextureMesh"} else command for command in commands]
    return commands, source_mode


def write_openmvs_run_script(path: Path, commands: list[list[str]] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "echo 'OpenMVS dense mesh baseline is relative/local-only; no geolocation, map projection, or metric route claims.'",
    ]
    if not commands:
        lines.append("echo 'OpenMVS commands are not available for this scene.' >&2")
        lines.append("exit 2")
    else:
        lines.extend(" ".join(shell_quote(part) for part in command) for command in commands)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        path.chmod(0o755)
    except Exception:
        pass


def openmvs_artifact_rows(project_dir: Path) -> list[dict]:
    rows = []
    suffixes = {".mvs", ".sfm", ".ply", ".obj", ".glb", ".gltf", ".log"}
    for artifact in sorted(project_dir.rglob("*")):
        if not artifact.is_file() or artifact.suffix.lower() not in suffixes:
            continue
        rows.append({"path": artifact.relative_to(ROOT).as_posix(), "size_bytes": artifact.stat().st_size})
    return rows


def prepare_openmvs_projects(selected: list[dict], deps: dict) -> dict:
    enabled = env_flag("OPTIONAL_RUN_OPENMVS", False) or env_flag("RUN_OPENMVS", False)
    ready = openmvs_ready_for_execution(deps)
    projects = []
    completed = 0
    failed = 0
    for scene in selected:
        clip_id = str(scene.get("clip_id"))
        project_name = safe_stem(clip_id)
        project_dir = OPENMVS_ROOT / project_name
        report_dir = OPENMVS_REPORTS_ROOT / project_name
        log_path = OPENMVS_LOGS_ROOT / f"{project_name}.log"
        images_dir = ROOT / str(scene.get("images_dir", ""))
        scene_dir = ROOT / str(scene.get("scene_dir", ""))
        sparse_model = find_colmap_sparse_model(scene_dir)
        row = {
            "clip_id": clip_id,
            "project_name": project_name,
            "project_dir": project_dir.relative_to(ROOT).as_posix(),
            "report_dir": report_dir.relative_to(ROOT).as_posix(),
            "log": log_path.relative_to(ROOT).as_posix(),
            "source_method": scene.get("source_method"),
            "source_scene_dir": scene.get("scene_dir"),
            "sparse_model": sparse_model.relative_to(ROOT).as_posix() if sparse_model else None,
            "status": "prepared",
            "warning": "relative OpenMVS dense baseline only; no geolocation, map projection, metric route, or operational interpretation",
        }
        try:
            project_dir.mkdir(parents=True, exist_ok=True)
            report_dir.mkdir(parents=True, exist_ok=True)
            image_count = len([path for path in images_dir.iterdir() if path.is_file()]) if images_dir.exists() else 0
            row["image_count"] = image_count
            write_json(report_dir / "input_scene.json", scene)
            write_dir_readme(
                project_dir,
                "OpenMVS Dense Mesh Baseline",
                "Prepared from a successful relative COLMAP/VGGT-COLMAP window. Treat outputs as local relative dense-reconstruction diagnostics only; no maps, coordinates, real route, or metric claims.",
            )
            commands, source_mode = openmvs_dense_commands(project_dir, images_dir, sparse_model, deps) if ready and image_count else (None, "missing_images_or_dependency")
            row["source_mode"] = source_mode
            row["commands"] = [" ".join(command) for command in commands] if commands else []
            write_openmvs_run_script(report_dir / "run_openmvs.sh", commands)
            if image_count == 0:
                row["status"] = "skipped_no_images"
            elif not ready or commands is None:
                row["status"] = "prepared_missing_dependency"
            elif not enabled:
                row["status"] = "ready_pending_execution_disabled"
            else:
                exit_codes = []
                for command in commands:
                    exit_code = run_logged(command, cwd=project_dir, log_path=log_path)
                    exit_codes.append(exit_code)
                    if exit_code != 0 and not env_flag("OPENMVS_CONTINUE_ON_FAILURE", False):
                        break
                row["exit_codes"] = exit_codes
                row["artifacts"] = openmvs_artifact_rows(project_dir)
                row["status"] = "done" if exit_codes and all(code == 0 for code in exit_codes) else "failed_soft"
                if row["status"] == "done":
                    completed += 1
                else:
                    failed += 1
        except Exception as exc:
            row["status"] = "failed_soft"
            row["error"] = str(exc)
            failed += 1
        write_json(report_dir / "project_report.json", row)
        projects.append(row)
    if completed:
        status = "done" if failed == 0 else "done_partial"
    elif projects and any(row.get("status") == "ready_pending_execution_disabled" for row in projects):
        status = "ready_pending_execution_disabled"
    elif projects and any(row.get("status") == "prepared_missing_dependency" for row in projects):
        status = "skipped_missing_dependency"
    elif projects:
        status = "failed_soft" if failed else "skipped_no_images"
    else:
        status = "skipped_no_successful_colmap_scene"
    return {
        "status": status,
        "execution_enabled": enabled,
        "ready_for_execution": ready,
        "generated_project_count": len(projects),
        "completed_project_count": completed,
        "failed_project_count": failed,
        "projects": projects,
    }

def odm_executable(deps: dict) -> str | None:
    if deps.get("odm_run"):
        return "odm_run"
    if deps.get("odm"):
        return "odm"
    return None


def odm_command(project_root: Path, project_name: str, deps: dict) -> list[str] | None:
    template = os.environ.get("ODM_COMMAND_TEMPLATE")
    if template:
        return template.format(project_path=str(project_root), project_name=project_name).split()
    executable = odm_executable(deps)
    if executable is None:
        return None
    args = [executable, "--project-path", str(project_root), project_name]
    extra = os.environ.get("ODM_EXTRA_ARGS", "").split()
    args.extend(extra)
    return args


def write_odm_run_script(path: Path, command: list[str] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if command is None:
        body = "#!/usr/bin/env bash\nset -euo pipefail\necho 'ODM executable is not available in this image.' >&2\nexit 2\n"
    else:
        body = "#!/usr/bin/env bash\nset -euo pipefail\n" + " ".join(shell_quote(part) for part in command) + "\n"
    path.write_text(body, encoding="utf-8")
    try:
        path.chmod(0o755)
    except Exception:
        pass


def prepare_odm_projects(selected: list[dict], deps: dict) -> dict:
    enabled = env_flag("OPTIONAL_RUN_ODM", False) or env_flag("RUN_ODM", False)
    projects = []
    completed = 0
    failed = 0
    for scene in selected:
        clip_id = str(scene.get("clip_id"))
        project_name = safe_stem(clip_id)
        project_dir = ODM_ROOT / project_name
        report_dir = ODM_REPORTS_ROOT / project_name
        log_path = ODM_LOGS_ROOT / f"{project_name}.log"
        images_dir = ROOT / str(scene.get("images_dir", ""))
        row = {
            "clip_id": clip_id,
            "project_name": project_name,
            "project_dir": project_dir.relative_to(ROOT).as_posix(),
            "report_dir": report_dir.relative_to(ROOT).as_posix(),
            "log": log_path.relative_to(ROOT).as_posix(),
            "source_method": scene.get("source_method"),
            "status": "prepared",
            "warning": "relative ODM baseline only; no geolocation, map projection, route, or coordinate claims",
        }
        try:
            image_count = copy_scene_images(images_dir, project_dir / "images")
            row["image_count"] = image_count
            write_json(report_dir / "input_scene.json", scene)
            write_dir_readme(
                project_dir,
                "ODM Relative Project",
                "Images were copied from a successful relative reconstruction window. Treat outputs as local relative photogrammetry diagnostics only; no maps or coordinates.",
            )
            command = odm_command(ODM_ROOT, project_name, deps)
            row["command"] = " ".join(command) if command else None
            write_odm_run_script(report_dir / "run_odm.sh", command)
            if image_count == 0:
                row["status"] = "skipped_no_images"
            elif command is None:
                row["status"] = "prepared_missing_dependency"
            elif not enabled:
                row["status"] = "ready_pending_execution_disabled"
            else:
                exit_code = run_logged(command, cwd=ODM_ROOT, log_path=log_path)
                row["exit_code"] = exit_code
                row["status"] = "done" if exit_code == 0 else "failed_soft"
                if exit_code == 0:
                    completed += 1
                else:
                    failed += 1
        except Exception as exc:
            row["status"] = "failed_soft"
            row["error"] = str(exc)
            failed += 1
        write_json(report_dir / "project_report.json", row)
        projects.append(row)
    if completed:
        status = "done" if failed == 0 else "done_partial"
    elif projects and any(row.get("status") == "ready_pending_execution_disabled" for row in projects):
        status = "ready_pending_execution_disabled"
    elif projects and any(row.get("status") == "prepared_missing_dependency" for row in projects):
        status = "skipped_missing_dependency"
    elif projects:
        status = "failed_soft" if failed else "skipped_no_images"
    else:
        status = "skipped_no_successful_colmap_scene"
    return {
        "status": status,
        "execution_enabled": enabled,
        "generated_project_count": len(projects),
        "completed_project_count": completed,
        "failed_project_count": failed,
        "projects": projects,
    }


def nerfstudio_ready_for_execution(deps: dict) -> bool:
    return bool(deps.get("ns_process_data") and deps.get("ns_train"))


def nerfstudio_process_command(raw_images: Path, processed_dir: Path) -> list[str]:
    template = os.environ.get("NERFSTUDIO_PROCESS_COMMAND_TEMPLATE")
    if template:
        return template.format(raw_images=str(raw_images), output_dir=str(processed_dir)).split()
    downscales = env_int("NERFSTUDIO_NUM_DOWNSCALES", 3, 0, 8)
    return [
        "ns-process-data",
        "images",
        "--data",
        str(raw_images),
        "--output-dir",
        str(processed_dir),
        "--num-downscales",
        str(downscales),
    ]


def nerfstudio_train_command(processed_dir: Path) -> list[str]:
    template = os.environ.get("NERFSTUDIO_TRAIN_COMMAND_TEMPLATE")
    if template:
        return template.format(data=str(processed_dir)).split()
    method = os.environ.get("NERFSTUDIO_METHOD", "splatfacto")
    command = ["ns-train", method, "--data", str(processed_dir)]
    extra = os.environ.get("NERFSTUDIO_EXTRA_ARGS", "").split()
    command.extend(extra)
    return command


def write_nerfstudio_run_script(path: Path, process_command: list[str] | None, train_command: list[str] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "echo 'Nerfstudio/gSplat showcase is visual-only and local-only; no geolocation or metric claims.'",
    ]
    if process_command is None or train_command is None:
        lines.append("echo 'Nerfstudio CLI commands are not available in this image.' >&2")
        lines.append("exit 2")
    else:
        lines.append(" ".join(shell_quote(part) for part in process_command))
        lines.append(" ".join(shell_quote(part) for part in train_command))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        path.chmod(0o755)
    except Exception:
        pass


def prepare_nerfstudio_projects(selected: list[dict], deps: dict) -> dict:
    enabled = env_flag("OPTIONAL_RUN_NERFSTUDIO", False) or env_flag("RUN_NERFSTUDIO", False)
    ready = nerfstudio_ready_for_execution(deps)
    projects = []
    completed = 0
    failed = 0
    for scene in selected:
        clip_id = str(scene.get("clip_id"))
        project_name = safe_stem(clip_id)
        project_dir = NERFSTUDIO_PROJECTS_ROOT / project_name
        raw_images = project_dir / "raw_images"
        processed_dir = project_dir / "processed"
        runs_dir = project_dir / "runs"
        report_path = project_dir / "project_report.json"
        log_path = SHOWCASE_RENDERS_ROOT / f"{project_name}.log"
        images_dir = ROOT / str(scene.get("images_dir", ""))
        row = {
            "clip_id": clip_id,
            "project_name": project_name,
            "project_dir": project_dir.relative_to(ROOT).as_posix(),
            "raw_images": raw_images.relative_to(ROOT).as_posix(),
            "processed_dir": processed_dir.relative_to(ROOT).as_posix(),
            "runs_dir": runs_dir.relative_to(ROOT).as_posix(),
            "log": log_path.relative_to(ROOT).as_posix(),
            "source_method": scene.get("source_method"),
            "status": "prepared",
            "warning": "visual showcase only; no geolocation, map, route, or metric depth claims",
        }
        try:
            image_count = copy_scene_images(images_dir, raw_images)
            row["image_count"] = image_count
            processed_dir.mkdir(parents=True, exist_ok=True)
            runs_dir.mkdir(parents=True, exist_ok=True)
            process_command = nerfstudio_process_command(raw_images, processed_dir) if ready else None
            train_command = nerfstudio_train_command(processed_dir) if ready else None
            row["process_command"] = " ".join(process_command) if process_command else None
            row["train_command"] = " ".join(train_command) if train_command else None
            write_nerfstudio_run_script(project_dir / "run_nerfstudio.sh", process_command, train_command)
            write_dir_readme(
                project_dir,
                "Nerfstudio gSplat Showcase Project",
                "Prepared from a successful relative reconstruction window. Visual showcase only; no maps, coordinates, real route, or metric claims.",
            )
            if image_count == 0:
                row["status"] = "skipped_no_images"
            elif not ready:
                row["status"] = "prepared_missing_dependency"
            elif not enabled:
                row["status"] = "ready_pending_execution_disabled"
            else:
                process_exit = run_logged(process_command, cwd=project_dir, log_path=log_path)
                row["process_exit_code"] = process_exit
                if process_exit == 0:
                    train_exit = run_logged(train_command, cwd=project_dir, log_path=log_path)
                    row["train_exit_code"] = train_exit
                    row["status"] = "done" if train_exit == 0 else "failed_soft"
                else:
                    row["status"] = "failed_soft"
                if row["status"] == "done":
                    completed += 1
                else:
                    failed += 1
        except Exception as exc:
            row["status"] = "failed_soft"
            row["error"] = str(exc)
            failed += 1
        write_json(report_path, row)
        projects.append(row)
    if completed:
        status = "done" if failed == 0 else "done_partial"
    elif projects and any(row.get("status") == "ready_pending_execution_disabled" for row in projects):
        status = "ready_pending_execution_disabled"
    elif projects and any(row.get("status") == "prepared_missing_dependency" for row in projects):
        status = "skipped_missing_dependency"
    elif projects:
        status = "failed_soft" if failed else "skipped_no_images"
    else:
        status = "skipped_no_successful_colmap_scene"
    return {
        "status": status,
        "execution_enabled": enabled,
        "ready_for_execution": ready,
        "generated_project_count": len(projects),
        "completed_project_count": completed,
        "failed_project_count": failed,
        "projects": projects,
    }



def research_method_definitions() -> list[dict]:
    return [
        {
            "method_id": "mast3r_optional_baseline",
            "label": "MASt3R optional relative baseline",
            "module_key": "mast3r_module",
            "template_env": "MAST3R_COMMAND_TEMPLATE",
            "builtin_runner": "mast3r",
            "default_model_env": "MAST3R_MODEL_NAME",
            "default_model_name": "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        },
        {
            "method_id": "dust3r_optional_baseline",
            "label": "DUSt3R optional relative baseline",
            "module_key": "dust3r_module",
            "template_env": "DUST3R_COMMAND_TEMPLATE",
            "builtin_runner": "dust3r",
            "default_model_env": "DUST3R_MODEL_NAME",
            "default_model_name": "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
        },
        {
            "method_id": "vggetr_candidate_baseline",
            "label": "VGGeTR candidate baseline",
            "module_key": "vggetr_module",
            "template_env": "VGGETR_COMMAND_TEMPLATE",
            "template_envs": ["VGGETR_COMMAND_TEMPLATE", "VG2GT_COMMAND_TEMPLATE"],
            "module_keys": ["vggetr_module", "vg2gt_module", "vg_2gt_module"],
            "builtin_runner": None,
            "default_model_env": "VGGETR_MODEL_NAME",
            "default_model_name": None,
        },
    ]


def research_method_builtin_runner_source() -> str:
    return r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

SAFETY = [
    "offline relative reconstruction diagnostic only",
    "no geolocation",
    "no meters or true speed/standoff/dive-angle claims",
    "no route, launch, target, guidance, or next-maneuver inference",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def image_files(raw_images: Path) -> list[Path]:
    return [p for p in sorted(raw_images.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]


def int_env(name: str, default: int, minimum: int = 1, maximum: int = 1000000) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def as_numpy(value):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def collect_scene_points(scene, max_points: int) -> np.ndarray:
    point_chunks = []
    try:
        pts3d = scene.get_pts3d()
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)
    try:
        masks = scene.get_masks()
    except Exception:
        masks = []
    for index, points in enumerate(pts3d):
        arr = as_numpy(points).reshape(-1, 3).astype(np.float32, copy=False)
        if index < len(masks):
            try:
                mask = as_numpy(masks[index]).reshape(-1).astype(bool)
                if len(mask) == len(arr):
                    arr = arr[mask]
            except Exception:
                pass
        arr = arr[np.isfinite(arr).all(axis=1)]
        if len(arr):
            point_chunks.append(arr)
    if not point_chunks:
        return np.zeros((0, 3), dtype=np.float32)
    points = np.concatenate(point_chunks, axis=0)
    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        points = points[idx]
    return points.astype(np.float32, copy=False)


def save_scene_npz(output_dir: Path, method_id: str, image_paths: list[Path], scene, extra: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    max_points = int_env("RESEARCH_METHOD_MAX_POINTS", 200000, 1000, 5000000)
    poses = np.zeros((0, 4, 4), dtype=np.float32)
    focals = np.zeros((0,), dtype=np.float32)
    try:
        poses = as_numpy(scene.get_im_poses()).astype(np.float32, copy=False)
    except Exception as exc:
        extra["pose_extract_error"] = str(exc)
    try:
        focals = as_numpy(scene.get_focals()).astype(np.float32, copy=False)
    except Exception as exc:
        extra["focal_extract_error"] = str(exc)
    points = collect_scene_points(scene, max_points=max_points)
    npz_path = output_dir / f"{method_id}_relative_output.npz"
    np.savez_compressed(
        npz_path,
        camera_poses=poses,
        focals=focals,
        points=points,
        image_names=np.asarray([p.name for p in image_paths]),
    )
    return {
        "npz": npz_path.name,
        "pose_count": int(poses.shape[0]) if poses.ndim >= 1 else 0,
        "point_count": int(points.shape[0]),
        "focal_count": int(focals.shape[0]) if focals.ndim >= 1 else 0,
        **extra,
    }


def run_dust3r(args, image_paths: list[Path]) -> dict:
    import torch
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.utils.image import load_images

    device = os.environ.get("RESEARCH_METHOD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    model_name = os.environ.get("DUST3R_MODEL_NAME", "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt")
    image_size = int_env("DUST3R_IMAGE_SIZE", 512, 128, 2048)
    batch_size = int_env("DUST3R_BATCH_SIZE", 1, 1, 64)
    niter = int_env("DUST3R_GLOBAL_ALIGN_ITER", 300, 0, 5000)
    schedule = os.environ.get("DUST3R_GLOBAL_ALIGN_SCHEDULE", "cosine")
    lr = float(os.environ.get("DUST3R_GLOBAL_ALIGN_LR", "0.01"))
    scene_graph = os.environ.get("DUST3R_SCENE_GRAPH", "complete")
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device).eval()
    images = load_images([str(path) for path in image_paths], size=image_size)
    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    with torch.no_grad():
        output = inference(pairs, model, device, batch_size=batch_size)
    mode = GlobalAlignerMode.PairViewer if len(images) <= 2 else GlobalAlignerMode.PointCloudOptimizer
    scene = global_aligner(output, device=device, mode=mode)
    loss = None
    if mode != GlobalAlignerMode.PairViewer and niter > 0:
        loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    return save_scene_npz(
        args.output_dir,
        args.method_id,
        image_paths,
        scene,
        {
            "runner": "builtin_dust3r_python_api",
            "model_name": model_name,
            "image_size": image_size,
            "scene_graph": scene_graph,
            "global_alignment_loss": float(loss) if loss is not None else None,
        },
    )


def run_mast3r(args, image_paths: list[Path]) -> dict:
    import torch
    import mast3r.utils.path_to_dust3r  # noqa: F401
    from mast3r.model import AsymmetricMASt3R
    from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images

    device = os.environ.get("RESEARCH_METHOD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    model_name = os.environ.get("MAST3R_MODEL_NAME", "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    image_size = int_env("MAST3R_IMAGE_SIZE", 512, 128, 2048)
    batch_size = int_env("MAST3R_BATCH_SIZE", 1, 1, 64)
    niter = int_env("MAST3R_GLOBAL_ALIGN_ITER", 300, 0, 5000)
    schedule = os.environ.get("MAST3R_GLOBAL_ALIGN_SCHEDULE", "cosine")
    lr = float(os.environ.get("MAST3R_GLOBAL_ALIGN_LR", "0.01"))
    scene_graph = os.environ.get("MAST3R_SCENE_GRAPH", "complete")
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device).eval()
    images = load_images([str(path) for path in image_paths], size=image_size)
    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    with torch.no_grad():
        output = inference(pairs, model, device, batch_size=batch_size, verbose=False)
    mode = GlobalAlignerMode.PairViewer if len(images) <= 2 else GlobalAlignerMode.PointCloudOptimizer
    scene = global_aligner(output, device=device, mode=mode)
    loss = None
    if mode != GlobalAlignerMode.PairViewer and niter > 0:
        loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    return save_scene_npz(
        args.output_dir,
        args.method_id,
        image_paths,
        scene,
        {
            "runner": "builtin_mast3r_python_api",
            "model_name": model_name,
            "image_size": image_size,
            "scene_graph": scene_graph,
            "global_alignment_loss": float(loss) if loss is not None else None,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--raw-images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "method_id": args.method_id,
        "status": "started",
        "raw_images": str(args.raw_images),
        "output_dir": str(args.output_dir),
        "contract": str(args.contract),
        "safety": SAFETY,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        images = image_files(args.raw_images)
        max_images = int_env("RESEARCH_METHOD_MAX_IMAGES", 64, 2, 512)
        if len(images) > max_images:
            indices = np.linspace(0, len(images) - 1, max_images).astype(np.int64)
            images = [images[int(index)] for index in indices]
            report["image_sampling"] = f"deterministic_even_sample_to_{max_images}"
        report["image_count"] = len(images)
        if len(images) < 2:
            raise RuntimeError("at least two images are required")
        if args.method_id == "dust3r_optional_baseline":
            result = run_dust3r(args, images)
        elif args.method_id == "mast3r_optional_baseline":
            result = run_mast3r(args, images)
        else:
            raise RuntimeError(f"no built-in runner for {args.method_id}")
        report.update(result)
        report["status"] = "done"
        write_json(args.output_dir / "method_report.json", report)
        return 0
    except Exception as exc:
        report["status"] = "failed_soft"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        write_json(args.output_dir / "method_report.json", report)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def write_research_method_builtin_runner(path: Path) -> None:
    path.write_text(research_method_builtin_runner_source(), encoding="utf-8")
    try:
        path.chmod(0o755)
    except Exception:
        pass


def research_method_command(method: dict, project_dir: Path, raw_images: Path, output_dir: Path, contract_path: Path, clip_id: str, dependency_available: bool) -> tuple[list[str] | None, str]:
    template = None
    for env_name in method.get("template_envs") or [method["template_env"]]:
        template = os.environ.get(str(env_name))
        if template:
            break
    if template is None:
        template = os.environ.get("RESEARCH_METHOD_COMMAND_TEMPLATE")
    if template is None:
        template = os.environ.get("RESEARCH_ADAPTER_COMMAND_TEMPLATE")
    if template:
        return template.format(
            method_id=method["method_id"],
            project_dir=str(project_dir),
            raw_images=str(raw_images),
            output_dir=str(output_dir),
            method_contract=str(contract_path),
            adapter_contract=str(contract_path),
            clip_id=clip_id,
        ).split(), "command_template"
    if method.get("builtin_runner") and dependency_available:
        return [
            sys.executable,
            str(project_dir / "run_research_method.py"),
            "--method-id",
            str(method["method_id"]),
            "--raw-images",
            str(raw_images),
            "--output-dir",
            str(output_dir),
            "--contract",
            str(contract_path),
        ], "builtin_python_runner"
    if method.get("builtin_runner"):
        return None, "builtin_runner_missing_dependency"
    return None, "missing_command_template"


def write_research_methods_run_script(path: Path, commands: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "echo 'Research methods are relative diagnostics only: no geolocation, meters, route, or operational interpretation.'",
    ]
    if not commands:
        lines.extend(
            [
                "echo 'No research method command template was provided.' >&2",
                "echo 'Set MAST3R_COMMAND_TEMPLATE, DUST3R_COMMAND_TEMPLATE, VGGETR_COMMAND_TEMPLATE, VG2GT_COMMAND_TEMPLATE, or RESEARCH_METHOD_COMMAND_TEMPLATE.' >&2",
                "exit 2",
            ]
        )
    else:
        for command in commands:
            lines.append(" ".join(shell_quote(part) for part in command))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        path.chmod(0o755)
    except Exception:
        pass


def run_research_method_projects(selected: list[dict], deps: dict) -> dict:
    enabled = not env_flag("SKIP_RESEARCH_METHODS", False)
    methods = research_method_definitions()
    projects = []
    completed = 0
    failed = 0
    for scene in selected:
        clip_id = str(scene.get("clip_id"))
        project_name = safe_stem(clip_id)
        project_dir = RESEARCH_METHOD_OUTPUTS_ROOT / project_name
        raw_images = project_dir / "raw_images"
        outputs_dir = project_dir / "outputs"
        logs_dir = project_dir / "logs"
        contract_dir = RESEARCH_METHODS_ROOT / project_name
        contract_path = contract_dir / "method_output_contract.json"
        report_path = project_dir / "project_report.json"
        log_path = logs_dir / "research_methods.log"
        images_dir = ROOT / str(scene.get("images_dir", ""))
        row = {
            "clip_id": clip_id,
            "project_name": project_name,
            "project_dir": project_dir.relative_to(ROOT).as_posix(),
            "raw_images": raw_images.relative_to(ROOT).as_posix(),
            "outputs_dir": outputs_dir.relative_to(ROOT).as_posix(),
            "contract": contract_path.relative_to(ROOT).as_posix(),
            "log": log_path.relative_to(ROOT).as_posix(),
            "source_method": scene.get("source_method"),
            "status": "prepared",
            "warning": "relative research baseline only; method output must be converted before local review; no geolocation, meters, route, or operational claims",
        }
        method_rows = []
        commands = []
        try:
            image_count = copy_scene_images(images_dir, raw_images)
            row["image_count"] = image_count
            outputs_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)
            contract_dir.mkdir(parents=True, exist_ok=True)
            contract = {
                "schema_version": "research-method-contract-v1",
                "clip_id": clip_id,
                "project_name": project_name,
                "raw_images": raw_images.relative_to(ROOT).as_posix(),
                "accepted_outputs": [
                    "repo-owned compact VGGT-style bundle",
                    "TUM-format relative trajectory",
                    "relative point cloud with optional RGB/confidence",
                    "method-specific report explaining scale and coordinate ambiguity",
                ],
                "required_semantics": [
                    "relative coordinate frame only",
                    "no meters",
                    "no geolocation",
                    "no launch/target/route inference",
                    "local-only real-media artifacts",
                ],
                "methods": methods,
            }
            write_json(contract_path, contract)
            write_json(project_dir / "method_input.json", {"scene": scene, "contract": contract})
            write_dir_readme(
                project_dir,
                "Research Method Project",
                "Prepared inputs for the single-run MASt3R/DUSt3R/VGGeTR baseline stage. Commands run during this cloud job when templates or built-in runners are available; outputs must stay relative and local-only.",
            )
            write_research_method_builtin_runner(project_dir / "run_research_method.py")
            for method in methods:
                method_output = outputs_dir / str(method["method_id"])
                method_output.mkdir(parents=True, exist_ok=True)
                dependency_available = any(bool(deps.get(key)) for key in method.get("module_keys", [method["module_key"]]))
                command, runner_kind = research_method_command(method, project_dir, raw_images, method_output, contract_path, clip_id, dependency_available)
                method_status = "ready_pending_execution_disabled" if command else "prepared_missing_command_template"
                if not command and not dependency_available:
                    method_status = "prepared_missing_dependency"
                method_row = {
                    "method_id": method["method_id"],
                    "label": method["label"],
                    "module_available": dependency_available,
                    "template_env": method["template_env"],
                    "template_envs": method.get("template_envs", [method["template_env"]]),
                    "module_keys": method.get("module_keys", [method["module_key"]]),
                    "builtin_runner": method.get("builtin_runner"),
                    "runner_kind": runner_kind,
                    "command": " ".join(command) if command else None,
                    "output_dir": method_output.relative_to(ROOT).as_posix(),
                    "status": method_status,
                }
                method_rows.append(method_row)
                if command:
                    commands.append((method_row, command))
            write_research_methods_run_script(project_dir / "run_research_methods.sh", [command for _method_row, command in commands])
            row["methods"] = method_rows
            row["command_count"] = len(commands)
            if image_count == 0:
                row["status"] = "skipped_no_images"
            elif commands and not enabled:
                row["status"] = "ready_pending_execution_disabled"
            elif commands and enabled:
                exit_codes = []
                for method_row, command in commands:
                    exit_code = run_logged(command, cwd=project_dir, log_path=log_path)
                    method_row["exit_code"] = exit_code
                    method_row["status"] = "done" if exit_code == 0 else "failed_soft"
                    exit_codes.append(exit_code)
                row["exit_codes"] = exit_codes
                row["status"] = "done" if all(code == 0 for code in exit_codes) else "failed_soft"
                if row["status"] == "done":
                    completed += 1
                else:
                    failed += 1
            elif any(deps.values()):
                row["status"] = "prepared_missing_command_template"
            else:
                row["status"] = "prepared_missing_dependency"
        except Exception as exc:
            row["status"] = "failed_soft"
            row["error"] = str(exc)
            failed += 1
        write_json(report_path, row)
        projects.append(row)
    if completed:
        status = "done" if failed == 0 else "done_partial"
    elif projects and any(row.get("status") == "ready_pending_execution_disabled" for row in projects):
        status = "ready_pending_execution_disabled"
    elif projects and any(row.get("status") == "prepared_missing_command_template" for row in projects):
        status = "ready_pending_method_command"
    elif projects and any(row.get("status") == "prepared_missing_dependency" for row in projects):
        status = "skipped_missing_dependency"
    elif projects:
        status = "failed_soft" if failed else "skipped_no_images"
    else:
        status = "skipped_no_successful_colmap_scene"
    return {
        "status": status,
        "execution_enabled": enabled,
        "generated_project_count": len(projects),
        "completed_project_count": completed,
        "failed_project_count": failed,
        "projects": projects,
    }

def write_optional_method_manifests(manifest: dict, vggt_stage: dict, classical_stage: dict) -> dict:
    for root in [
        ODM_ROOT,
        ODM_REPORTS_ROOT,
        ODM_LOGS_ROOT,
        OPENMVS_ROOT,
        OPENMVS_REPORTS_ROOT,
        OPENMVS_LOGS_ROOT,
        NERFSTUDIO_ROOT,
        NERFSTUDIO_PROJECTS_ROOT,
        GSPLAT_EXPORTS_ROOT,
        SHOWCASE_RENDERS_ROOT,
        RELATIVE_DEPTH_ROOT,
        RESEARCH_METHODS_ROOT,
        RESEARCH_METHOD_OUTPUTS_ROOT,
    ]:
        reset_dir(root)

    scenes = successful_method_scenes(vggt_stage, classical_stage)
    limit = optional_clip_limit()
    selected = scenes[:limit]
    deps = optional_dependency_probe()
    odm_ready = deps["odm"]["odm"] or deps["odm"]["odm_run"]
    openmvs_ready = openmvs_ready_for_execution(deps["openmvs"])
    nerfstudio_ready = deps["nerfstudio_gsplat"]["nerfstudio_module"] or deps["nerfstudio_gsplat"]["ns_train"] or deps["nerfstudio_gsplat"]["gsplat_module"]
    depth_ready = deps["relative_depth"]["depth_anything_v2_module"] or deps["relative_depth"]["transformers_module"]
    research_methods_ready = any(deps["research_methods"].values())
    depth_overlays = generate_relative_depth_overlays(manifest, selected)
    depth_overlay_count = int(depth_overlays.get("generated_overlay_count", 0))

    write_scene_selection(
        OPENMVS_REPORTS_ROOT,
        selected,
        "OpenMVS inputs selected from successful COLMAP/VGGT-COLMAP windows only; no coordinates or map projection.",
    )
    write_dir_readme(
        OPENMVS_ROOT,
        "OpenMVS Dense Mesh Baseline",
        "Recommended ODM replacement for selected successful windows. Relative dense/mesh artifacts only; no maps, coordinates, or route analysis.",
    )
    write_dir_readme(
        OPENMVS_LOGS_ROOT,
        "OpenMVS Logs",
        "Runtime logs belong here if OpenMVS is executed in the configured cloud image.",
    )
    openmvs_projects = prepare_openmvs_projects(selected, deps["openmvs"])
    openmvs_manifest = {
        "method_id": "openmvs_dense_mesh_baseline",
        "status": openmvs_projects["status"] if selected else "skipped_no_successful_colmap_scene" if openmvs_ready else "skipped_missing_dependency",
        "dependency_probe": deps["openmvs"],
        "selected_scene_count": len(selected),
        "selected_scenes": selected,
        "execution_enabled": openmvs_projects["execution_enabled"],
        "ready_for_execution": openmvs_projects["ready_for_execution"],
        "generated_project_count": openmvs_projects["generated_project_count"],
        "completed_project_count": openmvs_projects["completed_project_count"],
        "failed_project_count": openmvs_projects["failed_project_count"],
        "projects": openmvs_projects["projects"],
        "execution_policy": "Run only on selected post-QC windows; keep outputs local-only and relative. Set OPTIONAL_RUN_OPENMVS=1 to execute prepared projects.",
        "safety": SAFETY_WARNINGS,
    }
    write_json(ROOT / "openmvs_dense_mesh_manifest.json", openmvs_manifest)

    write_scene_selection(
        ODM_REPORTS_ROOT,
        selected,
        "ODM inputs selected from successful COLMAP/VGGT-COLMAP windows only; no coordinates or map projection.",
    )
    write_dir_readme(
        ODM_ROOT,
        "OpenDroneMap Relative Baseline",
        "Prepared as a local relative photogrammetry baseline. This is not geolocation, map projection, or route analysis.",
    )
    write_dir_readme(
        ODM_LOGS_ROOT,
        "ODM Logs",
        "Runtime logs belong here if ODM is executed in the configured cloud image.",
    )
    odm_projects = prepare_odm_projects(selected, deps["odm"])
    odm_manifest = {
        "method_id": "odm_offline_photogrammetry_baseline",
        "status": odm_projects["status"] if selected else "skipped_no_successful_colmap_scene" if odm_ready else "skipped_missing_dependency",
        "dependency_probe": deps["odm"],
        "selected_scene_count": len(selected),
        "selected_scenes": selected,
        "execution_enabled": odm_projects["execution_enabled"],
        "generated_project_count": odm_projects["generated_project_count"],
        "completed_project_count": odm_projects["completed_project_count"],
        "failed_project_count": odm_projects["failed_project_count"],
        "projects": odm_projects["projects"],
        "execution_policy": "Run only on selected post-QC windows; keep outputs local-only and relative. Set OPTIONAL_RUN_ODM=1 to execute prepared projects.",
        "safety": SAFETY_WARNINGS,
    }
    write_json(ROOT / "odm_relative_artifacts_manifest.json", odm_manifest)

    write_scene_selection(
        NERFSTUDIO_PROJECTS_ROOT,
        selected,
        "Nerfstudio/gSplat showcase candidates from successful COLMAP windows only.",
    )
    write_dir_readme(
        NERFSTUDIO_ROOT,
        "Nerfstudio gSplat Showcase",
        "Portfolio visualization staging area for top-ranked geometry windows. Visual only; no map or operational interpretation.",
    )
    write_dir_readme(GSPLAT_EXPORTS_ROOT, "gSplat Exports", "Gaussian-splat exports go here after a successful showcase run.")
    write_dir_readme(SHOWCASE_RENDERS_ROOT, "Showcase Renders", "Preview renders go here after successful Nerfstudio/gSplat execution.")
    nerfstudio_projects = prepare_nerfstudio_projects(selected, deps["nerfstudio_gsplat"])
    showcase_manifest = {
        "method_id": "gsplat_nerfstudio_showcase",
        "status": nerfstudio_projects["status"] if selected else "skipped_no_successful_colmap_scene" if nerfstudio_ready else "skipped_missing_dependency",
        "dependency_probe": deps["nerfstudio_gsplat"],
        "selected_scene_count": len(selected),
        "selected_scenes": selected,
        "suggested_limit": limit,
        "execution_enabled": nerfstudio_projects["execution_enabled"],
        "ready_for_execution": nerfstudio_projects["ready_for_execution"],
        "generated_project_count": nerfstudio_projects["generated_project_count"],
        "completed_project_count": nerfstudio_projects["completed_project_count"],
        "failed_project_count": nerfstudio_projects["failed_project_count"],
        "projects": nerfstudio_projects["projects"],
        "execution_policy": "Use only top-ranked COLMAP-success windows; set OPTIONAL_RUN_NERFSTUDIO=1 to run ns-process-data and ns-train splatfacto.",
        "safety": SAFETY_WARNINGS,
    }
    write_json(ROOT / "showcase_manifest.json", showcase_manifest)

    write_scene_selection(
        RELATIVE_DEPTH_ROOT,
        selected,
        "Relative-depth overlay candidates. These are qualitative review overlays only, never metric depth.",
    )
    depth_manifest = {
        "method_id": "relative_depth_overlay_optional",
        "status": "done" if depth_overlay_count else "ready_pending_execution" if depth_ready and selected else "skipped_missing_dependency" if not depth_ready else "skipped_no_successful_colmap_scene",
        "dependency_probe": deps["relative_depth"],
        "selected_scene_count": len(selected),
        "selected_scenes": selected,
        "generated_overlay_count": depth_overlay_count,
        "overlay_rows": depth_overlays.get("rows", []),
        "artifact_kind": "relative_vggt_depth_svg_contact_sheet",
        "safety": [*SAFETY_WARNINGS, "relative depth only; no meters"],
    }
    write_json(ROOT / "depth_overlay_manifest.json", depth_manifest)

    write_scene_selection(
        RESEARCH_METHOD_OUTPUTS_ROOT,
        selected,
        "MASt3R/DUSt3R/VGGeTR single-run baseline stage. Method output must be converted before review.",
    )
    write_dir_readme(
        RESEARCH_METHODS_ROOT,
        "Research Methods",
        "MASt3R/DUSt3R/VGGeTR methods are optional baselines and must preserve relative-only semantics.",
    )
    research_method_projects = run_research_method_projects(selected, deps["research_methods"])
    research_methods_manifest = {
        "method_id": "mast3r_dust3r_vggetr_single_run",
        "status": research_method_projects["status"] if selected else "skipped_no_successful_colmap_scene" if research_methods_ready else "skipped_missing_dependency",
        "dependency_probe": deps["research_methods"],
        "selected_scene_count": len(selected),
        "selected_scenes": selected,
        "execution_enabled": research_method_projects["execution_enabled"],
        "generated_project_count": research_method_projects["generated_project_count"],
        "completed_project_count": research_method_projects["completed_project_count"],
        "failed_project_count": research_method_projects["failed_project_count"],
        "projects": research_method_projects["projects"],
        "method_output_contract": "convert relative poses/points into the repo-owned compact bundle format before local review",
        "execution_policy": "Single-run stage: commands are executed during this cloud job when method command templates are present. Set SKIP_RESEARCH_METHODS=1 to skip this stage.",
        "command_templates": [
            "MAST3R_COMMAND_TEMPLATE",
            "DUST3R_COMMAND_TEMPLATE",
            "VGGETR_COMMAND_TEMPLATE",
            "VG2GT_COMMAND_TEMPLATE",
            "RESEARCH_METHOD_COMMAND_TEMPLATE",
        ],
        "safety": SAFETY_WARNINGS,
    }
    write_json(ROOT / "research_methods_manifest.json", research_methods_manifest)
    candidate_text = '''# VGGeTR / VG2GT Candidate Identity

Status: configured slot, runner not bundled. Public naming appears to overlap VGGeTR/VG^2GT/VG2GT references, so the cloud job accepts explicit `VGGETR_COMMAND_TEMPLATE` or `VG2GT_COMMAND_TEMPLATE` runners and records the exact command/output contract. Do not block VGGT+COLMAP production on this candidate, and do not claim VGGeTR ran unless the method row is `done`.
'''
    (ROOT / "candidate_identity.md").write_text(candidate_text, encoding="utf-8")
    feasibility = {
        "method_id": "vggetr_candidate_unresolved",
        "status": "needs_identifier_confirmation",
        "dependency_probe": {
            "vggetr_module": deps["research_methods"].get("vggetr_module", False),
            "vg2gt_module": deps["research_methods"].get("vg2gt_module", False),
            "vg_2gt_module": deps["research_methods"].get("vg_2gt_module", False),
        },
        "execution_enabled": research_method_projects["execution_enabled"],
        "requires_command_template": "VGGETR_COMMAND_TEMPLATE, VG2GT_COMMAND_TEMPLATE, or RESEARCH_METHOD_COMMAND_TEMPLATE",
        "candidate_project_count": research_method_projects["generated_project_count"],
        "required_before_execution": [
            "exact repository/checkpoint/API identity",
            "license compatibility",
            "output conversion into compact bundle or TUM trajectory format",
            "small-window smoke test",
        ],
        "safety": SAFETY_WARNINGS,
    }
    write_json(ROOT / "vggetr_feasibility_report.json", feasibility)

    return {
        "method_id": "optional_visual_and_research_methods",
        "status": "ready_pending_execution" if any([openmvs_ready, odm_ready, nerfstudio_ready, depth_ready, research_methods_ready, bool(research_method_projects["generated_project_count"])]) and selected else "planned_optional_after_qc",
        "selected_scene_count": len(selected),
        "available_scene_count": len(scenes),
        "clip_limit": limit,
        "dependency_probe": deps,
        "artifacts": {
            "openmvs_dense_mesh_manifest": "openmvs_dense_mesh_manifest.json",
            "odm_relative_artifacts_manifest": "odm_relative_artifacts_manifest.json",
            "showcase_manifest": "showcase_manifest.json",
            "depth_overlay_manifest": "depth_overlay_manifest.json",
            "research_methods_manifest": "research_methods_manifest.json",
            "candidate_identity": "candidate_identity.md",
            "vggetr_feasibility_report": "vggetr_feasibility_report.json",
        },
    }

def main() -> None:
    log("expanded method stage: start")
    manifest = load_json(ROOT / "job_manifest.json", {"clips": []})
    cloud_summary = load_json(ROOT / "cloud_summary.json", {"clips": []})
    stages = []
    vggt_colmap_stage = run_vggt_colmap_ba(manifest, cloud_summary)
    stages.append(vggt_colmap_stage)
    classical_colmap_stage = run_classical_colmap(manifest, cloud_summary)
    stages.append(classical_colmap_stage)
    stages.append(export_trajectories())
    stages.append(collect_points_ply_artifacts())
    stages.append(write_colmap_conversion_manifest(vggt_colmap_stage))
    stages.append(write_optional_method_manifests(manifest, vggt_colmap_stage, classical_colmap_stage))
    artifacts = {
        "colmap_sparse_zip": zip_colmap_sparse_models(VGGT_COLMAP_ROOT, ROOT / "colmap_sparse.zip"),
        "converted_colmap_bundles_zip": zip_dir(CONVERTED_COLMAP_ROOT, ROOT / "converted_colmap_bundles.zip"),
        "points_ply_zip": zip_dir(POINTS_PLY_ROOT, ROOT / "points_ply.zip"),
        "classical_colmap_sparse_zip": zip_colmap_sparse_models(CLASSICAL_COLMAP_ROOT, ROOT / "classical_colmap_sparse.zip"),
        "classical_colmap_logs_zip": zip_dir(ROOT / "classical_colmap_logs", ROOT / "classical_colmap_logs.zip"),
        "ba_logs_zip": zip_dir(ROOT / "ba_logs", ROOT / "ba_logs.zip"),
        "trajectories_zip": zip_dir(TUM_ROOT, ROOT / "trajectories.zip"),
        "evo_reports_zip": zip_dir(EVO_ROOT, ROOT / "evo_reports.zip"),
        "odm_artifacts_zip": zip_dir(ODM_ROOT, ROOT / "odm_artifacts.zip"),
        "odm_project_reports_zip": zip_dir(ODM_REPORTS_ROOT, ROOT / "odm_project_reports.zip"),
        "odm_logs_zip": zip_dir(ODM_LOGS_ROOT, ROOT / "odm_logs.zip"),
        "openmvs_artifacts_zip": zip_dir(OPENMVS_ROOT, ROOT / "openmvs_artifacts.zip"),
        "openmvs_project_reports_zip": zip_dir(OPENMVS_REPORTS_ROOT, ROOT / "openmvs_project_reports.zip"),
        "openmvs_logs_zip": zip_dir(OPENMVS_LOGS_ROOT, ROOT / "openmvs_logs.zip"),
        "nerfstudio_gsplat_showcase_zip": zip_dir(NERFSTUDIO_ROOT, ROOT / "nerfstudio_gsplat_showcase.zip"),
        "nerfstudio_projects_zip": zip_dir(NERFSTUDIO_PROJECTS_ROOT, ROOT / "nerfstudio_projects.zip"),
        "gsplat_exports_zip": zip_dir(GSPLAT_EXPORTS_ROOT, ROOT / "gsplat_exports.zip"),
        "showcase_renders_zip": zip_dir(SHOWCASE_RENDERS_ROOT, ROOT / "showcase_renders.zip"),
        "relative_depth_overlays_zip": zip_dir(RELATIVE_DEPTH_ROOT, ROOT / "relative_depth_overlays.zip"),
        "research_methods_zip": zip_dir(RESEARCH_METHODS_ROOT, ROOT / "research_methods.zip"),
        "research_methods_outputs_zip": zip_dir(RESEARCH_METHOD_OUTPUTS_ROOT, ROOT / "research_methods_outputs.zip"),
        "failure_packages_zip": zip_dir(FAILURE_ROOT, ROOT / "failure_packages.zip"),
    }
    report = {
        "schema_version": "expanded-method-stage-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "done_partial",
        "warnings": SAFETY_WARNINGS,
        "stages": stages,
        "artifacts": artifacts,
    }
    stage_statuses = [stage.get("status") for stage in stages]
    if stage_statuses and all(status == "done" for status in stage_statuses):
        report["status"] = "done"
    elif any(status == "done" for status in stage_statuses):
        report["status"] = "done_partial"
    elif any(str(status).startswith("skipped") or str(status).startswith("ready_pending") or str(status).startswith("planned") for status in stage_statuses):
        report["status"] = "done_partial"
    else:
        report["status"] = "failed_soft"
    write_json(REPORT_PATH, report)
    write_json(ROOT / "quality_report.json", report)
    log("expanded method stage: finished")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        FAILURE_ROOT.mkdir(exist_ok=True)
        write_json(FAILURE_ROOT / "expanded_method_stage_error.json", {"status": "failed_soft", "error": str(exc)})
        write_json(REPORT_PATH, {"status": "failed_soft", "error": str(exc), "warnings": SAFETY_WARNINGS})
        log(f"expanded method stage failed_soft: {exc}")
        raise SystemExit(0)
"""


def _return_packager_include_files() -> list[str]:
    base_files = [
        *CORE_RETURN_FILES,
        "cloud_summary.json",
        "run.log",
        "job_manifest.json",
        "METHOD_CONTRACT.md",
        "DOWNLOAD_ME.md",
        "selected_bundle_report.parquet",
        "selected_bundle_report.json",
        "tier_report.parquet",
        "tier_report.json",
        "quality_report.parquet",
        "quality_report.json",
        "hf_dataset_report.json",
        "hf_frame_pack_report.json",
        "cloud_run_status.json",
        "run_optimizer_status.json",
    ]
    return list(dict.fromkeys([*base_files, *expanded_return_artifact_files()]))


def _package_return_script() -> str:
    include_files = _return_packager_include_files()
    script = r'''from __future__ import annotations

import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIER_ORDER = {"smoke": 0, "scout": 1, "main": 2, "high_detail": 3}


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def discover_bundles() -> list[dict]:
    rows = []
    bundles_root = ROOT / "bundles"
    if not bundles_root.exists():
        return rows
    for metadata_path in bundles_root.rglob("metadata.json"):
        bundle = metadata_path.parent
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append({"bundle": str(bundle), "status": "invalid", "reason": str(exc)})
            continue
        relative = bundle.relative_to(bundles_root)
        tier = relative.parts[0] if relative.parts else "unknown"
        rows.append(
            {
                "tier": tier,
                "video_id": metadata.get("video_id"),
                "segment_id": metadata.get("segment_id"),
                "bundle": str(bundle),
                "status": "valid" if (bundle / "cameras.npz").exists() else "invalid",
                "reason": "valid bundle" if (bundle / "cameras.npz").exists() else "missing cameras.npz",
            }
        )
    return rows


def selected_rows(bundle_rows: list[dict]) -> list[dict]:
    best = {}
    for row in bundle_rows:
        if row.get("status") != "valid":
            continue
        key = (row.get("video_id"), row.get("segment_id"))
        if key[0] is None or key[1] is None:
            continue
        current = best.get(key)
        if current is None or TIER_ORDER.get(row["tier"], -1) > TIER_ORDER.get(current["tier"], -1):
            best[key] = row
    return list(best.values())


def copy_selected(rows: list[dict]) -> list[dict]:
    selected_root = ROOT / "selected"
    if selected_root.exists():
        shutil.rmtree(selected_root)
    copied = []
    for row in rows:
        destination = selected_root / str(row["video_id"]) / str(row["segment_id"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path(row["bundle"]), destination)
        copied.append(
            {
                "video_id": row["video_id"],
                "segment_id": row["segment_id"],
                "selected_tier": row["tier"],
                "status": "selected",
                "reason": "highest valid tier available",
                "bundle_path": destination.relative_to(ROOT).as_posix(),
            }
        )
    return copied


def ensure_method_stage_report() -> None:
    report_path = ROOT / "method_stage_report.json"
    if report_path.exists():
        return
    payload = {
        "schema_version": "expanded-method-stage-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "not_run",
        "warnings": ["no geolocation", "no meters", "relative method-consistency diagnostics only"],
        "reason": "expanded method stage did not run before packaging; returning partial package",
        "stages": [],
        "artifacts": {},
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_reports(selected: list[dict], bundles: list[dict]) -> None:
    (ROOT / "features").mkdir(exist_ok=True)
    try:
        import pandas as pd

        pd.DataFrame(selected).to_parquet(ROOT / "selected_bundle_report.parquet", index=False)
        pd.DataFrame(
            [
                {
                    "video_id": row.get("video_id"),
                    "segment_id": row.get("segment_id"),
                    "selected_tier": row.get("selected_tier"),
                    "reconstruction_reliability_score": 0.0,
                    "status": row.get("status"),
                }
                for row in selected
            ]
        ).to_parquet(ROOT / "features" / "segment_features.parquet", index=False)
        pd.DataFrame(bundles).to_parquet(ROOT / "tier_report.parquet", index=False)
    except Exception:
        (ROOT / "selected_bundle_report.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
        (ROOT / "tier_report.json").write_text(json.dumps(bundles, indent=2), encoding="utf-8")


def write_download_me(manifest: dict) -> None:
    status = manifest.get("status", "unknown")
    selected_count = manifest.get("selected_bundle_count", 0)
    text = f"""# Download This Return

Status: `{status}`
Selected bundles: `{selected_count}`

## Primary File

Download this file from the pod:

```text
/workspace/fpv-expanded/h100_return.zip
```

It contains compact selected VGGT bundles, cloud logs, environment reports,
method-stage reports, quality reports, and expanded artifacts when available.

## Sanity Check On Pod

```bash
ls -lh h100_return.zip cloud_summary.json cloud_run.log environment.json hf_dataset_report.json hf_frame_pack_report.json cloud_run_status.json method_stage_report.json quality_report.json DOWNLOAD_ME.md
```

## Local Validation After Download

```powershell
fpv h100 inspect-return --source <local_return_zip>
fpv h100 inspect-return --source <local_return_zip> --json
fpv h100 postflight-return --source <local_return_zip> --launch-manifest outputs/h100/full_161_run/launch/runpod_launch_4090_hf_manifest.json --output-dir outputs/reviews/h100_expanded_postflight
fpv h100 optional-report --source <local_return_zip> --output-dir outputs/reviews/h100_optional_methods
fpv h100 import-return --source <local_return_zip> --workdir outputs/h100/full_161_run --review-output outputs/reviews/h100_expanded_import --dry-run
```

## Safety

- no geolocation
- no meters
- relative VGGT frame
- local-only media
"""
    (ROOT / "DOWNLOAD_ME.md").write_text(text, encoding="utf-8")


def zip_return() -> None:
    target = ROOT / "h100_return.zip"
    if target.exists():
        target.unlink()
    include_files = __INCLUDE_FILES__
    include_dirs = ["selected", "features", "models", "model_reports"]
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in include_files:
            path = ROOT / name
            if path.is_file():
                archive.write(path, arcname=name)
        for dirname in include_dirs:
            root = ROOT / dirname
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=path.relative_to(ROOT))


def safe_repo_path_part(value: str) -> str:
    import re

    value = (value or "unknown").strip().replace("\\", "_").replace("/", "_")
    value = re.sub(r"[^A-Za-z0-9._=-]+", "_", value)
    return value[:120] or "unknown"


def auto_upload_returns(manifest: dict) -> None:
    """Upload only the approved return files. Never uploads raw media or frame caches."""
    approved_names = [
        "h100_return.zip",
        "cloud_summary.json",
        "cloud_run.log",
        "environment.json",
        "return_upload_manifest.json",
    ]
    run_id = safe_repo_path_part(
        os.environ.get("FPV_RUN_ID")
        or os.environ.get("RUN_ID")
        or str(manifest.get("run_id") or "unknown")
    )
    if run_id == "unknown":
        run_id = "runpod_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    repo_id = os.environ.get("HF_RETURN_REPO_ID") or os.environ.get("HF_REPO_ID") or "Grimster/FPV_Hezbo"
    repo_type = os.environ.get("HF_RETURN_REPO_TYPE") or "dataset"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    upload_manifest = {
        "schema_version": "fpv-return-upload-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "skipped_no_token" if not token else "pending",
        "repo_id": repo_id,
        "repo_type": repo_type,
        "path_prefix": f"returns/{run_id}",
        "approved_files_only": approved_names,
        "uploaded": [],
        "missing": [],
        "errors": [],
        "warnings": [
            "private return upload only",
            "no raw videos, extracted frames, or media caches are uploaded by this hook",
        ],
    }
    manifest_path = ROOT / "return_upload_manifest.json"
    manifest_path.write_text(json.dumps(upload_manifest, indent=2, sort_keys=True), encoding="utf-8")
    if not token:
        print("HF return auto-upload skipped: HF_TOKEN is not set.")
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        for name in approved_names:
            path = ROOT / name
            if not path.is_file():
                upload_manifest["missing"].append(name)
                continue
            path_in_repo = f"returns/{run_id}/{name}"
            try:
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=token,
                )
                upload_manifest["uploaded"].append(
                    {"local": name, "path_in_repo": path_in_repo, "size_bytes": path.stat().st_size}
                )
                print(f"HF return auto-upload uploaded {name} -> {path_in_repo}")
            except Exception as exc:
                upload_manifest["errors"].append({"file": name, "error": repr(exc)[:500]})
        if upload_manifest["uploaded"] and not upload_manifest["errors"]:
            upload_manifest["status"] = "done"
        elif upload_manifest["uploaded"]:
            upload_manifest["status"] = "done_partial"
        else:
            upload_manifest["status"] = "failed_soft"
    except Exception as exc:
        upload_manifest["status"] = "failed_soft"
        upload_manifest["errors"].append({"stage": "init", "error": repr(exc)[:500]})
        print(f"HF return auto-upload failed softly: {exc!r}")
    finally:
        manifest_path.write_text(json.dumps(upload_manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    cloud_summary = load_json(ROOT / "cloud_summary.json", {"status": "failed_soft", "clips": []})
    cloud_run_status = load_json(ROOT / "cloud_run_status.json", {})
    bundles = discover_bundles()
    selected = copy_selected(selected_rows(bundles))
    write_reports(selected, bundles)
    ensure_method_stage_report()
    cloud_summary_status = cloud_summary.get("status")
    cloud_run_stage_status = cloud_run_status.get("status")
    partial_stage_statuses = {"done_partial", "failed_soft", "failed", "blocked", "blocked_environment"}
    if selected and cloud_summary_status == "done" and cloud_run_stage_status not in partial_stage_statuses:
        status = "done"
    elif selected:
        status = "done_partial"
    else:
        status = "failed_soft"
    manifest = {
        "schema_version": "h100-return-v1",
        "run_id": load_json(ROOT / "job_manifest.json", {}).get("run_id", "unknown"),
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "warnings": ["no geolocation", "no meters", "relative VGGT frame", "local-only media"],
        "selected_bundle_count": len(selected),
        "feature_files": ["features/segment_features.parquet"],
        "selected_bundle_report": "selected_bundle_report.parquet",
        "method_contract": "METHOD_CONTRACT.md",
        "method_matrix": "method_matrix.json",
        "return_contract": "return_contract.json",
        "method_stage_plan": "method_stage_plan.json",
        "method_stage_report": "method_stage_report.json",
        "expanded_runtime_check": "expanded_runtime_check.json",
        "cloud_run_status": "cloud_run_status.json",
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_download_me(manifest)
    zip_return()
    auto_upload_returns(manifest)
    if status == "failed_soft":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
'''
    return script.replace("__INCLUDE_FILES__", json.dumps(include_files, indent=8))


def _runpod_readme() -> str:
    return """# RunPod GPU Inference Job

This package contains sampled frame packs only. Raw videos are intentionally not
included.

Run:

```bash
bash run_all.sh
```

Expanded stage:

- `scripts/05_expanded_runtime_check.py` writes `expanded_runtime_check.json` before expensive stages.
- `scripts/06_hf_dataset_preflight.py` writes `hf_dataset_report.json`; set `HF_DATASET_ID`, `HF_TOKEN`, and optionally `REQUIRE_HF_DATASET=1` for cloud-side dataset checks.
- `scripts/07_hf_dataset_frame_packs.py` can replace packaged frame packs from a downloaded/local HF snapshot when `HF_DATASET_BUILD_FRAME_PACKS=1`.
- `scripts/80_run_expanded_methods.py` runs after VGGT feed-forward.
- It uses official VGGT `demo_colmap.py --scene_dir ... --use_ba` when `VGGT_REPO_DIR` or `/workspace/vggt` is available.
- It runs classical COLMAP only when `colmap` is on PATH.
- It always writes or packages `method_stage_report.json` so partial returns are inspectable.

Return:

- `h100_return.zip`
- `cloud_summary.json`
- `run.log` / `cloud_run.log`
- `environment.json`
- `hf_dataset_report.json` when the preflight stage ran
- `hf_frame_pack_report.json` when the optional HF frame-pack stage ran
- `method_stage_report.json`
- `quality_report.json` when the expanded stage ran

Safety: no geolocation, no meters, no true speed/standoff/dive-angle claims,
no route/guidance/targeting analysis.
"""


def _job_yaml(run_id: str) -> str:
    return f"""schema_version: runpod-job-v1
run_id: {run_id}
gpu: RTX 4090 24GB pilot; 48GB GPU or H100 fallback for full/high-detail runs
requires_pinned_image: true
live_dependency_installs: false
"""




