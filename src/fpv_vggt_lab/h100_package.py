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
from .schemas import FrameManifest, model_to_dict
from .segments import read_annotations


H100_WARNINGS = [
    "no geolocation",
    "no meters",
    "relative VGGT frame",
    "local-only media",
]

RAW_MEDIA_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SECRET_PATTERN = re.compile(r"\b(api[_-]?key|token|secret|sk-[A-Za-z0-9])\b", re.IGNORECASE)
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
unzip /workspace/runpod_job.zip
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
            logger.log("sampling " + tier.name + " frame pack for " + annotation.video_id + "/" + annotation.segment_id)
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
        "schema_version": "h100-job-v1",
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
        "clips": clips,
    }
    (runpod_job / "job_manifest.json").write_text(
        json.dumps(job_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (runpod_job / "run_vggt_job.py").write_text(_h100_vggt_runner_script(), encoding="utf-8")
    (runpod_job / "run_all.sh").write_text(_run_all_script(), encoding="utf-8")
    scripts = runpod_job / "scripts"
    scripts.mkdir()
    (scripts / "00_env_check.py").write_text(_env_check_script(), encoding="utf-8")
    (scripts / "10_validate_inputs.py").write_text(_stage_placeholder("validate inputs"), encoding="utf-8")
    (scripts / "20_run_vggt_tier.py").write_text(_stage_placeholder("run VGGT tiers"), encoding="utf-8")
    (scripts / "30_validate_bundle.py").write_text(_stage_placeholder("validate bundles"), encoding="utf-8")
    (scripts / "40_select_best_tier.py").write_text(_stage_placeholder("select best tier"), encoding="utf-8")
    (scripts / "50_extract_features.py").write_text(_stage_placeholder("extract features"), encoding="utf-8")
    (scripts / "60_train_diagnostics.py").write_text(_stage_placeholder("train diagnostics"), encoding="utf-8")
    (scripts / "70_package_return.py").write_text(_package_return_script(), encoding="utf-8")
    (runpod_job / "README.md").write_text(_runpod_readme(), encoding="utf-8")
    (runpod_job / "job.yaml").write_text(_job_yaml(run_id), encoding="utf-8")
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


def _run_all_script() -> str:
    return """#!/usr/bin/env bash
set +e
python scripts/00_env_check.py
ENV_STATUS=$?
if [ "$ENV_STATUS" -eq 0 ]; then
  python run_vggt_job.py
  JOB_STATUS=$?
else
  JOB_STATUS=$ENV_STATUS
fi
python scripts/70_package_return.py
PACK_STATUS=$?
if [ "$JOB_STATUS" -ne 0 ]; then
  exit "$JOB_STATUS"
fi
exit "$PACK_STATUS"
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


def _package_return_script() -> str:
    return r'''from __future__ import annotations

import json
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


def zip_return() -> None:
    target = ROOT / "h100_return.zip"
    if target.exists():
        target.unlink()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in ROOT.rglob("*"):
            if not path.is_file():
                continue
            if path.name == "h100_return.zip" or path.name == "runpod_job.zip":
                continue
            if path.parts[-2:] and "frame_packs" in path.parts:
                continue
            archive.write(path, arcname=path.relative_to(ROOT))


def main() -> None:
    cloud_summary = load_json(ROOT / "cloud_summary.json", {"status": "failed_soft", "clips": []})
    bundles = discover_bundles()
    selected = copy_selected(selected_rows(bundles))
    write_reports(selected, bundles)
    status = "done" if selected and cloud_summary.get("status") == "done" else "done_partial" if selected else "failed_soft"
    manifest = {
        "schema_version": "h100-return-v1",
        "run_id": load_json(ROOT / "job_manifest.json", {}).get("run_id", "unknown"),
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "warnings": ["no geolocation", "no meters", "relative VGGT frame", "local-only media"],
        "selected_bundle_count": len(selected),
        "feature_files": ["features/segment_features.parquet"],
        "selected_bundle_report": "selected_bundle_report.parquet",
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    zip_return()
    if status == "failed_soft":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
'''


def _runpod_readme() -> str:
    return """# H100 RunPod Job

This package contains sampled frame packs only. Raw videos are intentionally not
included.

Run:

```bash
bash run_all.sh
```

Return:

- `h100_return.zip`
- `cloud_summary.json`
- `run.log` / `cloud_run.log`
- `environment.json`

Safety: no geolocation, no meters, no true speed/standoff/dive-angle claims,
no route/guidance/targeting analysis.
"""


def _job_yaml(run_id: str) -> str:
    return f"""schema_version: h100-job-v1
run_id: {run_id}
gpu: NVIDIA H100
requires_pinned_image: true
live_dependency_installs: false
"""




