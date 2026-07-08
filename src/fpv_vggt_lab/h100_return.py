from __future__ import annotations

import html
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .feature_policy import forbidden_feature_issues
from .h100_package import H100_WARNINGS
from .method_contract import CORE_RETURN_REPORT_FILES, expanded_method_matrix, expanded_return_artifact_files
from .vggt import validate_bundle

OPTIONAL_METHOD_REPORT_FILES = [
    ("odm_relative_artifacts_manifest.json", "odm_offline_photogrammetry_baseline"),
    ("showcase_manifest.json", "gsplat_nerfstudio_showcase"),
    ("depth_overlay_manifest.json", "relative_depth_overlay_optional"),
    ("research_methods_manifest.json", "mast3r_dust3r_vggetr_single_run"),
    ("adapter_report.json", "mast3r_dust3r_vggetr_single_run"),
    ("vggetr_feasibility_report.json", "vggetr_candidate_unresolved"),
    ("adapter_feasibility_report.json", "vggetr_candidate_unresolved"),
    ("candidate_identity.md", "vggetr_candidate_unresolved"),
    ("r3_dataset_prep_report.json", "r3_relative_regression_optional"),
    ("r3_training_report.json", "r3_relative_regression_optional"),
    ("r3_reconstruction_manifest.json", "r3_relative_regression_optional"),
    ("lingbot_map_manifest.json", "lingbot_map_streaming_optional"),
]



def inspect_h100_return(source: Path) -> str:
    report = validate_h100_return(source, extract_root=None)
    manifest = report.get("manifest") or {}
    runtime = report.get("expanded_runtime_check") or {}
    method_stage = report.get("method_stage_report") or {}
    emergency = report.get("emergency_return_manifest") or {}
    lines = [
        "H100 return inspection",
        f"source: {source.resolve()}",
        f"status: {manifest.get('status', 'unknown')}",
        f"schema: {manifest.get('schema_version', 'missing')}",
        f"selected bundles: {report['selected_bundle_count']}",
        f"feature files: {len(report['feature_files'])}",
        f"expanded artifacts: {len(report['expanded_artifacts'])}",
        f"runtime check: {runtime.get('status', 'missing')}",
        f"method stage: {method_stage.get('status', 'missing')}",
    ]
    if report.get("is_emergency_return"):
        lines.append("emergency return: yes")
        lines.append(f"emergency reason: {emergency.get('reason', 'unknown')}")
    for artifact in report.get("expanded_artifacts", []):
        lines.append(
            "artifact "
            f"{artifact['path']}: {artifact['size_bytes']} bytes, sha256 {artifact['sha256']}"
        )
    for key, value in (runtime.get("checks") or {}).items():
        lines.append(f"runtime {key}: {value}")
    for stage in method_stage.get("stages") or []:
        method_id = stage.get("method_id", stage.get("stage", "unknown"))
        lines.append(f"method {method_id}: {stage.get('status', 'unknown')}")
    for audit in report.get("method_artifact_audit", []):
        method_id = audit.get("method_id", "unknown")
        status = audit.get("status", "unknown")
        found = audit.get("found_count", 0)
        expected = audit.get("expected_count", 0)
        lines.append(f"method artifacts {method_id}: {status} ({found}/{expected})")
    for optional in report.get("optional_method_reports", []):
        method_id = optional.get("method_id", "unknown")
        status = optional.get("status", "unknown")
        path = optional.get("path", "unknown")
        selected = optional.get("selected_scene_count")
        suffix = f", selected scenes {selected}" if selected is not None else ""
        lines.append(f"optional {method_id}: {status} ({path}{suffix})")
    if report["issues"]:
        lines.append("issues:")
        lines.extend(f"- {issue}" for issue in report["issues"])
    if report.get("next_actions"):
        lines.append("next actions:")
        lines.extend(f"- {action}" for action in report["next_actions"])
    lines.append("safety: no geolocation, no meters, relative VGGT frame, local-only media")
    return "\n".join(lines)



def write_h100_optional_method_report(
    *,
    source: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    extract_root = output_dir / "h100_return_optional_inspect"
    validation = validate_h100_return(source, extract_root=extract_root)
    optional_reports = list(validation.get("optional_method_reports") or [])
    status = "failed_soft" if validation.get("issues") else "done" if optional_reports else "done_no_optional_reports"
    report = {
        "schema_version": "h100-optional-method-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "source": str(source.resolve()),
        "output_dir": str(output_dir),
        "optional_method_report_count": len(optional_reports),
        "optional_method_reports": optional_reports,
        "issues": list(validation.get("issues") or []),
        "next_actions": _optional_method_next_actions(optional_reports),
        "warnings": H100_WARNINGS,
    }
    (output_dir / "optional_method_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "OPTIONAL_METHOD_REPORT.md").write_text(
        _optional_method_report_markdown(validation),
        encoding="utf-8",
    )
    return report
def import_h100_return(
    *,
    source: Path,
    workdir: Path,
    vggt_root: Path,
    review_output: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    review_output = review_output.resolve()
    review_output.mkdir(parents=True, exist_ok=True)
    extracted = review_output / "h100_return_extracted"
    report = validate_h100_return(source, extract_root=extracted)
    report.update(
        {
            "dry_run": dry_run,
            "workdir": str(workdir),
            "vggt_root": str(vggt_root),
            "review_output": str(review_output),
            "imported_count": 0,
            "would_import_count": 0,
            "clips": [],
            "warnings": H100_WARNINGS,
        }
    )

    if report["issues"]:
        report["status"] = "failed_soft"
        _write_import_outputs(review_output, report)
        return report

    selected_root = Path(report["extracted_to"]) / "selected"
    for bundle_path in sorted(selected_root.rglob("metadata.json")):
        source_bundle = bundle_path.parent
        validation = validate_bundle(source_bundle)
        clip_report = {
            "source": str(source_bundle),
            "valid": validation.valid,
            "errors": list(validation.errors),
            "warnings": list(validation.warnings),
            "metadata": validation.metadata,
            "copied": False,
            "would_copy": False,
            "destination": None,
        }
        if validation.valid and validation.metadata is not None:
            video_id = validation.metadata["video_id"]
            segment_id = validation.metadata["segment_id"]
            destination = vggt_root.resolve() / video_id / segment_id
            clip_report["destination"] = str(destination)
            if destination.exists() and not overwrite and not dry_run:
                clip_report["errors"].append(f"destination already exists: {destination}")
            if not clip_report["errors"]:
                if dry_run:
                    clip_report["would_copy"] = True
                    report["would_import_count"] += 1
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        shutil.rmtree(destination)
                    shutil.copytree(source_bundle, destination)
                    clip_report["copied"] = True
                    report["imported_count"] += 1
        report["clips"].append(clip_report)

    if any(clip["errors"] for clip in report["clips"]):
        report["status"] = "failed_soft"
    else:
        report["status"] = "done"

    _copy_return_tables(Path(report["extracted_to"]), review_output)
    _write_import_outputs(review_output, report)
    return report


def verify_h100_return_against_launch(
    *,
    source: Path,
    launch_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    launch = _load_launch_manifest(launch_manifest)
    extract_root = output_dir / "h100_return_extracted"
    validation = validate_h100_return(source, extract_root=extract_root)
    extracted_root = Path(validation["extracted_to"])
    expected = launch.get("expected_return") or {}
    required_files = list(expected.get("required_files_inside_return") or [])
    missing_required = [name for name in required_files if not (extracted_root / name).exists()]
    manifest = validation.get("manifest") or {}
    return_matrix_ids = _return_method_ids(extracted_root)
    launch_methods = list(launch.get("methods") or [])
    missing_methods = sorted(set(launch_methods) - set(return_matrix_ids))
    method_stage = validation.get("method_stage_report") or {}
    stage_warnings = _method_stage_warnings(method_stage)
    found_artifacts = [artifact["path"] for artifact in validation.get("expanded_artifacts", [])]
    optional_method_reports = list(validation.get("optional_method_reports") or [])
    method_artifact_audit = list(validation.get("method_artifact_audit") or [])
    core_artifact_hints = ["trajectories.zip", "evo_reports.zip"]
    missing_core_artifact_hints = [name for name in core_artifact_hints if name not in found_artifacts]
    warnings = []
    warnings.extend(stage_warnings)
    warnings.extend(_optional_method_warnings(optional_method_reports))
    if missing_core_artifact_hints:
        warnings.append(
            "missing expected trajectory/EVO artifact hints: "
            + ", ".join(missing_core_artifact_hints)
        )
    if validation.get("selected_bundle_count", 0) == 0:
        missing_required.append("selected VGGT bundles")

    issues = list(validation.get("issues") or [])
    issues.extend(f"missing required return file: {name}" for name in missing_required)
    issues.extend(f"return method_matrix missing method: {method_id}" for method_id in missing_methods)
    for required_warning in H100_WARNINGS:
        if required_warning not in manifest.get("warnings", []):
            issues.append(f"return manifest missing warning: {required_warning}")

    status = "failed_soft" if issues else "done_with_warnings" if warnings else "done"
    report = {
        "schema_version": "h100-return-postflight-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "source": str(source.resolve()),
        "launch_manifest": str(launch_manifest.resolve()),
        "output_dir": str(output_dir),
        "launch_package_sha256": (launch.get("package") or {}).get("sha256"),
        "launch_status": launch.get("status"),
        "return_validation": validation,
        "required_files": required_files,
        "missing_required_files": missing_required,
        "launch_methods": launch_methods,
        "return_methods": return_matrix_ids,
        "missing_methods": missing_methods,
        "found_expanded_artifacts": found_artifacts,
        "optional_method_reports": optional_method_reports,
        "method_artifact_audit": method_artifact_audit,
        "warnings": warnings,
        "issues": issues,
        "next_actions": _postflight_next_actions(status, issues, warnings),
    }
    (output_dir / "return_postflight.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "RETURN_POSTFLIGHT.md").write_text(
        _postflight_markdown(report),
        encoding="utf-8",
    )
    return report


def _load_launch_manifest(path: Path) -> dict[str, Any]:
    try:
        launch = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"could not read launch manifest: {exc}") from exc
    if not isinstance(launch, dict):
        raise ValueError("launch manifest must contain a JSON object")
    allowed_schemas = {"runpod-launch-manifest-v1", "runpod-4090-hf-launch-v1"}
    if launch.get("schema_version") not in allowed_schemas:
        raise ValueError(
            "launch manifest schema_version must be runpod-launch-manifest-v1 "
            "or runpod-4090-hf-launch-v1"
        )
    return launch


def _return_method_ids(root: Path) -> list[str]:
    path = root / "method_matrix.json"
    if not path.exists():
        return []
    try:
        matrix = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(matrix, list):
        return []
    return [str(row.get("method_id")) for row in matrix if isinstance(row, dict) and row.get("method_id")]


def _method_artifact_audit(
    *,
    root: Path,
    selected_bundle_count: int,
    expanded_artifacts: list[dict[str, Any]],
    method_stage_report: dict[str, Any] | None,
    optional_method_reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    found_paths = {str(item.get("path")) for item in expanded_artifacts if item.get("path")}
    stage_status = _method_stage_status_map(method_stage_report or {})
    optional_statuses: dict[str, list[str]] = {}
    for item in optional_method_reports:
        method_id = str(item.get("method_id", "unknown"))
        optional_statuses.setdefault(method_id, []).append(str(item.get("status", "unknown")))

    rows: list[dict[str, Any]] = []
    for method in _return_method_contract_rows(root):
        method_id = str(method.get("method_id", "unknown"))
        expected = [str(item) for item in method.get("return_contract") or []]
        found: list[str] = []
        missing: list[str] = []
        substitutions: list[dict[str, str]] = []
        for artifact in expected:
            resolved = _resolve_method_artifact(root, artifact, found_paths, selected_bundle_count)
            if resolved is None:
                missing.append(artifact)
                continue
            found.append(resolved)
            if resolved != artifact:
                substitutions.append({"expected": artifact, "satisfied_by": resolved})

        status = _method_artifact_status(
            expected_count=len(expected),
            found_count=len(found),
            method_id=method_id,
            selected_bundle_count=selected_bundle_count,
            stage_status=stage_status.get(method_id),
            optional_statuses=optional_statuses.get(method_id, []),
        )
        rows.append(
            {
                "method_id": method_id,
                "status": status,
                "method_contract_status": method.get("status"),
                "expected_artifacts": expected,
                "found_artifacts": sorted(set(found)),
                "missing_artifacts": missing,
                "substitutions": substitutions,
                "found_count": len(set(found)),
                "expected_count": len(expected),
                "method_stage_status": stage_status.get(method_id),
                "optional_report_statuses": optional_statuses.get(method_id, []),
            }
        )
    return rows


def _return_method_contract_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "method_matrix.json"
    if path.exists():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            value = None
        if isinstance(value, list):
            rows = [row for row in value if isinstance(row, dict) and row.get("method_id")]
            if rows:
                return rows
    return expanded_method_matrix()


def _method_stage_status_map(method_stage: dict[str, Any]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for stage in method_stage.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        method_id = stage.get("method_id")
        if method_id:
            statuses[str(method_id)] = str(stage.get("status", "unknown"))
    return statuses


def _resolve_method_artifact(
    root: Path,
    artifact: str,
    found_paths: set[str],
    selected_bundle_count: int,
) -> str | None:
    if artifact in found_paths or (root / artifact).is_file():
        return artifact
    if artifact.endswith("/") and (root / artifact.rstrip("/")).exists():
        return artifact
    if artifact == "compact_bundles.zip" and selected_bundle_count > 0:
        return "selected/"
    if artifact == "quality_report.parquet":
        for alternate in ["quality_report.parquet", "quality_report.json", "tier_report.parquet", "tier_report.json"]:
            if alternate in found_paths or (root / alternate).is_file():
                return alternate
    return None


def _method_artifact_status(
    *,
    expected_count: int,
    found_count: int,
    method_id: str,
    selected_bundle_count: int,
    stage_status: str | None,
    optional_statuses: list[str],
) -> str:
    if expected_count == 0:
        return "not_contractualized"
    if found_count == expected_count:
        return "artifact_complete"
    if method_id == "vggt_feedforward_full" and selected_bundle_count > 0 and found_count > 0:
        return "usable_compact_partial"
    if stage_status == "skipped_missing_dependency" and found_count == 0:
        return "skipped_missing_dependency"
    if any(status == "skipped_missing_dependency" for status in optional_statuses) and found_count == 0:
        return "skipped_missing_dependency"
    if optional_statuses and found_count > 0:
        return "staged_artifacts_partial"
    if found_count > 0:
        return "artifact_partial"
    return "artifact_missing"


def _method_stage_warnings(method_stage: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for stage in method_stage.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        status = stage.get("status")
        if status in {"skipped_missing_dependency", "failed_soft", "failed"}:
            method_id = stage.get("method_id", stage.get("stage", "unknown"))
            reason = stage.get("reason") or stage.get("error") or stage.get("warning") or status
            warnings.append(f"{method_id}: {status}: {reason}")
    return warnings


def _optional_method_warnings(optional_reports: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for report in optional_reports:
        status = str(report.get("status", "unknown"))
        if status not in {"skipped_missing_dependency", "failed_soft", "failed", "needs_identifier_confirmation", "unresolved"}:
            continue
        method_id = report.get("method_id", "unknown")
        path = report.get("path", "unknown")
        warnings.append(f"{method_id}: {status}: {path}")
    return warnings


def _postflight_next_actions(status: str, issues: list[str], warnings: list[str]) -> list[str]:
    if status == "failed_soft":
        return [
            "Do not import as final. Read return_postflight.json and fix the missing/corrupt return files first.",
            "If the pod produced partial results, download the complete h100_return.zip again and rerun postflight.",
        ]
    actions = [
        "Run fpv h100 import-return --dry-run using this return package.",
        "If dry-run passes, import and render local review artifacts.",
    ]
    if warnings:
        actions.insert(0, "Review warnings before treating COLMAP/EVO/optional-method outputs as usable diagnostics.")
    return actions


def _postflight_markdown(report: dict[str, Any]) -> str:
    issues = "\n".join(f"- {issue}" for issue in report.get("issues", [])) or "- none"
    warnings = "\n".join(f"- {warning}" for warning in report.get("warnings", [])) or "- none"
    artifacts = "\n".join(f"- `{artifact}`" for artifact in report.get("found_expanded_artifacts", [])) or "- none"
    optional = "\n".join(
        f"- `{item.get('method_id', 'unknown')}`: `{item.get('status', 'unknown')}` from `{item.get('path', 'unknown')}`"
        for item in report.get("optional_method_reports", [])
    ) or "- none"
    method_audit = "\n".join(
        f"- `{item.get('method_id', 'unknown')}`: `{item.get('status', 'unknown')}` "
        f"({item.get('found_count', 0)}/{item.get('expected_count', 0)} artifacts)"
        for item in report.get("method_artifact_audit", [])
    ) or "- none"
    actions = "\n".join(f"- {action}" for action in report.get("next_actions", [])) or "- none"
    return f"""# H100 Return Postflight

Status: `{report['status']}`

## Return

- Source: `{report['source']}`
- Launch manifest: `{report['launch_manifest']}`
- Selected bundles: `{report['return_validation'].get('selected_bundle_count', 0)}`
- Launch package SHA256: `{report.get('launch_package_sha256')}`

## Expanded Artifacts Found

{artifacts}

## Optional Method Reports

{optional}

## Method Artifact Audit

{method_audit}

## Warnings

{warnings}

## Issues

{issues}

## Next Actions

{actions}

## Safety

- no geolocation
- no meters
- relative VGGT frame
- local-only media
"""


def validate_h100_return(source: Path, extract_root: Path | None) -> dict[str, Any]:
    source = source.resolve()
    owned_extract = extract_root is not None
    if extract_root is None:
        extract_root = source.parent / f"{source.stem}_inspect"
    if owned_extract and extract_root.exists():
        shutil.rmtree(extract_root)
    if source.suffix.lower() == ".zip":
        _extract_zip_safely(source, extract_root)
        root = extract_root
    else:
        root = source

    issues: list[str] = []
    emergency_return_manifest = _optional_json_report(root, "emergency_return_manifest.json", issues)
    is_emergency_return = emergency_return_manifest is not None
    manifest_path = root / "manifest.json"
    manifest: dict[str, Any] | None = None
    if not manifest_path.exists():
        if is_emergency_return:
            manifest = emergency_return_manifest
            reason = emergency_return_manifest.get("reason") or "normal return packaging failed"
            issues.append(f"emergency return package: {reason}")
        else:
            issues.append("missing manifest.json")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append(f"invalid manifest.json: {exc}")
            manifest = None

    if manifest is not None:
        expected_schema = "h100-emergency-return-v1" if is_emergency_return and not manifest_path.exists() else "h100-return-v1"
        if manifest.get("schema_version") != expected_schema:
            issues.append(f"manifest schema_version must be {expected_schema}")
        warnings = manifest.get("warnings", [])
        for required in H100_WARNINGS:
            if required not in warnings:
                issues.append(f"manifest missing warning: {required}")

    selected_bundles = sorted(path.parent for path in (root / "selected").rglob("metadata.json"))
    if not selected_bundles and not is_emergency_return:
        issues.append("no selected VGGT bundles found")
    for bundle in selected_bundles:
        validation = validate_bundle(bundle)
        if not validation.valid:
            issues.extend(f"{bundle}: {error}" for error in validation.errors)

    feature_files = _feature_files(root)
    for feature_file in feature_files:
        issues.extend(_feature_file_issues(feature_file, root))

    expanded_runtime_check = _optional_json_report(root, "expanded_runtime_check.json", issues)
    method_stage_report = _optional_json_report(root, "method_stage_report.json", issues)
    quality_report = _optional_json_report(root, "quality_report.json", issues)
    expanded_artifacts = _expanded_artifacts(root, issues)
    optional_method_reports = _optional_method_reports(root, issues)
    method_artifact_audit = _method_artifact_audit(
        root=root,
        selected_bundle_count=len(selected_bundles),
        expanded_artifacts=expanded_artifacts,
        method_stage_report=method_stage_report,
        optional_method_reports=optional_method_reports,
    )

    result = {
        "source": str(source),
        "source_was_zip": source.suffix.lower() == ".zip",
        "extracted_to": str(root),
        "manifest": manifest,
        "emergency_return_manifest": emergency_return_manifest,
        "is_emergency_return": is_emergency_return,
        "selected_bundle_count": len(selected_bundles),
        "feature_files": [path.relative_to(root).as_posix() for path in feature_files],
        "expanded_runtime_check": expanded_runtime_check,
        "method_stage_report": method_stage_report,
        "quality_report": quality_report,
        "expanded_artifacts": expanded_artifacts,
        "optional_method_reports": optional_method_reports,
        "method_artifact_audit": method_artifact_audit,
        "issues": issues,
    }
    result["next_actions"] = _expanded_return_next_actions(result)
    return result



def _optional_method_reports(root: Path, issues: list[str]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for filename, fallback_method_id in OPTIONAL_METHOD_REPORT_FILES:
        path = root / filename
        if not path.exists() or not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                issues.append(f"invalid optional method report {relative}: {exc}")
                continue
            if not isinstance(payload, dict):
                issues.append(f"optional method report {relative} must contain a JSON object")
                continue
            reports.append(
                {
                    "path": relative,
                    "method_id": str(payload.get("method_id") or fallback_method_id),
                    "status": str(payload.get("status") or "unknown"),
                    "selected_scene_count": payload.get("selected_scene_count"),
                    "dependency_probe": payload.get("dependency_probe"),
                    "summary": payload.get("execution_policy")
                    or payload.get("method_output_contract")
                    or payload.get("required_adapter_contract")
                    or payload.get("required_before_execution")
                    or payload.get("warning"),
                }
            )
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            issues.append(f"could not read optional method report {relative}: {exc}")
            continue
        status = "documented"
        for line in text.splitlines():
            if line.lower().startswith("status:"):
                status = line.split(":", 1)[1].strip().strip(".") or status
                if "." in status:
                    status = status.split(".", 1)[0].strip() or status
                break
        reports.append(
            {
                "path": relative,
                "method_id": fallback_method_id,
                "status": status,
                "selected_scene_count": None,
                "dependency_probe": None,
                "summary": text[:280],
            }
        )
    return reports

def _expanded_return_next_actions(report: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    emergency = report.get("emergency_return_manifest") or {}
    if report.get("is_emergency_return"):
        reason = emergency.get("reason") or "normal return packaging failed"
        actions.append(
            "Emergency return package: "
            f"{reason}. Inspect cloud_run.log, environment.json, expanded_runtime_check.json, "
            "method_stage_report.json, and failure_packages before rerunning."
        )
        if report.get("selected_bundle_count", 0) == 0:
            actions.append("No selected VGGT bundles were packaged; do not import this return as final output.")
    elif report.get("issues"):
        actions.append("Fix return validation issues before import; inspect the listed files and rerun dry-run import.")
    runtime = report.get("expanded_runtime_check") or {}
    checks = runtime.get("checks") or {}
    if runtime.get("status") is None:
        actions.append("Return is missing expanded_runtime_check.json; use the expanded package or patch ZIP on the next cloud run.")
    if checks.get("vggt_feedforward_ready") is False:
        actions.append("VGGT feed-forward runtime was not ready; verify CUDA PyTorch, vggt install, and model cache/token access.")
    if checks.get("vggt_colmap_ba_ready") is False:
        actions.append("VGGT COLMAP BA was not ready; set VGGT_REPO_DIR or place VGGT at /workspace/vggt and install pycolmap/COLMAP.")
    if checks.get("classical_colmap_ready") is False:
        actions.append("Classical COLMAP baseline was not ready; install colmap and ensure it is on PATH.")
    if checks.get("trajectory_evo_ready") is False:
        actions.append("EVO was not importable; install evo if trajectory consistency plots are needed.")
    if checks.get("odm_ready") is False:
        actions.append("ODM was not available; keep ODM as optional post-QC unless the pod image includes OpenDroneMap.")
    if checks.get("nerfstudio_gsplat_ready") is False:
        actions.append("Nerfstudio/gsplat was not ready; run showcase rendering later on top-ranked successful COLMAP windows.")
    method_stage = report.get("method_stage_report") or {}
    if method_stage.get("status") is None:
        actions.append("Return is missing method_stage_report.json; confirm scripts/80_run_expanded_methods.py ran or package a partial report.")
    for stage in method_stage.get("stages") or []:
        if stage.get("status") == "skipped_missing_dependency":
            method_id = stage.get("method_id", "unknown")
            reason = stage.get("reason", "missing dependency")
            actions.append(f"{method_id} skipped: {reason}")
    for optional in report.get("optional_method_reports", []):
        method_id = optional.get("method_id", "unknown")
        status = str(optional.get("status", "unknown"))
        if status == "skipped_missing_dependency":
            actions.append(f"{method_id} optional report says dependency is missing; keep it post-QC or use a richer cloud image.")
        elif status in {"ready_pending_execution", "ready_pending_method_command"}:
            actions.append(f"{method_id} has staged inputs/status `{status}`; review the manifest before spending more GPU time.")
        elif status == "needs_identifier_confirmation":
            actions.append("VGGeTR/VG2GT candidate slot needs a verified runner; set VGGETR_COMMAND_TEMPLATE or VG2GT_COMMAND_TEMPLATE only after confirming repo/checkpoint/license.")
    if not actions and report.get("selected_bundle_count", 0) > 0:
        actions.append("Run fpv h100 import-return --dry-run, then import and render local review artifacts.")
    return actions


def _optional_json_report(root: Path, filename: str, issues: list[str]) -> dict[str, Any] | None:
    path = root / filename
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.append(f"invalid {filename}: {exc}")
        return None
    if not isinstance(report, dict):
        issues.append(f"{filename} must contain a JSON object")
        return None
    return report


def _feature_files(root: Path) -> list[Path]:
    features = root / "features"
    if not features.exists():
        return []
    return sorted(
        [
            path
            for path in features.rglob("*")
            if path.suffix.lower() in {".parquet", ".json", ".jsonl", ".csv"}
        ]
    )


def _feature_file_issues(path: Path, root: Path) -> list[str]:
    relative = path.relative_to(root).as_posix()
    try:
        if path.suffix.lower() == ".parquet":
            columns = list(pd.read_parquet(path).columns)
        elif path.suffix.lower() == ".csv":
            columns = list(pd.read_csv(path, nrows=1).columns)
        elif path.suffix.lower() == ".jsonl":
            first = path.read_text(encoding="utf-8").splitlines()[0]
            columns = list(json.loads(first).keys())
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, list) and value:
                columns = list(value[0].keys())
            elif isinstance(value, dict):
                columns = list(value.keys())
            else:
                columns = []
    except Exception as exc:
        return [f"could not read feature file {relative}: {exc}"]
    return [f"{relative}: {issue}" for issue in forbidden_feature_issues(columns)]

def _expanded_artifact_names() -> list[str]:
    names = set(expanded_return_artifact_files())
    names.difference_update(CORE_RETURN_REPORT_FILES)
    return sorted(names)


def _expanded_artifacts(root: Path, issues: list[str]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for name in _expanded_artifact_names():
        path = root / name
        if not path.exists() or not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() == ".zip":
            zip_issue = _zip_artifact_issue(path, relative)
            if zip_issue is not None:
                issues.append(zip_issue)
        artifacts.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return artifacts


def _zip_artifact_issue(path: Path, relative: str) -> str | None:
    if not zipfile.is_zipfile(path):
        return f"{relative} is not a valid zip file"
    try:
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
    except Exception as exc:
        return f"{relative} could not be read as zip: {exc}"
    if bad is not None:
        return f"{relative} contains corrupt zip member: {bad}"
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_zip_safely(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"unsafe path in h100 return zip: {member.filename}")
        archive.extractall(target)


def _copy_return_tables(extracted_root: Path, review_output: Path) -> None:
    for filename in ["selected_bundle_report.parquet", "tier_report.parquet"]:
        source = extracted_root / filename
        if source.exists():
            shutil.copyfile(source, review_output / filename)
    for filename in [
        "expanded_runtime_check.json",
        "method_stage_report.json",
        "quality_report.json",
        "method_stage_plan.json",
        "method_matrix.json",
        "return_contract.json",
        "METHOD_CONTRACT.md",
    ]:
        source = extracted_root / filename
        if source.exists():
            shutil.copyfile(source, review_output / filename)
    source_features = extracted_root / "features"
    if source_features.exists():
        target_features = review_output / "feature_tables"
        if target_features.exists():
            shutil.rmtree(target_features)
        shutil.copytree(source_features, target_features)
    _copy_expanded_artifacts(extracted_root, review_output)


def _copy_expanded_artifacts(extracted_root: Path, review_output: Path) -> None:
    target_root = review_output / "expanded_artifacts"
    copied_any = False
    for name in _expanded_artifact_names():
        source = extracted_root / name
        if not source.exists() or not source.is_file():
            continue
        if not copied_any:
            if target_root.exists():
                shutil.rmtree(target_root)
            target_root.mkdir(parents=True, exist_ok=True)
            copied_any = True
        shutil.copyfile(source, target_root / source.name)


def _write_import_outputs(review_output: Path, report: dict[str, Any]) -> None:
    (review_output / "import_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    optional_reports = list(report.get("optional_method_reports") or [])
    optional_status_counts: dict[str, int] = {}
    for item in optional_reports:
        status = str(item.get("status", "unknown"))
        optional_status_counts[status] = optional_status_counts.get(status, 0) + 1
    method_artifact_audit = list(report.get("method_artifact_audit") or [])
    method_artifact_status_counts: dict[str, int] = {}
    for item in method_artifact_audit:
        status = str(item.get("status", "unknown"))
        method_artifact_status_counts[status] = method_artifact_status_counts.get(status, 0) + 1
    summary = {
        "schema_version": "h100-local-import-summary-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": report["status"],
        "dry_run": report["dry_run"],
        "imported_count": report["imported_count"],
        "would_import_count": report["would_import_count"],
        "warnings": H100_WARNINGS,
        "issues": report["issues"],
        "runtime_check_status": (report.get("expanded_runtime_check") or {}).get("status"),
        "method_stage_status": (report.get("method_stage_report") or {}).get("status"),
        "expanded_artifact_count": len(report.get("expanded_artifacts") or []),
        "expanded_artifact_dir": "expanded_artifacts" if report.get("expanded_artifacts") else None,
        "optional_method_report_count": len(optional_reports),
        "optional_method_status_counts": optional_status_counts,
        "optional_method_reports": optional_reports,
        "method_artifact_audit": method_artifact_audit,
        "method_artifact_status_counts": method_artifact_status_counts,
    }
    (review_output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (review_output / "NEXT_STEPS.md").write_text(_next_steps(report), encoding="utf-8")
    (review_output / "OPTIONAL_METHOD_REPORT.md").write_text(
        _optional_method_report_markdown(report),
        encoding="utf-8",
    )
    (review_output / "index.html").write_text(_index_html(report), encoding="utf-8")


def _next_steps(report: dict[str, Any]) -> str:
    if report["status"] == "done":
        return """# H100 Import Next Steps

Status: `done`

- Open `index.html`.
- Use imported selected VGGT bundles for local review rendering.
- Review `expanded_artifacts/` for COLMAP sparse outputs, trajectory exports, EVO reports, and optional method artifacts when present.
- Read `OPTIONAL_METHOD_REPORT.md` before spending more cloud time on ODM, Nerfstudio/gSplat, relative-depth, or research methods.
- Keep real-media artifacts local-only.

Warning block: No geolocation. No meters. Relative VGGT frame. Local-only media.
"""
    return """# H100 Import Next Steps

Status: `failed_soft`

Review `import_report.json`, fix the listed issues, then rerun `fpv h100 import-return`.

Warning block: No geolocation. No meters. Relative VGGT frame. Local-only media.
"""



def _optional_method_report_markdown(report: dict[str, Any]) -> str:
    rows = list(report.get("optional_method_reports") or [])
    lines = [
        "# Optional Method Report",
        "",
        "This report summarizes optional cloud-return method manifests. Treat these as workflow diagnostics, not truth claims.",
        "",
        "## Safety",
        "",
        "- no geolocation",
        "- no meters",
        "- relative reconstruction diagnostics only",
        "- local-only real-media artifacts",
        "",
        "## Statuses",
        "",
    ]
    if not rows:
        lines.extend([
            "- No optional method manifests were returned.",
            "",
            "## Next Actions",
            "",
            "- Use the expanded RunPod package for the next cloud run if optional method readiness is needed.",
        ])
        return "\n".join(lines) + "\n"
    for item in rows:
        method_id = item.get("method_id", "unknown")
        status = item.get("status", "unknown")
        path = item.get("path", "unknown")
        selected = item.get("selected_scene_count")
        lines.append(f"- `{method_id}`: `{status}` from `{path}`; selected scenes: `{selected if selected is not None else 'n/a'}`")
    actions = _optional_method_next_actions(rows)
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {action}" for action in actions)
    return "\n".join(lines) + "\n"


def _optional_method_next_actions(rows: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    statuses = {str(row.get("status", "unknown")) for row in rows}
    methods = {str(row.get("method_id", "unknown")) for row in rows}
    if "ready_pending_execution" in statuses:
        actions.append("Review staged-scene manifests, then run expensive optional methods only on top-ranked successful COLMAP windows.")
    if "ready_pending_method_command" in statuses:
        actions.append("Research method candidates exist, but their outputs must be converted into the repo compact bundle or TUM trajectory format before review.")
    if "skipped_missing_dependency" in statuses:
        actions.append("Missing optional dependencies are not fatal; use a richer pod image only for methods you actually want to execute.")
    if "vggetr_candidate_unresolved" in methods:
        actions.append("Confirm the exact VGGeTR/VG2GT repository, checkpoint, license, and output API, then configure VGGETR_COMMAND_TEMPLATE or VG2GT_COMMAND_TEMPLATE before execution.")
    if not actions:
        actions.append("No optional-method follow-up is required before reviewing the primary VGGT/COLMAP outputs.")
    return actions
def _index_html(report: dict[str, Any]) -> str:
    rows = []
    for clip in report.get("clips", []):
        metadata = clip.get("metadata") or {}
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(metadata.get('video_id', 'unknown')))}</td>"
            f"<td>{html.escape(str(metadata.get('segment_id', 'unknown')))}</td>"
            f"<td>{html.escape('copied' if clip.get('copied') else 'dry-run' if clip.get('would_copy') else 'not copied')}</td>"
            f"<td>{html.escape('; '.join(clip.get('errors') or []))}</td>"
            "</tr>"
        )
    issues = "".join(f"<li>{html.escape(issue)}</li>" for issue in report.get("issues", []))
    runtime = report.get("expanded_runtime_check") or {}
    method_stage = report.get("method_stage_report") or {}
    runtime_rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in (runtime.get("checks") or {}).items()
    )
    method_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(stage.get('method_id', stage.get('stage', 'unknown'))))}</td>"
        f"<td>{html.escape(str(stage.get('status', 'unknown')))}</td>"
        f"<td>{html.escape(str(stage.get('reason', stage.get('warning', ''))))}</td>"
        "</tr>"
        for stage in (method_stage.get("stages") or [])
    )
    artifact_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(artifact.get('path', 'unknown')))}</td>"
        f"<td>{html.escape(str(artifact.get('size_bytes', 0)))}</td>"
        f"<td>{html.escape(str(artifact.get('sha256', '')))}</td>"
        "</tr>"
        for artifact in report.get("expanded_artifacts", [])
    )
    method_audit_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('method_id', 'unknown')))}</td>"
        f"<td>{html.escape(str(item.get('status', 'unknown')))}</td>"
        f"<td>{html.escape(str(item.get('found_count', 0)))}/{html.escape(str(item.get('expected_count', 0)))}</td>"
        f"<td>{html.escape(', '.join(item.get('missing_artifacts') or []))}</td>"
        "</tr>"
        for item in report.get("method_artifact_audit", [])
    )
    optional_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item.get('method_id', 'unknown')))}</td>"
        f"<td>{html.escape(str(item.get('status', 'unknown')))}</td>"
        f"<td>{html.escape(str(item.get('path', 'unknown')))}</td>"
        f"<td>{html.escape(str(item.get('selected_scene_count', '')))}</td>"
        "</tr>"
        for item in report.get("optional_method_reports", [])
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>H100 Import Review</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; background: #f7f7f4; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #d8ddd5; padding: 0.6rem; text-align: left; }}
    .warning {{ background: #fff7df; border: 1px solid #e2bd63; padding: 1rem; margin: 1rem 0; }}
  </style>
</head>
<body>
  <h1>H100 Import Review</h1>
  <p>Status: <strong>{html.escape(str(report.get("status")))}</strong></p>
  <div class="warning">No geolocation. No meters. Relative VGGT frame. Local-only media.</div>
  <p>Imported: {report.get("imported_count", 0)} | Would import: {report.get("would_import_count", 0)}</p>
  <h2>Expanded Runtime</h2>
  <p>Runtime check: <strong>{html.escape(str(runtime.get("status", "missing")))}</strong></p>
  <table><thead><tr><th>Check</th><th>Ready</th></tr></thead><tbody>{runtime_rows}</tbody></table>
  <h2>Expanded Method Stage</h2>
  <p>Method stage: <strong>{html.escape(str(method_stage.get("status", "missing")))}</strong></p>
  <table><thead><tr><th>Method</th><th>Status</th><th>Reason</th></tr></thead><tbody>{method_rows}</tbody></table>
  <h2>Expanded Artifacts</h2>
  <p>Copied into <code>expanded_artifacts/</code> during import when present.</p>
  <table><thead><tr><th>Artifact</th><th>Bytes</th><th>SHA256</th></tr></thead><tbody>{artifact_rows}</tbody></table>
  <h2>Method Artifact Audit</h2>
  <table><thead><tr><th>Method</th><th>Status</th><th>Found</th><th>Missing</th></tr></thead><tbody>{method_audit_rows}</tbody></table>
  <h2>Optional Method Manifests</h2>
  <table><thead><tr><th>Method</th><th>Status</th><th>Manifest</th><th>Selected Scenes</th></tr></thead><tbody>{optional_rows}</tbody></table>
  <h2>Issues</h2>
  <ul>{issues}</ul>
  <h2>Selected Bundles</h2>
  <table>
    <thead><tr><th>Video</th><th>Segment</th><th>Status</th><th>Errors</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""
