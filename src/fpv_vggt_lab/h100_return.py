from __future__ import annotations

import html
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .feature_policy import forbidden_feature_issues
from .h100_package import H100_WARNINGS
from .vggt import validate_bundle


def inspect_h100_return(source: Path) -> str:
    report = validate_h100_return(source, extract_root=None)
    manifest = report.get("manifest") or {}
    lines = [
        "H100 return inspection",
        f"source: {source.resolve()}",
        f"status: {manifest.get('status', 'unknown')}",
        f"schema: {manifest.get('schema_version', 'missing')}",
        f"selected bundles: {report['selected_bundle_count']}",
        f"feature files: {len(report['feature_files'])}",
    ]
    if report["issues"]:
        lines.append("issues:")
        lines.extend(f"- {issue}" for issue in report["issues"])
    lines.append("safety: no geolocation, no meters, relative VGGT frame, local-only media")
    return "\n".join(lines)


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
    manifest_path = root / "manifest.json"
    manifest: dict[str, Any] | None = None
    if not manifest_path.exists():
        issues.append("missing manifest.json")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append(f"invalid manifest.json: {exc}")
            manifest = None

    if manifest is not None:
        if manifest.get("schema_version") != "h100-return-v1":
            issues.append("manifest schema_version must be h100-return-v1")
        warnings = manifest.get("warnings", [])
        for required in H100_WARNINGS:
            if required not in warnings:
                issues.append(f"manifest missing warning: {required}")

    selected_bundles = sorted(path.parent for path in (root / "selected").rglob("metadata.json"))
    if not selected_bundles:
        issues.append("no selected VGGT bundles found")
    for bundle in selected_bundles:
        validation = validate_bundle(bundle)
        if not validation.valid:
            issues.extend(f"{bundle}: {error}" for error in validation.errors)

    feature_files = _feature_files(root)
    for feature_file in feature_files:
        issues.extend(_feature_file_issues(feature_file, root))

    return {
        "source": str(source),
        "source_was_zip": source.suffix.lower() == ".zip",
        "extracted_to": str(root),
        "manifest": manifest,
        "selected_bundle_count": len(selected_bundles),
        "feature_files": [path.relative_to(root).as_posix() for path in feature_files],
        "issues": issues,
    }


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
    source_features = extracted_root / "features"
    if source_features.exists():
        target_features = review_output / "feature_tables"
        if target_features.exists():
            shutil.rmtree(target_features)
        shutil.copytree(source_features, target_features)


def _write_import_outputs(review_output: Path, report: dict[str, Any]) -> None:
    (review_output / "import_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "h100-local-import-summary-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": report["status"],
        "dry_run": report["dry_run"],
        "imported_count": report["imported_count"],
        "would_import_count": report["would_import_count"],
        "warnings": H100_WARNINGS,
        "issues": report["issues"],
    }
    (review_output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (review_output / "NEXT_STEPS.md").write_text(_next_steps(report), encoding="utf-8")
    (review_output / "index.html").write_text(_index_html(report), encoding="utf-8")


def _next_steps(report: dict[str, Any]) -> str:
    if report["status"] == "done":
        return """# H100 Import Next Steps

Status: `done`

- Open `index.html`.
- Use imported selected VGGT bundles for local review rendering.
- Keep real-media artifacts local-only.

Warning block: No geolocation. No meters. Relative VGGT frame. Local-only media.
"""
    return """# H100 Import Next Steps

Status: `failed_soft`

Review `import_report.json`, fix the listed issues, then rerun `fpv h100 import-return`.

Warning block: No geolocation. No meters. Relative VGGT frame. Local-only media.
"""


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
