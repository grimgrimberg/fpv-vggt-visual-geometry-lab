from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import cv2

from .frames import sample_video_frames
from .geometry import summarize_bundle
from .schemas import model_to_dict
from .synthetic import create_synthetic_video
from .vggt import create_mock_bundle


SAFETY_WARNINGS = [
    "no geolocation",
    "no meters",
    "relative VGGT frame",
    "local-only media",
]


def run_synthetic_pipeline(workdir: Path, frames: int = 8) -> dict[str, Any]:
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    log_path = workdir / "run.log"
    stage_status: dict[str, str] = {}
    artifacts: dict[str, Any] = {}

    def log(message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {message}\n")

    log("start synthetic run")

    video_path = workdir / "synthetic.mp4"
    create_synthetic_video(video_path, frames=max(frames * 3, 24), width=160, height=120)
    stage_status["synthetic_video"] = "done"
    artifacts["synthetic_video"] = str(video_path)
    log(f"created synthetic video: {video_path}")

    frames_dir = workdir / "frames"
    manifest = sample_video_frames(
        video=video_path,
        output=frames_dir,
        count=frames,
        video_id="synthetic-video",
        segment_id="segment-001",
    )
    stage_status["frame_sampling"] = "done"
    artifacts["frame_manifest"] = str(frames_dir / "frames.json")
    log(f"sampled {len(manifest.frames)} frames")

    bundle_dir = workdir / "vggt_bundle"
    create_mock_bundle(frames_dir / "frames.json", bundle_dir)
    stage_status["mock_vggt_bundle"] = "done"
    artifacts["vggt_bundle"] = str(bundle_dir)
    log(f"created mocked VGGT bundle: {bundle_dir}")

    reconstruction_path = workdir / "reconstruction_summary.json"
    reconstruction_summary = summarize_bundle(bundle_dir, reconstruction_path)
    stage_status["reconstruction_summary"] = "done"
    artifacts["reconstruction_summary"] = str(reconstruction_path)
    log(f"wrote reconstruction summary: {reconstruction_path}")

    summary = {
        "status": "done",
        "stage_status": stage_status,
        "warnings": SAFETY_WARNINGS,
        "artifacts": artifacts,
        "environment": environment_snapshot(),
        "reconstruction": model_to_dict(reconstruction_summary),
    }
    _write_run_summary(workdir, summary, _next_steps_text(summary))
    log("finished synthetic run")
    return summary


def run_review_pipeline(
    workdir: Path,
    frame_manifests: list[Path],
    bundles: list[Path],
    smooth: bool = False,
    heatmaps: bool = False,
    export_video: bool = False,
) -> dict[str, Any]:
    from .review import audit_three_clip_inputs, run_three_clip_review

    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    log_path = workdir / "run.log"
    stage_status: dict[str, str] = {}
    artifacts: dict[str, str] = {}
    failures: list[str] = []

    def log(message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {message}\n")

    log("start three-clip review run")
    log(f"frame manifests: {[str(path) for path in frame_manifests]}")
    log(f"bundles: {[str(path) for path in bundles]}")

    summary: dict[str, Any] = {
        "status": "failed_soft",
        "inputs": {
            "frame_manifests": [str(path) for path in frame_manifests],
            "bundles": [str(path) for path in bundles],
            "smooth": smooth,
            "heatmaps": heatmaps,
            "export_video": export_video,
        },
        "decisions": [
            "audit inputs before rendering review artifacts",
            "render only when all VGGT bundles validate",
            "keep real-media-derived outputs local-only",
        ],
        "stage_status": stage_status,
        "warnings": SAFETY_WARNINGS,
        "artifacts": artifacts,
        "data_checks": {},
        "environment": environment_snapshot(),
        "failures": failures,
    }

    try:
        audit_path = workdir / "audit.json"
        audit = audit_three_clip_inputs(frame_manifests, bundles, audit_path)
        artifacts["audit_report"] = str(audit_path)
        if not audit["ready"]:
            summary["status"] = "needs_vggt_bundle"
            stage_status["audit_three"] = "needs_vggt_bundle"
            failures.extend(_audit_failures(audit))
            log("audit result: needs_vggt_bundle")
            _write_run_summary(workdir, summary, _review_next_steps_text(summary))
            return summary

        stage_status["audit_three"] = "done"
        log("audit result: ready")

        review_dir = workdir / "review_artifacts"
        report_path = run_three_clip_review(
            frame_manifests,
            bundles,
            review_dir,
            smooth=smooth,
            generate_heatmaps=heatmaps,
            export_video=export_video,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        artifacts["review_report"] = str(report_path)
        artifacts["comparison_html"] = report["comparison_html"]
        artifacts["side_by_side_videos"] = [
            clip["side_by_side_video"]
            for clip in report.get("clips", [])
            if clip.get("side_by_side_video")
        ]
        artifacts["heatmap_manifests"] = [
            clip["heatmap_manifest"]
            for clip in report.get("clips", [])
            if clip.get("heatmap_manifest")
        ]
        _record_heatmap_diagnostics(report, heatmaps, stage_status, failures)
        stage_status["review_three"] = "done"
        summary["status"] = "done"
        log(f"wrote review report: {report_path}")
        _write_run_summary(workdir, summary, _review_next_steps_text(summary))
        return summary
    except Exception as exc:
        failures.append(str(exc))
        stage_status.setdefault("audit_three", "failed_soft")
        summary["status"] = "failed_soft"
        log(f"failed_soft: {exc}")
        _write_run_summary(workdir, summary, _review_next_steps_text(summary))
        return summary


def run_video_id_pipeline(
    workdir: Path,
    video_ids: list[str],
    catalog_path: Path,
    media_dir: Path,
    media_inventory: Path,
    annotations: Path,
    frames_root: Path,
    vggt_root: Path,
    cloud_job_output: Path,
    cloud_return: Path | None = None,
    cloud_import_report: Path | None = None,
    overwrite_vggt: bool = False,
    segment_id: str = "segment-001",
    frame_count: int = 64,
    resized_long_edge: int | None = None,
    fetch_media_enabled: bool = False,
    smooth: bool = False,
    heatmaps: bool = False,
    export_video: bool = False,
) -> dict[str, Any]:
    from .cloud_job import create_cloud_job_package
    from .frames import sample_accepted_segment_frames
    from .media import (
        audit_media_inventory,
        fetch_media,
        find_media_record,
        write_media_audit_report,
    )
    from .readiness import audit_data_readiness, write_readiness_report
    from .review import audit_three_clip_inputs, run_three_clip_review
    from .segments import (
        get_accepted_segment,
        get_annotation,
        propose_segment,
        render_segment_contact_sheet,
    )
    from .vggt import import_cloud_job_bundles

    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "run.log"
    stage_status: dict[str, str] = {}
    artifacts: dict[str, Any] = {
        "frame_manifests": [],
        "bundles": [],
        "contact_sheets": [],
    }
    failures: list[str] = []

    def log(message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {message}\n")

    summary: dict[str, Any] = {
        "status": "failed_soft",
        "inputs": {
            "video_ids": video_ids,
            "catalog": str(catalog_path),
            "media_dir": str(media_dir),
            "media_inventory": str(media_inventory),
            "annotations": str(annotations),
            "frames_root": str(frames_root),
            "vggt_root": str(vggt_root),
            "cloud_job_output": str(cloud_job_output),
            "cloud_return": str(cloud_return) if cloud_return is not None else None,
            "cloud_import_report": str(cloud_import_report)
            if cloud_import_report is not None
            else "outputs/reviews/cloud_bundle_import.json",
            "overwrite_vggt": overwrite_vggt,
            "segment_id": segment_id,
            "frame_count": frame_count,
            "resized_long_edge": resized_long_edge,
            "fetch_media": fetch_media_enabled,
            "smooth": smooth,
            "heatmaps": heatmaps,
            "export_video": export_video,
        },
        "decisions": [
            "fetch media only when --fetch-media is set",
            "sample frames only from accepted segment annotations",
            "render review artifacts only when all VGGT bundles validate",
            "keep real-media-derived outputs local-only",
        ],
        "stage_status": stage_status,
        "warnings": SAFETY_WARNINGS,
        "artifacts": artifacts,
        "data_checks": {},
        "environment": environment_snapshot(),
        "failures": failures,
    }

    log("start video-id review run")
    log(f"video ids: {video_ids}")

    if len(video_ids) < 3:
        failures.append("video-id run requires at least three clips")
        stage_status["input_selection"] = "failed_soft"
        _write_run_summary(workdir, summary, _video_id_next_steps_text(summary))
        return summary
    stage_status["input_selection"] = "done"

    missing_media: list[str] = []
    for video_id in video_ids:
        try:
            find_media_record(media_inventory, video_id)
        except Exception:
            if not fetch_media_enabled:
                missing_media.append(video_id)
                continue
            try:
                record = fetch_media(catalog_path, media_dir, media_inventory, video_id)
                log(f"fetched local media for {video_id}: {record.local_path}")
            except Exception as exc:
                missing_media.append(video_id)
                failures.append(f"{video_id}: {exc}")
    if missing_media:
        stage_status["local_media"] = "needs_human_review"
        summary["status"] = "needs_human_review"
        failures.extend(
            f"local media is missing for {video_id}; rerun with --fetch-media"
            for video_id in missing_media
        )
        log("local media missing")
        _write_run_summary(workdir, summary, _video_id_next_steps_text(summary))
        return summary
    stage_status["local_media"] = "done"

    media_audit_path = workdir / "media_audit_report.json"
    media_audit = audit_media_inventory(media_inventory, video_ids=video_ids)
    write_media_audit_report(media_audit, media_audit_path)
    artifacts["media_audit_report"] = str(media_audit_path)
    summary["data_checks"]["media_audit"] = _media_audit_status_summary(
        media_audit, media_audit_path
    )
    if media_audit["status"] != "ready":
        stage_status["media_audit"] = "needs_human_review"
        summary["status"] = "needs_human_review"
        failures.extend(_media_audit_failures(media_audit))
        log("media audit needs_human_review")
        _write_run_summary(workdir, summary, _video_id_next_steps_text(summary))
        return summary
    stage_status["media_audit"] = "done"

    pending_segments: list[str] = []
    contact_dir = workdir / "segment_review"
    for video_id in video_ids:
        try:
            get_accepted_segment(annotations, video_id, segment_id)
        except Exception:
            existing = get_annotation(annotations, video_id, segment_id)
            try:
                if existing is None:
                    existing = propose_segment(media_inventory, annotations, video_id, segment_id)
                contact_sheet = render_segment_contact_sheet(
                    inventory_path=media_inventory,
                    annotations_path=annotations,
                    video_id=video_id,
                    segment_id=segment_id,
                    output=contact_dir / f"{video_id}__{segment_id}.jpg",
                )
                artifacts["contact_sheets"].append(str(contact_sheet))
            except Exception as exc:
                failures.append(f"{video_id}: could not prepare segment review: {exc}")
            pending_segments.append(video_id)
    if pending_segments:
        stage_status["accepted_segments"] = "needs_human_review"
        summary["status"] = "needs_human_review"
        failures.extend(
            f"segment is not accepted for {video_id}/{segment_id}" for video_id in pending_segments
        )
        log("accepted segment gate needs human review")
        _write_run_summary(workdir, summary, _video_id_next_steps_text(summary))
        return summary
    stage_status["accepted_segments"] = "done"

    frame_manifest_paths: list[Path] = []
    for video_id in video_ids:
        frame_output = frames_root / video_id / segment_id
        try:
            manifest = sample_accepted_segment_frames(
                media_inventory=media_inventory,
                annotations=annotations,
                video_id=video_id,
                segment_id=segment_id,
                output=frame_output,
                count=frame_count,
                resized_long_edge=resized_long_edge,
            )
        except Exception as exc:
            failures.append(f"{video_id}: frame sampling failed: {exc}")
            continue
        frame_manifest_path = frame_output / "frames.json"
        frame_manifest_paths.append(frame_manifest_path)
        artifacts["frame_manifests"].append(str(frame_manifest_path))
        log(f"sampled {len(manifest.frames)} frames for {video_id}")
    if len(frame_manifest_paths) != len(video_ids):
        stage_status["frame_sampling"] = "failed_soft"
        summary["status"] = "failed_soft"
        _write_run_summary(workdir, summary, _video_id_next_steps_text(summary))
        return summary
    stage_status["frame_sampling"] = "done"

    readiness_path = workdir / "readiness_report.json"
    readiness = audit_data_readiness(
        media_inventory=media_inventory,
        annotations=annotations,
        frames_root=frames_root,
        video_ids=video_ids,
        segment_id=segment_id,
        min_expected_frames=frame_count,
    )
    write_readiness_report(readiness, readiness_path)
    artifacts["readiness_report"] = str(readiness_path)
    summary["data_checks"]["frame_readiness"] = _readiness_status_summary(
        readiness, readiness_path
    )
    if readiness["status"] != "ready_for_cloud":
        for video_id in video_ids:
            contact_sheet_path = contact_dir / f"{video_id}__{segment_id}.jpg"
            try:
                contact_sheet = render_segment_contact_sheet(
                    inventory_path=media_inventory,
                    annotations_path=annotations,
                    video_id=video_id,
                    segment_id=segment_id,
                    output=contact_sheet_path,
                )
                contact_sheet_text = str(contact_sheet)
                if contact_sheet_text not in artifacts["contact_sheets"]:
                    artifacts["contact_sheets"].append(contact_sheet_text)
            except Exception as exc:
                failures.append(f"{video_id}: could not render readiness contact sheet: {exc}")
        stage_status["data_readiness"] = "needs_human_review"
        summary["status"] = "needs_human_review"
        failures.extend(_readiness_failures(readiness))
        log("data readiness needs_human_review")
        _write_run_summary(workdir, summary, _video_id_next_steps_text(summary))
        return summary
    stage_status["data_readiness"] = "done"

    bundle_paths = [vggt_root / video_id / segment_id for video_id in video_ids]
    artifacts["bundles"] = [str(path) for path in bundle_paths]
    if cloud_return is not None:
        import_report_path = cloud_import_report or (workdir / "cloud_bundle_import.json")
        import_report = import_cloud_job_bundles(
            source=cloud_return,
            output_root=vggt_root,
            report_path=import_report_path,
            overwrite=overwrite_vggt,
            expected_bundles={(video_id, segment_id) for video_id in video_ids},
        )
        artifacts["cloud_import_report"] = str(import_report_path)
        if import_report["status"] != "done":
            stage_status["cloud_return_import"] = "failed_soft"
            failures.append(
                f"cloud return import {import_report['status']}: "
                f"{import_report['imported_count']}/{import_report['discovered_count']} bundles"
            )
            log("cloud return import failed_soft; continuing to expected bundle audit")
        else:
            stage_status["cloud_return_import"] = "done"
            log(f"imported cloud return: {cloud_return}")

    audit_path = workdir / "audit.json"
    audit = audit_three_clip_inputs(frame_manifest_paths, bundle_paths, audit_path)
    artifacts["audit_report"] = str(audit_path)
    if not audit["ready"]:
        cloud_package = create_cloud_job_package(frame_manifest_paths, cloud_job_output)
        cloud_package_zip = cloud_package / "cloud_vggt_job.zip"
        artifacts["cloud_job_package"] = str(cloud_package_zip)
        artifacts["cloud_job_package_bytes"] = cloud_package_zip.stat().st_size
        artifacts["cloud_job_package_sha256"] = _sha256_file(cloud_package_zip)
        stage_status["cloud_job"] = "done"
        stage_status["audit_three"] = "needs_vggt_bundle"
        summary["status"] = "needs_vggt_bundle"
        failures.extend(_audit_failures(audit))
        log("VGGT bundles missing or invalid; wrote cloud job package")
        _write_run_summary(workdir, summary, _video_id_next_steps_text(summary))
        return summary

    stage_status["audit_three"] = "done"
    review_dir = workdir / "review_artifacts"
    report_path = run_three_clip_review(
        frame_manifest_paths,
        bundle_paths,
        review_dir,
        smooth=smooth,
        generate_heatmaps=heatmaps,
        export_video=export_video,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifacts["review_report"] = str(report_path)
    artifacts["comparison_html"] = report["comparison_html"]
    artifacts["side_by_side_videos"] = [
        clip["side_by_side_video"]
        for clip in report.get("clips", [])
        if clip.get("side_by_side_video")
    ]
    artifacts["heatmap_manifests"] = [
        clip["heatmap_manifest"]
        for clip in report.get("clips", [])
        if clip.get("heatmap_manifest")
    ]
    _record_heatmap_diagnostics(report, heatmaps, stage_status, failures)
    stage_status["review_three"] = "done"
    summary["status"] = "done"
    log(f"wrote review report: {report_path}")
    _write_run_summary(workdir, summary, _video_id_next_steps_text(summary))
    return summary


def environment_snapshot() -> dict[str, Any]:
    packages = {}
    for package in [
        "fpv-vggt-visual-geometry-lab",
        "numpy",
        "opencv-python",
        "pandas",
        "pyarrow",
        "pydantic",
        "typer",
    ]:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = "not installed"
    return {
        "python": sys.version.split()[0],
        "os": platform.system(),
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "packages": packages,
        "ffmpeg": _ffmpeg_version(),
    }


def _ffmpeg_version() -> str:
    try:
        completed = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "not found"
    first_line = completed.stdout.splitlines()[0] if completed.stdout else ""
    return first_line or f"ffmpeg exited with code {completed.returncode}"


def _write_run_summary(workdir: Path, summary: dict[str, Any], next_steps: str) -> None:
    _record_run_folder_artifacts(workdir, summary)
    (workdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (workdir / "NEXT_STEPS.md").write_text(next_steps, encoding="utf-8")


def _record_run_folder_artifacts(workdir: Path, summary: dict[str, Any]) -> None:
    artifacts = summary.setdefault("artifacts", {})
    artifacts.setdefault("run_log", str(workdir / "run.log"))
    artifacts.setdefault("summary_json", str(workdir / "summary.json"))
    artifacts.setdefault("next_steps", str(workdir / "NEXT_STEPS.md"))


def _next_steps_text(summary: dict[str, Any]) -> str:
    artifacts = summary["artifacts"]
    return "\n".join(
        [
            "# Next Steps",
            "",
            "Synthetic Milestone 1 run completed.",
            "",
            "Safety reminders:",
            "- no geolocation",
            "- no meters",
            "- relative VGGT frame",
            "- local-only media",
            "",
            "Useful artifacts:",
            f"- Frame manifest: {artifacts['frame_manifest']}",
            f"- VGGT bundle: {artifacts['vggt_bundle']}",
            f"- Reconstruction summary: {artifacts['reconstruction_summary']}",
            "",
        ]
    )


def _audit_failures(audit: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if audit["clip_count"] < 3:
        failures.append("three-clip review requires at least three clips")
    for clip in audit["clips"]:
        for error in clip.get("errors", []):
            failures.append(error)
    return failures


def _record_heatmap_diagnostics(
    report: dict[str, Any],
    heatmaps: bool,
    stage_status: dict[str, str],
    failures: list[str],
) -> None:
    if not heatmaps:
        return

    degraded = False
    for index, clip in enumerate(report.get("clips", []), start=1):
        status = clip.get("heatmap_status") or "missing"
        if status == "done":
            continue
        degraded = True
        label = _clip_label(clip, index)
        message = f"{label}: heatmap_status={status}"
        error = _sanitize_diagnostic_text(clip.get("heatmap_error"))
        if error:
            message = f"{message}; heatmap_error={error}"
        failures.append(message)

    stage_status["heatmaps"] = "failed_soft" if degraded else "done"


def _clip_label(clip: dict[str, Any], index: int) -> str:
    video_id = clip.get("video_id")
    segment_id = clip.get("segment_id")
    if video_id and segment_id:
        return f"{video_id}/{segment_id}"
    if video_id:
        return str(video_id)
    if segment_id:
        return str(segment_id)
    return f"clip {index}"


def _sanitize_diagnostic_text(value: Any) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    if len(text) > 300:
        return f"{text[:297]}..."
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _readiness_failures(readiness: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for clip in readiness.get("clips", []):
        if clip.get("ready_for_cloud"):
            continue
        video_id = clip.get("video_id", "unknown-video")
        for check_name, check in clip.get("checks", {}).items():
            if check.get("status") == "pass":
                continue
            messages = check.get("messages") or [check.get("status", "needs_review")]
            for message in messages:
                failures.append(f"{video_id}: {check_name}: {message}")
    return failures


def _media_audit_failures(media_audit: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for clip in media_audit.get("clips", []):
        if clip.get("ready"):
            continue
        video_id = clip.get("video_id", "unknown-video")
        for check_name, check in clip.get("checks", {}).items():
            if check.get("status") == "pass":
                continue
            messages = check.get("messages") or [check.get("status", "needs_review")]
            for message in messages:
                failures.append(f"{video_id}: {check_name}: {message}")
    return failures


def _media_audit_status_summary(media_audit: dict[str, Any], report_path: Path) -> dict[str, Any]:
    return {
        "status": media_audit.get("status", "unknown"),
        "checked_count": media_audit.get("checked_count", 0),
        "ready_count": media_audit.get("ready_count", 0),
        "report": str(report_path),
    }


def _readiness_status_summary(readiness: dict[str, Any], report_path: Path) -> dict[str, Any]:
    return {
        "status": readiness.get("status", "unknown"),
        "clip_count": readiness.get("clip_count", 0),
        "ready_count": readiness.get("ready_count", 0),
        "report": str(report_path),
    }


def _review_next_steps_text(summary: dict[str, Any]) -> str:
    lines = [
        "# Next Steps",
        "",
        f"Status: {summary['status']}",
        "",
        "Safety reminders:",
        "- no geolocation",
        "- no meters",
        "- relative VGGT frame",
        "- local-only media",
        "",
        "Artifacts:",
    ]
    artifacts = summary.get("artifacts", {})
    if artifacts:
        lines.extend([f"- {name}: {path}" for name, path in artifacts.items()])
    else:
        lines.append("- none yet")

    if summary["status"] == "needs_vggt_bundle":
        inputs = summary["inputs"]
        lines.extend(
            [
                "",
                "Create a portable GPU job package:",
                "",
                "```powershell",
                _cloud_job_command(inputs["frame_manifests"]),
                "```",
                "",
                "Upload or copy the cloud job package to a CUDA GPU machine, unzip it,",
                "then run the packaged script from inside the extracted folder:",
                "",
                "```bash",
                "python run_vggt_job.py",
                "```",
                "",
                "Reruns skip clips that already have valid normalized bundles.",
                "Return `bundles.zip`, `cloud_summary.json`, and `cloud_run.log`,",
                "then validate the returned `bundles.zip` without installing it:",
                "",
                "```powershell",
                _cloud_import_dry_run_command(),
                "```",
                "",
                "If the dry run reports `done`, import and validate `bundles.zip` directly:",
                "",
                "```powershell",
                "fpv vggt import-cloud-job --source <returned-bundles.zip> --output-root data/vggt --report outputs/reviews/cloud_bundle_import.json",
                "```",
                "",
                "Then rerun:",
                "",
                "```powershell",
                _rerun_command(inputs["frame_manifests"], inputs["bundles"], inputs["smooth"]),
                "```",
                "",
            ]
        )
    elif summary["status"] == "done":
        lines.extend(
            [
                "",
                "Review the local HTML artifacts listed above. Treat diagnostics as",
                "relative VGGT-frame review signals, not physical truth claims.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Inspect `run.log`, `summary.json`, and `audit.json` if present, then",
                "rerun with corrected inputs.",
                "",
            ]
        )
    return "\n".join(lines)


def _cloud_job_command(frame_manifests: list[str]) -> str:
    parts = ["fpv vggt cloud-job"]
    for frame_manifest in frame_manifests:
        parts.append(f"  --frame-manifest {frame_manifest}")
    parts.append("  --output outputs/cloud_vggt_job")
    return " `\n".join(parts)


def _cloud_import_dry_run_command(
    output_root: str = "data/vggt",
    report: str = "outputs/reviews/cloud_bundle_import.json",
    expected_from_run: str | None = "outputs/reviews/three_clip_run/summary.json",
) -> str:
    parts = [
        "fpv vggt import-cloud-job",
        "  --source <returned-bundles.zip>",
        f"  --output-root {output_root}",
        f"  --report {_dry_run_report_path(report)}",
    ]
    if expected_from_run is not None:
        parts.append(f"  --expected-from-run {expected_from_run}")
    parts.append("  --dry-run")
    return " `\n".join(parts)


def _dry_run_report_path(report: str) -> str:
    report_path = Path(report)
    suffix = report_path.suffix or ".json"
    return str(report_path.with_name(f"{report_path.stem}_dry_run{suffix}"))


def _rerun_command(frame_manifests: list[str], bundles: list[str], smooth: bool) -> str:
    parts = ["fpv run --workdir outputs/reviews/three_clip_run"]
    for frame_manifest, bundle in zip(frame_manifests, bundles):
        parts.append(f"  --frame-manifest {frame_manifest} --bundle {bundle}")
    if smooth:
        parts.append("  --smooth")
    return " `\n".join(parts)


def _video_id_next_steps_text(summary: dict[str, Any]) -> str:
    inputs = summary["inputs"]
    lines = [
        "# Next Steps",
        "",
        f"Status: {summary['status']}",
        "",
        "Safety reminders:",
        "- no geolocation",
        "- no meters",
        "- relative VGGT frame",
        "- local-only media",
        "",
        "Artifacts:",
    ]
    artifacts = summary.get("artifacts", {})
    for name, value in artifacts.items():
        if value:
            if isinstance(value, list):
                lines.append(f"- {name}:")
                lines.extend(f"  - {item}" for item in value)
            else:
                lines.append(f"- {name}: {value}")
    if summary.get("stage_status", {}).get("local_media") == "needs_human_review":
        lines.extend(
            [
                "",
                "Fetch explicitly selected local media, then continue:",
                "",
                "```powershell",
                _video_id_rerun_command(inputs, fetch_media=True),
                "```",
                "",
            ]
        )
    elif summary.get("stage_status", {}).get("media_audit") == "needs_human_review":
        lines.extend(
            [
                "",
                "Review the media audit report, refresh any stale or corrupted local",
                "cache entries, then continue. The usual repair path is to rerun with",
                "`--fetch-media` for the explicitly selected clips:",
                "",
                "```powershell",
                _video_id_rerun_command(inputs, fetch_media=True),
                "```",
                "",
            ]
        )
    elif summary["status"] == "needs_human_review":
        lines.extend(
            [
                "",
                "Review the contact sheets, then accept segment bounds for each clip:",
                "",
                "```powershell",
                "fpv segment accept --annotations <annotations.jsonl> --video-id <video_id> --segment-id segment-001 --start <sec> --end <sec>",
                "```",
                "",
            ]
        )
    elif summary["status"] == "needs_vggt_bundle":
        lines.extend(
            [
                "",
                "Upload or copy the cloud job package listed above to a CUDA GPU machine,",
                "unzip it, then run the packaged script from inside the extracted folder:",
                "",
                "```bash",
                "python run_vggt_job.py",
                "```",
                "",
                "Reruns skip clips that already have valid normalized bundles.",
                "Return `bundles.zip`, `cloud_summary.json`, and `cloud_run.log`, then",
                "validate the returned ZIP without installing it:",
                "",
                "```powershell",
                _cloud_import_dry_run_command(
                    output_root=inputs["vggt_root"],
                    report=inputs["cloud_import_report"],
                ),
                "```",
                "",
                "If the dry run reports `done`, continue the same workflow with the returned ZIP:",
                "",
                "```powershell",
                _video_id_rerun_command(inputs, fetch_media=False, cloud_return="<returned-bundles.zip>"),
                "```",
                "",
            ]
        )
    elif summary["status"] == "done":
        lines.extend(
            [
                "",
                "Review the local HTML artifacts listed above. Treat diagnostics as",
                "relative VGGT-frame review signals, not physical truth claims.",
                "",
            ]
        )
    else:
        lines.extend(["", "Inspect `run.log` and `summary.json`, then rerun with corrected inputs.", ""])
    return "\n".join(lines)


def _video_id_rerun_command(
    inputs: dict[str, Any],
    fetch_media: bool,
    cloud_return: str | None = None,
) -> str:
    parts = [
        "fpv run",
        "  --workdir outputs/reviews/three_clip_run",
        f"  --catalog {inputs['catalog']}",
        f"  --media-dir {inputs['media_dir']}",
        f"  --media-inventory {inputs['media_inventory']}",
        f"  --annotations {inputs['annotations']}",
        f"  --frames-root {inputs['frames_root']}",
        f"  --vggt-root {inputs['vggt_root']}",
        f"  --cloud-job-output {inputs['cloud_job_output']}",
        f"  --frames {inputs['frame_count']}",
    ]
    if inputs.get("resized_long_edge") is not None:
        parts.append(f"  --resized-long-edge {inputs['resized_long_edge']}")
    if fetch_media:
        parts.append("  --fetch-media")
    if cloud_return is not None:
        parts.append(f"  --cloud-return {cloud_return}")
        parts.append(f"  --cloud-import-report {inputs['cloud_import_report']}")
    if inputs.get("smooth"):
        parts.append("  --smooth")
    for video_id in inputs["video_ids"]:
        parts.append(f"  --video-id {video_id}")
    return " `\n".join(parts)
