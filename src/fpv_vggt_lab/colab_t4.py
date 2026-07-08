from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cloud_job import create_cloud_job_package
from .pipeline import environment_snapshot
from .windows import (
    WINDOW_WARNINGS,
    propose_stable_windows,
    selected_window_frame_manifests,
)


def prepare_colab_t4_window_run(
    *,
    media_inventory: Path,
    annotations: Path,
    frames_root: Path,
    output_dir: Path,
    video_ids: list[str],
    segment_id: str = "segment-001",
    window_sec: float = 8.0,
    stride_sec: float = 3.0,
    target_fps: float = 2.0,
    max_frames: int = 20,
    candidate_limit: int = 2,
    resized_long_edge: int | None = 1024,
    eval_samples: int = 12,
    neighbor_radius: int = 2,
) -> dict[str, Any]:
    if not video_ids:
        raise ValueError("Colab T4 packaging requires at least one explicit --video-id")
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")

    frame_count = max(2, min(max_frames, int(round(window_sec * target_fps))))
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    log_path.write_text("", encoding="utf-8")

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")

    log("start Colab T4 stable-window package")
    summary: dict[str, Any] = {
        "schema_version": "colab-t4-window-run-v1",
        "status": "failed_soft",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "media_inventory": str(media_inventory),
            "annotations": str(annotations),
            "frames_root": str(frames_root),
            "output_dir": str(output_dir),
            "video_ids": video_ids,
            "segment_id": segment_id,
            "window_sec": window_sec,
            "stride_sec": stride_sec,
            "target_fps": target_fps,
            "max_frames": max_frames,
            "actual_frames_per_window": frame_count,
            "candidate_limit": candidate_limit,
            "resized_long_edge": resized_long_edge,
            "eval_samples": eval_samples,
            "neighbor_radius": neighbor_radius,
        },
        "stage_status": {
            "window_proposal": "pending",
            "cloud_package": "pending",
        },
        "warnings": WINDOW_WARNINGS
        + [
            "Colab T4 memory is limited; run one or a few clips per package",
            "2 FPS stable windows are preferred over full edited segments",
        ],
        "artifacts": {
            "run_log": str(log_path),
            "summary_json": str(output_dir / "summary.json"),
            "next_steps": str(output_dir / "NEXT_STEPS.md"),
        },
        "environment": environment_snapshot(),
        "failures": [],
    }

    try:
        proposal_dir = output_dir / "stable_windows"
        proposal = propose_stable_windows(
            media_inventory=media_inventory,
            annotations=annotations,
            output_dir=proposal_dir,
            frames_root=frames_root,
            video_ids=video_ids,
            segment_id=segment_id,
            window_sec=window_sec,
            stride_sec=stride_sec,
            candidate_limit=candidate_limit,
            frame_count=frame_count,
            resized_long_edge=resized_long_edge,
            eval_samples=eval_samples,
            neighbor_radius=neighbor_radius,
        )
        summary["artifacts"]["proposal_dashboard"] = proposal["artifacts"]["dashboard"]
        summary["artifacts"]["proposal_summary"] = proposal["artifacts"]["summary_json"]
        summary["stage_status"]["window_proposal"] = proposal["status"]
        if proposal["status"] != "done":
            summary["status"] = proposal["status"]
            summary["failures"].extend(proposal.get("failures", []))
            _write_summary(output_dir, summary)
            return summary

        frame_manifests = selected_window_frame_manifests(proposal_dir)
        (output_dir / "frame_manifests.txt").write_text(
            "\n".join(str(path) for path in frame_manifests) + "\n",
            encoding="utf-8",
        )
        summary["artifacts"]["frame_manifest_list"] = str(output_dir / "frame_manifests.txt")
        log(f"selected {len(frame_manifests)} window frame manifests")

        package_dir = output_dir / "cloud_vggt_job"
        create_cloud_job_package(frame_manifests, package_dir)
        zip_path = package_dir / "cloud_vggt_job.zip"
        summary["stage_status"]["cloud_package"] = "done"
        summary["status"] = "done"
        summary["window_count"] = len(frame_manifests)
        summary["artifacts"].update(
            {
                "cloud_job_dir": str(package_dir),
                "cloud_job_zip": str(zip_path),
                "job_manifest": str(package_dir / "job_manifest.json"),
            }
        )
        log(f"wrote Colab T4 package: {zip_path}")
    except Exception as exc:
        summary["stage_status"].setdefault("window_proposal", "failed_soft")
        summary["stage_status"]["cloud_package"] = "failed_soft"
        summary["status"] = "failed_soft"
        summary["failures"].append(str(exc))
        log(f"failed_soft: {exc}")

    _write_summary(output_dir, summary)
    return summary


def _write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "NEXT_STEPS.md").write_text(_next_steps(summary), encoding="utf-8")


def _next_steps(summary: dict[str, Any]) -> str:
    artifacts = summary.get("artifacts", {})
    lines = [
        "# Colab T4 Next Steps",
        "",
        f"Status: `{summary['status']}`",
        "",
        "Warning block:",
        "- no geolocation",
        "- no meters",
        "- relative VGGT frame",
        "- local-only media",
        "",
    ]
    if summary["status"] != "done":
        lines.extend(
            [
                "The Colab package was not completed. Inspect `run.log` and `summary.json`,",
                "then rerun with fewer clips, fewer windows, or a smaller resize.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "Upload this ZIP to Colab T4:",
            "",
            "```text",
            artifacts["cloud_job_zip"],
            "```",
            "",
            "In Colab, use `docs/colab_vggt_t4_runner.ipynb` or",
            "`docs/colab_drive_vggt_runner.ipynb`, then download:",
            "",
            "- `bundles.zip`",
            "- `cloud_summary.json`",
            "- `cloud_run.log`",
            "",
            "Import the returned `bundles.zip` locally:",
            "",
            "```powershell",
            "fpv vggt import-cloud-job --source <returned-bundles.zip> --output-root data/vggt --report outputs/reviews/colab_t4_window_import.json",
            "```",
            "",
            "Then rank the window bundles:",
            "",
            "```powershell",
            f"fpv windows rank --proposal-run {artifacts['proposal_summary']} --vggt-root data/vggt --output-dir outputs/reviews/colab_t4_window_ranked --stitched-review",
            "```",
            "",
        ]
    )
    return "\n".join(lines)
