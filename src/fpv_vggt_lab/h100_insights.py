from __future__ import annotations

import csv
import html
import json
import math
import os
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SAFETY_BOUNDARY = [
    "Offline historical-media review only.",
    "VGGT coordinates are relative, scale ambiguous, and not meters.",
    "No geolocation, no map projection, no route/approach-corridor claims.",
    "No real speed, standoff, dive-angle, guidance, targeting, or next-maneuver claims.",
    "Treat scores as transparent diagnostics, not ground truth.",
]


def build_h100_insights(
    *,
    review_run: Path,
    output_dir: Path | None = None,
    import_output: Path | None = None,
    source_return: Path | None = None,
) -> dict[str, Any]:
    """Build a local-only H100 insight dashboard from a rendered review run."""
    review_run = review_run.resolve()
    run_report_path = review_run if review_run.is_file() else review_run / "run_report.json"
    if not run_report_path.exists():
        raise FileNotFoundError(f"missing review run report: {run_report_path}")
    review_dir = run_report_path.parent
    output_dir = (output_dir or review_dir / "insights").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_report = _load_json(run_report_path)
    rows = [_clip_row(clip, output_dir) for clip in run_report.get("clips", [])]
    if not rows:
        raise ValueError("review run report does not contain clips")

    summary = _summary(rows, review_dir, import_output, source_return)
    insight_payload = {"summary": summary, "clips": rows}

    json_path = output_dir / "h100_insights.json"
    json_path.write_text(json.dumps(insight_payload, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = _write_csv(output_dir / "h100_clip_metrics.csv", rows)
    verification_path = _write_verification_manifest(
        output_dir=output_dir,
        review_dir=review_dir,
        rows=rows,
        import_output=import_output,
        source_return=source_return,
    )
    html_path = output_dir / "index.html"
    html_path.write_text(
        _render_html(
            rows=rows,
            summary=summary,
            verification_path=verification_path,
            json_path=json_path,
            csv_path=csv_path,
        ),
        encoding="utf-8",
    )
    markdown_path = output_dir / "H100_INSIGHTS.md"
    markdown_path.write_text(
        _render_markdown(
            rows=rows,
            summary=summary,
            html_path=html_path,
            json_path=json_path,
            csv_path=csv_path,
            verification_path=verification_path,
            review_dir=review_dir,
            import_output=import_output,
        ),
        encoding="utf-8",
    )

    return {
        "status": "done",
        "clip_count": len(rows),
        "descriptor_safe_count": summary["safe_descriptor_count"],
        "descriptor_gated_count": summary["descriptor_gated_count"],
        "artifacts": {
            "dashboard": str(html_path),
            "markdown_report": str(markdown_path),
            "metrics_json": str(json_path),
            "metrics_csv": str(csv_path),
            "verification_manifest": str(verification_path),
        },
        "summary": summary,
    }


def _clip_row(clip: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    summary = _load_json(Path(clip["summary"]))
    frame_manifest = _load_json(Path(clip["frame_manifest"]))
    bundle = Path(clip["bundle"])
    metadata = _load_json(bundle / "metadata.json")
    cameras = np.load(bundle / "cameras.npz", allow_pickle=False)
    points_path = bundle / "points.npz"
    points = np.load(points_path, allow_pickle=False) if points_path.exists() else None

    centers = np.asarray(cameras["camera_centers"], dtype=np.float64)
    quats = np.asarray(cameras["quaternions_xyzw"], dtype=np.float64)
    valid_mask = np.asarray(cameras["valid_pose_mask"]).astype(bool)
    pose_conf = np.asarray(cameras["pose_confidence"], dtype=np.float64)
    finite_mask = valid_mask & np.isfinite(centers).all(axis=1)
    valid_centers = centers[finite_mask]
    steps = np.linalg.norm(np.diff(valid_centers, axis=0), axis=1) if len(valid_centers) >= 2 else np.array([])
    bbox = np.ptp(valid_centers, axis=0) if len(valid_centers) else np.array([0.0, 0.0, 0.0])
    quat_deltas = (
        np.asarray([_quat_angle(a, b) for a, b in zip(quats[:-1], quats[1:])], dtype=np.float64)
        if len(quats) >= 2
        else np.array([])
    )

    frame_qualities = [frame.get("quality", {}) for frame in frame_manifest.get("frames", [])]
    timestamps = [float(frame["timestamp_sec"]) for frame in frame_manifest.get("frames", [])]
    descriptors = summary.get("descriptors", {})
    reliability = summary.get("reliability", {})
    failure_flags = reliability.get("failure_flags", [])
    point_count = int(points["points"].shape[0]) if points is not None and "points" in points else 0
    review_html = Path(clip["review_html"]).resolve()

    row = {
        "video_id": clip["video_id"],
        "segment_id": clip["segment_id"],
        "review_html": Path(os.path.relpath(review_html, output_dir)).as_posix(),
        "summary_json": Path(os.path.relpath(Path(clip["summary"]).resolve(), output_dir)).as_posix(),
        "bundle_path": str(bundle.resolve()),
        "frame_manifest": str(Path(clip["frame_manifest"]).resolve()),
        "selected_tier_inferred": _infer_tier(frame_manifest),
        "source_tool": metadata.get("source_tool"),
        "frame_count": len(frame_manifest.get("frames", [])),
        "camera_count": int(centers.shape[0]),
        "valid_pose_count": int(valid_mask.sum()),
        "valid_pose_fraction": round(float(valid_mask.mean()), 6) if len(valid_mask) else 0.0,
        "duration_sec_sampled_window": round(max(timestamps) - min(timestamps), 4) if len(timestamps) >= 2 else 0.0,
        "reliability_score": _finite_float(reliability.get("score")),
        "reliability_label": reliability.get("label"),
        "safe_to_use_for_descriptors": bool(reliability.get("safe_to_use_for_descriptors")),
        "failure_flags": ";".join(failure_flags),
        "pose_jump_count": int(descriptors.get("pose_jump_count") or 0),
        "normalized_path_length": _finite_float(descriptors.get("normalized_path_length")),
        "displacement_ratio": _finite_float(descriptors.get("displacement_ratio")),
        "mean_turn_angle_rad": _finite_float(descriptors.get("mean_turn_angle_rad")),
        "max_turn_angle_rad": _finite_float(descriptors.get("max_turn_angle_rad")),
        "raw_relative_step_median": _rounded_median(steps),
        "raw_relative_step_p95": _rounded_percentile(steps, 0.95),
        "raw_relative_step_max": _rounded(float(steps.max())) if len(steps) else None,
        "raw_relative_path_sum": _rounded(float(steps.sum())) if len(steps) else None,
        "raw_relative_bbox_diag": _rounded(float(np.linalg.norm(bbox))),
        "raw_quat_delta_median_rad": _rounded_median(quat_deltas),
        "raw_quat_delta_p95_rad": _rounded_percentile(quat_deltas, 0.95),
        "pose_conf_mean": _rounded(float(np.nanmean(pose_conf))) if len(pose_conf) else None,
        "pose_conf_min": _rounded(float(np.nanmin(pose_conf))) if len(pose_conf) else None,
        "pose_conf_max": _rounded(float(np.nanmax(pose_conf))) if len(pose_conf) else None,
        "point_count": point_count,
        "points_has_rgb": bool(points is not None and "point_colors_rgb" in points),
        "points_has_confidence": bool(points is not None and "point_confidence" in points),
        "points_has_depth": bool(points is not None and "point_depth" in points),
        "frame_blur_mean": _quality_mean(frame_qualities, "blur_score"),
        "frame_brightness_mean": _quality_mean(frame_qualities, "brightness_mean"),
        "frame_contrast_mean": _quality_mean(frame_qualities, "contrast_std"),
        "smoothing_status": clip.get("smoothing_status"),
        "smoothing_error": clip.get("smoothing_error"),
        "heatmap_status": clip.get("heatmap_status"),
        "side_by_side_video_status": clip.get("side_by_side_video_status"),
    }
    row["review_priority_score"] = round(
        100.0 * (row["reliability_score"] or 0.0)
        + 5.0 * (1 if row["point_count"] >= 50000 else 0)
        + 5.0 * row["valid_pose_fraction"]
        - 8.0 * min(row["pose_jump_count"], 5),
        3,
    )
    return row


def _summary(
    rows: list[dict[str, Any]],
    review_dir: Path,
    import_output: Path | None,
    source_return: Path | None,
) -> dict[str, Any]:
    labels = Counter(row["reliability_label"] for row in rows)
    flags = Counter(
        flag for row in rows for flag in str(row.get("failure_flags") or "").split(";") if flag
    )
    smoothing = Counter(row["smoothing_status"] for row in rows)
    safe_rows = [row for row in rows if row["safe_to_use_for_descriptors"]]
    by_date: dict[str, dict[str, int]] = defaultdict(
        lambda: {"clips": 0, "good": 0, "mixed": 0, "descriptor_safe": 0, "pose_jump_gated": 0}
    )
    for row in rows:
        date = row["video_id"][:10] if len(row["video_id"]) >= 10 else "unknown"
        bucket = by_date[date]
        bucket["clips"] += 1
        if row["reliability_label"] == "good":
            bucket["good"] += 1
        if row["reliability_label"] == "mixed":
            bucket["mixed"] += 1
        if row["safe_to_use_for_descriptors"]:
            bucket["descriptor_safe"] += 1
        if row["pose_jump_count"] > 0:
            bucket["pose_jump_gated"] += 1

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_return": str(source_return.resolve()) if source_return else None,
        "review_run": str(review_dir),
        "import_run": str(import_output.resolve()) if import_output else None,
        "clip_count": len(rows),
        "selected_tier_inferred": dict(Counter(row["selected_tier_inferred"] for row in rows)),
        "frame_count": _min_median_max(row["frame_count"] for row in rows),
        "camera_count": _min_median_max(row["camera_count"] for row in rows),
        "point_count": _min_median_max(row["point_count"] for row in rows),
        "reliability_labels": dict(labels),
        "safe_descriptor_count": len(safe_rows),
        "descriptor_gated_count": len(rows) - len(safe_rows),
        "failure_flags": dict(flags),
        "smoothing_statuses": dict(smoothing),
        "point_fields": {
            "rgb": sum(row["points_has_rgb"] for row in rows),
            "confidence": sum(row["points_has_confidence"] for row in rows),
            "depth": sum(row["points_has_depth"] for row in rows),
        },
        "reliability_score": _min_median_max(
            row["reliability_score"] for row in rows if row["reliability_score"] is not None
        ),
        "pose_jump_count": {
            "zero": sum(row["pose_jump_count"] == 0 for row in rows),
            "one_or_more": sum(row["pose_jump_count"] > 0 for row in rows),
            "median": float(st.median(row["pose_jump_count"] for row in rows)),
            "max": max(row["pose_jump_count"] for row in rows),
        },
        "source_date_distribution": dict(sorted(by_date.items())),
        "safety": SAFETY_BOUNDARY,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_verification_manifest(
    *,
    output_dir: Path,
    review_dir: Path,
    rows: list[dict[str, Any]],
    import_output: Path | None,
    source_return: Path | None,
) -> Path:
    import_report = (import_output / "import_report.json") if import_output else None
    local_audit = review_dir / "local_only_audit.json"
    checks = {
        "review_run_report": {"status": "passed", "clip_count": len(rows)},
        "review_pages": {
            "status": "passed" if all((output_dir / row["review_html"]).exists() for row in rows) else "failed_soft",
            "count": sum((output_dir / row["review_html"]).exists() for row in rows),
        },
        "source_return": {
            "status": "passed" if source_return is None or source_return.exists() else "missing",
            "path": str(source_return.resolve()) if source_return else None,
        },
        "import_report": {
            "status": "passed" if import_report is not None and import_report.exists() else "not_provided",
            "path": str(import_report.resolve()) if import_report is not None and import_report.exists() else None,
        },
        "local_only_audit": {
            "status": _local_audit_status(local_audit),
            "path": str(local_audit.resolve()) if local_audit.exists() else None,
        },
    }
    status = "passed" if all(value["status"] in {"passed", "not_provided"} for value in checks.values()) else "failed_soft"
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "checks": checks,
        "safety_boundary": SAFETY_BOUNDARY,
    }
    path = output_dir / "verification_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _render_html(
    *,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    verification_path: Path,
    json_path: Path,
    csv_path: Path,
) -> str:
    labels = Counter(row["reliability_label"] for row in rows)
    flags = Counter(
        flag for row in rows for flag in str(row.get("failure_flags") or "").split(";") if flag
    )
    safe = [row for row in rows if row["safe_to_use_for_descriptors"]]
    gated = [row for row in rows if not row["safe_to_use_for_descriptors"]]
    best_review = sorted(
        rows,
        key=lambda row: (row["review_priority_score"], -row["pose_jump_count"], row["video_id"]),
        reverse=True,
    )[:20]
    best_safe = sorted(safe, key=lambda row: (row["reliability_score"] or 0, row["review_priority_score"]), reverse=True)[:20]
    needs_attention = sorted(
        rows,
        key=lambda row: (row["safe_to_use_for_descriptors"], -row["pose_jump_count"], row["reliability_score"] or 0),
    )[:25]
    most_path = sorted(safe, key=lambda row: row["normalized_path_length"] or -1, reverse=True)[:15]
    most_direct = sorted(safe, key=lambda row: row["displacement_ratio"] or -1, reverse=True)[:15]
    most_turning = sorted(safe, key=lambda row: row["mean_turn_angle_rad"] or -1, reverse=True)[:15]
    columns = _table_columns()

    cards = "".join(
        [
            _card("accepted clips", len(rows), "full H100 review set"),
            _card("review pages", len(rows), "interactive local HTML"),
            _card("frames per clip", summary["frame_count"]["median"], "high-detail tier"),
            _card("point cloud / clip", f"{summary['point_count']['median']:,}", "relative VGGT points"),
            _card("descriptor-safe", f"{len(safe)}/{len(rows)}", "pose-jump gate passed"),
            _card("gated for attention", len(gated), "reviewable but not descriptor-safe"),
        ]
    )
    label_bars = "".join(_bar(str(key), value, len(rows)) for key, value in labels.items())
    flag_bars = "".join(_bar(str(key), value, len(rows), "warn") for key, value in (flags or {"none": 0}).items())
    date_rows = "".join(
        "<tr>"
        f"<td>{_esc(date)}</td><td>{values['clips']}</td><td>{values['good']}</td>"
        f"<td>{values['mixed']}</td><td>{values['descriptor_safe']}</td>"
        f"<td>{values['pose_jump_gated']}</td></tr>"
        for date, values in summary["source_date_distribution"].items()
    )
    all_rows = "".join(_table_row(row, columns) for row in rows)
    safety = "".join(f"<li>{_esc(item)}</li>" for item in summary["safety"])

    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>H100 VGGT Insight Review</title>
<style>
:root {{ color-scheme: dark; --bg:#0f1113; --panel:#171b1f; --line:#2b343d; --text:#edf2f6; --muted:#a7b2bb; --accent:#55d6b5; --blue:#8db8ff; --warn:#f2c766; --bad:#ff8175; --ok:#7bd88f; }}
* {{ box-sizing:border-box; }} body {{ margin:0; font-family:Inter,Segoe UI,Arial,sans-serif; background:radial-gradient(circle at 18% 0%, #182026 0, #0f1113 34%); color:var(--text); }}
header {{ padding:30px 34px 20px; border-bottom:1px solid var(--line); background:linear-gradient(180deg,#151a1f,#101316); }}
h1 {{ margin:0; font-size:30px; letter-spacing:0; }} p {{ color:var(--muted); line-height:1.55; }} main {{ max-width:1480px; margin:0 auto; padding:24px 28px 60px; }}
.kicker {{ color:var(--accent); font-size:12px; text-transform:uppercase; letter-spacing:.08em; font-weight:700; }}
.cards {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:12px; margin:20px 0; }} .cards article {{ background:linear-gradient(180deg,var(--panel),#13171b); border:1px solid var(--line); border-radius:8px; padding:14px; min-height:98px; }} .cards span {{ display:block; color:var(--muted); font-size:12px; }} .cards strong {{ display:block; margin-top:9px; font-size:25px; }} .cards em {{ display:block; margin-top:8px; color:#7f8b95; font-style:normal; font-size:12px; }}
.grid,.findings {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }} section {{ margin:18px 0; background:rgba(23,27,31,.94); border:1px solid var(--line); border-radius:8px; padding:18px; }} h2 {{ margin:0 0 12px; font-size:18px; }} .section-head {{ display:flex; align-items:center; justify-content:space-between; gap:12px; }} .section-head span {{ color:var(--muted); font-size:12px; }}
.warning {{ border-color:#6e5a25; background:#1b1811; }} .warning strong {{ color:var(--warn); }} ul {{ margin:8px 0 0; padding-left:20px; color:var(--muted); }}
.bar-row {{ display:grid; grid-template-columns:210px 52px 1fr; gap:10px; align-items:center; margin:10px 0; color:var(--muted); }} .bar-row b {{ color:var(--text); text-align:right; }} .bar-row div {{ height:10px; background:#0b0e11; border-radius:999px; overflow:hidden; border:1px solid var(--line); }} .bar-row i {{ display:block; height:100%; background:linear-gradient(90deg,var(--accent),var(--blue)); }} .bar-row.warn i {{ background:linear-gradient(90deg,var(--warn),var(--bad)); }}
.toolbar {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:0 0 12px; }} input {{ background:#0c1013; color:var(--text); border:1px solid var(--line); border-radius:7px; padding:10px 12px; min-width:280px; }} .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:7px; }} table {{ width:100%; border-collapse:collapse; min-width:960px; }} th,td {{ padding:9px 10px; border-bottom:1px solid var(--line); text-align:left; font-size:13px; white-space:nowrap; }} th {{ color:#d8e0e6; background:#10151a; position:sticky; top:0; }} td {{ color:var(--muted); }} a {{ color:var(--accent); text-decoration:none; }} code {{ color:#d9e5ee; background:#0b0d0f; padding:2px 5px; border-radius:4px; }} .pill {{ display:inline-block; padding:3px 8px; border-radius:999px; font-size:12px; }} .pill.ok {{ color:#102017; background:var(--ok); }} .pill.warn {{ color:#241a08; background:var(--warn); }} .note {{ color:var(--muted); font-size:13px; }}
@media (max-width: 1050px) {{ .cards {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .grid,.findings {{ grid-template-columns:1fr; }} header {{ padding:22px 18px; }} main {{ padding:18px; }} }}
</style>
</head>
<body>
<header><div class="kicker">Offline relative visual geometry</div><h1>H100 VGGT Insight Review</h1><p>High-detail VGGT return rendered for {len(rows)} accepted clips. This page is for reconstruction triage, provenance, and relative-geometry review only.</p></header>
<main>
<section class="warning"><strong>Safety and interpretation boundary</strong><ul>{safety}</ul></section>
<div class="cards">{cards}</div>
<div class="findings"><section><h2>What The H100 Run Gives You</h2><p>Every accepted clip now has a high-detail VGGT bundle, a relative point cloud, an interactive review page, provenance warnings, and reliability gates. The strongest dataset-level insight is reconstruction triage: which clips are reviewable, which are descriptor-safe, and which need human inspection because relative pose jumps were detected.</p><p class="note">RGB point colors are absent when the return lacks them; confidence/depth availability is tracked per clip.</p></section><section><h2>Verification</h2><p><span class="pill ok">recorded</span> Artifact checks are captured in <a href="{_esc(verification_path.name)}">verification_manifest.json</a>.</p><p class="note">Run <code>fpv review audit-local-only</code> after moving or exporting real-media-derived artifacts.</p></section></div>
<div class="grid"><section><h2>Reliability Labels</h2>{label_bars}</section><section><h2>Descriptor Gates</h2>{flag_bars}<p class="note">Descriptor and smoothing gates are conservative. A clip can be visually useful and still be gated for derived path descriptors.</p></section></div>
<section><div class="section-head"><h2>Source-Date Distribution</h2><span>source metadata only</span></div><div class="table-wrap"><table><thead><tr><th>date</th><th>clips</th><th>good</th><th>mixed</th><th>descriptor-safe</th><th>pose-jump gated</th></tr></thead><tbody>{date_rows}</tbody></table></div></section>
<section><div class="section-head"><h2>Search All Clip Metrics</h2><span>{len(rows)} rows</span></div><div class="toolbar"><input id="clipSearch" placeholder="Search video id, label, flags..."></div><div class="table-wrap"><table id="allClips"><thead>{_table_head(columns)}</thead><tbody>{all_rows}</tbody></table></div></section>
{_table('Best Review Candidates', best_review, columns)}
{_table('Descriptor-Safe Relative Path Candidates', best_safe, columns)}
{_table('Needs Human Attention', needs_attention, columns)}
{_table('Longest Scale-Free Relative Paths', most_path, columns)}
{_table('Most Direct Scale-Free Relative Paths', most_direct, columns)}
{_table('Highest Relative Turning Diagnostics', most_turning, columns)}
<section><h2>Files</h2><p><code>{_esc(json_path.name)}</code> · <code>{_esc(csv_path.name)}</code> · <code>H100_INSIGHTS.md</code> · <code>{_esc(verification_path.name)}</code></p><p><a href="../index.html">Full review index</a> · <a href="../comparison.html">Comparison page</a></p></section>
</main>
<script>
const input = document.getElementById('clipSearch');
const clipRows = [...document.querySelectorAll('#allClips tbody tr')];
input.addEventListener('input', () => {{
  const q = input.value.toLowerCase().trim();
  clipRows.forEach(row => {{ row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none'; }});
}});
</script>
</body></html>'''


def _render_markdown(
    *,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    html_path: Path,
    json_path: Path,
    csv_path: Path,
    verification_path: Path,
    review_dir: Path,
    import_output: Path | None,
) -> str:
    labels = summary["reliability_labels"]
    best_review = sorted(
        rows,
        key=lambda row: (row["review_priority_score"], -row["pose_jump_count"], row["video_id"]),
        reverse=True,
    )[:15]
    attention = sorted(
        rows,
        key=lambda row: (row["safe_to_use_for_descriptors"], -row["pose_jump_count"], row["reliability_score"] or 0),
    )[:15]
    lines = [
        "# H100 VGGT Insight Report",
        "",
        f"Generated: {summary['created_at']}",
        "",
        "## Boundary",
        "",
        *[f"- {item}" for item in SAFETY_BOUNDARY],
        "",
        "## Headline Findings",
        "",
        f"- Imported/reviewed clips: {len(rows)}.",
        f"- Frames per clip: median {summary['frame_count']['median']}.",
        f"- Point cloud per clip: median {summary['point_count']['median']:,} relative points.",
        f"- Reliability labels: {labels}.",
        f"- Descriptor-safe clips: {summary['safe_descriptor_count']} / {len(rows)}.",
        f"- Pose-jump gated clips: {summary['pose_jump_count']['one_or_more']} / {len(rows)}.",
        "",
        "## Interpretation",
        "",
        "The H100 output is most useful as a reconstruction triage layer: every accepted clip has a local review page, while conservative gates separate visually reviewable clips from clips that should not feed derived path descriptors without human review.",
        "",
        "## Top Review Candidates",
        "",
    ]
    lines.extend(
        f"- {row['video_id']}: score={row['reliability_score']}, jumps={row['pose_jump_count']}, descriptor_safe={row['safe_to_use_for_descriptors']}"
        for row in best_review
    )
    lines.extend(["", "## Needs Human Attention", ""])
    lines.extend(
        f"- {row['video_id']}: score={row['reliability_score']}, flags={row['failure_flags'] or 'none'}, smoothing={row['smoothing_status']}"
        for row in attention
    )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Dashboard: {html_path}",
            f"- Metrics JSON: {json_path}",
            f"- Metrics CSV: {csv_path}",
            f"- Verification manifest: {verification_path}",
            f"- Full review index: {review_dir / 'index.html'}",
        ]
    )
    if import_output:
        lines.append(f"- Import index: {import_output / 'index.html'}")
    return "\n".join(lines) + "\n"


def _table_columns() -> list[tuple[str, str]]:
    return [
        ("video_id", "video id"),
        ("reliability_score", "score"),
        ("reliability_label", "label"),
        ("pose_jump_count", "jumps"),
        ("safe_to_use_for_descriptors", "descriptor gate"),
        ("normalized_path_length", "norm path"),
        ("displacement_ratio", "displacement"),
        ("point_count", "points"),
        ("review_html", "review"),
    ]


def _table(title: str, rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    return (
        f'<section><div class="section-head"><h2>{_esc(title)}</h2><span>{len(rows)} clips</span></div>'
        f'<div class="table-wrap"><table><thead>{_table_head(columns)}</thead>'
        f'<tbody>{"".join(_table_row(row, columns) for row in rows)}</tbody></table></div></section>'
    )


def _table_head(columns: list[tuple[str, str]]) -> str:
    return "<tr>" + "".join(f"<th>{_esc(label)}</th>" for _, label in columns) + "</tr>"


def _table_row(row: dict[str, Any], columns: list[tuple[str, str]]) -> str:
    return "<tr>" + "".join(f"<td>{_cell(row, key)}</td>" for key, _ in columns) + "</tr>"


def _cell(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if key == "review_html":
        return f'<a href="{_esc(value)}">open</a>'
    if key == "safe_to_use_for_descriptors":
        return '<span class="pill ok">yes</span>' if value else '<span class="pill warn">gated</span>'
    if key == "failure_flags":
        return _esc(value or "none")
    return _esc(value)


def _card(label: str, value: Any, note: str) -> str:
    return f"<article><span>{_esc(label)}</span><strong>{_esc(value)}</strong><em>{_esc(note)}</em></article>"


def _bar(label: str, value: int, total: int, class_name: str = "") -> str:
    width = 0 if total == 0 else 100.0 * value / total
    return (
        f'<div class="bar-row {class_name}"><span>{_esc(label)}</span><b>{value}</b>'
        f'<div><i style="width:{width:.2f}%"></i></div></div>'
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _local_audit_status(path: Path) -> str:
    if not path.exists():
        return "not_provided"
    try:
        return str(_load_json(path).get("status", "unknown"))
    except Exception:
        return "invalid"


def _infer_tier(frame_manifest: dict[str, Any]) -> str:
    count = len(frame_manifest.get("frames", []))
    long_edges = {frame.get("resized_long_edge") for frame in frame_manifest.get("frames", [])}
    if count >= 128:
        return "high_detail"
    if count >= 96:
        return "main"
    if count >= 32:
        return "scout"
    edge_text = ",".join(str(edge) for edge in sorted(long_edges, key=str))
    return f"unknown_{count}_frames_{edge_text}"


def _quat_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = q1 / max(float(np.linalg.norm(q1)), 1e-12)
    q2 = q2 / max(float(np.linalg.norm(q2)), 1e-12)
    dot = abs(float(np.dot(q1, q2)))
    return float(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def _rounded(value: float | None, digits: int = 6) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def _rounded_median(values: np.ndarray) -> float | None:
    return _rounded(float(np.median(values))) if len(values) else None


def _rounded_percentile(values: np.ndarray, q: float) -> float | None:
    if not len(values):
        return None
    vals = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not vals:
        return None
    position = (len(vals) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return _rounded(vals[low])
    return _rounded(vals[low] * (high - position) + vals[high] * (position - low))


def _quality_mean(qualities: list[dict[str, Any]], key: str) -> float | None:
    values = [_finite_float(item.get(key)) for item in qualities]
    finite = [value for value in values if value is not None]
    return round(sum(finite) / len(finite), 4) if finite else None


def _min_median_max(values) -> dict[str, float | int]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {"min": 0, "median": 0, "max": 0}
    median = st.median(finite)
    return {"min": round(min(finite), 4), "median": round(median, 4), "max": round(max(finite), 4)}


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))
