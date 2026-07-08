from __future__ import annotations

from datetime import datetime, timezone

import json
from pathlib import Path

import typer

from .catalog import DEFAULT_BANNER_AUDIT_URL, DEFAULT_GEO_RECORDS_URL, DEFAULT_MANIFEST_URL, DEFAULT_README_URL, DEFAULT_WEBSITE_URL, shortlist_catalog, sync_catalog
from .cloud_job import create_cloud_job_package
from .colab_t4 import prepare_colab_t4_window_run
from .export_video import export_side_by_side_mp4
from .frames import sample_accepted_segment_frames, sample_video_frames
from .geometry import summarize_bundle
from .glb_export import export_hf_style_glb
from .heatmaps import generate_heatmaps_from_manifest
from .h100_insights import build_h100_insights
from .h100_package import (
    audit_runpod_4090_hf_launch_kit,
    runpod_4090_hf_profile,
    smoke_runpod_job_package,
    write_runpod_4090_hf_profile,
    write_runpod_4090_hf_start_here,
    inspect_runpod_job_package,
    prepare_h100_run,
    write_runpod_launch_manifest,
)
from .h100_return import (
    import_h100_return,
    inspect_h100_return,
    validate_h100_return,
    verify_h100_return_against_launch,
    write_h100_optional_method_report,
)
from .local_only import audit_local_only_artifacts
from .method_contract import (
    validate_method_matrix,
    validate_return_layout,
    write_method_contract_files,
)
from .media import audit_media_inventory, fetch_media, write_media_audit_report
from .pipeline import run_review_pipeline, run_synthetic_pipeline, run_video_id_pipeline
from .readiness import audit_data_readiness, write_readiness_report
from .review import audit_three_clip_inputs, run_three_clip_review
from .schemas import model_to_dict
from .segment_qa import run_segment_qa
from .segments import (
    accept_segment,
    edit_segment,
    propose_segment,
    read_annotations,
    reject_segment,
    render_segment_contact_sheet,
)
from .smoothing import smooth_relative_poses
from .synthetic import create_synthetic_video
from .vggt import (
    create_mock_bundle,
    expected_bundles_from_run_summary,
    export_vggt_request,
    format_bundle_inspection,
    import_cloud_job_bundles,
    import_converted_bundle,
    import_predictions_npz,
    load_bundle,
    validate_bundle,
)
from .vggt_installed import run_installed_vggt
from .viz import render_comparison_html, render_review_html
from .windows import (
    propose_stable_windows,
    rank_window_bundles,
    selected_window_frame_manifests,
)


app = typer.Typer(no_args_is_help=True, help="Offline FPV visual-geometry tools.")
catalog_app = typer.Typer(no_args_is_help=True, help="Dataset catalog helpers.")
media_app = typer.Typer(no_args_is_help=True, help="Local media cache helpers.")
segment_app = typer.Typer(no_args_is_help=True, help="Segment proposal and review.")
synthetic_app = typer.Typer(no_args_is_help=True, help="Synthetic media helpers.")
frames_app = typer.Typer(no_args_is_help=True, help="Frame extraction helpers.")
vggt_app = typer.Typer(no_args_is_help=True, help="VGGT bundle helpers.")
reconstruct_app = typer.Typer(no_args_is_help=True, help="Reconstruction summaries.")
heatmap_app = typer.Typer(no_args_is_help=True, help="Image-space heatmap diagnostics.")
viz_app = typer.Typer(no_args_is_help=True, help="Local HTML review artifacts.")
compare_app = typer.Typer(no_args_is_help=True, help="Comparison artifacts.")
smooth_app = typer.Typer(no_args_is_help=True, help="Relative pose smoothing diagnostics.")
review_app = typer.Typer(no_args_is_help=True, help="End-to-end local review runners.")
h100_app = typer.Typer(no_args_is_help=True, help="RunPod GPU full-dataset workflow helpers (legacy h100 namespace).")
methods_app = typer.Typer(no_args_is_help=True, help="Expanded reconstruction method contracts.")
windows_app = typer.Typer(
    no_args_is_help=True,
    help="Stable-window proposal, quality-aware sampling, and VGGT window ranking.",
)

app.add_typer(catalog_app, name="catalog")
app.add_typer(media_app, name="media")
app.add_typer(segment_app, name="segment")
app.add_typer(synthetic_app, name="synthetic-video")
app.add_typer(frames_app, name="frames")
app.add_typer(vggt_app, name="vggt")
app.add_typer(reconstruct_app, name="reconstruct")
app.add_typer(heatmap_app, name="heatmap")
app.add_typer(viz_app, name="viz")
app.add_typer(compare_app, name="compare")
app.add_typer(smooth_app, name="smooth")
app.add_typer(review_app, name="review")
app.add_typer(methods_app, name="methods")
app.add_typer(h100_app, name="h100")
app.add_typer(h100_app, name="runpod")
app.add_typer(windows_app, name="windows")


@catalog_app.command("sync")
def catalog_sync(
    readme: str = typer.Option(DEFAULT_README_URL, "--readme", help="README source path or URL."),
    manifest: str | None = typer.Option(
        DEFAULT_MANIFEST_URL, "--manifest", help="Manifest TSV source path or URL."
    ),
    banner_audit: str | None = typer.Option(
        None,
        "--banner-audit",
        help=f"Optional banner/title audit TSV source. Current upstream default: {DEFAULT_BANNER_AUDIT_URL}",
    ),
    geo_records: str | None = typer.Option(
        None,
        "--geo-records",
        help=f"Optional upstream geo CSV source, stored as source metadata only. Current upstream default: {DEFAULT_GEO_RECORDS_URL}",
    ),
    website: str | None = typer.Option(
        None,
        "--website",
        help=f"Optional public website snapshot source for provenance. Current upstream default: {DEFAULT_WEBSITE_URL}",
    ),
    fallback_catalog: Path | None = typer.Option(
        None,
        "--fallback-catalog",
        help="Optional previous local catalog used only to reconcile missing storage rows when upstream declares more MP4s than its current metadata exposes.",
    ),
    output: Path = typer.Option(Path("data/catalog"), "--output", "-o", help="Catalog output directory."),
) -> None:
    catalog_path, snapshot_path = sync_catalog(
        readme_source=readme,
        manifest_source=manifest,
        banner_audit_source=banner_audit,
        geo_records_source=geo_records,
        website_source=website,
        fallback_catalog_source=fallback_catalog,
        output=output,
    )
    typer.echo(f"wrote catalog: {catalog_path}")
    typer.echo(f"wrote snapshot: {snapshot_path}")


@catalog_app.command("show")
def catalog_show(
    catalog: Path = typer.Option(..., "--catalog", help="Catalog parquet path."),
    limit: int = typer.Option(10, "--limit", min=1, help="Rows to print."),
) -> None:
    typer.echo(shortlist_catalog(catalog, limit=limit).to_string(index=False))


@catalog_app.command("shortlist")
def catalog_shortlist(
    catalog: Path = typer.Option(..., "--catalog", help="Catalog parquet path."),
    limit: int = typer.Option(10, "--limit", min=1, help="Rows to print."),
) -> None:
    typer.echo(shortlist_catalog(catalog, limit=limit).to_string(index=False))


@media_app.command("fetch")
def media_fetch(
    catalog: Path = typer.Option(..., "--catalog", help="Catalog parquet path."),
    media_dir: Path = typer.Option(Path("data/media"), "--media-dir", help="Local media directory."),
    inventory: Path = typer.Option(
        Path("data/media/media_inventory.parquet"),
        "--inventory",
        help="Media inventory parquet path.",
    ),
    video_id: str | None = typer.Option(None, "--video-id", help="Explicit video id to fetch."),
) -> None:
    if not video_id:
        typer.echo("missing required option: --video-id")
        raise typer.Exit(1)
    record = fetch_media(
        catalog_path=catalog,
        media_dir=media_dir,
        inventory_path=inventory,
        video_id=video_id,
    )
    typer.echo(f"fetched {record.video_id}: {record.local_path}")


@media_app.command("audit")
def media_audit(
    inventory: Path = typer.Option(
        Path("data/media/media_inventory.parquet"),
        "--inventory",
        help="Media inventory parquet path.",
    ),
    video_id: list[str] = typer.Option(
        [], "--video-id", help="Video id to audit. Repeat for multiple clips."
    ),
    report: Path = typer.Option(..., "--report", help="Output media audit JSON."),
) -> None:
    result = audit_media_inventory(inventory, video_ids=video_id or None)
    write_media_audit_report(result, report)
    typer.echo(
        f"media audit {result['status']}: "
        f"{result['ready_count']}/{result['checked_count']} clips ready"
    )
    typer.echo(f"report: {report}")
    if result["status"] != "ready":
        for clip in result["clips"]:
            if clip["ready"]:
                continue
            typer.echo(f"{clip['video_id']}: needs_review")
            for name, check in clip["checks"].items():
                if check["status"] != "pass":
                    messages = "; ".join(check["messages"]) or check["status"]
                    typer.echo(f"  - {name}: {messages}")
        raise typer.Exit(1)


@segment_app.command("propose")
def segment_propose(
    media_inventory: Path = typer.Option(..., "--media-inventory", help="Media inventory parquet."),
    annotations: Path = typer.Option(
        Path("data/annotations/segments.jsonl"),
        "--annotations",
        help="Segment annotations JSONL.",
    ),
    video_id: str = typer.Option(..., "--video-id", help="Video id."),
    segment_id: str = typer.Option("segment-001", "--segment-id", help="Segment id."),
) -> None:
    annotation = propose_segment(media_inventory, annotations, video_id, segment_id)
    typer.echo(f"proposed {annotation.video_id}/{annotation.segment_id}: {annotation.start_sec}-{annotation.end_sec}")


@segment_app.command("accept")
def segment_accept(
    annotations: Path = typer.Option(
        Path("data/annotations/segments.jsonl"),
        "--annotations",
        help="Segment annotations JSONL.",
    ),
    video_id: str = typer.Option(..., "--video-id", help="Video id."),
    segment_id: str = typer.Option(..., "--segment-id", help="Segment id."),
    start: float = typer.Option(..., "--start", min=0, help="Accepted start seconds."),
    end: float = typer.Option(..., "--end", min=0, help="Accepted end seconds."),
    notes: str = typer.Option("", "--notes", help="Review notes."),
) -> None:
    annotation = accept_segment(annotations, video_id, segment_id, start, end, notes)
    typer.echo(f"accepted {annotation.video_id}/{annotation.segment_id}: {annotation.start_sec}-{annotation.end_sec}")


@segment_app.command("edit")
def segment_edit(
    annotations: Path = typer.Option(
        Path("data/annotations/segments.jsonl"),
        "--annotations",
        help="Segment annotations JSONL.",
    ),
    video_id: str = typer.Option(..., "--video-id", help="Video id."),
    segment_id: str = typer.Option(..., "--segment-id", help="Segment id."),
    start: float | None = typer.Option(None, "--start", min=0, help="Edited start seconds."),
    end: float | None = typer.Option(None, "--end", min=0, help="Edited end seconds."),
    notes: str | None = typer.Option(None, "--notes", help="Edited review notes."),
    confidence: str | None = typer.Option(
        None,
        "--confidence",
        help="Edited annotation confidence: low, medium, or high.",
    ),
) -> None:
    try:
        annotation = edit_segment(
            annotations_path=annotations,
            video_id=video_id,
            segment_id=segment_id,
            start_sec=start,
            end_sec=end,
            notes=notes,
            confidence=confidence,
        )
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(
        f"edited {annotation.video_id}/{annotation.segment_id}: "
        f"{annotation.status} {annotation.start_sec}-{annotation.end_sec}"
    )


@segment_app.command("reject")
def segment_reject(
    annotations: Path = typer.Option(Path("data/annotations/segments.jsonl"), "--annotations"),
    video_id: str = typer.Option(..., "--video-id"),
    segment_id: str = typer.Option(..., "--segment-id"),
) -> None:
    annotation = reject_segment(annotations, video_id, segment_id)
    typer.echo(f"rejected {annotation.video_id}/{annotation.segment_id}")


@segment_app.command("list")
def segment_list(
    annotations: Path = typer.Option(Path("data/annotations/segments.jsonl"), "--annotations"),
) -> None:
    for annotation in read_annotations(annotations):
        typer.echo(
            f"{annotation.video_id}\t{annotation.segment_id}\t{annotation.status}\t"
            f"{annotation.start_sec:.3f}\t{annotation.end_sec:.3f}"
        )


@segment_app.command("contact-sheet")
def segment_contact_sheet(
    media_inventory: Path = typer.Option(..., "--media-inventory", help="Media inventory parquet."),
    annotations: Path = typer.Option(..., "--annotations", help="Segment annotations JSONL."),
    video_id: str = typer.Option(..., "--video-id", help="Video id."),
    segment_id: str = typer.Option(..., "--segment-id", help="Segment id."),
    output: Path = typer.Option(..., "--output", "-o", help="Output local contact sheet image."),
    samples: int = typer.Option(12, "--samples", min=1, help="Number of sampled thumbnails."),
) -> None:
    try:
        path = render_segment_contact_sheet(
            inventory_path=media_inventory,
            annotations_path=annotations,
            video_id=video_id,
            segment_id=segment_id,
            output=output,
            samples=samples,
        )
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(f"wrote local-only segment contact sheet: {path}")


@segment_app.command("qa")
def segment_qa(
    media_inventory: Path = typer.Option(
        Path("data/media/media_inventory.parquet"),
        "--media-inventory",
        help="Local media inventory parquet.",
    ),
    annotations: Path = typer.Option(
        Path("data/annotations/segments.jsonl"),
        "--annotations",
        help="Segment annotations JSONL.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/segment_qa/latest"),
        "--output-dir",
        help="Output QA run folder.",
    ),
    frames_root: Path | None = typer.Option(
        Path("data/frames"),
        "--frames-root",
        help="Optional frame root for first/last sampled-frame sheets.",
    ),
    video_id: list[str] = typer.Option(
        [], "--video-id", help="Video id to QA. Omit to process all local inventory rows."
    ),
    segment_id: str = typer.Option("segment-001", "--segment-id", help="Segment id to compare/write."),
    probe_samples: int = typer.Option(64, "--probe-samples", min=6, help="Decoded frames used for edit-boundary detection."),
    contact_samples: int = typer.Option(16, "--contact-samples", min=4, help="Source-video contact-sheet samples."),
    boundary_samples: int = typer.Option(12, "--boundary-samples", min=4, help="Start/end boundary contact-sheet samples."),
    write_proposals: bool = typer.Option(
        False,
        "--write-proposals/--no-write-proposals",
        help="Write proposed annotations only when no accepted segment already exists.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Reuse existing per-clip segment_qa.json files in the output directory.",
    ),
) -> None:
    try:
        summary = run_segment_qa(
            media_inventory=media_inventory,
            annotations=annotations,
            output_dir=output_dir,
            frames_root=frames_root,
            video_ids=video_id or None,
            segment_id=segment_id,
            probe_samples=probe_samples,
            contact_samples=contact_samples,
            boundary_samples=boundary_samples,
            write_proposals=write_proposals,
            resume=resume,
        )
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(
        f"segment QA {summary['status']}: "
        f"{summary['ready_count']} ready, {summary['needs_review_count']} need review, "
        f"{summary['failed_count']} failed"
    )
    typer.echo(f"dashboard: {summary['artifacts']['dashboard']}")
    typer.echo(f"next steps: {summary['artifacts']['next_steps']}")
    if summary["status"] != "ready_for_vggt":
        raise typer.Exit(1)


@synthetic_app.command("create")
def synthetic_create(
    output: Path = typer.Option(..., "--output", "-o", help="Output MP4 path."),
    frames: int = typer.Option(48, "--frames", min=1, help="Number of frames."),
    width: int = typer.Option(320, "--width", min=16, help="Frame width."),
    height: int = typer.Option(180, "--height", min=16, help="Frame height."),
    fps: float = typer.Option(12.0, "--fps", min=0.1, help="Frames per second."),
) -> None:
    path = create_synthetic_video(output=output, frames=frames, width=width, height=height, fps=fps)
    typer.echo(f"created synthetic video: {path}")


@frames_app.command("sample")
def frames_sample(
    video: Path = typer.Option(..., "--video", help="Input video path."),
    output: Path = typer.Option(..., "--output", "-o", help="Output frame directory."),
    count: int = typer.Option(64, "--count", min=1, help="Number of frames."),
    video_id: str = typer.Option("synthetic-video", "--video-id", help="Video id."),
    segment_id: str = typer.Option("segment-001", "--segment-id", help="Segment id."),
    resized_long_edge: int | None = typer.Option(
        None, "--resized-long-edge", min=1, help="Optional max long edge."
    ),
) -> None:
    manifest = sample_video_frames(
        video=video,
        output=output,
        count=count,
        video_id=video_id,
        segment_id=segment_id,
        resized_long_edge=resized_long_edge,
    )
    typer.echo(f"sampled {len(manifest.frames)} frames: {output / 'frames.json'}")


@frames_app.command("sample-accepted")
def frames_sample_accepted(
    media_inventory: Path = typer.Option(..., "--media-inventory", help="Media inventory parquet."),
    annotations: Path = typer.Option(..., "--annotations", help="Segment annotations JSONL."),
    video_id: str = typer.Option(..., "--video-id", help="Video id."),
    segment_id: str = typer.Option(..., "--segment-id", help="Accepted segment id."),
    output: Path = typer.Option(..., "--output", "-o", help="Output frame directory."),
    count: int = typer.Option(64, "--count", min=1, help="Number of frames."),
    resized_long_edge: int | None = typer.Option(
        None, "--resized-long-edge", min=1, help="Optional max long edge."
    ),
) -> None:
    try:
        manifest = sample_accepted_segment_frames(
            media_inventory=media_inventory,
            annotations=annotations,
            video_id=video_id,
            segment_id=segment_id,
            output=output,
            count=count,
            resized_long_edge=resized_long_edge,
        )
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(f"sampled {len(manifest.frames)} accepted segment frames: {output / 'frames.json'}")


@vggt_app.command("mock")
def vggt_mock(
    frame_manifest: Path = typer.Option(
        ..., "--frame-manifest", help="Frame manifest JSON."
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output bundle directory."),
) -> None:
    bundle = create_mock_bundle(frame_manifest, output)
    typer.echo(f"created mocked VGGT bundle: {bundle}")


@vggt_app.command("export-request")
def vggt_export_request(
    frame_manifest: Path = typer.Option(..., "--frame-manifest", help="Frame manifest JSON."),
    output: Path = typer.Option(..., "--output", "-o", help="Output request package directory."),
) -> None:
    path = export_vggt_request(frame_manifest, output)
    typer.echo(f"wrote VGGT request package: {path}")


@vggt_app.command("import-bundle")
def vggt_import_bundle(
    source: Path = typer.Option(..., "--source", help="Converted bundle source directory."),
    frame_manifest: Path = typer.Option(..., "--frame-manifest", help="Frame manifest JSON."),
    output: Path = typer.Option(..., "--output", "-o", help="Normalized output bundle."),
    source_tool: str = typer.Option(..., "--source-tool", help="VGGT source tool."),
    source_url_or_repo: str | None = typer.Option(None, "--source-url-or-repo", help="Public source URL/repo."),
    source_commit_or_version: str | None = typer.Option(
        None, "--source-commit-or-version", help="Source version or commit."
    ),
    export_notes: str | None = typer.Option(None, "--export-notes", help="Conversion notes."),
) -> None:
    try:
        path = import_converted_bundle(
            source=source,
            frame_manifest_path=frame_manifest,
            output=output,
            source_tool=source_tool,
            source_url_or_repo=source_url_or_repo,
            source_commit_or_version=source_commit_or_version,
            export_notes=export_notes,
        )
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(f"imported VGGT bundle: {path}")


@vggt_app.command("import-predictions")
def vggt_import_predictions(
    predictions: Path = typer.Option(..., "--predictions", help="VGGT predictions.npz path."),
    frame_manifest: Path = typer.Option(..., "--frame-manifest", help="Frame manifest JSON."),
    output: Path = typer.Option(..., "--output", "-o", help="Normalized output bundle."),
    source_tool: str = typer.Option("huggingface-space", "--source-tool", help="VGGT source tool."),
    source_url_or_repo: str | None = typer.Option(None, "--source-url-or-repo", help="Public source URL/repo."),
    source_commit_or_version: str | None = typer.Option(
        None, "--source-commit-or-version", help="Source version or commit."
    ),
    export_notes: str | None = typer.Option(None, "--export-notes", help="Conversion notes."),
    max_points: int = typer.Option(50000, "--max-points", min=1, help="Maximum point-cloud points."),
) -> None:
    try:
        path = import_predictions_npz(
            predictions_path=predictions,
            frame_manifest_path=frame_manifest,
            output=output,
            source_tool=source_tool,
            source_url_or_repo=source_url_or_repo,
            source_commit_or_version=source_commit_or_version,
            export_notes=export_notes,
            max_points=max_points,
        )
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(f"imported VGGT predictions bundle: {path}")


@vggt_app.command("run-installed")
def vggt_run_installed(
    frame_manifest: Path = typer.Option(..., "--frame-manifest", help="Frame manifest JSON."),
    predictions_output: Path = typer.Option(..., "--predictions-output", help="Output predictions.npz."),
    bundle_output: Path = typer.Option(..., "--bundle-output", help="Normalized output bundle."),
    source_url_or_repo: str = typer.Option(
        "https://github.com/facebookresearch/vggt",
        "--source-url-or-repo",
        help="VGGT source URL/repo.",
    ),
    source_commit_or_version: str | None = typer.Option(
        None, "--source-commit-or-version", help="VGGT version or commit."
    ),
    allow_cpu: bool = typer.Option(False, "--allow-cpu", help="Allow CPU inference despite model size."),
    max_points: int = typer.Option(50000, "--max-points", min=1, help="Maximum point-cloud points."),
) -> None:
    try:
        path = run_installed_vggt(
            frame_manifest_path=frame_manifest,
            predictions_output=predictions_output,
            bundle_output=bundle_output,
            source_url_or_repo=source_url_or_repo,
            source_commit_or_version=source_commit_or_version,
            allow_cpu=allow_cpu,
            max_points=max_points,
        )
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(f"ran installed VGGT and wrote bundle: {path}")


@vggt_app.command("cloud-job")
def vggt_cloud_job(
    frame_manifest: list[Path] = typer.Option(
        ..., "--frame-manifest", help="Frame manifest JSON. Repeat for each clip."
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output cloud job package directory."),
) -> None:
    try:
        path = create_cloud_job_package(frame_manifest, output)
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    zip_path = path / "cloud_vggt_job.zip"
    typer.echo(f"wrote cloud VGGT job package: {path}")
    typer.echo(f"upload ZIP: {zip_path}")
    typer.echo("on CUDA GPU: extract with `python -m zipfile -e cloud_vggt_job.zip .`, then run `python run_vggt_job.py`")
    typer.echo("return: bundles.zip, cloud_summary.json, cloud_run.log")


@vggt_app.command("import-cloud-job")
def vggt_import_cloud_job(
    source: Path = typer.Option(..., "--source", help="Returned bundles.zip or cloud job bundles directory."),
    output_root: Path = typer.Option(Path("data/vggt"), "--output-root", help="Local VGGT bundle root."),
    report: Path = typer.Option(..., "--report", help="Output import report JSON."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace existing local bundle directories."),
    expected_from_run: Path | None = typer.Option(
        None,
        "--expected-from-run",
        help="Run summary.json used to restrict returned bundles to this run's expected video/segment ids.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate returned cloud bundles and report what would be installed without copying.",
    ),
) -> None:
    try:
        expected_bundles = (
            expected_bundles_from_run_summary(expected_from_run)
            if expected_from_run is not None
            else None
        )
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    result = import_cloud_job_bundles(
        source=source,
        output_root=output_root,
        report_path=report,
        overwrite=overwrite,
        expected_bundles=expected_bundles,
        dry_run=dry_run,
    )
    if dry_run:
        typer.echo(
            f"cloud VGGT dry-run {result['status']}: "
            f"{result['would_import_count']}/{result['discovered_count']} bundles would import"
        )
    else:
        typer.echo(
            f"cloud VGGT import {result['status']}: "
            f"{result['imported_count']}/{result['discovered_count']} bundles"
        )
    typer.echo(f"report: {report}")
    if result["status"] != "done":
        raise typer.Exit(1)


@vggt_app.command("validate")
def vggt_validate(
    bundle: Path = typer.Option(..., "--bundle", help="VGGT bundle directory."),
    report: Path | None = typer.Option(None, "--report", help="Optional validation report JSON."),
) -> None:
    validation = validate_bundle(bundle)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(model_to_dict(validation), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if not validation.valid:
        typer.echo(f"invalid VGGT bundle: {'; '.join(validation.errors)}")
        raise typer.Exit(1)
    loaded = load_bundle(bundle)
    typer.echo(f"valid VGGT bundle: {loaded.metadata.video_id}/{loaded.metadata.segment_id}")


@vggt_app.command("inspect")
def vggt_inspect(
    bundle: Path = typer.Option(..., "--bundle", help="VGGT bundle directory."),
) -> None:
    text = format_bundle_inspection(bundle)
    typer.echo(text)
    if "Validation: invalid" in text:
        raise typer.Exit(1)


@reconstruct_app.command("summarize")
def reconstruct_summarize(
    bundle: Path = typer.Option(..., "--bundle", help="VGGT bundle directory."),
    output: Path = typer.Option(..., "--output", "-o", help="Output summary JSON."),
) -> None:
    summary = summarize_bundle(bundle_path=bundle, output=output)
    typer.echo(
        f"wrote reconstruction summary: {output} ({summary.reliability.label})"
    )


@heatmap_app.command("generate")
def heatmap_generate(
    frame_manifest: Path = typer.Option(
        ..., "--frame-manifest", help="Frame manifest JSON."
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Output directory for heatmap PNGs and heatmaps.json."
    ),
) -> None:
    path = generate_heatmaps_from_manifest(frame_manifest, output)
    typer.echo(f"wrote heatmap manifest: {path}")


@viz_app.command("render")
def viz_render(
    frame_manifest: Path = typer.Option(..., "--frame-manifest", help="Frame manifest JSON."),
    bundle: Path = typer.Option(..., "--bundle", help="VGGT bundle directory."),
    summary: Path = typer.Option(..., "--summary", help="Reconstruction summary JSON."),
    output: Path = typer.Option(..., "--output", "-o", help="Output local HTML path."),
    smoothed_poses: Path | None = typer.Option(
        None,
        "--smoothed-poses",
        help="Optional smoothed_poses.npz for raw-vs-smoothed review toggle.",
    ),
    heatmap_manifest: Path | None = typer.Option(
        None,
        "--heatmap-manifest",
        help="Optional heatmaps.json manifest for image-space review overlays.",
    ),
    glb_scene: Path | None = typer.Option(
        None,
        "--glb-scene",
        help="Optional HF-style GLB scene for external Model3D-style review.",
    ),
    segment_qa: Path | None = typer.Option(
        None,
        "--segment-qa",
        help="Optional segment_qa.json report for pre-VGGT cut warnings.",
    ),
) -> None:
    path = render_review_html(
        frame_manifest,
        bundle,
        summary,
        output,
        smoothed_poses,
        heatmap_manifest,
        glb_scene,
        segment_qa,
    )
    typer.echo(f"wrote local review HTML: {path}")


@viz_app.command("export-glb")
def viz_export_glb(
    bundle: Path = typer.Option(..., "--bundle", help="VGGT bundle directory."),
    output: Path = typer.Option(..., "--output", "-o", help="Output HF-style GLB scene path."),
    metadata_output: Path | None = typer.Option(
        None,
        "--metadata-output",
        help="Optional GLB export metadata JSON path.",
    ),
    confidence_percentile: float = typer.Option(
        20.0,
        "--confidence-percentile",
        min=0.0,
        max=100.0,
        help="Drop points below this confidence percentile.",
    ),
    max_points: int = typer.Option(
        300_000,
        "--max-points",
        min=1,
        help="Maximum exported point count.",
    ),
    show_cameras: bool = typer.Option(
        True,
        "--show-cameras/--hide-cameras",
        help="Include schematic camera cones.",
    ),
) -> None:
    try:
        result = export_hf_style_glb(
            bundle_path=bundle,
            output=output,
            metadata_output=metadata_output,
            confidence_percentile=confidence_percentile,
            max_points=max_points,
            show_cameras=show_cameras,
        )
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(f"wrote HF-style local GLB scene: {result.output}")
    typer.echo(f"metadata: {result.metadata_output}")
    typer.echo(
        f"points={result.point_count} cameras={result.camera_count} "
        f"pose_jumps={result.pose_jump_count}"
    )


@viz_app.command("export-video")
def viz_export_video(
    frame_manifest: Path = typer.Option(..., "--frame-manifest", help="Frame manifest JSON."),
    bundle: Path = typer.Option(..., "--bundle", help="VGGT bundle directory."),
    summary: Path = typer.Option(..., "--summary", help="Reconstruction summary JSON."),
    output: Path = typer.Option(..., "--output", "-o", help="Output side-by-side MP4 path."),
    fps: float = typer.Option(8.0, "--fps", min=0.1, help="Output MP4 frames per second."),
    width: int = typer.Option(1280, "--width", min=320, help="Output MP4 width."),
    height: int = typer.Option(720, "--height", min=240, help="Output MP4 height."),
    point_budget: int = typer.Option(
        8000, "--point-budget", min=1, help="Maximum point-cloud points to draw per frame."
    ),
    confidence_min: float = typer.Option(
        0.0,
        "--confidence-min",
        min=0.0,
        max=1.0,
        help="Drop points below this optional point-confidence threshold.",
    ),
    relative_depth_quantile: float = typer.Option(
        1.0,
        "--relative-depth-quantile",
        min=0.0,
        max=1.0,
        help="Keep points up to this relative-depth quantile.",
    ),
) -> None:
    path = export_side_by_side_mp4(
        frame_manifest_path=frame_manifest,
        bundle_path=bundle,
        summary_path=summary,
        output=output,
        fps=fps,
        width=width,
        height=height,
        point_budget=point_budget,
        confidence_min=confidence_min,
        relative_depth_quantile=relative_depth_quantile,
    )
    typer.echo(f"wrote side-by-side MP4: {path}")


@compare_app.command("render")
def compare_render(
    summary: list[Path] = typer.Option(..., "--summary", help="Summary JSON path. Repeat for 3+ clips."),
    output: Path = typer.Option(..., "--output", "-o", help="Output comparison HTML."),
) -> None:
    if len(summary) < 3:
        typer.echo("comparison requires at least three summaries")
        raise typer.Exit(1)
    path = render_comparison_html(summary, output)
    typer.echo(f"wrote comparison HTML: {path}")


@smooth_app.command("poses")
def smooth_poses(
    bundle: Path = typer.Option(..., "--bundle", help="VGGT bundle directory."),
    summary: Path = typer.Option(..., "--summary", help="Reconstruction summary JSON."),
    output: Path = typer.Option(..., "--output", "-o", help="Output smoothed NPZ."),
    metadata_output: Path | None = typer.Option(
        None, "--metadata-output", help="Optional smoothing metadata JSON."
    ),
    window: int = typer.Option(3, "--window", min=1, help="Moving-average window."),
) -> None:
    try:
        path, meta = smooth_relative_poses(bundle, summary, output, metadata_output, window)
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(f"wrote relative pose smoother output: {path}")
    if meta is not None:
        typer.echo(f"wrote smoothing metadata: {meta}")


@review_app.command("run-three")
def review_run_three(
    frame_manifest: list[Path] = typer.Option(
        ..., "--frame-manifest", help="Frame manifest JSON. Repeat for each clip."
    ),
    bundle: list[Path] = typer.Option(
        ..., "--bundle", help="Imported VGGT bundle. Repeat for each clip."
    ),
    output_dir: Path = typer.Option(..., "--output-dir", help="Output review run directory."),
    smooth: bool = typer.Option(False, "--smooth", help="Also write relative pose smoother outputs."),
    export_video: bool = typer.Option(
        False, "--export-video", help="Also write side-by-side MP4 exports for each clip."
    ),
    heatmaps: bool = typer.Option(
        False,
        "--heatmaps",
        help="Generate image-space heatmap overlays for each review clip.",
    ),
) -> None:
    try:
        report = run_three_clip_review(
            frame_manifest,
            bundle,
            output_dir,
            smooth=smooth,
            export_video=export_video,
            generate_heatmaps=heatmaps,
        )
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(f"wrote three-clip review report: {report}")


@review_app.command("audit-three")
def review_audit_three(
    frame_manifest: list[Path] = typer.Option(
        ..., "--frame-manifest", help="Frame manifest JSON. Repeat for each clip."
    ),
    bundle: list[Path] = typer.Option(
        ..., "--bundle", help="Imported VGGT bundle. Repeat for each clip."
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output audit JSON."),
) -> None:
    try:
        report = audit_three_clip_inputs(frame_manifest, bundle, output)
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(f"wrote three-clip audit report: {output}")
    if not report["ready"]:
        raise typer.Exit(1)


@review_app.command("audit-readiness")
def review_audit_readiness(
    media_inventory: Path = typer.Option(..., "--media-inventory", help="Media inventory parquet."),
    annotations: Path = typer.Option(..., "--annotations", help="Segment annotations JSONL."),
    frames_root: Path = typer.Option(..., "--frames-root", help="Root containing sampled frame manifests."),
    video_id: list[str] = typer.Option(..., "--video-id", help="Video id to audit. Repeat for multiple clips."),
    segment_id: str = typer.Option("segment-001", "--segment-id", help="Segment id to audit."),
    report: Path = typer.Option(..., "--report", help="Output readiness report JSON."),
    min_expected_frames: int = typer.Option(
        1, "--min-expected-frames", min=1, help="Minimum sampled frames expected per clip."
    ),
) -> None:
    readiness = audit_data_readiness(
        media_inventory=media_inventory,
        annotations=annotations,
        frames_root=frames_root,
        video_ids=video_id,
        segment_id=segment_id,
        min_expected_frames=min_expected_frames,
    )
    write_readiness_report(readiness, report)
    typer.echo(
        f"data readiness {readiness['status']}: "
        f"{readiness['ready_count']}/{readiness['clip_count']} clips ready"
    )
    typer.echo(f"report: {report}")
    if readiness["status"] != "ready_for_cloud":
        for clip in readiness["clips"]:
            if clip["ready_for_cloud"]:
                continue
            typer.echo(f"{clip['video_id']}: needs_review")
            for name, check in clip["checks"].items():
                if check["status"] != "pass":
                    messages = "; ".join(check["messages"]) or check["status"]
                    typer.echo(f"  - {name}: {messages}")
        raise typer.Exit(1)


@review_app.command("audit-local-only")
def review_audit_local_only(
    root: Path = typer.Option(Path("."), "--root", help="Workspace root to scan."),
    report: Path = typer.Option(..., "--report", help="Output local-only audit JSON."),
) -> None:
    result = audit_local_only_artifacts(root=root, report=report)
    if result["status"] != "passed":
        typer.echo(f"local-only audit failed: {result['flagged_count']} flagged files")
        typer.echo(f"report: {report}")
        raise typer.Exit(1)
    typer.echo(f"local-only audit passed: {result['checked_count']} media-derived files checked")
    typer.echo(f"report: {report}")


@app.command("run")
def run(
    synthetic: bool = typer.Option(
        False,
        "--synthetic",
        help="Run the Milestone 1 synthetic mocked pipeline.",
    ),
    workdir: Path = typer.Option(..., "--workdir", help="Run folder."),
    frames: int = typer.Option(64, "--frames", min=1, help="Sampled frame count."),
    frame_manifest: list[Path] = typer.Option(
        [], "--frame-manifest", help="Frame manifest JSON. Repeat for each real review clip."
    ),
    bundle: list[Path] = typer.Option(
        [], "--bundle", help="Imported VGGT bundle. Repeat for each real review clip."
    ),
    video_id: list[str] = typer.Option(
        [], "--video-id", help="Dataset video id. Repeat for each real workflow clip."
    ),
    catalog: Path = typer.Option(Path("data/catalog/catalog.parquet"), "--catalog", help="Catalog parquet path."),
    media_dir: Path = typer.Option(Path("data/media"), "--media-dir", help="Local media directory."),
    media_inventory: Path = typer.Option(
        Path("data/media/media_inventory.parquet"),
        "--media-inventory",
        help="Media inventory parquet path.",
    ),
    annotations: Path = typer.Option(
        Path("data/annotations/segments.jsonl"),
        "--annotations",
        help="Segment annotations JSONL.",
    ),
    frames_root: Path = typer.Option(Path("data/frames"), "--frames-root", help="Sampled frame root."),
    vggt_root: Path = typer.Option(Path("data/vggt"), "--vggt-root", help="Imported VGGT bundle root."),
    cloud_job_output: Path = typer.Option(
        Path("outputs/cloud_vggt_job"),
        "--cloud-job-output",
        help="Cloud VGGT job package output directory.",
    ),
    cloud_return: Path | None = typer.Option(
        None,
        "--cloud-return",
        help="Returned cloud bundles.zip or bundles directory to import before audit.",
    ),
    cloud_import_report: Path | None = typer.Option(
        None,
        "--cloud-import-report",
        help="Cloud return import report JSON path.",
    ),
    overwrite_vggt: bool = typer.Option(
        False,
        "--overwrite-vggt",
        help="Replace existing local VGGT bundle directories when importing cloud return.",
    ),
    segment_id: str = typer.Option("segment-001", "--segment-id", help="Segment id to process."),
    resized_long_edge: int | None = typer.Option(
        768, "--resized-long-edge", min=1, help="Optional sampled frame max long edge."
    ),
    fetch_media_enabled: bool = typer.Option(
        False, "--fetch-media", help="Explicitly fetch selected videos if missing locally."
    ),
    smooth: bool = typer.Option(False, "--smooth", help="Also write relative pose smoother outputs."),
    heatmaps: bool = typer.Option(
        False,
        "--heatmaps",
        help="Generate image-space heatmap overlays only for review clips.",
    ),
    export_video: bool = typer.Option(
        False, "--export-video", help="Also write side-by-side MP4 exports during review rendering."
    ),
) -> None:
    if synthetic:
        summary = run_synthetic_pipeline(workdir=workdir, frames=frames)
        typer.echo(f"synthetic run {summary['status']}: {workdir}")
        return
    if video_id:
        if frame_manifest or bundle:
            typer.echo("use either --video-id workflow inputs or --frame-manifest/--bundle review inputs, not both")
            raise typer.Exit(1)
        summary = run_video_id_pipeline(
            workdir=workdir,
            video_ids=video_id,
            catalog_path=catalog,
            media_dir=media_dir,
            media_inventory=media_inventory,
            annotations=annotations,
            frames_root=frames_root,
            vggt_root=vggt_root,
            cloud_job_output=cloud_job_output,
            cloud_return=cloud_return,
            cloud_import_report=cloud_import_report,
            overwrite_vggt=overwrite_vggt,
            segment_id=segment_id,
            frame_count=frames,
            resized_long_edge=resized_long_edge,
            fetch_media_enabled=fetch_media_enabled,
            smooth=smooth,
            heatmaps=heatmaps,
            export_video=export_video,
        )
        typer.echo(f"video-id run {summary['status']}: {workdir}")
        typer.echo(f"summary: {workdir / 'summary.json'}")
        typer.echo(f"next steps: {workdir / 'NEXT_STEPS.md'}")
        if summary["status"] != "done":
            raise typer.Exit(1)
        return
    if not frame_manifest and not bundle:
        typer.echo("provide --synthetic, repeat --video-id, or repeat --frame-manifest/--bundle")
        raise typer.Exit(1)
    summary = run_review_pipeline(
        workdir=workdir,
        frame_manifests=frame_manifest,
        bundles=bundle,
        smooth=smooth,
        heatmaps=heatmaps,
        export_video=export_video,
    )
    typer.echo(f"review run {summary['status']}: {workdir}")
    typer.echo(f"summary: {workdir / 'summary.json'}")
    typer.echo(f"next steps: {workdir / 'NEXT_STEPS.md'}")
    if summary["status"] != "done":
        raise typer.Exit(1)



@methods_app.command("write-contract")
def methods_write_contract(
    output: Path = typer.Option(..., "--output", "-o", help="Output directory for method contract files."),
) -> None:
    artifacts = write_method_contract_files(output)
    typer.echo(f"method contract: {artifacts['method_contract']}")
    typer.echo(f"method matrix: {artifacts['method_matrix']}")
    typer.echo(f"return contract: {artifacts['return_contract']}")


@methods_app.command("validate-matrix")
def methods_validate_matrix(
    matrix: Path = typer.Option(..., "--matrix", help="method_matrix.json path."),
) -> None:
    report = validate_method_matrix(matrix)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise typer.Exit(1)


@methods_app.command("validate-return")
def methods_validate_return(
    source: Path = typer.Option(..., "--source", help="Return zip or extracted return directory."),
    extract_root: Path | None = typer.Option(
        None,
        "--extract-root",
        help="Optional directory used to extract and inspect a return zip.",
    ),
) -> None:
    report = validate_return_layout(source, extract_root)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise typer.Exit(1)


@h100_app.command("prepare")
def h100_prepare(
    dataset: str = typer.Option("latest", "--dataset", help="Dataset mode: latest or none."),
    workdir: Path = typer.Option(..., "--workdir", help="H100 run folder."),
    media_dir: Path = typer.Option(Path("data/media"), "--media-dir", help="Local media directory."),
    media_inventory: Path = typer.Option(
        Path("data/media/media_inventory.parquet"),
        "--media-inventory",
        help="Local media inventory parquet.",
    ),
    annotations: Path = typer.Option(
        Path("data/annotations/segments.jsonl"),
        "--annotations",
        help="Segment annotations JSONL.",
    ),
    frames_root: Path = typer.Option(Path("data/frames"), "--frames-root", help="Frame root."),
    frame_manifest: list[Path] = typer.Option(
        [], "--frame-manifest", help="Existing frame manifest to package. Repeat for multiple clips."
    ),
    frame_scout: int = typer.Option(32, "--frame-scout", min=1, help="Scout tier frame count."),
    frame_main: int = typer.Option(96, "--frame-main", min=1, help="Main tier frame count."),
    frame_high_detail: int = typer.Option(
        128, "--frame-high-detail", min=1, help="High-detail tier frame count."
    ),
    resize_scout: int = typer.Option(768, "--resize-scout", min=1, help="Scout tier long edge."),
    resize_main: int = typer.Option(1024, "--resize-main", min=1, help="Main tier long edge."),
    resize_high_detail: int = typer.Option(
        1024, "--resize-high-detail", min=1, help="High-detail tier long edge."
    ),
    auto_segment: str = typer.Option("strict", "--auto-segment", help="Segmentation mode; v1 supports strict."),
    metadata_policy: str = typer.Option(
        "provenance-only", "--metadata-policy", help="Source metadata policy."
    ),
) -> None:
    try:
        summary = prepare_h100_run(
            dataset=dataset,
            workdir=workdir,
            media_dir=media_dir,
            media_inventory=media_inventory,
            annotations=annotations,
            frames_root=frames_root,
            frame_manifests=frame_manifest,
            frame_scout=frame_scout,
            frame_main=frame_main,
            frame_high_detail=frame_high_detail,
            resize_scout=resize_scout,
            resize_main=resize_main,
            resize_high_detail=resize_high_detail,
            auto_segment=auto_segment,
            metadata_policy=metadata_policy,
        )
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(f"H100 prepare {summary['status']}: {workdir}")
    typer.echo(f"summary: {workdir / 'summary.json'}")
    typer.echo(f"next steps: {workdir / 'NEXT_STEPS.md'}")
    runpod_zip = summary.get("artifacts", {}).get("runpod_job_zip")
    if runpod_zip:
        typer.echo(f"upload ZIP: {runpod_zip}")
    if summary["status"] != "done":
        raise typer.Exit(1)


def _display_ps_path(path: Path) -> str:
    resolved = path.resolve()
    slash = chr(92)
    try:
        rel = resolved.relative_to(Path.cwd().resolve())
        return "." + slash + str(rel).replace("/", slash)
    except ValueError:
        return str(resolved)




@h100_app.command("smoke-package")
def h100_smoke_package(
    source: Path = typer.Option(
        Path("outputs/h100/full_161_run/runpod_job_4090_hf_full_stack_clean.zip"),
        "--source",
        help="RunPod job package ZIP or extracted package directory.",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Run full ZIP integrity/SHA checks before syntax checks.",
    ),
) -> None:
    report = smoke_runpod_job_package(source, fast=not full)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise typer.Exit(1)

@h100_app.command("profile-4090-launch")
def h100_profile_4090_launch(
    launch_dir: Path = typer.Option(
        Path("outputs/h100/full_161_run/launch"),
        "--launch-dir",
        help="Directory where RUNPOD_4090_PILOT_PROFILE.json should be written.",
    ),
    write: bool = typer.Option(
        False,
        "--write",
        help="Write RUNPOD_4090_PILOT_PROFILE.json into the launch directory.",
    ),
) -> None:
    if write:
        report = write_runpod_4090_hf_profile(launch_dir)
    else:
        report = {
            "status": "preview",
            "profile": runpod_4090_hf_profile(),
            "path": str((launch_dir / "RUNPOD_4090_PILOT_PROFILE.json").resolve()),
        }
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@h100_app.command("status-4090-launch")
def h100_status_4090_launch(
    package: Path = typer.Option(
        Path("outputs/h100/full_161_run/runpod_job_4090_hf_full_stack_clean.zip"),
        "--package",
        help="4090/HF RunPod package ZIP or extracted package directory.",
    ),
    launch_dir: Path = typer.Option(
        Path("outputs/h100/full_161_run/launch"),
        "--launch-dir",
        help="Directory containing runpod_launch_4090_hf_manifest.json and operator helpers.",
    ),
    fast: bool = typer.Option(
        True,
        "--fast/--full",
        help="Use --full to include package SHA256 hashing and full ZIP integrity checks.",
    ),
) -> None:
    report = audit_runpod_4090_hf_launch_kit(package=package, launch_dir=launch_dir, fast=fast)
    status = "ready_to_run_4090_pilot" if report["valid"] else "blocked"
    package_report = report.get("package_inspection", {})
    docs = report.get("docs", {})
    helpers = report.get("helpers", {})
    start_here = Path(docs.get("start_here_markdown", {}).get("path", launch_dir / "START_HERE_4090_HF.md"))
    pod_profile = Path(docs.get("pilot_profile", {}).get("path", launch_dir / "RUNPOD_4090_PILOT_PROFILE.json"))
    prelaunch = Path(helpers.get("prelaunch_audit_powershell", {}).get("path", launch_dir / "PRELAUNCH_AUDIT_4090_HF.ps1"))
    upload = Path(helpers.get("upload_and_start_powershell", {}).get("path", launch_dir / "UPLOAD_AND_OPTIONALLY_START_4090_HF.ps1"))
    one_shot = Path(helpers.get("one_shot_full_powershell", {}).get("path", launch_dir / "RUN_FULL_4090_R3_LINGBOT.ps1"))
    monitor = Path(helpers.get("monitor_powershell", {}).get("path", launch_dir / "MONITOR_RUNPOD_4090_HF.ps1"))
    download = Path(helpers.get("download_and_validate_powershell", {}).get("path", launch_dir / "DOWNLOAD_AND_VALIDATE_RETURN.ps1"))
    identity_file = "$env:USERPROFILE" + chr(92) + ".ssh" + chr(92) + "id_ed25519"

    typer.echo("RunPod 4090/HF launch status")
    typer.echo(f"status: {status}")
    typer.echo(f"package: {report['package']}")
    typer.echo(f"clip jobs: {package_report.get('clip_count', 0)}")
    manifest_path = launch_dir / "runpod_launch_4090_hf_manifest.json"
    optional_runner = None
    manifest_method_count = None
    if manifest_path.exists():
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            optional_runner = manifest_data.get("r3_lingbot_runner")
            manifest_method_count = len(manifest_data.get("methods") or [])
        except Exception:
            optional_runner = None
    typer.echo(f"base package method count: {len(package_report.get('method_ids', []))}")
    if manifest_method_count is not None:
        typer.echo(f"launch manifest method count: {manifest_method_count}")
    if optional_runner:
        typer.echo(f"R3/LingBot runner: {optional_runner}")
    typer.echo(f"start here: {_display_ps_path(start_here)}")
    typer.echo(f"pod profile: {_display_ps_path(pod_profile)}")
    typer.echo("")
    typer.echo("local gates before upload:")
    typer.echo("  fpv runpod smoke-package --source outputs/h100/full_161_run/runpod_job_4090_hf_full_stack_clean.zip")
    audit_command = "fpv runpod audit-4090-launch --fast" if fast else "fpv runpod audit-4090-launch"
    typer.echo(f"  {audit_command}")
    typer.echo(f"  {_display_ps_path(prelaunch)} -Fast")
    typer.echo("")
    typer.echo("one-shot full 4090/R3/LingBot command:")
    typer.echo(f"  {_display_ps_path(one_shot)} -HostName <runpod_ip> -Port <tcp_port> -IdentityFile {identity_file}")
    typer.echo("")
    typer.echo("first pilot command:")
    typer.echo(f"  {_display_ps_path(upload)} -HostName <runpod_ip> -Port <tcp_port> -IdentityFile {identity_file} -Start -RunMode pilot_hf_scout3")
    typer.echo("R3/LingBot full wrapper command:")
    typer.echo(f"  {_display_ps_path(upload)} -HostName <runpod_ip> -Port <tcp_port> -IdentityFile {identity_file} -Start -RunMode full_hf_rebuild -UseR3LingBotRunner -EnableFullOptionalMethods")
    typer.echo("")
    typer.echo("monitor:")
    typer.echo(f"  {_display_ps_path(monitor)} -HostName <runpod_ip> -Port <tcp_port> -IdentityFile {identity_file}")
    typer.echo("")
    typer.echo("download and validate:")
    typer.echo(f"  {_display_ps_path(download)} -HostName <runpod_ip> -Port <tcp_port> -IdentityFile {identity_file}")
    typer.echo("")
    typer.echo("final audit after download:")
    typer.echo("  fpv runpod final-audit-4090-return --source <local_return_zip>")
    typer.echo("")
    safety_warnings = ", ".join(runpod_4090_hf_profile()["safety_warnings"])
    typer.echo(f"safety: {safety_warnings}")
    typer.echo("remaining proof: run on RunPod, download fresh h100_return.zip, validate locally")
    if report["issues"]:
        typer.echo("")
        typer.echo("issues:")
        for issue in report["issues"]:
            typer.echo(f"- {issue}")
        raise typer.Exit(1)


@h100_app.command("write-4090-start-here")
def h100_write_4090_start_here(
    package: Path = typer.Option(
        Path("outputs/h100/full_161_run/runpod_job_4090_hf_full_stack_clean.zip"),
        "--package",
        help="4090/HF RunPod package ZIP or extracted package directory.",
    ),
    launch_dir: Path = typer.Option(
        Path("outputs/h100/full_161_run/launch"),
        "--launch-dir",
        help="Directory containing runpod_launch_4090_hf_manifest.json and operator helpers.",
    ),
    status: str = typer.Option(
        "ready_to_run_4090_pilot",
        "--status",
        help="Human-readable current status to write into START_HERE_4090_HF.md.",
    ),
) -> None:
    try:
        report = write_runpod_4090_hf_start_here(package=package, launch_dir=launch_dir, status=status)
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "done":
        raise typer.Exit(1)


@h100_app.command("audit-4090-launch")
def h100_audit_4090_launch(
    package: Path = typer.Option(
        Path("outputs/h100/full_161_run/runpod_job_4090_hf_full_stack_clean.zip"),
        "--package",
        help="4090/HF RunPod package ZIP or extracted package directory.",
    ),
    launch_dir: Path = typer.Option(
        Path("outputs/h100/full_161_run/launch"),
        "--launch-dir",
        help="Directory containing runpod_launch_4090_hf_manifest.json and operator helpers.",
    ),
    fast: bool = typer.Option(
        False,
        "--fast",
        help="Skip package SHA256 hashing and full ZIP integrity checks for the large package.",
    ),
) -> None:
    report = audit_runpod_4090_hf_launch_kit(package=package, launch_dir=launch_dir, fast=fast)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise typer.Exit(1)


@h100_app.command("inspect-package")
def h100_inspect_package(
    source: Path = typer.Option(..., "--source", help="runpod_job.zip or extracted runpod_job directory."),
    fast: bool = typer.Option(False, "--fast", help="Skip full ZIP integrity test and SHA256 hashing for very large packages."),
) -> None:
    report = inspect_runpod_job_package(source, fast=fast, compute_sha256=not fast)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise typer.Exit(1)


@h100_app.command("launch-manifest")
def h100_launch_manifest(
    source: Path = typer.Option(..., "--source", help="runpod_job.zip or extracted runpod_job directory."),
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="Output directory for runpod_launch_manifest.json and RUNPOD_LAUNCH.md."
    ),
) -> None:
    try:
        manifest = write_runpod_launch_manifest(source, output_dir)
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(f"RunPod launch manifest {manifest['status']}: {output_dir}")
    typer.echo(f"json: {output_dir / 'runpod_launch_manifest.json'}")
    typer.echo(f"markdown: {output_dir / 'RUNPOD_LAUNCH.md'}")
    if manifest["status"] not in {"ready_to_upload", "ready_unhashed"}:
        raise typer.Exit(1)


@h100_app.command("inspect-return")
def h100_inspect_return(
    source: Path = typer.Option(..., "--source", help="h100_return.zip or extracted return directory."),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable inspection JSON."),
) -> None:
    try:
        if json_output:
            report = validate_h100_return(source, extract_root=None)
            typer.echo(json.dumps(report, indent=2, sort_keys=True))
            if report["issues"]:
                raise typer.Exit(1)
            return
        text = inspect_h100_return(source)
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(text)
    if "issues:" in text:
        raise typer.Exit(1)


@h100_app.command("postflight-return")
def h100_postflight_return(
    source: Path = typer.Option(..., "--source", help="h100_return.zip or extracted return directory."),
    launch_manifest: Path = typer.Option(
        ..., "--launch-manifest", help="runpod_launch_manifest.json created before cloud execution."
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="Output directory for return_postflight.json and RETURN_POSTFLIGHT.md."
    ),
) -> None:
    try:
        report = verify_h100_return_against_launch(
            source=source,
            launch_manifest=launch_manifest,
            output_dir=output_dir,
        )
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(f"H100 return postflight {report['status']}: {output_dir}")
    typer.echo(f"json: {output_dir / 'return_postflight.json'}")
    typer.echo(f"markdown: {output_dir / 'RETURN_POSTFLIGHT.md'}")
    if report["status"] == "failed_soft":
        raise typer.Exit(1)



@h100_app.command("final-audit-4090-return")
def h100_final_audit_4090_return(
    source: Path = typer.Option(..., "--source", help="Downloaded h100_return.zip or extracted return directory."),
    launch_manifest: Path = typer.Option(
        Path("outputs/h100/full_161_run/launch/runpod_launch_4090_hf_manifest.json"),
        "--launch-manifest",
        help="Current 4090/HF launch manifest.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/reviews/runpod_4090_final_audit"),
        "--output-dir",
        help="Output directory for runpod_4090_final_audit.json and RUNPOD_4090_FINAL_AUDIT.md.",
    ),
) -> None:
    try:
        postflight = verify_h100_return_against_launch(
            source=source,
            launch_manifest=launch_manifest,
            output_dir=output_dir,
        )
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc

    launch = json.loads(launch_manifest.read_text(encoding="utf-8"))
    validation = postflight.get("return_validation", {})
    manifest = validation.get("manifest") or {}
    method_stage = validation.get("method_stage_report") or {}
    status = (
        "failed_soft"
        if postflight["status"] == "failed_soft"
        else "validated_with_warnings"
        if postflight.get("warnings")
        else "validated"
    )
    audit = {
        "schema_version": "runpod-4090-final-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "source": str(source.resolve()),
        "launch_manifest": str(launch_manifest.resolve()),
        "output_dir": str(output_dir.resolve()),
        "postflight_json": str((output_dir / "return_postflight.json").resolve()),
        "postflight_status": postflight["status"],
        "launch_schema_version": launch.get("schema_version"),
        "launch_status": launch.get("status"),
        "launch_package_sha256": (launch.get("package") or {}).get("sha256"),
        "return_manifest_status": manifest.get("status"),
        "selected_bundle_count": validation.get("selected_bundle_count", 0),
        "method_stage_status": method_stage.get("status"),
        "missing_required_files": postflight.get("missing_required_files", []),
        "missing_methods": postflight.get("missing_methods", []),
        "warnings": postflight.get("warnings", []),
        "issues": postflight.get("issues", []),
        "next_actions": postflight.get("next_actions", []),
        "safety_warnings": runpod_4090_hf_profile()["safety_warnings"],
        "completion_boundary": (
            "This audit validates the cloud return package against the 4090/HF launch contract. "
            "It does not add geolocation, meters, route, approach, guidance, or next-maneuver claims."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runpod_4090_final_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    warnings = "\n".join(f"- {item}" for item in audit["warnings"]) or "- none"
    issues = "\n".join(f"- {item}" for item in audit["issues"]) or "- none"
    actions = "\n".join(f"- {item}" for item in audit["next_actions"]) or "- none"
    safety = "\n".join(f"- {item}" for item in audit["safety_warnings"])
    markdown = f"""# RunPod 4090 Final Audit

Status: `{audit['status']}`
Postflight status: `{audit['postflight_status']}`

## Inputs

- Return source: `{audit['source']}`
- Launch manifest: `{audit['launch_manifest']}`
- Postflight JSON: `{audit['postflight_json']}`

## Summary

- Selected bundles: `{audit['selected_bundle_count']}`
- Method stage status: `{audit['method_stage_status']}`
- Launch package SHA256: `{audit['launch_package_sha256']}`
- Missing required files: `{len(audit['missing_required_files'])}`
- Missing methods: `{len(audit['missing_methods'])}`

## Warnings

{warnings}

## Issues

{issues}

## Next Actions

{actions}

## Safety Boundary

{safety}
"""
    (output_dir / "RUNPOD_4090_FINAL_AUDIT.md").write_text(markdown, encoding="utf-8")

    typer.echo(f"RunPod 4090 final audit {audit['status']}: {output_dir}")
    typer.echo(f"json: {output_dir / 'runpod_4090_final_audit.json'}")
    typer.echo(f"markdown: {output_dir / 'RUNPOD_4090_FINAL_AUDIT.md'}")
    if audit["status"] == "failed_soft":
        raise typer.Exit(1)

@h100_app.command("optional-report")
def h100_optional_report(
    source: Path = typer.Option(..., "--source", help="h100_return.zip or extracted return directory."),
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="Output directory for optional_method_report.json and OPTIONAL_METHOD_REPORT.md."
    ),
) -> None:
    try:
        report = write_h100_optional_method_report(source=source, output_dir=output_dir)
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(f"H100 optional report {report['status']}: {output_dir}")
    typer.echo(f"json: {output_dir / 'optional_method_report.json'}")
    typer.echo(f"markdown: {output_dir / 'OPTIONAL_METHOD_REPORT.md'}")
    if report["status"] == "failed_soft":
        raise typer.Exit(1)

@h100_app.command("import-return")
def h100_import_return(
    source: Path = typer.Option(..., "--source", help="h100_return.zip or extracted return directory."),
    workdir: Path = typer.Option(..., "--workdir", help="Original H100 local run folder."),
    vggt_root: Path = typer.Option(Path("data/vggt"), "--vggt-root", help="Local VGGT bundle root."),
    review_output: Path = typer.Option(
        ..., "--review-output", help="Output directory for import report and review landing page."
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace existing imported bundles."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without copying bundles."),
) -> None:
    try:
        report = import_h100_return(
            source=source,
            workdir=workdir,
            vggt_root=vggt_root,
            review_output=review_output,
            overwrite=overwrite,
            dry_run=dry_run,
        )
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    if dry_run:
        typer.echo(
            f"H100 return dry-run {report['status']}: "
            f"{report['would_import_count']}/{report['selected_bundle_count']} bundles would import"
        )
    else:
        typer.echo(
            f"H100 return import {report['status']}: "
            f"{report['imported_count']}/{report['selected_bundle_count']} bundles imported"
        )
    typer.echo(f"report: {review_output / 'import_report.json'}")
    if report["status"] != "done":
        raise typer.Exit(1)


@h100_app.command("insights")
def h100_insights(
    review_run: Path = typer.Option(
        ..., "--review-run", help="Review run directory or run_report.json."
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Output directory for H100 insight artifacts."
    ),
    import_output: Path | None = typer.Option(
        None, "--import-output", help="Optional H100 import output directory."
    ),
    source_return: Path | None = typer.Option(
        None, "--source-return", help="Optional h100_return.zip path used for provenance."
    ),
) -> None:
    try:
        report = build_h100_insights(
            review_run=review_run,
            output_dir=output_dir,
            import_output=import_output,
            source_return=source_return,
        )
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(
        f"H100 insights {report['status']}: "
        f"{report['clip_count']} clips, "
        f"{report['descriptor_safe_count']} descriptor-safe"
    )
    typer.echo(f"dashboard: {report['artifacts']['dashboard']}")
    typer.echo(f"metrics: {report['artifacts']['metrics_csv']}")

@windows_app.command("propose")
def windows_propose(
    media_inventory: Path = typer.Option(
        Path("data/media/media_inventory.parquet"),
        "--media-inventory",
        help="Local media inventory parquet.",
    ),
    annotations: Path = typer.Option(
        Path("data/annotations/segments.jsonl"),
        "--annotations",
        help="Accepted segment annotations JSONL.",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/windows/stable_window_run"),
        "--output-dir",
        help="Output run folder for window tables and dashboard.",
    ),
    frames_root: Path = typer.Option(
        Path("data/frames"), "--frames-root", help="Root for sampled window frames."
    ),
    video_id: list[str] = typer.Option(
        [], "--video-id", help="Video id to process. Repeat for a subset."
    ),
    segment_id: str | None = typer.Option(
        "segment-001", "--segment-id", help="Accepted segment id to window."
    ),
    window_sec: float = typer.Option(6.0, "--window-sec", min=0.25, help="Candidate window length."),
    stride_sec: float = typer.Option(3.0, "--stride-sec", min=0.1, help="Sliding-window stride."),
    candidate_limit: int = typer.Option(3, "--candidate-limit", min=1, help="Windows kept per segment."),
    frames: int = typer.Option(64, "--frames", min=1, help="Quality-aware sampled frames per window."),
    resized_long_edge: int | None = typer.Option(
        1024, "--resized-long-edge", min=1, help="Optional sampled frame long edge."
    ),
    eval_samples: int = typer.Option(12, "--eval-samples", min=2, help="Frames used to score each window."),
    neighbor_radius: int = typer.Option(
        2, "--neighbor-radius", min=0, help="Local frame search radius around deterministic anchors."
    ),
    mask_static_overlays: bool = typer.Option(
        True,
        "--mask-static-overlays/--no-mask-static-overlays",
        help="Neutralize detected static lower-left overlay/censor regions in exported window frames.",
    ),
) -> None:
    try:
        summary = propose_stable_windows(
            media_inventory=media_inventory,
            annotations=annotations,
            output_dir=output_dir,
            frames_root=frames_root,
            video_ids=video_id or None,
            segment_id=segment_id,
            window_sec=window_sec,
            stride_sec=stride_sec,
            candidate_limit=candidate_limit,
            frame_count=frames,
            resized_long_edge=resized_long_edge,
            eval_samples=eval_samples,
            neighbor_radius=neighbor_radius,
            mask_static_overlays=mask_static_overlays,
        )
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(
        f"stable-window proposal {summary['status']}: "
        f"{summary['window_count']} selected windows"
    )
    typer.echo(f"dashboard: {summary['artifacts']['dashboard']}")
    typer.echo(f"frame manifests: {summary['artifacts']['frame_manifest_list']}")
    typer.echo(f"next steps: {summary['artifacts']['next_steps']}")
    if summary["status"] != "done":
        raise typer.Exit(1)



@windows_app.command("colab-t4")
def windows_colab_t4(
    media_inventory: Path = typer.Option(
        Path("data/media/media_inventory.parquet"),
        "--media-inventory",
        help="Local media inventory parquet.",
    ),
    annotations: Path = typer.Option(
        Path("data/annotations/segments.jsonl"),
        "--annotations",
        help="Accepted segment annotations JSONL.",
    ),
    frames_root: Path = typer.Option(Path("data/frames"), "--frames-root", help="Frame root."),
    output_dir: Path = typer.Option(
        Path("outputs/colab/t4_window_run"),
        "--output-dir",
        help="Colab T4 package output folder.",
    ),
    video_id: list[str] = typer.Option(
        ..., "--video-id", help="Video id to process. Repeat for a small T4 batch."
    ),
    segment_id: str = typer.Option("segment-001", "--segment-id", help="Accepted segment id."),
    window_sec: float = typer.Option(
        8.0, "--window-sec", min=0.5, help="Stable window length in seconds."
    ),
    stride_sec: float = typer.Option(3.0, "--stride-sec", min=0.1, help="Window stride in seconds."),
    target_fps: float = typer.Option(
        2.0, "--target-fps", min=0.1, help="Approximate sampled FPS inside each stable window."
    ),
    max_frames: int = typer.Option(
        20,
        "--max-frames",
        min=2,
        help="T4 memory cap for frames per window after applying target FPS.",
    ),
    candidate_limit: int = typer.Option(
        2, "--candidate-limit", min=1, help="Windows kept per accepted segment."
    ),
    resized_long_edge: int | None = typer.Option(
        1024, "--resized-long-edge", min=1, help="High-quality source frame long edge."
    ),
    eval_samples: int = typer.Option(12, "--eval-samples", min=2, help="Frames used to score each window."),
    neighbor_radius: int = typer.Option(
        2, "--neighbor-radius", min=0, help="Local frame search radius around deterministic anchors."
    ),
) -> None:
    try:
        summary = prepare_colab_t4_window_run(
            media_inventory=media_inventory,
            annotations=annotations,
            frames_root=frames_root,
            output_dir=output_dir,
            video_ids=video_id,
            segment_id=segment_id,
            window_sec=window_sec,
            stride_sec=stride_sec,
            target_fps=target_fps,
            max_frames=max_frames,
            candidate_limit=candidate_limit,
            resized_long_edge=resized_long_edge,
            eval_samples=eval_samples,
            neighbor_radius=neighbor_radius,
        )
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(
        f"Colab T4 package {summary['status']}: "
        f"{summary.get('window_count', 0)} window frame packs"
    )
    typer.echo(f"upload ZIP: {summary.get('artifacts', {}).get('cloud_job_zip')}")
    typer.echo(f"proposal dashboard: {summary.get('artifacts', {}).get('proposal_dashboard')}")
    typer.echo(f"next steps: {output_dir / 'NEXT_STEPS.md'}")
    if summary["status"] != "done":
        raise typer.Exit(1)


@windows_app.command("h100-package")
def windows_h100_package(
    proposal_run: Path = typer.Option(
        ..., "--proposal-run", help="Stable-window run directory or summary.json."
    ),
    workdir: Path = typer.Option(..., "--workdir", help="H100 package output folder."),
    media_dir: Path = typer.Option(Path("data/media"), "--media-dir", help="Local media directory."),
    media_inventory: Path = typer.Option(
        Path("data/media/media_inventory.parquet"),
        "--media-inventory",
        help="Local media inventory parquet.",
    ),
    annotations: Path = typer.Option(
        Path("data/annotations/segments.jsonl"), "--annotations", help="Segment annotations JSONL."
    ),
    frames_root: Path = typer.Option(Path("data/frames"), "--frames-root", help="Frame root."),
    metadata_policy: str = typer.Option(
        "provenance-only", "--metadata-policy", help="Source metadata policy."
    ),
) -> None:
    try:
        frame_manifests = selected_window_frame_manifests(proposal_run)
        summary = prepare_h100_run(
            dataset="none",
            workdir=workdir,
            media_dir=media_dir,
            media_inventory=media_inventory,
            annotations=annotations,
            frames_root=frames_root,
            frame_manifests=frame_manifests,
            frame_scout=32,
            frame_main=64,
            frame_high_detail=64,
            resize_scout=768,
            resize_main=1024,
            resize_high_detail=1024,
            auto_segment="strict",
            metadata_policy=metadata_policy,
        )
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(
        f"window H100 package {summary['status']}: "
        f"{summary['clip_count']} window frame packs"
    )
    typer.echo(f"upload ZIP: {summary.get('artifacts', {}).get('runpod_job_zip')}")
    typer.echo(f"next steps: {workdir / 'NEXT_STEPS.md'}")
    if summary["status"] != "done":
        raise typer.Exit(1)


@windows_app.command("rank")
def windows_rank(
    proposal_run: Path = typer.Option(
        ..., "--proposal-run", help="Stable-window run directory or summary.json."
    ),
    vggt_root: Path = typer.Option(Path("data/vggt"), "--vggt-root", help="Imported VGGT bundle root."),
    output_dir: Path = typer.Option(
        Path("outputs/reviews/window_ranked"), "--output-dir", help="Window ranking/review output folder."
    ),
    render_review: bool = typer.Option(
        True, "--render-review/--no-render-review", help="Render per-window local HTML reviews."
    ),
    stitched_review: bool = typer.Option(
        False,
        "--stitched-review/--no-stitched-review",
        help="Write a local best-window review index. This is not coordinate alignment.",
    ),
) -> None:
    try:
        report = rank_window_bundles(
            proposal_run=proposal_run,
            vggt_root=vggt_root,
            output_dir=output_dir,
            render_review=render_review,
            stitched_review=stitched_review,
        )
    except Exception as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    typer.echo(
        f"window ranking {report['status']}: "
        f"{report['best_window_count']}/{report['window_count']} best windows ready"
    )
    typer.echo(f"dashboard: {report['artifacts']['dashboard']}")
    if report["artifacts"].get("stitched_review"):
        typer.echo(f"best-window review: {report['artifacts']['stitched_review']}")
    if report["status"] != "done":
        raise typer.Exit(1)


def main() -> None:
    app()


