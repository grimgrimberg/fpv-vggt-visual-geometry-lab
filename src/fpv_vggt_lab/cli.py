from __future__ import annotations

import json
from pathlib import Path

import typer

from .catalog import DEFAULT_MANIFEST_URL, DEFAULT_README_URL, shortlist_catalog, sync_catalog
from .cloud_job import create_cloud_job_package
from .export_video import export_side_by_side_mp4
from .frames import sample_accepted_segment_frames, sample_video_frames
from .geometry import summarize_bundle
from .heatmaps import generate_heatmaps_from_manifest
from .local_only import audit_local_only_artifacts
from .media import audit_media_inventory, fetch_media, write_media_audit_report
from .pipeline import run_review_pipeline, run_synthetic_pipeline, run_video_id_pipeline
from .readiness import audit_data_readiness, write_readiness_report
from .review import audit_three_clip_inputs, run_three_clip_review
from .schemas import model_to_dict
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


@catalog_app.command("sync")
def catalog_sync(
    readme: str = typer.Option(DEFAULT_README_URL, "--readme", help="README source path or URL."),
    manifest: str | None = typer.Option(
        DEFAULT_MANIFEST_URL, "--manifest", help="Manifest TSV source path or URL."
    ),
    output: Path = typer.Option(Path("data/catalog"), "--output", "-o", help="Catalog output directory."),
) -> None:
    catalog_path, snapshot_path = sync_catalog(readme_source=readme, manifest_source=manifest, output=output)
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
    typer.echo("on CUDA GPU: unzip, then run `python run_vggt_job.py`")
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
) -> None:
    path = render_review_html(
        frame_manifest,
        bundle,
        summary,
        output,
        smoothed_poses,
        heatmap_manifest,
    )
    typer.echo(f"wrote local review HTML: {path}")


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


def main() -> None:
    app()
