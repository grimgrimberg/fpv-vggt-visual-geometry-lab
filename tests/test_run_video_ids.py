import json
import zipfile
import hashlib
from pathlib import Path

import cv2
import numpy as np
from typer.testing import CliRunner

from fpv_vggt_lab.cli import app
from fpv_vggt_lab.pipeline import run_review_pipeline
from fpv_vggt_lab.synthetic import create_synthetic_video
from tests.test_viz_compare_smoothing import create_mocked_summary, extract_review_payload


runner = CliRunner()


def write_three_clip_catalog(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    video_ids: list[str] = []
    readme_lines = [
        "# Fixture Dataset",
        "",
        "| Date | Image | Description | Link |",
        "|---|---|---|---|",
    ]
    manifest_lines = ["current_stem\ttarget_stem\tdate\tslug\tconfidence\tnotes"]
    for index in range(3):
        video_id = f"2026-06-22_fixture_clip_{index}"
        video_ids.append(video_id)
        video_path = tmp_path / f"{video_id}.mp4"
        create_synthetic_video(video_path, frames=24, width=160, height=120, fps=12)
        readme_lines.append(
            f"| 2026-06-22 | <img src=\"https://example.test/{index}.jpg\" width=\"180\"> "
            f"| Fixture clip {index} | [Download]({video_path.as_uri()}) |"
        )
        manifest_lines.append(
            f"{video_id}\t{video_id}\t2026-06-22\tfixture_clip_{index}\thigh\tFixture row."
        )

    readme = tmp_path / "README.md"
    manifest = tmp_path / "manifest.tsv"
    readme.write_text("\n".join(readme_lines), encoding="utf-8")
    manifest.write_text("\n".join(manifest_lines), encoding="utf-8")
    return readme, manifest, video_ids


def write_video_with_blue_glitch(path: Path, frames: int = 24, fps: float = 12.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (160, 120),
    )
    assert writer.isOpened()
    try:
        for index in range(frames):
            if index == 10:
                frame = np.full((120, 160, 3), (255, 0, 0), dtype=np.uint8)
            else:
                frame = np.full((120, 160, 3), (80, 130, 160), dtype=np.uint8)
                cv2.circle(frame, (20 + index * 3, 60), 12, (240, 240, 240), -1)
                cv2.line(frame, (0, 100), (159, 95 - index), (0, 220, 255), 2)
            writer.write(frame)
    finally:
        writer.release()


def test_run_from_video_ids_fetches_samples_and_packages_cloud_job(tmp_path: Path):
    readme, manifest, video_ids = write_three_clip_catalog(tmp_path)
    catalog_dir = tmp_path / "catalog"
    annotations = tmp_path / "segments.jsonl"

    synced = runner.invoke(
        app,
        [
            "catalog",
            "sync",
            "--readme",
            str(readme),
            "--manifest",
            str(manifest),
            "--output",
            str(catalog_dir),
        ],
    )
    assert synced.exit_code == 0, synced.output

    for video_id in video_ids:
        accepted = runner.invoke(
            app,
            [
                "segment",
                "accept",
                "--annotations",
                str(annotations),
                "--video-id",
                video_id,
                "--segment-id",
                "segment-001",
                "--start",
                "0",
                "--end",
                "1.5",
            ],
        )
        assert accepted.exit_code == 0, accepted.output

    workdir = tmp_path / "run"
    result = runner.invoke(
        app,
        [
            "run",
            "--workdir",
            str(workdir),
            "--catalog",
            str(catalog_dir / "catalog.parquet"),
            "--media-dir",
            str(tmp_path / "media"),
            "--media-inventory",
            str(tmp_path / "media" / "media_inventory.parquet"),
            "--annotations",
            str(annotations),
            "--frames-root",
            str(tmp_path / "frames"),
            "--vggt-root",
            str(tmp_path / "vggt"),
            "--cloud-job-output",
            str(tmp_path / "cloud_job"),
            "--fetch-media",
            "--frames",
            "6",
            "--video-id",
            video_ids[0],
            "--video-id",
            video_ids[1],
            "--video-id",
            video_ids[2],
            "--smooth",
        ],
    )

    assert result.exit_code != 0
    assert "needs_vggt_bundle" in result.output
    summary = json.loads((workdir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "needs_vggt_bundle"
    assert summary["stage_status"]["local_media"] == "done"
    assert summary["stage_status"]["accepted_segments"] == "done"
    assert summary["stage_status"]["frame_sampling"] == "done"
    assert summary["stage_status"]["cloud_job"] == "done"
    assert summary["stage_status"]["audit_three"] == "needs_vggt_bundle"
    assert summary["data_checks"]["media_audit"] == {
        "status": "ready",
        "checked_count": 3,
        "ready_count": 3,
        "report": summary["artifacts"]["media_audit_report"],
    }
    assert summary["data_checks"]["frame_readiness"] == {
        "status": "ready_for_cloud",
        "clip_count": 3,
        "ready_count": 3,
        "report": summary["artifacts"]["readiness_report"],
    }
    assert len(summary["artifacts"]["frame_manifests"]) == 3
    cloud_package = Path(summary["artifacts"]["cloud_job_package"])
    assert cloud_package.exists()
    assert summary["artifacts"]["cloud_job_package_bytes"] == cloud_package.stat().st_size
    assert summary["artifacts"]["cloud_job_package_sha256"] == hashlib.sha256(
        cloud_package.read_bytes()
    ).hexdigest()
    assert Path(summary["artifacts"]["run_log"]).exists()
    assert Path(summary["artifacts"]["summary_json"]).exists()
    assert Path(summary["artifacts"]["next_steps"]).exists()
    assert (tmp_path / "cloud_job" / "cloud_vggt_job.zip").exists()
    next_steps = (workdir / "NEXT_STEPS.md").read_text(encoding="utf-8")
    assert "--cloud-return <returned-bundles.zip>" in next_steps
    assert "cloud_bundle_import_dry_run.json" in next_steps
    assert "--expected-from-run outputs/reviews/three_clip_run/summary.json" in next_steps
    assert "--dry-run" in next_steps
    assert "Upload or copy the cloud job package" in next_steps
    assert "python run_vggt_job.py" in next_steps
    assert "Reruns skip clips that already have valid normalized bundles" in next_steps
    assert summary["artifacts"]["cloud_job_package_sha256"] in next_steps
    assert "no geolocation" in next_steps


def test_run_from_video_ids_missing_media_uses_allowed_stage_status(tmp_path: Path):
    readme, manifest, video_ids = write_three_clip_catalog(tmp_path)
    catalog_dir = tmp_path / "catalog"
    annotations = tmp_path / "segments.jsonl"

    synced = runner.invoke(
        app,
        [
            "catalog",
            "sync",
            "--readme",
            str(readme),
            "--manifest",
            str(manifest),
            "--output",
            str(catalog_dir),
        ],
    )
    assert synced.exit_code == 0, synced.output

    workdir = tmp_path / "run-missing-media"
    result = runner.invoke(
        app,
        [
            "run",
            "--workdir",
            str(workdir),
            "--catalog",
            str(catalog_dir / "catalog.parquet"),
            "--media-dir",
            str(tmp_path / "media"),
            "--media-inventory",
            str(tmp_path / "media" / "media_inventory.parquet"),
            "--annotations",
            str(annotations),
            "--frames-root",
            str(tmp_path / "frames"),
            "--vggt-root",
            str(tmp_path / "vggt"),
            "--cloud-job-output",
            str(tmp_path / "cloud_job"),
            "--video-id",
            video_ids[0],
            "--video-id",
            video_ids[1],
            "--video-id",
            video_ids[2],
        ],
    )

    assert result.exit_code != 0
    assert "needs_human_review" in result.output
    summary = json.loads((workdir / "summary.json").read_text(encoding="utf-8"))
    allowed_statuses = {
        "done",
        "needs_human_review",
        "needs_vggt_bundle",
        "failed_soft",
        "blocked",
    }
    assert summary["status"] == "needs_human_review"
    assert set(summary["stage_status"].values()).issubset(allowed_statuses)
    assert summary["stage_status"]["local_media"] == "needs_human_review"
    assert any("--fetch-media" in failure for failure in summary["failures"])
    next_steps = (workdir / "NEXT_STEPS.md").read_text(encoding="utf-8")
    assert "--fetch-media" in next_steps


def test_run_from_video_ids_audits_cached_media_before_segments(tmp_path: Path):
    readme, manifest, video_ids = write_three_clip_catalog(tmp_path)
    catalog_dir = tmp_path / "catalog"
    media_dir = tmp_path / "media"
    inventory = media_dir / "media_inventory.parquet"
    annotations = tmp_path / "segments.jsonl"

    synced = runner.invoke(
        app,
        [
            "catalog",
            "sync",
            "--readme",
            str(readme),
            "--manifest",
            str(manifest),
            "--output",
            str(catalog_dir),
        ],
    )
    assert synced.exit_code == 0, synced.output
    for video_id in video_ids:
        fetched = runner.invoke(
            app,
            [
                "media",
                "fetch",
                "--catalog",
                str(catalog_dir / "catalog.parquet"),
                "--media-dir",
                str(media_dir),
                "--inventory",
                str(inventory),
                "--video-id",
                video_id,
            ],
        )
        assert fetched.exit_code == 0, fetched.output
        accepted = runner.invoke(
            app,
            [
                "segment",
                "accept",
                "--annotations",
                str(annotations),
                "--video-id",
                video_id,
                "--segment-id",
                "segment-001",
                "--start",
                "0",
                "--end",
                "1.5",
            ],
        )
        assert accepted.exit_code == 0, accepted.output

    create_synthetic_video(media_dir / f"{video_ids[1]}.mp4", frames=30, width=160, height=120, fps=12)

    workdir = tmp_path / "run-media-audit"
    result = runner.invoke(
        app,
        [
            "run",
            "--workdir",
            str(workdir),
            "--catalog",
            str(catalog_dir / "catalog.parquet"),
            "--media-dir",
            str(media_dir),
            "--media-inventory",
            str(inventory),
            "--annotations",
            str(annotations),
            "--frames-root",
            str(tmp_path / "frames"),
            "--vggt-root",
            str(tmp_path / "vggt"),
            "--cloud-job-output",
            str(tmp_path / "cloud_job"),
            "--frames",
            "6",
            "--video-id",
            video_ids[0],
            "--video-id",
            video_ids[1],
            "--video-id",
            video_ids[2],
        ],
    )

    assert result.exit_code != 0
    assert "needs_human_review" in result.output
    summary = json.loads((workdir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "needs_human_review"
    assert summary["stage_status"]["local_media"] == "done"
    assert summary["stage_status"]["media_audit"] == "needs_human_review"
    assert "accepted_segments" not in summary["stage_status"]
    assert Path(summary["artifacts"]["media_audit_report"]).exists()
    assert any("sha256 mismatch" in failure for failure in summary["failures"])
    next_steps = (workdir / "NEXT_STEPS.md").read_text(encoding="utf-8")
    assert "media audit" in next_steps.lower()
    assert "--fetch-media" in next_steps


def test_run_from_video_ids_stops_before_cloud_job_when_readiness_needs_review(tmp_path: Path):
    readme, manifest, video_ids = write_three_clip_catalog(tmp_path)
    write_video_with_blue_glitch(tmp_path / f"{video_ids[0]}.mp4")
    catalog_dir = tmp_path / "catalog"
    annotations = tmp_path / "segments.jsonl"
    cloud_job = tmp_path / "cloud_job"

    assert runner.invoke(
        app,
        [
            "catalog",
            "sync",
            "--readme",
            str(readme),
            "--manifest",
            str(manifest),
            "--output",
            str(catalog_dir),
        ],
    ).exit_code == 0
    for video_id in video_ids:
        accepted = runner.invoke(
            app,
            [
                "segment",
                "accept",
                "--annotations",
                str(annotations),
                "--video-id",
                video_id,
                "--segment-id",
                "segment-001",
                "--start",
                "0",
                "--end",
                "1.5",
            ],
        )
        assert accepted.exit_code == 0, accepted.output

    workdir = tmp_path / "run"
    result = runner.invoke(
        app,
        [
            "run",
            "--workdir",
            str(workdir),
            "--catalog",
            str(catalog_dir / "catalog.parquet"),
            "--media-dir",
            str(tmp_path / "media"),
            "--media-inventory",
            str(tmp_path / "media" / "media_inventory.parquet"),
            "--annotations",
            str(annotations),
            "--frames-root",
            str(tmp_path / "frames"),
            "--vggt-root",
            str(tmp_path / "vggt"),
            "--cloud-job-output",
            str(cloud_job),
            "--fetch-media",
            "--frames",
            "6",
            "--video-id",
            video_ids[0],
            "--video-id",
            video_ids[1],
            "--video-id",
            video_ids[2],
        ],
    )

    assert result.exit_code != 0
    assert "needs_human_review" in result.output
    summary = json.loads((workdir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "needs_human_review"
    assert summary["stage_status"]["data_readiness"] == "needs_human_review"
    assert Path(summary["artifacts"]["readiness_report"]).exists()
    assert "cloud_job_package" not in summary["artifacts"]
    assert not (cloud_job / "cloud_vggt_job.zip").exists()
    readiness = json.loads(Path(summary["artifacts"]["readiness_report"]).read_text(encoding="utf-8"))
    assert readiness["status"] == "needs_review"
    assert readiness["clips"][0]["checks"]["frame_quality"]["status"] == "warn"
    assert "color-dominant sampled frames: 1" in readiness["clips"][0]["checks"]["frame_quality"]["messages"]
    next_steps = (workdir / "NEXT_STEPS.md").read_text(encoding="utf-8")
    assert "readiness_report" in next_steps
    assert "Review the contact sheets" in next_steps


def test_run_from_video_ids_renders_review_when_bundles_exist(tmp_path: Path):
    readme, manifest, video_ids = write_three_clip_catalog(tmp_path)
    catalog_dir = tmp_path / "catalog"
    annotations = tmp_path / "segments.jsonl"
    frames_root = tmp_path / "frames"
    vggt_root = tmp_path / "vggt"

    assert runner.invoke(
        app,
        [
            "catalog",
            "sync",
            "--readme",
            str(readme),
            "--manifest",
            str(manifest),
            "--output",
            str(catalog_dir),
        ],
    ).exit_code == 0
    for video_id in video_ids:
        assert runner.invoke(
            app,
            [
                "segment",
                "accept",
                "--annotations",
                str(annotations),
                "--video-id",
                video_id,
                "--segment-id",
                "segment-001",
                "--start",
                "0",
                "--end",
                "1.5",
            ],
        ).exit_code == 0

    common_args = [
        "--catalog",
        str(catalog_dir / "catalog.parquet"),
        "--media-dir",
        str(tmp_path / "media"),
        "--media-inventory",
        str(tmp_path / "media" / "media_inventory.parquet"),
        "--annotations",
        str(annotations),
        "--frames-root",
        str(frames_root),
        "--vggt-root",
        str(vggt_root),
        "--cloud-job-output",
        str(tmp_path / "cloud_job"),
        "--fetch-media",
        "--frames",
        "6",
        "--video-id",
        video_ids[0],
        "--video-id",
        video_ids[1],
        "--video-id",
        video_ids[2],
        "--smooth",
    ]

    first = runner.invoke(app, ["run", "--workdir", str(tmp_path / "first"), *common_args])
    assert first.exit_code != 0
    assert "needs_vggt_bundle" in first.output

    for video_id in video_ids:
        mocked = runner.invoke(
            app,
            [
                "vggt",
                "mock",
                "--frame-manifest",
                str(frames_root / video_id / "segment-001" / "frames.json"),
                "--output",
                str(vggt_root / video_id / "segment-001"),
            ],
        )
        assert mocked.exit_code == 0, mocked.output

    done = runner.invoke(app, ["run", "--workdir", str(tmp_path / "done"), *common_args])

    assert done.exit_code == 0, done.output
    summary = json.loads((tmp_path / "done" / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "done"
    assert summary["stage_status"]["review_three"] == "done"
    assert Path(summary["artifacts"]["comparison_html"]).exists()
    report = Path(summary["artifacts"]["review_report"])
    assert report.exists()
    run_report = json.loads(report.read_text(encoding="utf-8"))
    assert all(Path(clip["review_html"]).exists() for clip in run_report["clips"])
    assert all(Path(clip["smoothing_output"]).exists() for clip in run_report["clips"])


def test_run_from_video_ids_renders_review_with_heatmaps(tmp_path: Path):
    readme, manifest, video_ids = write_three_clip_catalog(tmp_path)
    catalog_dir = tmp_path / "catalog"
    annotations = tmp_path / "segments.jsonl"
    frames_root = tmp_path / "frames"
    vggt_root = tmp_path / "vggt"

    assert runner.invoke(
        app,
        [
            "catalog",
            "sync",
            "--readme",
            str(readme),
            "--manifest",
            str(manifest),
            "--output",
            str(catalog_dir),
        ],
    ).exit_code == 0
    for video_id in video_ids:
        assert runner.invoke(
            app,
            [
                "segment",
                "accept",
                "--annotations",
                str(annotations),
                "--video-id",
                video_id,
                "--segment-id",
                "segment-001",
                "--start",
                "0",
                "--end",
                "1.5",
            ],
        ).exit_code == 0

    common_args = [
        "--catalog",
        str(catalog_dir / "catalog.parquet"),
        "--media-dir",
        str(tmp_path / "media"),
        "--media-inventory",
        str(tmp_path / "media" / "media_inventory.parquet"),
        "--annotations",
        str(annotations),
        "--frames-root",
        str(frames_root),
        "--vggt-root",
        str(vggt_root),
        "--cloud-job-output",
        str(tmp_path / "cloud_job"),
        "--fetch-media",
        "--frames",
        "6",
        "--video-id",
        video_ids[0],
        "--video-id",
        video_ids[1],
        "--video-id",
        video_ids[2],
        "--heatmaps",
    ]

    first = runner.invoke(app, ["run", "--workdir", str(tmp_path / "first"), *common_args])
    assert first.exit_code != 0
    assert "needs_vggt_bundle" in first.output

    for video_id in video_ids:
        mocked = runner.invoke(
            app,
            [
                "vggt",
                "mock",
                "--frame-manifest",
                str(frames_root / video_id / "segment-001" / "frames.json"),
                "--output",
                str(vggt_root / video_id / "segment-001"),
            ],
        )
        assert mocked.exit_code == 0, mocked.output

    done = runner.invoke(app, ["run", "--workdir", str(tmp_path / "done"), *common_args])

    assert done.exit_code == 0, done.output
    summary = json.loads((tmp_path / "done" / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "done"
    assert summary["inputs"]["heatmaps"] is True
    heatmap_manifests = summary["artifacts"]["heatmap_manifests"]
    assert len(heatmap_manifests) == 3
    assert all(Path(path).exists() for path in heatmap_manifests)
    report = Path(summary["artifacts"]["review_report"])
    run_report = json.loads(report.read_text(encoding="utf-8"))
    for clip in run_report["clips"]:
        html = Path(clip["review_html"]).read_text(encoding="utf-8")
        assert "heatmap-layer-select" in html
        assert "image-space heatmaps only" in html
        payload = extract_review_payload(html)
        assert payload["heatmapStatus"] == "available"
        assert payload["heatmapLayers"]


def test_run_review_pipeline_reports_failed_soft_heatmap_degradation(
    tmp_path: Path, monkeypatch
):
    frame_manifests: list[Path] = []
    bundles: list[Path] = []
    for index in range(3):
        frame_manifest, bundle, _summary, _video = create_mocked_summary(
            tmp_path, f"heatmap-degraded-video-{index}"
        )
        frame_manifests.append(frame_manifest)
        bundles.append(bundle)

    def fake_run_three_clip_review(
        frame_manifests_arg: list[Path],
        bundles_arg: list[Path],
        output_dir: Path,
        *,
        smooth: bool = False,
        generate_heatmaps: bool = False,
        export_video: bool = False,
    ) -> Path:
        assert frame_manifests_arg == frame_manifests
        assert bundles_arg == bundles
        assert generate_heatmaps is True
        output_dir.mkdir(parents=True, exist_ok=True)
        success_manifest = output_dir / "clip_00" / "heatmaps" / "heatmaps.json"
        success_manifest.parent.mkdir(parents=True, exist_ok=True)
        success_manifest.write_text("{}", encoding="utf-8")
        comparison = output_dir / "comparison.html"
        comparison.write_text("<html></html>", encoding="utf-8")
        for index in range(3):
            (output_dir / f"clip_{index:02d}.html").write_text(
                "<html></html>", encoding="utf-8"
            )
        report_path = output_dir / "three_clip_review_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "status": "done",
                    "comparison_html": str(comparison),
                    "clips": [
                        {
                            "video_id": "video-ok",
                            "segment_id": "segment-001",
                            "review_html": str(output_dir / "clip_00.html"),
                            "heatmap_manifest": str(success_manifest),
                            "heatmap_status": "done",
                            "heatmap_error": None,
                        },
                        {
                            "video_id": "video-bad",
                            "segment_id": "segment-001",
                            "review_html": str(output_dir / "clip_01.html"),
                            "heatmap_manifest": None,
                            "heatmap_status": "failed_soft",
                            "heatmap_error": "opencv failed\nlayer unavailable",
                        },
                        {
                            "video_id": "video-skipped",
                            "segment_id": "segment-002",
                            "review_html": str(output_dir / "clip_02.html"),
                            "heatmap_manifest": None,
                            "heatmap_status": "skipped",
                            "heatmap_error": None,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return report_path

    import fpv_vggt_lab.review as review

    monkeypatch.setattr(review, "run_three_clip_review", fake_run_three_clip_review)

    summary = run_review_pipeline(
        workdir=tmp_path / "run",
        frame_manifests=frame_manifests,
        bundles=bundles,
        heatmaps=True,
    )

    assert summary["status"] == "done"
    assert summary["stage_status"]["heatmaps"] == "failed_soft"
    assert len(summary["artifacts"]["heatmap_manifests"]) == 1
    assert Path(summary["artifacts"]["heatmap_manifests"][0]).exists()
    failures = summary["failures"]
    assert any(
        "video-bad/segment-001" in failure
        and "failed_soft" in failure
        and "opencv failed layer unavailable" in failure
        for failure in failures
    )
    assert any(
        "video-skipped/segment-002" in failure and "skipped" in failure
        for failure in failures
    )
    assert all("\n" not in failure for failure in failures)


def test_run_from_video_ids_imports_cloud_return_zip_and_renders_review(tmp_path: Path):
    readme, manifest, video_ids = write_three_clip_catalog(tmp_path)
    catalog_dir = tmp_path / "catalog"
    annotations = tmp_path / "segments.jsonl"
    frames_root = tmp_path / "frames"
    vggt_root = tmp_path / "vggt"

    assert runner.invoke(
        app,
        [
            "catalog",
            "sync",
            "--readme",
            str(readme),
            "--manifest",
            str(manifest),
            "--output",
            str(catalog_dir),
        ],
    ).exit_code == 0
    for video_id in video_ids:
        assert runner.invoke(
            app,
            [
                "segment",
                "accept",
                "--annotations",
                str(annotations),
                "--video-id",
                video_id,
                "--segment-id",
                "segment-001",
                "--start",
                "0",
                "--end",
                "1.5",
            ],
        ).exit_code == 0

    common_args = [
        "--catalog",
        str(catalog_dir / "catalog.parquet"),
        "--media-dir",
        str(tmp_path / "media"),
        "--media-inventory",
        str(tmp_path / "media" / "media_inventory.parquet"),
        "--annotations",
        str(annotations),
        "--frames-root",
        str(frames_root),
        "--vggt-root",
        str(vggt_root),
        "--cloud-job-output",
        str(tmp_path / "cloud_job"),
        "--fetch-media",
        "--frames",
        "6",
        "--video-id",
        video_ids[0],
        "--video-id",
        video_ids[1],
        "--video-id",
        video_ids[2],
        "--smooth",
    ]

    first = runner.invoke(app, ["run", "--workdir", str(tmp_path / "first"), *common_args])
    assert first.exit_code != 0

    returned_root = tmp_path / "returned"
    returned_bundles = returned_root / "bundles"
    for video_id in video_ids:
        mocked = runner.invoke(
            app,
            [
                "vggt",
                "mock",
                "--frame-manifest",
                str(frames_root / video_id / "segment-001" / "frames.json"),
                "--output",
                str(returned_bundles / video_id / "segment-001"),
            ],
        )
        assert mocked.exit_code == 0, mocked.output
    bundle_zip = returned_root / "bundles.zip"
    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in returned_bundles.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(returned_root))
    (returned_root / "cloud_summary.json").write_text(
        json.dumps({"status": "done"}), encoding="utf-8"
    )
    (returned_root / "cloud_run.log").write_text("done\n", encoding="utf-8")

    done = runner.invoke(
        app,
        [
            "run",
            "--workdir",
            str(tmp_path / "done"),
            "--cloud-return",
            str(bundle_zip),
            "--cloud-import-report",
            str(tmp_path / "done" / "cloud_import.json"),
            *common_args,
        ],
    )

    assert done.exit_code == 0, done.output
    summary = json.loads((tmp_path / "done" / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "done"
    assert summary["stage_status"]["cloud_return_import"] == "done"
    assert Path(summary["artifacts"]["cloud_import_report"]).exists()
    assert Path(summary["artifacts"]["comparison_html"]).exists()
    assert all((vggt_root / video_id / "segment-001" / "metadata.json").exists() for video_id in video_ids)


def test_run_from_video_ids_skips_unexpected_bundles_in_cloud_return(tmp_path: Path):
    readme, manifest, video_ids = write_three_clip_catalog(tmp_path)
    catalog_dir = tmp_path / "catalog"
    annotations = tmp_path / "segments.jsonl"
    frames_root = tmp_path / "frames"
    vggt_root = tmp_path / "vggt"

    assert runner.invoke(
        app,
        [
            "catalog",
            "sync",
            "--readme",
            str(readme),
            "--manifest",
            str(manifest),
            "--output",
            str(catalog_dir),
        ],
    ).exit_code == 0
    for video_id in video_ids:
        assert runner.invoke(
            app,
            [
                "segment",
                "accept",
                "--annotations",
                str(annotations),
                "--video-id",
                video_id,
                "--segment-id",
                "segment-001",
                "--start",
                "0",
                "--end",
                "1.5",
            ],
        ).exit_code == 0

    common_args = [
        "--catalog",
        str(catalog_dir / "catalog.parquet"),
        "--media-dir",
        str(tmp_path / "media"),
        "--media-inventory",
        str(tmp_path / "media" / "media_inventory.parquet"),
        "--annotations",
        str(annotations),
        "--frames-root",
        str(frames_root),
        "--vggt-root",
        str(vggt_root),
        "--cloud-job-output",
        str(tmp_path / "cloud_job"),
        "--fetch-media",
        "--frames",
        "6",
        "--video-id",
        video_ids[0],
        "--video-id",
        video_ids[1],
        "--video-id",
        video_ids[2],
        "--smooth",
    ]

    first = runner.invoke(app, ["run", "--workdir", str(tmp_path / "first"), *common_args])
    assert first.exit_code != 0

    returned_root = tmp_path / "returned"
    returned_bundles = returned_root / "bundles"
    for video_id in video_ids:
        mocked = runner.invoke(
            app,
            [
                "vggt",
                "mock",
                "--frame-manifest",
                str(frames_root / video_id / "segment-001" / "frames.json"),
                "--output",
                str(returned_bundles / video_id / "segment-001"),
            ],
        )
        assert mocked.exit_code == 0, mocked.output
    _extra_manifest, extra_bundle, _extra_summary, _extra_video = create_mocked_summary(
        tmp_path, "unexpected-extra-video"
    )
    extra_destination = returned_bundles / "unexpected-extra-video" / "segment-001"
    extra_destination.parent.mkdir(parents=True, exist_ok=True)
    for path in extra_bundle.iterdir():
        if path.is_file():
            target = extra_destination / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())

    bundle_zip = returned_root / "bundles.zip"
    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in returned_bundles.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(returned_root))
    (returned_root / "cloud_summary.json").write_text(
        json.dumps({"status": "done"}), encoding="utf-8"
    )
    (returned_root / "cloud_run.log").write_text("done with extra bundle\n", encoding="utf-8")

    run_dir = tmp_path / "done"
    cloud_import_report = run_dir / "cloud_import.json"
    done = runner.invoke(
        app,
        [
            "run",
            "--workdir",
            str(run_dir),
            "--cloud-return",
            str(bundle_zip),
            "--cloud-import-report",
            str(cloud_import_report),
            *common_args,
        ],
    )

    assert done.exit_code == 0, done.output
    assert not (vggt_root / "unexpected-extra-video").exists()
    report = json.loads(cloud_import_report.read_text(encoding="utf-8"))
    assert report["status"] == "done"
    assert report["imported_count"] == 3
    assert report["skipped_unexpected_count"] == 1
    extra_clip = next(
        clip for clip in report["clips"] if clip["metadata"]["video_id"] == "unexpected-extra-video"
    )
    assert extra_clip["copied"] is False
    assert extra_clip["skipped_reason"] == "unexpected bundle for this run"


def test_run_from_video_ids_audits_after_partial_cloud_return(tmp_path: Path):
    readme, manifest, video_ids = write_three_clip_catalog(tmp_path)
    catalog_dir = tmp_path / "catalog"
    annotations = tmp_path / "segments.jsonl"
    frames_root = tmp_path / "frames"
    vggt_root = tmp_path / "vggt"

    assert runner.invoke(
        app,
        [
            "catalog",
            "sync",
            "--readme",
            str(readme),
            "--manifest",
            str(manifest),
            "--output",
            str(catalog_dir),
        ],
    ).exit_code == 0
    for video_id in video_ids:
        assert runner.invoke(
            app,
            [
                "segment",
                "accept",
                "--annotations",
                str(annotations),
                "--video-id",
                video_id,
                "--segment-id",
                "segment-001",
                "--start",
                "0",
                "--end",
                "1.5",
            ],
        ).exit_code == 0

    common_args = [
        "--catalog",
        str(catalog_dir / "catalog.parquet"),
        "--media-dir",
        str(tmp_path / "media"),
        "--media-inventory",
        str(tmp_path / "media" / "media_inventory.parquet"),
        "--annotations",
        str(annotations),
        "--frames-root",
        str(frames_root),
        "--vggt-root",
        str(vggt_root),
        "--cloud-job-output",
        str(tmp_path / "cloud_job"),
        "--fetch-media",
        "--frames",
        "6",
        "--video-id",
        video_ids[0],
        "--video-id",
        video_ids[1],
        "--video-id",
        video_ids[2],
        "--smooth",
    ]

    first = runner.invoke(app, ["run", "--workdir", str(tmp_path / "first"), *common_args])
    assert first.exit_code != 0

    returned_root = tmp_path / "returned"
    returned_bundles = returned_root / "bundles"
    for video_id in video_ids[:2]:
        mocked = runner.invoke(
            app,
            [
                "vggt",
                "mock",
                "--frame-manifest",
                str(frames_root / video_id / "segment-001" / "frames.json"),
                "--output",
                str(returned_bundles / video_id / "segment-001"),
            ],
        )
        assert mocked.exit_code == 0, mocked.output
    invalid_bundle = returned_bundles / video_ids[2] / "segment-001"
    invalid_bundle.mkdir(parents=True)
    (invalid_bundle / "metadata.json").write_text("{}", encoding="utf-8")

    bundle_zip = returned_root / "bundles.zip"
    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in returned_bundles.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(returned_root))
    (returned_root / "cloud_summary.json").write_text(
        json.dumps({"status": "failed_soft"}), encoding="utf-8"
    )
    (returned_root / "cloud_run.log").write_text("one clip failed\n", encoding="utf-8")

    run_dir = tmp_path / "partial"
    result = runner.invoke(
        app,
        [
            "run",
            "--workdir",
            str(run_dir),
            "--cloud-return",
            str(bundle_zip),
            "--cloud-import-report",
            str(run_dir / "cloud_import.json"),
            *common_args,
        ],
    )

    assert result.exit_code != 0
    assert "needs_vggt_bundle" in result.output
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "needs_vggt_bundle"
    assert summary["stage_status"]["cloud_return_import"] == "failed_soft"
    assert summary["stage_status"]["audit_three"] == "needs_vggt_bundle"
    assert Path(summary["artifacts"]["audit_report"]).exists()
    assert Path(summary["artifacts"]["cloud_import_report"]).exists()
    assert not (vggt_root / video_ids[0] / "segment-001" / "metadata.json").exists()
    assert not (vggt_root / video_ids[1] / "segment-001" / "metadata.json").exists()
    assert not (vggt_root / video_ids[2] / "segment-001" / "metadata.json").exists()
    report = json.loads((run_dir / "cloud_import.json").read_text(encoding="utf-8"))
    assert report["cloud_summary_audit"]["status"] == "needs_review"
    assert report["cloud_summary_audit"]["issues"] == ["cloud_summary status is failed_soft"]


def test_run_from_video_ids_rejects_bundle_with_mismatched_frame_metadata(tmp_path: Path):
    readme, manifest, video_ids = write_three_clip_catalog(tmp_path)
    catalog_dir = tmp_path / "catalog"
    annotations = tmp_path / "segments.jsonl"
    frames_root = tmp_path / "frames"
    vggt_root = tmp_path / "vggt"

    assert runner.invoke(
        app,
        [
            "catalog",
            "sync",
            "--readme",
            str(readme),
            "--manifest",
            str(manifest),
            "--output",
            str(catalog_dir),
        ],
    ).exit_code == 0
    for video_id in video_ids:
        assert runner.invoke(
            app,
            [
                "segment",
                "accept",
                "--annotations",
                str(annotations),
                "--video-id",
                video_id,
                "--segment-id",
                "segment-001",
                "--start",
                "0",
                "--end",
                "1.5",
            ],
        ).exit_code == 0

    common_args = [
        "--catalog",
        str(catalog_dir / "catalog.parquet"),
        "--media-dir",
        str(tmp_path / "media"),
        "--media-inventory",
        str(tmp_path / "media" / "media_inventory.parquet"),
        "--annotations",
        str(annotations),
        "--frames-root",
        str(frames_root),
        "--vggt-root",
        str(vggt_root),
        "--cloud-job-output",
        str(tmp_path / "cloud_job"),
        "--fetch-media",
        "--frames",
        "6",
        "--video-id",
        video_ids[0],
        "--video-id",
        video_ids[1],
        "--video-id",
        video_ids[2],
        "--smooth",
    ]

    first = runner.invoke(app, ["run", "--workdir", str(tmp_path / "first"), *common_args])
    assert first.exit_code != 0

    for video_id in video_ids:
        mocked = runner.invoke(
            app,
            [
                "vggt",
                "mock",
                "--frame-manifest",
                str(frames_root / video_id / "segment-001" / "frames.json"),
                "--output",
                str(vggt_root / video_id / "segment-001"),
            ],
        )
        assert mocked.exit_code == 0, mocked.output

    bad_metadata_path = vggt_root / video_ids[1] / "segment-001" / "metadata.json"
    bad_metadata = json.loads(bad_metadata_path.read_text(encoding="utf-8"))
    bad_metadata["frame_indices"] = [index + 1000 for index in bad_metadata["frame_indices"]]
    bad_metadata_path.write_text(json.dumps(bad_metadata), encoding="utf-8")

    run_dir = tmp_path / "mismatch"
    result = runner.invoke(app, ["run", "--workdir", str(run_dir), *common_args])

    assert result.exit_code != 0
    assert "needs_vggt_bundle" in result.output
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "needs_vggt_bundle"
    assert summary["stage_status"]["audit_three"] == "needs_vggt_bundle"
    audit = json.loads(Path(summary["artifacts"]["audit_report"]).read_text(encoding="utf-8"))
    mismatched_clip = next(clip for clip in audit["clips"] if clip["bundle"].endswith(f"{video_ids[1]}\\segment-001"))
    assert any("frame_indices do not match frame manifest" in error for error in mismatched_clip["errors"])
