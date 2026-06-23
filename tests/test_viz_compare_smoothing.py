import json
from pathlib import Path

import cv2
import numpy as np
from typer.testing import CliRunner

from fpv_vggt_lab.cli import app
from fpv_vggt_lab.heatmaps import generate_heatmaps_from_manifest
from fpv_vggt_lab.review import run_three_clip_review


runner = CliRunner()


def extract_review_payload(html: str) -> dict:
    start = html.index("<script>const reviewPayload = ") + len("<script>const reviewPayload = ")
    end = html.index(";\nconst frames=", start)
    return json.loads(html[start:end])


def create_mocked_summary(tmp_path: Path, name: str) -> tuple[Path, Path, Path, Path]:
    video = tmp_path / f"{name}.mp4"
    frames_dir = tmp_path / f"{name}_frames"
    bundle_dir = tmp_path / f"{name}_bundle"
    summary_path = tmp_path / f"{name}_summary.json"

    assert runner.invoke(
        app,
        [
            "synthetic-video",
            "create",
            "--output",
            str(video),
            "--frames",
            "24",
            "--width",
            "160",
            "--height",
            "120",
        ],
    ).exit_code == 0
    assert runner.invoke(
        app,
        [
            "frames",
            "sample",
            "--video",
            str(video),
            "--output",
            str(frames_dir),
            "--count",
            "6",
            "--video-id",
            name,
            "--segment-id",
            "segment-001",
        ],
    ).exit_code == 0
    assert runner.invoke(
        app,
        [
            "vggt",
            "mock",
            "--frame-manifest",
            str(frames_dir / "frames.json"),
            "--output",
            str(bundle_dir),
        ],
    ).exit_code == 0
    assert runner.invoke(
        app,
        [
            "reconstruct",
            "summarize",
            "--bundle",
            str(bundle_dir),
            "--output",
            str(summary_path),
        ],
    ).exit_code == 0
    return frames_dir / "frames.json", bundle_dir, summary_path, video


def test_local_review_html_contains_synchronized_review_payload(tmp_path: Path):
    frame_manifest, bundle_dir, summary_path, _video = create_mocked_summary(
        tmp_path, "review-video"
    )
    output = tmp_path / "review.html"

    result = runner.invoke(
        app,
        [
            "viz",
            "render",
            "--frame-manifest",
            str(frame_manifest),
            "--bundle",
            str(bundle_dir),
            "--summary",
            str(summary_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    html = output.read_text(encoding="utf-8")
    assert "Relative VGGT Review" in html
    assert "Local diagnostic workbench" in html
    assert "review-shell" in html
    assert "review-metric-strip" in html
    assert "metric-card" in html
    assert "media-card" in html
    assert "viz-card" in html
    assert "control-deck" in html
    assert "safety-strip" in html
    assert "<video id='source-video'" in html
    assert "play-segment" in html
    assert "toggleSegmentPlayback" in html
    assert "Playback is ready. Use the native video controls" in html
    assert "segment-start" in html
    assert "segment-end" in html
    assert "prev-sample" in html
    assert "next-sample" in html
    assert "playback-rate" in html
    assert "loop-segment" in html
    assert "video-progress" in html
    assert "video-time-label" in html
    assert "animate-drone" in html
    assert "point-budget" in html
    assert "show-points" in html
    assert "sixDofStates" in html
    assert "cameraOrientations" in html
    assert "drawDronePose" in html
    assert "orbitProject" in html
    assert "no geolocation" in html
    assert "reliability-timeline" in html
    assert "descriptor-panel" in html
    assert "provenance-panel" in html
    assert "reliability-bars" in html
    assert "renderReliabilityTimeline" in html
    assert "setActiveReliabilityBar" in html
    assert "data-frame-index" in html
    assert "cameraPath" in html
    assert "pointCloud" in html
    assert "sixdof-state-panel" in html
    assert "6DoF State-Space Diagnostic" in html
    assert "state_vector" in html
    assert "updateSixDofStatePanel" in html
    assert "drawDroneBody" in html
    assert "sixdof-drone-animation-panel" in html
    assert "drone-state-canvas" in html
    assert "drawDroneStateAnimation" in html
    assert "drawDroneStateFrustum" in html
    assert "droneAttitudeProject" in html
    assert "State-space drone animation" in html
    assert "cameraConvention" in html
    assert "visual_up" in html
    assert "bundle-quality-panel" in html
    assert "Bundle Quality" in html
    assert "relative-depth fallback" in html
    assert "console.log" not in html
    payload = extract_review_payload(html)
    first_frame_path = payload["frames"][0]["path"]
    expected_frame_path = Path(
        json.loads(frame_manifest.read_text(encoding="utf-8"))["frames"][0]["path"]
    ).relative_to(output.parent).as_posix()
    assert first_frame_path == expected_frame_path
    assert not Path(first_frame_path).is_absolute()
    expected_video_path = Path(
        json.loads(frame_manifest.read_text(encoding="utf-8"))["source_video"]
    ).relative_to(output.parent).as_posix()
    assert payload["sourceVideo"] == expected_video_path
    assert payload["sourceVideo"].endswith(".mp4")
    assert len(payload["sixDofStates"]) == len(payload["frames"])
    assert len(payload["cameraOrientations"]) == len(payload["frames"])
    first_state = payload["sixDofStates"][0]
    assert len(first_state["state_vector"]) == 6
    assert len(first_state["euler_rpy_rad"]) == 3
    assert len(first_state["velocity_relative_per_sec"]) == 3
    assert first_state["coordinate_convention"] == "camera +Z forward, +X right, visual up is -Y"
    assert payload["cameraConvention"]["forward"] == [0, 0, 1]
    assert payload["cameraConvention"]["visual_up"] == [0, -1, 0]
    assert payload["pointCloudStats"]["count"] == len(payload["pointCloud"])
    assert payload["bundleQuality"]["point_count"] == len(payload["pointCloud"])
    assert payload["bundleQuality"]["uses_relative_depth_fallback"] is True


def test_local_review_html_embeds_image_space_heatmap_controls(tmp_path: Path):
    frame_manifest, bundle_dir, summary_path, _video = create_mocked_summary(
        tmp_path, "heatmap-review-video"
    )
    heatmap_manifest = generate_heatmaps_from_manifest(
        frame_manifest_path=frame_manifest,
        output_dir=tmp_path / "heatmaps",
    )
    output = tmp_path / "review.html"

    result = runner.invoke(
        app,
        [
            "viz",
            "render",
            "--frame-manifest",
            str(frame_manifest),
            "--bundle",
            str(bundle_dir),
            "--summary",
            str(summary_path),
            "--heatmap-manifest",
            str(heatmap_manifest),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    html = output.read_text(encoding="utf-8")
    assert "heatmap-layer-select" in html
    assert "heatmap-opacity" in html
    assert "heatmap-overlay" in html
    assert "heatmapLayers" in html
    assert "image-space heatmaps only" in html
    assert "drawHeatmapOverlay" in html
    payload = extract_review_payload(html)
    assert payload["heatmapStatus"] == "available"
    assert payload["heatmapWarnings"] == [
        "image-space diagnostics only",
        "no geolocation",
        "no meters",
        "local-only media",
    ]
    assert {row["layer_name"] for row in payload["heatmapLayers"]} == {
        "frame_difference",
        "optical_flow",
        "blur",
        "visibility_change",
    }
    first_heatmap_path = payload["heatmapLayers"][0]["heatmap_path"]
    assert not Path(first_heatmap_path).is_absolute()
    assert first_heatmap_path.endswith(".png")


def test_local_review_html_fails_soft_without_leaking_heatmap_manifest_path(tmp_path: Path):
    frame_manifest, bundle_dir, summary_path, _video = create_mocked_summary(
        tmp_path, "missing-heatmap-review-video"
    )
    missing_heatmap_manifest = tmp_path / "missing-heatmaps" / "heatmaps.json"
    output = tmp_path / "review.html"

    result = runner.invoke(
        app,
        [
            "viz",
            "render",
            "--frame-manifest",
            str(frame_manifest),
            "--bundle",
            str(bundle_dir),
            "--summary",
            str(summary_path),
            "--heatmap-manifest",
            str(missing_heatmap_manifest),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    html = output.read_text(encoding="utf-8")
    payload = extract_review_payload(html)
    warning_text = " | ".join(payload["heatmapWarnings"])
    escaped_tmp_path = str(tmp_path).replace("\\", "\\\\")
    escaped_missing_path = str(missing_heatmap_manifest).replace("\\", "\\\\")

    assert payload["heatmapStatus"] == "failed_soft"
    assert payload["heatmapWarnings"] == [
        "heatmap manifest unavailable: failed to read or validate heatmap manifest"
    ]
    assert str(tmp_path) not in warning_text
    assert str(missing_heatmap_manifest) not in warning_text
    assert str(tmp_path) not in html
    assert str(missing_heatmap_manifest) not in html
    assert escaped_tmp_path not in html
    assert escaped_missing_path not in html


def test_local_review_html_supports_point_attributes_and_orbit_controls(tmp_path: Path):
    frame_manifest, bundle_dir, summary_path, _video = create_mocked_summary(
        tmp_path, "point-attributes-video"
    )
    points_path = bundle_dir / "points.npz"
    points = np.load(points_path)["points"]
    colors = np.column_stack(
        [
            np.linspace(40, 240, len(points)),
            np.linspace(220, 50, len(points)),
            np.full(len(points), 150),
        ]
    ).astype(np.uint8)
    confidence = np.linspace(0.2, 1.0, len(points), dtype=np.float32)
    depth = np.linalg.norm(points - points.mean(axis=0, keepdims=True), axis=1).astype(np.float32)
    np.savez_compressed(
        points_path,
        points=points,
        point_colors_rgb=colors,
        point_confidence=confidence,
        point_depth=depth,
    )
    output = tmp_path / "review.html"

    result = runner.invoke(
        app,
        [
            "viz",
            "render",
            "--frame-manifest",
            str(frame_manifest),
            "--bundle",
            str(bundle_dir),
            "--summary",
            str(summary_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    html = output.read_text(encoding="utf-8")
    assert "point-color-mode" in html
    assert "confidence-min" in html
    assert "relative-depth-max" in html
    assert "show-frustums" in html
    assert "reset-orbit" in html
    assert "orbitProject" in html
    assert "drawCameraFrustum" in html
    assert "geometry.addEventListener('wheel'" in html
    assert "VGGT-frame 6DoF" in html
    assert "bundle-quality-panel" in html
    assert "RGB point colors: available" in html
    payload = extract_review_payload(html)
    assert payload["pointCloudStats"]["has_rgb"] is True
    assert payload["pointCloudStats"]["has_confidence"] is True
    assert payload["pointCloudStats"]["has_depth"] is True
    assert payload["pointCloudStats"]["uses_relative_depth_fallback"] is False
    assert len(payload["pointColors"]) == len(payload["pointCloud"])
    assert len(payload["pointConfidence"]) == len(payload["pointCloud"])
    assert len(payload["pointDepth"]) == len(payload["pointCloud"])
    assert payload["pointColors"][0] == colors[0].tolist()
    assert payload["bundleQuality"]["rgb_status"] == "available"


def test_side_by_side_mp4_export_is_readable(tmp_path: Path):
    frame_manifest, bundle_dir, summary_path, _video = create_mocked_summary(
        tmp_path, "export-video"
    )
    output = tmp_path / "side_by_side.mp4"

    result = runner.invoke(
        app,
        [
            "viz",
            "export-video",
            "--frame-manifest",
            str(frame_manifest),
            "--bundle",
            str(bundle_dir),
            "--summary",
            str(summary_path),
            "--output",
            str(output),
            "--fps",
            "4",
            "--width",
            "640",
            "--height",
            "360",
            "--point-budget",
            "500",
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) >= 1
        ok, frame = capture.read()
        assert ok
        assert frame.shape[0] == 360
        assert frame.shape[1] == 640
    finally:
        capture.release()


def test_comparison_page_and_reliability_gated_smoothing(tmp_path: Path):
    summaries: list[Path] = []
    bundle_dir: Path | None = None
    for index in range(3):
        _frames, bundle, summary, _video = create_mocked_summary(
            tmp_path, f"compare-video-{index}"
        )
        summaries.append(summary)
        bundle_dir = bundle

    compare_output = tmp_path / "compare.html"
    result = runner.invoke(
        app,
        [
            "compare",
            "render",
            "--output",
            str(compare_output),
            "--summary",
            str(summaries[0]),
            "--summary",
            str(summaries[1]),
            "--summary",
            str(summaries[2]),
        ],
    )
    assert result.exit_code == 0, result.output
    html = compare_output.read_text(encoding="utf-8")
    assert "Three-Clip Reconstruction Comparison" in html
    assert html.count("<tr") >= 4
    assert "normalized_path_length" in html
    assert "comparisonPayload" in html
    assert "comparison-bars" in html
    assert "comparison-table" in html
    assert "renderComparisonBars" in html
    assert "data-comparison-metric" in html

    smooth_npz = tmp_path / "smoothed.npz"
    smooth_meta = tmp_path / "smoothed.json"
    good = runner.invoke(
        app,
        [
            "smooth",
            "poses",
            "--bundle",
            str(bundle_dir),
            "--summary",
            str(summaries[-1]),
            "--output",
            str(smooth_npz),
            "--metadata-output",
            str(smooth_meta),
        ],
    )
    assert good.exit_code == 0, good.output
    assert smooth_npz.exists()
    metadata = json.loads(smooth_meta.read_text(encoding="utf-8"))
    assert metadata["label"] == "experimental VGGT-frame relative pose smoother"
    assert metadata["not_physical_dynamics"] is True

    failed_summary = tmp_path / "failed_summary.json"
    failed_data = json.loads(summaries[0].read_text(encoding="utf-8"))
    failed_data["reliability"]["label"] = "failed"
    failed_data["reliability"]["safe_to_use_for_descriptors"] = False
    failed_summary.write_text(json.dumps(failed_data), encoding="utf-8")
    blocked = runner.invoke(
        app,
        [
            "smooth",
            "poses",
            "--bundle",
            str(bundle_dir),
            "--summary",
            str(failed_summary),
            "--output",
            str(tmp_path / "blocked.npz"),
        ],
    )
    assert blocked.exit_code != 0
    assert "not safe" in blocked.output.lower()


def test_review_and_comparison_pages_escape_external_payload_strings(tmp_path: Path):
    frame_manifest, bundle_dir, summary_path, _video = create_mocked_summary(
        tmp_path, "escape-video"
    )
    marker = "clip</script><img src=x onerror=alert(1)>"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["video_id"] = marker
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    review_output = tmp_path / "review.html"
    review = runner.invoke(
        app,
        [
            "viz",
            "render",
            "--frame-manifest",
            str(frame_manifest),
            "--bundle",
            str(bundle_dir),
            "--summary",
            str(summary_path),
            "--output",
            str(review_output),
        ],
    )
    assert review.exit_code == 0, review.output
    review_html = review_output.read_text(encoding="utf-8")
    assert "<img src=x" not in review_html
    assert "\\u003c/script\\u003e" in review_html
    assert extract_review_payload(review_html)["summary"]["video_id"] == marker

    compare_output = tmp_path / "compare.html"
    compare = runner.invoke(
        app,
        [
            "compare",
            "render",
            "--output",
            str(compare_output),
            "--summary",
            str(summary_path),
            "--summary",
            str(summary_path),
            "--summary",
            str(summary_path),
        ],
    )
    assert compare.exit_code == 0, compare.output
    comparison_html = compare_output.read_text(encoding="utf-8")
    assert "<img src=x" not in comparison_html
    assert "&lt;/script&gt;" in comparison_html
    assert "\\u003c/script\\u003e" in comparison_html


def test_three_clip_review_embeds_smoothed_path_toggle_for_safe_clips(tmp_path: Path):
    frame_manifests: list[Path] = []
    bundles: list[Path] = []
    for index in range(3):
        frame_manifest, bundle, _summary, _video = create_mocked_summary(
            tmp_path, f"smoothed-toggle-video-{index}"
        )
        frame_manifests.append(frame_manifest)
        bundles.append(bundle)

    report_path = run_three_clip_review(
        frame_manifests=frame_manifests,
        bundles=bundles,
        output_dir=tmp_path / "review",
        smooth=True,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    first_review = Path(report["clips"][0]["review_html"])
    html = first_review.read_text(encoding="utf-8")
    assert "use-smoothed-path" in html
    assert "smoothedCameraPath" in html
    assert "activeCameraPath" in html
    assert "Smoothed pose path" in html
    payload = extract_review_payload(html)
    assert payload["smoothedPathStatus"] == "done"
    assert len(payload["smoothedCameraPath"]) == len(payload["cameraPath"])
    assert payload["bundleQuality"]["smoothed_path"] == "available"


def test_three_clip_review_skips_unsafe_smoothing_without_blocking_outputs(tmp_path: Path):
    frame_manifests: list[Path] = []
    bundles: list[Path] = []
    for index in range(3):
        frame_manifest, bundle, _summary, _video = create_mocked_summary(
            tmp_path, f"review-smooth-video-{index}"
        )
        frame_manifests.append(frame_manifest)
        bundles.append(bundle)

    cameras_path = bundles[0] / "cameras.npz"
    cameras = np.load(cameras_path)
    camera_centers = cameras["camera_centers"].copy()
    camera_centers[3] += np.asarray([100.0, 0.0, 0.0], dtype=np.float32)
    np.savez_compressed(
        cameras_path,
        camera_centers=camera_centers,
        quaternions_xyzw=cameras["quaternions_xyzw"],
        valid_pose_mask=cameras["valid_pose_mask"],
        pose_confidence=cameras["pose_confidence"],
    )

    report_path = run_three_clip_review(
        frame_manifests=frame_manifests,
        bundles=bundles,
        output_dir=tmp_path / "review",
        smooth=True,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "done"
    assert Path(report["comparison_html"]).exists()
    assert report["clips"][0]["smoothing_output"] is None
    assert report["clips"][0]["smoothing_status"] == "skipped"
    assert "not safe" in report["clips"][0]["smoothing_error"].lower()
    assert report["clips"][1]["smoothing_status"] == "done"
    assert Path(report["clips"][1]["smoothing_output"]).exists()
    assert report["clips"][2]["smoothing_status"] == "done"
    assert Path(report["clips"][2]["smoothing_output"]).exists()


def test_three_clip_review_can_generate_heatmap_artifacts(tmp_path: Path):
    frame_manifests: list[Path] = []
    bundles: list[Path] = []
    for index in range(3):
        frame_manifest, bundle, _summary, _video = create_mocked_summary(
            tmp_path, f"heatmap-run-video-{index}"
        )
        frame_manifests.append(frame_manifest)
        bundles.append(bundle)

    output_dir = tmp_path / "review-run"
    report_path = run_three_clip_review(
        frame_manifests,
        bundles,
        output_dir,
        generate_heatmaps=True,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "done"
    assert all(clip["heatmap_status"] == "done" for clip in report["clips"])
    for clip in report["clips"]:
        heatmap_manifest = Path(clip["heatmap_manifest"])
        assert heatmap_manifest.exists()
        html = Path(clip["review_html"]).read_text(encoding="utf-8")
        assert "heatmap-layer-select" in html
        assert "image-space heatmaps only" in html
