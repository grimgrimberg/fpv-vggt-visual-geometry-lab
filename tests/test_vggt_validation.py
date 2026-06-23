import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from fpv_vggt_lab.cli import app
from tests.test_viz_compare_smoothing import create_mocked_summary


runner = CliRunner()


def test_vggt_validation_report_and_fail_soft_summary(tmp_path: Path):
    bad_bundle = tmp_path / "bad_bundle"
    bad_bundle.mkdir()
    (bad_bundle / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "video_id": "bad-video",
                "segment_id": "segment-001",
                "source_tool": "huggingface-space",
                "source_url_or_repo": "https://huggingface.co/demo/vggt",
                "generated_at": "2026-06-22T00:00:00+00:00",
                "frame_indices": [0],
                "frame_timestamps_sec": [0.0],
                "warnings": [],
                "coordinate_frame": "VGGT relative coordinate frame",
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "validation.json"
    summary_path = tmp_path / "summary.json"

    validation = runner.invoke(
        app,
        [
            "vggt",
            "validate",
            "--bundle",
            str(bad_bundle),
            "--report",
            str(report_path),
        ],
    )
    assert validation.exit_code != 0
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["valid"] is False
    assert any("cameras.npz" in error for error in report["errors"])
    assert report["metadata"]["source_tool"] == "huggingface-space"

    summarized = runner.invoke(
        app,
        [
            "reconstruct",
            "summarize",
            "--bundle",
            str(bad_bundle),
            "--output",
            str(summary_path),
        ],
    )
    assert summarized.exit_code == 0, summarized.output
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["video_id"] == "bad-video"
    assert summary["reliability"]["label"] == "failed"
    assert summary["reliability"]["safe_to_use_for_descriptors"] is False
    assert "bundle_validation_failed" in summary["reliability"]["failure_flags"]


def test_vggt_validation_rejects_secret_looking_provenance(tmp_path: Path):
    _frame_manifest, bundle, _summary, _video = create_mocked_summary(
        tmp_path, "secret-provenance-video"
    )
    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["source_tool"] = "runpod"
    metadata["source_url_or_repo"] = "https://example.test/vggt?token=secret-value"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    report_path = tmp_path / "secret_validation.json"

    validation = runner.invoke(
        app,
        [
            "vggt",
            "validate",
            "--bundle",
            str(bundle),
            "--report",
            str(report_path),
        ],
    )

    assert validation.exit_code != 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["valid"] is False
    assert "provenance contains secret-looking text" in report["errors"]


def test_vggt_inspect_prints_plain_english_bundle_contract(tmp_path: Path):
    _frame_manifest, bundle, _summary, _video = create_mocked_summary(
        tmp_path, "inspect-video"
    )
    points_path = bundle / "points.npz"
    points = np.load(points_path)["points"]
    colors = np.tile(np.asarray([[10, 120, 240]], dtype=np.uint8), (len(points), 1))
    confidence = np.linspace(0.25, 0.95, len(points), dtype=np.float32)
    depth = np.linspace(0.1, 1.5, len(points), dtype=np.float32)
    np.savez_compressed(
        points_path,
        points=points,
        point_colors_rgb=colors,
        point_confidence=confidence,
        point_depth=depth,
    )

    result = runner.invoke(app, ["vggt", "inspect", "--bundle", str(bundle)])

    assert result.exit_code == 0, result.output
    assert "VGGT bundle inspection" in result.output
    assert "Video: inspect-video" in result.output
    assert "Frames: 6" in result.output
    assert "camera_centers: present shape=(6, 3)" in result.output
    assert f"points: present shape=({len(points)}, 3)" in result.output
    assert "point_colors_rgb: present" in result.output
    assert "point_confidence: present" in result.output
    assert "point_depth: present" in result.output
    assert "RGB point colors: available" in result.output
    assert "Confidence filter: uses point_confidence" in result.output
    assert "Depth filter: uses point_depth" in result.output
    assert "No meters, no geolocation" in result.output


@pytest.mark.parametrize(
    ("metadata_update", "expected_error"),
    [
        ({"frame_indices": []}, "frame_indices must not be empty"),
        ({"frame_indices": [0, 5, 5, 14, 18, 23]}, "frame_indices must be strictly increasing"),
        (
            {"frame_timestamps_sec": [0.0, 0.4, 0.3, 1.0, 1.5, 1.9]},
            "frame_timestamps_sec must be non-decreasing",
        ),
    ],
)
def test_vggt_validation_rejects_ambiguous_frame_metadata(
    tmp_path: Path, metadata_update: dict, expected_error: str
):
    _frame_manifest, bundle, _summary, _video = create_mocked_summary(
        tmp_path, "ambiguous-frame-metadata-video"
    )
    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(metadata_update)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    report_path = tmp_path / "metadata_validation.json"

    validation = runner.invoke(
        app,
        [
            "vggt",
            "validate",
            "--bundle",
            str(bundle),
            "--report",
            str(report_path),
        ],
    )

    assert validation.exit_code != 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["valid"] is False
    assert any(expected_error in error for error in report["errors"])


@pytest.mark.parametrize(
    ("array_name", "expected_error"),
    [
        ("camera_centers", "camera_centers contains non-finite values"),
        ("quaternions_xyzw", "quaternions_xyzw contains non-finite values"),
        ("pose_confidence", "pose_confidence contains non-finite values"),
        ("points", "points contains non-finite values"),
    ],
)
def test_vggt_validation_rejects_non_finite_arrays(
    tmp_path: Path, array_name: str, expected_error: str
):
    _frame_manifest, bundle, _summary, _video = create_mocked_summary(
        tmp_path, "non-finite-video"
    )
    cameras_path = bundle / "cameras.npz"
    cameras = np.load(cameras_path)
    camera_centers = cameras["camera_centers"].copy()
    quaternions = cameras["quaternions_xyzw"].copy()
    pose_confidence = cameras["pose_confidence"].copy()
    if array_name == "camera_centers":
        camera_centers[0, 0] = np.nan
    if array_name == "quaternions_xyzw":
        quaternions[0, 0] = np.inf
    if array_name == "pose_confidence":
        pose_confidence[1] = np.inf
    np.savez_compressed(
        cameras_path,
        camera_centers=camera_centers,
        quaternions_xyzw=quaternions,
        valid_pose_mask=cameras["valid_pose_mask"],
        pose_confidence=pose_confidence,
    )
    if array_name == "points":
        points_path = bundle / "points.npz"
        point_data = np.load(points_path)
        points = point_data["points"].copy()
        points[0, 2] = np.inf
        np.savez_compressed(points_path, points=points)
    report_path = tmp_path / "finite_validation.json"

    validation = runner.invoke(
        app,
        [
            "vggt",
            "validate",
            "--bundle",
            str(bundle),
            "--report",
            str(report_path),
        ],
    )

    assert validation.exit_code != 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["valid"] is False
    assert expected_error in report["errors"]
