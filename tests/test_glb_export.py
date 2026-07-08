from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from fpv_vggt_lab.glb_export import export_hf_style_glb
from fpv_vggt_lab.viz import render_review_html


def test_export_hf_style_glb_writes_valid_glb_and_metadata(tmp_path: Path) -> None:
    bundle = _mock_bundle(tmp_path, pose_jump=True)
    output = tmp_path / "scene.glb"

    result = export_hf_style_glb(
        bundle_path=bundle,
        output=output,
        confidence_percentile=20,
        max_points=8,
    )

    assert result.output == output.resolve()
    assert output.read_bytes()[:4] == b"glTF"
    version, total_length = struct.unpack("<II", output.read_bytes()[4:12])
    assert version == 2
    assert total_length == output.stat().st_size
    metadata = json.loads(result.metadata_output.read_text(encoding="utf-8"))
    assert metadata["point_count"] == 8
    assert metadata["camera_count"] == 4
    assert metadata["pose_jump_count"] >= 1
    assert "no meters" in metadata["warnings"]


def test_review_html_links_hf_style_glb_scene(tmp_path: Path) -> None:
    bundle = _mock_bundle(tmp_path)
    manifest = _frame_manifest(tmp_path)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "video_id": "video-001",
                "segment_id": "segment-001",
                "reliability": {
                    "score": 1.0,
                    "label": "good",
                    "failure_flags": [],
                    "safe_to_use_for_descriptors": True,
                },
                "descriptors": {
                    "sampled_frame_count": 4,
                    "valid_pose_count": 4,
                    "normalized_path_length": 1.0,
                    "displacement_ratio": 1.0,
                    "mean_turn_angle_rad": 0.0,
                    "max_turn_angle_rad": 0.0,
                    "pose_jump_count": 0,
                },
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    glb = tmp_path / "scene.glb"
    glb.write_bytes(b"glTF" + b"\x00" * 32)
    html = tmp_path / "review.html"

    render_review_html(manifest, bundle, summary, html, glb_scene_path=glb)

    text = html.read_text(encoding="utf-8")
    assert "Open HF-style Scene GLB" in text
    assert "scene.glb" in text
    assert "relative VGGT frame only" in text


def _mock_bundle(tmp_path: Path, pose_jump: bool = False) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [1.0 if pose_jump else 0.15, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        bundle / "cameras.npz",
        camera_centers=centers,
        quaternions_xyzw=np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (4, 1)),
        valid_pose_mask=np.ones(4, dtype=bool),
        pose_confidence=np.ones(4, dtype=np.float32),
    )
    points = np.stack(
        [
            np.linspace(-0.5, 0.5, 12),
            np.linspace(0.0, 0.25, 12),
            np.linspace(1.0, 2.0, 12),
        ],
        axis=1,
    ).astype(np.float32)
    colors = np.tile(np.array([[220, 180, 90]], dtype=np.uint8), (12, 1))
    np.savez_compressed(
        bundle / "points.npz",
        points=points,
        point_colors_rgb=colors,
        point_confidence=np.linspace(0.0, 1.0, 12).astype(np.float32),
        point_depth=points[:, 2],
    )
    (bundle / "metadata.json").write_text(
        json.dumps(
            {
                "video_id": "video-001",
                "segment_id": "segment-001",
                "schema_version": "1.0",
                "source_tool": "mock",
                "source_url_or_repo": None,
                "source_commit_or_version": None,
                "export_notes": None,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "frame_indices": [0, 1, 2, 3],
                "frame_timestamps_sec": [0.0, 1.0, 2.0, 3.0],
                "warnings": [],
                "coordinate_frame": "VGGT relative coordinate frame",
            }
        ),
        encoding="utf-8",
    )
    return bundle


def _frame_manifest(tmp_path: Path) -> Path:
    frames = []
    for index in range(4):
        frame = tmp_path / f"frame_{index:03d}.jpg"
        frame.write_bytes(b"not-a-real-image")
        frames.append(
            {
                "video_id": "video-001",
                "segment_id": "segment-001",
                "frame_index": index,
                "timestamp_sec": float(index),
                "path": str(frame),
                "width": 10,
                "height": 10,
                "resized_long_edge": None,
                "quality": {
                    "blur_score": 1.0,
                    "brightness_mean": 120.0,
                    "contrast_std": 10.0,
                },
            }
        )
    manifest = tmp_path / "frames.json"
    manifest.write_text(
        json.dumps(
            {
                "video_id": "video-001",
                "segment_id": "segment-001",
                "source_video": str(tmp_path / "source.mp4"),
                "frame_count": 4,
                "source_fps": 4.0,
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    return manifest
