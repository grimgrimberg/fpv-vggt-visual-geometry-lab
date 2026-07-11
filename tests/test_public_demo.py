from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fpv_vggt_lab.public_demo import (
    PublicDemoConfig,
    PublicDemoError,
    audit_public_demo,
    build_public_demo,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _synthetic_sources(tmp_path: Path) -> tuple[Path, Path]:
    scene = tmp_path / "private_scene"
    archive = tmp_path / "private_archive"
    viewer = scene / "viewer"
    diagnostics = scene / "diagnostics"
    visualization = scene / "visualization"

    sample_count = 24
    phase = np.linspace(0.0, 1.0, sample_count)
    raw_path = np.column_stack(
        (phase * 4.0, np.sin(phase * np.pi * 2.0), np.cos(phase * np.pi) * 0.5)
    )
    paths = {
        "raw": raw_path.tolist(),
        "bspline": (raw_path * np.array([0.98, 0.92, 0.96])).tolist(),
        "kalman": (raw_path * np.array([0.96, 0.90, 0.94])).tolist(),
        "rts": (raw_path * np.array([0.95, 0.88, 0.92])).tolist(),
    }
    _write_json(
        viewer / "scene_meta.json",
        {
            "schema_version": "2.0.0",
            "scene_id": "private_absolute_scene_name",
            "title": r"D:\private\recognizable\source",
            "calibration": {"state": "relative_only", "evidence": None},
            "display_units": "relative units",
            "pose_semantics": "camera_pose_proxy",
            "body_attitude": "unavailable",
            "reconstruction": {
                "backend": "synthetic-vggt",
                "frame_count": sample_count,
                "point_count_source": 9000,
                "point_count_viewer": 9000,
                "coordinate_frame": "synthetic_relative_frame",
            },
            "quality": {
                "hero_score": 0.82,
                "score_confidence": 0.7,
                "transparent_heuristic": True,
                "components": {
                    "reconstruction_quality": 0.9,
                    "path_continuity": 0.85,
                    "match_connectivity": 1.0,
                    "contamination": 0.25,
                    "visual_clarity": 0.65,
                },
            },
            "warnings": ["relative only", "original media withheld"],
        },
    )
    _write_json(
        viewer / "camera_path.json",
        {
            "schema_version": "2.0.0",
            "scale_status": "relative_only",
            "timestamps_sec": (phase * 12.0 + 8.0).tolist(),
            "layers": paths,
        },
    )
    _write_json(
        viewer / "trajectory_profiles.json",
        {
            "time_normalized": phase.tolist(),
            "speed_relative": (0.5 + np.sin(phase * np.pi) ** 2).tolist(),
            "acceleration_relative": np.abs(np.cos(phase * np.pi * 2.0)).tolist(),
            "curvature": (phase**2).tolist(),
            "jerk_proxy": np.abs(np.sin(phase * np.pi * 4.0)).tolist(),
            "rts_residual": (0.1 + phase * 0.05).tolist(),
            "pose_jump": [False] * (sample_count - 1) + [True],
        },
    )
    samples = []
    for index, value in enumerate(phase):
        angle = value * np.pi * 0.4
        samples.append(
            {
                "sample_index": index,
                "timestamp_sec": float(8.0 + value * 12.0),
                "quaternion_wxyz": [float(np.cos(angle / 2)), 0.0, float(np.sin(angle / 2)), 0.0],
                "confidence": float(0.7 + value * 0.2),
            }
        )
    _write_json(
        visualization / "pose_6dof_animation.json",
        {
            "scale_status": "relative_only",
            "pose_semantics": "camera_pose_proxy",
            "body_attitude": "unavailable",
            "quaternion_order": "wxyz",
            "samples": samples,
        },
    )

    rng = np.random.default_rng(7)
    points = np.column_stack(
        (
            rng.normal(0.0, 1.8, 9000),
            rng.normal(0.0, 0.8, 9000),
            rng.normal(0.0, 1.1, 9000),
        )
    ).astype("<f4")
    viewer.mkdir(parents=True, exist_ok=True)
    points.tofile(viewer / "points_preview.bin")
    colors = rng.integers(0, 256, size=(len(points), 3), dtype=np.uint8)
    colors.tofile(viewer / "points_preview_colors.bin")

    methods = {
        "vggt_omega": {"status": "done", "role": "primary relative reconstruction"},
        "hloc_lightglue_colmap": {
            "status": "done",
            "role": "feature matching and sparse reconstruction",
            "input_frames": sample_count,
            "registered_images": sample_count,
            "pairs": 48,
            "points3D": 321,
        },
        "r3": {"status": "done", "role": "independent depth and pose reconstruction"},
        "lingbot_map": {"status": "done", "role": "independent depth and confidence"},
        "mast3r_sfm": {"status": "failed", "role": "retained failure case"},
    }
    _write_json(
        viewer / "methods" / "method_comparison.json",
        {
            "scale_status": "relative_only",
            "alignment_policy": "method_native_not_aligned",
            "comparison_rule": "No visual agreement claim without explicit alignment.",
            "methods": methods,
            "masks": {"status": "not_run", "reason": "optional lane not run"},
        },
    )
    edges = [
        {"source_id": index, "target_id": index + 1, "matches": 40 + index}
        for index in range(sample_count - 1)
    ]
    _write_json(
        diagnostics / "match_graph.json",
        {
            "node_count": sample_count,
            "edge_count": len(edges),
            "connected_component_count": 1,
            "adjacent_pair_coverage": 1.0,
            "isolated_frames": [],
            "edges": edges,
            "database": r"D:\private\database.db",
        },
    )
    _write_json(
        diagnostics / "geometry_consistency.json",
        {
            "scale_status": "relative_only",
            "depth_alignment": {"status": "pending_array_comparison"},
            "artifact_masks": {"status": "not_run", "reason": "optional lane not run"},
        },
    )
    _write_json(
        diagnostics / "colmap_failure_taxonomy.json",
        {
            "categories": ["unclassified"],
            "bounded_recommendations": ["preserve_vggt_only"],
            "executes_fallback": False,
            "root": r"D:\private\other_models",
        },
    )

    r3 = archive / "other_models" / "r3"
    lingbot = archive / "other_models" / "lingbot_map"
    for index in range(4):
        depth = (np.arange(120, dtype=np.float32).reshape(10, 12) + index) / 30.0 + 0.2
        confidence = np.full((10, 12), 1.1 + index * 0.1, dtype=np.float32)
        (r3 / "depth").mkdir(parents=True, exist_ok=True)
        (r3 / "conf").mkdir(parents=True, exist_ok=True)
        np.save(r3 / "depth" / f"{index:06d}.npy", depth)
        np.save(r3 / "conf" / f"{index:06d}.npy", confidence)
        (lingbot / "frames").mkdir(parents=True, exist_ok=True)
        np.savez(
            lingbot / "frames" / f"frame_{index:06d}.npz",
            depth=(depth[..., None] * 0.8).astype(np.float32),
            depth_conf=(confidence * 1.2).astype(np.float32),
            images=np.zeros((3, 10, 12), dtype=np.float32),
        )
    _write_json(
        lingbot / "batch_results.json",
        {
            "total_duration": 2.5,
            "input_folder": "/workspace/private/frames",
            "results": [{"success": True, "duration": 2.4, "output_video": "/tmp/private.mp4"}],
        },
    )
    return scene, archive


def test_build_public_demo_emits_only_sanitized_aggregates(tmp_path: Path) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    output = tmp_path / "docs"

    result = build_public_demo(
        PublicDemoConfig(
            scene_root=scene,
            archive_root=archive,
            output_root=output,
            slug="relative-geometry-study",
            public_title="Relative Geometry Study",
            path_sample_count=16,
            max_density_cells=80,
            generated_at="2026-07-11T12:00:00Z",
        )
    )

    assert result.status == "built"
    assert (output / "index.html").is_file()
    assert (output / ".nojekyll").is_file()
    assert (output / "scenes" / "relative-geometry-study" / "index.html").is_file()
    payload = json.loads(
        (output / "scenes" / "relative-geometry-study" / "scene.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["scale_status"] == "relative_only"
    assert payload["pose_semantics"] == "camera_pose_proxy"
    assert payload["body_attitude"] == "unavailable"
    assert len(payload["paths"]["raw"]) == 16
    assert len(payload["orientations"]) == 16
    assert len(payload["geometry_density"]["cells"]) <= 80
    assert payload["geometry_density"]["kind"] == "coarse_normalized_voxel_density"
    assert "point_cloud" not in payload
    assert payload["publication_boundary"]["real_point_sample_published"] is False
    assert payload["publication_boundary"]["full_backend_arrays_published"] is False
    assert payload["depth_summaries"]["r3"]["scale_status"] == "relative_only"
    assert payload["depth_summaries"]["lingbot_map"]["scale_status"] == "relative_only"
    assert payload["match_connectivity"]["node_count"] == 24
    assert "edges" not in payload["match_connectivity"]
    serialized = json.dumps(payload)
    assert str(scene) not in serialized
    assert str(archive) not in serialized
    assert "D:\\" not in serialized
    assert "/workspace/" not in serialized
    assert "/tmp/" not in serialized
    assert '"images":' not in serialized
    assert not list(output.rglob("*.bin"))


def test_public_site_is_offline_accessible_and_presentation_complete(tmp_path: Path) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    output = tmp_path / "docs"
    build_public_demo(
        PublicDemoConfig(
            scene_root=scene,
            archive_root=archive,
            output_root=output,
            slug="relative-geometry-study",
            public_title="Relative Geometry Study",
            path_sample_count=16,
            max_density_cells=80,
            generated_at="2026-07-11T12:00:00Z",
        )
    )

    gallery = (output / "index.html").read_text(encoding="utf-8")
    scene_html = (output / "scenes" / "relative-geometry-study" / "index.html").read_text(
        encoding="utf-8"
    )
    styles = (output / "assets" / "site.css").read_text(encoding="utf-8")
    app = (output / "assets" / "scene.js").read_text(encoding="utf-8")
    combined = "\n".join((gallery, scene_html, styles, app))
    assert '<link rel="icon" href="data:,">' in gallery
    assert '<link rel="icon" href="data:,">' in scene_html

    for required in (
        'id="scene-canvas"',
        'id="pose-slider"',
        'id="trajectory-canvas"',
        'id="method-tabs"',
        'id="method-panel"',
        'id="provenance"',
        'id="failure-case"',
        'id="media-notice"',
        "camera pose proxy",
        "relative only",
        "original media is not redistributed",
    ):
        assert required in combined
    assert "prefers-reduced-motion" in styles
    assert "drawGeometryDensity" in app
    assert "drawCameraProxy" in app
    assert "drawTrajectoryProfiles" in app
    assert "renderMethodTabs" in app
    assert "https://" not in combined
    assert "http://" not in combined
    assert "cdn" not in combined.lower()


def test_audit_public_demo_rejects_private_file_types_and_absolute_paths(tmp_path: Path) -> None:
    output = tmp_path / "docs"
    output.mkdir()
    (output / "index.html").write_text('<a href="D:\\private\\source.mp4">leak</a>', encoding="utf-8")
    (output / "frame.jpg").write_bytes(b"private frame")

    report = audit_public_demo(output)

    assert report["status"] == "failed"
    reasons = {finding["reason"] for finding in report["findings"]}
    assert "forbidden public file type" in reasons
    assert "absolute machine path" in reasons


def test_build_public_demo_rejects_non_relative_scene(tmp_path: Path) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    meta_path = scene / "viewer" / "scene_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["calibration"]["state"] = "calibrated"
    _write_json(meta_path, meta)

    with pytest.raises(PublicDemoError, match="relative_only"):
        build_public_demo(
            PublicDemoConfig(
                scene_root=scene,
                archive_root=archive,
                output_root=tmp_path / "docs",
                slug="relative-geometry-study",
                public_title="Relative Geometry Study",
            )
        )
