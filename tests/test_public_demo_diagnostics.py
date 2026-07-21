from __future__ import annotations

import json
from pathlib import Path

from fpv_vggt_lab.public_demo import PublicDemoConfig, build_public_demo
from tests.test_public_demo import _synthetic_sources, _write_json


def test_public_demo_retains_successful_colmap_warning_nuance(tmp_path: Path) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    diagnostic_path = scene / "diagnostics" / "colmap_failure_taxonomy.json"
    _write_json(
        diagnostic_path,
        {
            "status": "completed_with_warnings",
            "categories": ["ba_numerical_instability"],
            "successful_reconstruction": True,
            "registered_frame_max": 23,
            "ba_invocations": 18,
            "fallback_eligible": False,
            "bounded_recommendations": ["alternate_sparse_linear_solver"],
            "executes_fallback": False,
            "root": r"D:\private\other_models",
        },
    )

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

    payload = json.loads(
        (output / "scenes" / "relative-geometry-study" / "scene.json").read_text(
            encoding="utf-8"
        )
    )
    failure_case = payload["failure_case"]
    assert failure_case["status"] == "completed_with_warnings"
    assert failure_case["successful_reconstruction_retained"] is True
    assert failure_case["fallback_eligible"] is False
    assert failure_case["declared_registered_images"] == 24
    assert failure_case["registered_frame_max_observed"] == 23
    assert failure_case["ba_log_events"] == 18
