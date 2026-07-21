from __future__ import annotations

import json
from pathlib import Path

from fpv_vggt_lab.public_demo import PublicDemoConfig, build_public_demo
from tests.test_public_demo import _synthetic_sources, _write_json


def test_public_demo_allows_only_explicit_catalog_display_fields(tmp_path: Path) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    meta_path = scene / "viewer" / "scene_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["source"] = {
        "catalog_date": "2026-04-24",
        "town": "Taybeh",
        "source_path": r"D:\private\must-not-publish",
    }
    _write_json(meta_path, meta)

    output = tmp_path / "docs"
    build_public_demo(
        PublicDemoConfig(
            scene_root=scene,
            archive_root=archive,
            output_root=output,
            slug="relative-geometry-study",
            public_title="Engineering Vehicle · Taybeh",
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
    assert payload["provenance"]["catalog_date"] == "2026-04-24"
    assert payload["provenance"]["catalog_town"] == "Taybeh"
    assert "source_path" not in payload["provenance"]
    assert "D:\\" not in json.dumps(payload)
