from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import typer

from fpv_vggt_lab.public_demo import (
    PublicDemoConfig,
    PublicDemoError,
    _catalog_public_scene_url,
    audit_public_demo,
    build_public_demo,
    main,
)
from tests.test_public_demo import _synthetic_sources, _write_json


BLOCKED_PUBLIC_SCENE_URLS = [
    f"https://{hostname}/private-scene/"
    for hostname in (
        "localhost",
        "localhost.",
        "foo.localhost",
        "foo.localhost.",
        "localdomain",
        "localhost.localdomain",
        "host.localdomain",
        "service.local",
        "service.internal",
        "home.arpa",
        "router.home.arpa",
        "printer.lan",
        "server.home",
        "service.corp",
        "reserved.invalid",
        "reserved.test",
        "reserved.example",
        "127.1",
        "0177.0.0.1",
        "10.0.0.1",
        "[::1]",
        "[fe80::1]",
    )
]


def _collection_fixture(path: Path) -> None:
    records = [
        {
            "video_file": "2026-04-24_d9_engineering_vehicle_between_taybeh_and_deir_siryan.mp4",
            "slug": "2026-04-24_d9_engineering_vehicle_between_taybeh_and_deir_siryan",
            "date": "2026-04-24",
            "description": "D9 engineering vehicle on the road between Taybeh and Deir Siryan",
            "town": "Taybeh",
            "source_record_url": (
                "https://www.itamarweiss.com/fpv/video/"
                "2026-04-24_d9_engineering_vehicle_between_taybeh_and_deir_siryan/"
            ),
            "annotation_state": "manual_ground_truth",
            "edit_segmentation": {
                "segment_count": 3,
                "manual_segment_count": 3,
                "auto_segment_count": 0,
                "source_span_seconds": 42.129,
                "summary": (
                    "3 intervals · banner_start → flight_start → replay_start · "
                    "42.1 source s"
                ),
                "types": ["banner_start", "flight_start", "replay_start"],
                "segments": [
                    {"time": 0.0, "type": "banner_start"},
                    {"time": 8.067, "type": "flight_start"},
                    {"time": 42.129, "type": "replay_start"},
                ],
            },
            "research": {
                "public_scene_available": True,
                "public_scene_url": "scenes/stale-route/",
                "backend": "VGGT-Omega-1B-512",
                "calibration_state": "relative_only",
                "status": "ready_with_failures",
                "hero_score": 0.878,
                "method_status": {
                    "omega": "done",
                    "r3": "done",
                    "lingbot": "done",
                    "hloc_colmap": "done",
                    "mast3r": "failed",
                },
                "method_coverage": {
                    "bucket": "strong",
                    "done": 4,
                    "total": 5,
                    "ratio": 0.8,
                },
                "warnings": [
                    "relative_only: no physical scale calibration is documented",
                    "camera_pose_proxy: vehicle-body attitude is unavailable",
                ],
            },
        },
        {
            "video_file": "2026-06-14_engineering_vehicle_arnoun_ababil_drone.mp4",
            "slug": "2026-06-14_engineering_vehicle_arnoun_ababil_drone",
            "date": "2026-06-14",
            "description": "Israeli army engineering vehicle, outskirts of Arnoun",
            "town": "Arnoun",
            "source_record_url": (
                "https://www.itamarweiss.com/fpv/video/"
                "2026-06-14_engineering_vehicle_arnoun_ababil_drone/"
            ),
            "annotation_state": "manual_ground_truth",
            "edit_segmentation": {
                "segment_count": 2,
                "types": ["banner_start", "flight_start"],
                "segments": [
                    {"time": 0.0, "type": "banner_start"},
                    {"time": 4.2, "type": "flight_start"},
                ],
            },
            "research": {"public_scene_available": False},
        },
        {
            "video_file": "2026-06-17_force_position_kfar_tebnit_mmirleb_17590.mp4",
            "slug": "2026-06-17_force_position_kfar_tebnit_mmirleb_17590",
            "date": "2026-06-17",
            "description": "Infiltrating force position, outskirts of Kfar Tebnit",
            "town": "Kfar Tebnit",
            "source_record_url": (
                "https://www.itamarweiss.com/fpv/video/"
                "2026-06-17_force_position_kfar_tebnit_mmirleb_17590/"
            ),
            "annotation_state": "auto_generated",
            "edit_segmentation": {
                "segment_count": 1,
                "types": ["flight_start"],
                "segments": [{"time": 0.0, "type": "flight_start"}],
            },
            "research": {"public_scene_available": False},
        },
    ]
    _write_json(
        path,
        {
            "schema_version": "wow-public-videos-v1",
            "generated_at": "2026-07-11T12:00:00Z",
            "source": {
                "kind": "local_annotation_snapshots",
                "selection_policy": (
                    "one record per video_file; manual preferred over auto; filename tie-break"
                ),
            },
            "provenance": {
                "generated_by": "tools.wow_layer.public_catalog",
                "annotation_files_considered": 189,
                "gallery_manifest_merged": True,
                "scene_routes": "explicit_only",
            },
            "publication_boundary": {
                "third_party_media_embedded": False,
                "real_media_assets_published": False,
                "source_record_links_only": True,
            },
            "counts": {
                "records": 3,
                "manual_ground_truth": 2,
                "auto_generated": 1,
                "public_scenes": 1,
            },
            "records": records,
        },
    )


def _add_scene_catalog_context(scene: Path) -> None:
    meta_path = scene / "viewer" / "scene_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["source"] = {
        "video_file": "2026-04-24_d9_engineering_vehicle_between_taybeh_and_deir_siryan.mp4",
        "catalog_date": "2026-04-24",
        "town": "Taybeh",
        "source_path": r"D:\private\must-not-publish",
    }
    meta["annotation"] = {
        "kind": "manual_ground_truth",
        "source_path": "/workspace/private/annotation.json",
        "intervals": [
            {
                "segment_id": "seg01",
                "start_s": 8.067,
                "end_s": 42.129,
                "start_type": "flight_start",
                "end_type": "replay_start",
            }
        ],
    }
    meta["edit_segments"] = [
        {
            "start_s": 0.0,
            "end_s": 8.067,
            "type": "banner_start",
            "label": "Banner Start",
            "review_status": "manual_ground_truth",
        },
        {
            "start_s": 8.067,
            "end_s": 42.129,
            "type": "flight_start",
            "label": "Flight Start",
            "review_status": "manual_ground_truth",
        },
        {
            "start_s": 42.129,
            "end_s": 69.781,
            "type": "replay_start",
            "label": "Replay Start",
            "review_status": "manual_ground_truth",
        },
    ]
    _write_json(meta_path, meta)


def test_public_demo_builds_scalable_catalog_and_edit_aware_scene(tmp_path: Path) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    _add_scene_catalog_context(scene)
    catalog_path = tmp_path / "public-catalog.json"
    _collection_fixture(catalog_path)
    output = tmp_path / "docs"

    result = build_public_demo(
        PublicDemoConfig(
            scene_root=scene,
            archive_root=archive,
            output_root=output,
            catalog_path=catalog_path,
            slug="relative-geometry-study",
            public_title="Engineering Vehicle · Taybeh",
            path_sample_count=16,
            max_density_cells=80,
            generated_at="2026-07-11T12:00:00Z",
        )
    )

    catalog = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    assert catalog["counts"]["records"] == 3
    assert [record["date"] for record in catalog["records"]] == [
        "2026-06-17",
        "2026-06-14",
        "2026-04-24",
    ]
    target = next(
        record
        for record in catalog["records"]
        if record["video_file"].startswith("2026-04-24_")
    )
    assert target["research"]["public_scene_available"] is True
    assert target["research"]["public_scene_url"] == (
        "scenes/relative-geometry-study/"
    )
    assert target["research"]["calibration_state"] == "relative_only"
    assert target["research"]["method_coverage"] == {
        "bucket": "strong",
        "done": 4,
        "total": 5,
        "ratio": 0.8,
    }
    assert target["research"]["warnings"] == [
        "relative_only: no physical scale calibration is documented",
        "camera_pose_proxy: vehicle-body attitude is unavailable",
    ]
    assert target["edit_segmentation"]["types"] == [
        "banner_start",
        "flight_start",
        "replay_start",
    ]
    assert target["edit_segmentation"]["manual_segment_count"] == 3
    assert target["edit_segmentation"]["auto_segment_count"] == 0
    assert target["edit_segmentation"]["source_span_seconds"] == 42.129
    assert catalog["provenance"]["annotation_files_considered"] == 189
    assert catalog["publication_boundary"]["source_record_links_only"] is True
    serialized_catalog = json.dumps(catalog)
    assert "thumbnail_url" not in serialized_catalog
    assert "video_url" not in serialized_catalog
    assert "D:\\" not in serialized_catalog
    assert "/workspace/" not in serialized_catalog

    payload = json.loads(
        (output / "scenes" / "relative-geometry-study" / "scene.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["annotation"]["kind"] == "manual_ground_truth"
    assert payload["annotation"]["source_interval"] == {
        "segment_id": "seg01",
        "start_s": 8.067,
        "end_s": 42.129,
    }
    assert [item["type"] for item in payload["edit_segments"]] == [
        "banner_start",
        "flight_start",
        "replay_start",
    ]
    assert "/workspace/" not in json.dumps(payload)
    assert "D:\\" not in json.dumps(payload)

    gallery_html = (output / "index.html").read_text(encoding="utf-8")
    for identifier in (
        'id="catalog"',
        'id="catalog-search"',
        'id="catalog-sort"',
        'id="catalog-scenes-only"',
        'id="catalog-annotation"',
        'id="catalog-grid"',
        'id="catalog-count"',
    ):
        assert identifier in gallery_html
    assert 'href="#catalog"' in gallery_html
    assert gallery_html.index('id="catalog"') < gallery_html.index(
        'class="gallery-proof'
    )
    assert '<script src="assets/catalog.js" defer></script>' in gallery_html

    scene_html = (
        output / "scenes" / "relative-geometry-study" / "index.html"
    ).read_text(encoding="utf-8")
    assert 'id="scene-catalog-meta"' in scene_html
    assert 'id="edit-timeline"' in scene_html
    assert 'id="edit-cursor"' in scene_html
    assert any(item["path"] == "catalog.json" for item in json.loads(
        result.manifest.read_text(encoding="utf-8")
    )["files"])
    assert audit_public_demo(
        output,
        authorized_binary_paths=(),
    )["status"] == "passed"


def test_public_demo_preserves_repeated_edit_types_in_segment_order(
    tmp_path: Path,
) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    _add_scene_catalog_context(scene)
    catalog_path = tmp_path / "repeated-edit-types.json"
    _collection_fixture(catalog_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    edit = catalog["records"][0]["edit_segmentation"]
    edit.update(
        {
            "types": ["flight_start", "pause_start", "flight_start"],
            "segments": [
                {"time": 0.0, "type": "flight_start"},
                {"time": 1.0, "type": "pause_start"},
                {"time": 2.0, "type": "flight_start"},
            ],
            "source_span_seconds": 2.0,
            "summary": (
                "3 intervals · flight_start → pause_start → flight_start · "
                "2.0 source s"
            ),
        }
    )
    _write_json(catalog_path, catalog)

    build_public_demo(
        PublicDemoConfig(
            scene_root=scene,
            archive_root=archive,
            output_root=tmp_path / "docs",
            catalog_path=catalog_path,
            slug="relative-geometry-study",
            public_title="Relative Geometry Study",
            generated_at="2026-07-11T12:00:00Z",
        )
    )

    published = json.loads(
        (tmp_path / "docs" / "catalog.json").read_text(encoding="utf-8")
    )
    target = next(
        record for record in published["records"] if record["date"] == "2026-04-24"
    )
    assert target["edit_segmentation"]["types"] == [
        "flight_start",
        "pause_start",
        "flight_start",
    ]


def test_public_demo_defaults_missing_research_to_unavailable(
    tmp_path: Path,
) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    _add_scene_catalog_context(scene)
    catalog_path = tmp_path / "missing-research.json"
    _collection_fixture(catalog_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["records"][1].pop("research")
    _write_json(catalog_path, catalog)

    build_public_demo(
        PublicDemoConfig(
            scene_root=scene,
            archive_root=archive,
            output_root=tmp_path / "docs",
            catalog_path=catalog_path,
            slug="relative-geometry-study",
            public_title="Relative Geometry Study",
            generated_at="2026-07-11T12:00:00Z",
        )
    )

    published = json.loads(
        (tmp_path / "docs" / "catalog.json").read_text(encoding="utf-8")
    )
    record = next(item for item in published["records"] if item["town"] == "Arnoun")
    assert record["research"] == {"public_scene_available": False}


def test_public_demo_preserves_bounded_long_edit_summary(tmp_path: Path) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    _add_scene_catalog_context(scene)
    catalog_path = tmp_path / "long-edit-summary.json"
    _collection_fixture(catalog_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    summary = " · ".join(["flight_start"] * 64)
    assert 800 < len(summary) < 1200
    catalog["records"][0]["edit_segmentation"]["summary"] = summary
    _write_json(catalog_path, catalog)

    build_public_demo(
        PublicDemoConfig(
            scene_root=scene,
            archive_root=archive,
            output_root=tmp_path / "docs",
            catalog_path=catalog_path,
            slug="relative-geometry-study",
            public_title="Relative Geometry Study",
            generated_at="2026-07-11T12:00:00Z",
        )
    )

    published = json.loads(
        (tmp_path / "docs" / "catalog.json").read_text(encoding="utf-8")
    )
    record = next(item for item in published["records"] if item["date"] == "2026-04-24")
    assert record["edit_segmentation"]["summary"] == summary


@pytest.mark.parametrize(
    "mutation",
    [
        lambda record: record.__setitem__(
            "thumbnail_url", "https://d2fioemadmrru3.cloudfront.net/private.jpg"
        ),
        lambda record: record.__setitem__(
            "source_record_url", "https://example.com/not-an-approved-record/"
        ),
        lambda record: record["research"].__setitem__(
            "public_scene_url", "../../private/"
        ),
    ],
)
def test_public_demo_rejects_unsafe_catalog_fields(
    tmp_path: Path,
    mutation,
) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    _add_scene_catalog_context(scene)
    catalog_path = tmp_path / "unsafe-catalog.json"
    _collection_fixture(catalog_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    mutation(catalog["records"][0])
    _write_json(catalog_path, catalog)

    with pytest.raises(PublicDemoError):
        build_public_demo(
            PublicDemoConfig(
                scene_root=scene,
                archive_root=archive,
                output_root=tmp_path / "docs",
                catalog_path=catalog_path,
                slug="relative-geometry-study",
                public_title="Relative Geometry Study",
                path_sample_count=16,
                max_density_cells=80,
                generated_at="2026-07-11T12:00:00Z",
            )
        )


def test_public_demo_synthesizes_metadata_only_catalog_for_legacy_callers(
    tmp_path: Path,
) -> None:
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

    catalog = json.loads((output / "catalog.json").read_text(encoding="utf-8"))
    assert catalog["schema_version"] == "wow-public-videos-v1"
    assert catalog["counts"] == {
        "records": 1,
        "manual_ground_truth": 0,
        "auto_generated": 0,
        "unreviewed": 1,
        "public_scenes": 1,
    }
    record = catalog["records"][0]
    assert record["research"] == {
        "public_scene_available": True,
        "public_scene_url": "scenes/relative-geometry-study/",
        "calibration_state": "relative_only",
    }
    assert "source_record_url" not in record
    assert audit_public_demo(output)["status"] == "passed"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda catalog: catalog.__setitem__("schema_version", "unsafe-v0"),
        lambda catalog: catalog["records"][0].__setitem__(
            "frame_path", "/workspace/private/frame.jpg"
        ),
        lambda catalog: catalog["records"][0]["research"].__setitem__(
            "hero_score", float("nan")
        ),
        lambda catalog: catalog["records"][0]["research"].__setitem__(
            "public_scene_url", "http://public.example/scene/"
        ),
        lambda catalog: catalog["records"][0].__setitem__(
            "town", r"\\private-server\catalog"
        ),
        lambda catalog: catalog["records"][0]["research"].__setitem__(
            "public_scene_url", "https://127.0.0.1/private-scene/"
        ),
        lambda catalog: catalog["records"][0]["research"].__setitem__(
            "public_scene_url", "https://127.1/private-scene/"
        ),
        lambda catalog: catalog["records"][0]["research"].__setitem__(
            "public_scene_url", "https://0177.0.0.1/private-scene/"
        ),
        lambda catalog: catalog["records"][0]["research"].__setitem__(
            "public_scene_url", "https://localhost./private-scene/"
        ),
        lambda catalog: catalog["records"][0]["research"].__setitem__(
            "public_scene_url", "https://service.internal./private-scene/"
        ),
        lambda catalog: catalog["records"][0]["research"].__setitem__(
            "public_scene_url", "https://router.home.arpa/private-scene/"
        ),
    ],
)
def test_public_demo_rejects_invalid_catalog_contract_values(
    tmp_path: Path,
    mutation,
) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    _add_scene_catalog_context(scene)
    catalog_path = tmp_path / "invalid-catalog.json"
    _collection_fixture(catalog_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    mutation(catalog)
    _write_json(catalog_path, catalog)

    with pytest.raises(PublicDemoError):
        build_public_demo(
            PublicDemoConfig(
                scene_root=scene,
                archive_root=archive,
                output_root=tmp_path / "docs",
                catalog_path=catalog_path,
                slug="relative-geometry-study",
                public_title="Relative Geometry Study",
                generated_at="2026-07-11T12:00:00Z",
            )
        )


def test_collection_frontend_assets_are_dependency_free_and_edit_aware() -> None:
    asset_root = (
        Path(__file__).parents[1]
        / "src"
        / "fpv_vggt_lab"
        / "public_demo_assets"
    )
    catalog_js = (asset_root / "catalog.js").read_text(encoding="utf-8")
    scene_js = (asset_root / "scene.js").read_text(encoding="utf-8")
    styles = (asset_root / "site.css").read_text(encoding="utf-8")

    for required in (
        'fetch("catalog.json", { cache: "no-store" })',
        "IntersectionObserver",
        "catalog-search",
        "catalog-sort",
        "catalog-scenes-only",
        "catalog-annotation",
        "source_record_url",
        "public_scene_url",
    ):
        assert required in catalog_js
    assert "createElement(\"img\")" not in catalog_js
    assert "createElement(\"video\")" not in catalog_js
    assert "edit-cursor" in scene_js
    assert "edit_segments" in scene_js
    assert "repeat(4, minmax(0, 1fr))" in styles
    assert "prefers-reduced-motion: reduce" in styles


@pytest.mark.parametrize("url", BLOCKED_PUBLIC_SCENE_URLS)
def test_catalog_python_rejects_reserved_local_hostnames(url: str) -> None:
    with pytest.raises(PublicDemoError):
        _catalog_public_scene_url(url)


def test_catalog_frontend_rejects_normalized_private_hosts() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for catalog URL behavior validation")
    asset = (
        Path(__file__).parents[1]
        / "src"
        / "fpv_vggt_lab"
        / "public_demo_assets"
        / "catalog.js"
    )
    harness = f"""
      const fs = require("fs");
      const vm = require("vm");
      const source = fs.readFileSync({json.dumps(str(asset))}, "utf8");
      const browser = {{}};
      const context = {{
        window: browser,
        document: {{ getElementById: () => null }},
        URL,
        console,
      }};
      vm.runInNewContext(source, context);
      const safe = browser.FpvCatalogSafety.safePublicSceneUrl;
      const record = (url) => ({{
        research: {{ public_scene_url: url }},
      }});
      const rejected = {json.dumps(BLOCKED_PUBLIC_SCENE_URLS)};
      const accepted = [
        "https://grimgrimberg.github.io/private-scene/",
        "https://[2001:4860:4860::8888]/private-scene/",
      ];
      if (rejected.some((url) => safe(record(url)) !== null)) {{
        throw new Error("private or legacy host was accepted");
      }}
      if (accepted.some((url) => safe(record(url)) !== url)) {{
        throw new Error("public DNS or IPv6 host was rejected");
      }}
      console.log("catalog URL host policy passed");
    """

    result = subprocess.run(
        [node, "-e", harness],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    assert "catalog URL host policy passed" in result.stdout


def test_synthesized_catalog_preserves_sanitized_scene_edit_structure(
    tmp_path: Path,
) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    _add_scene_catalog_context(scene)
    output = tmp_path / "docs"

    build_public_demo(
        PublicDemoConfig(
            scene_root=scene,
            archive_root=archive,
            output_root=output,
            slug="relative-geometry-study",
            public_title="Relative Geometry Study",
            generated_at="2026-07-11T12:00:00Z",
        )
    )

    record = json.loads((output / "catalog.json").read_text(encoding="utf-8"))[
        "records"
    ][0]
    assert record["edit_segmentation"]["types"] == [
        "banner_start",
        "flight_start",
        "replay_start",
    ]
    assert record["edit_segmentation"]["segments"][1] == {
        "start_s": 8.067,
        "end_s": 42.129,
        "type": "flight_start",
        "label": "Flight Start",
        "review_status": "manual_ground_truth",
    }


def test_synthesized_catalog_preserves_duplicate_edit_type_sequence(
    tmp_path: Path,
) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    _add_scene_catalog_context(scene)
    meta_path = scene / "viewer" / "scene_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["edit_segments"] = [
        {
            "start_s": 0.0,
            "end_s": 1.0,
            "type": "flight_start",
            "label": "Flight",
            "review_status": "manual_ground_truth",
        },
        {
            "start_s": 1.0,
            "end_s": 2.0,
            "type": "pause_start",
            "label": "Pause",
            "review_status": "manual_ground_truth",
        },
        {
            "start_s": 2.0,
            "end_s": 3.0,
            "type": "flight_start",
            "label": "Flight resumes",
            "review_status": "manual_ground_truth",
        },
    ]
    _write_json(meta_path, meta)

    build_public_demo(
        PublicDemoConfig(
            scene_root=scene,
            archive_root=archive,
            output_root=tmp_path / "docs",
            slug="relative-geometry-study",
            public_title="Relative Geometry Study",
            generated_at="2026-07-11T12:00:00Z",
        )
    )

    record = json.loads(
        (tmp_path / "docs" / "catalog.json").read_text(encoding="utf-8")
    )["records"][0]
    assert record["edit_segmentation"]["types"] == [
        "flight_start",
        "pause_start",
        "flight_start",
    ]


def test_public_audit_rejects_noncanonical_catalog_even_without_external_url(
    tmp_path: Path,
) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    output = tmp_path / "docs"
    build_public_demo(
        PublicDemoConfig(
            scene_root=scene,
            archive_root=archive,
            output_root=output,
            slug="relative-geometry-study",
            public_title="Relative Geometry Study",
            generated_at="2026-07-11T12:00:00Z",
        )
    )
    catalog_path = output / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["debug_only"] = True
    _write_json(catalog_path, catalog)

    report = audit_public_demo(output)

    assert report["status"] == "failed"
    assert any(
        finding["path"] == "catalog.json"
        and finding["reason"].startswith("invalid public catalog")
        for finding in report["findings"]
    )


def test_public_demo_cli_accepts_catalog_json() -> None:
    app = typer.Typer()
    app.command()(main)
    command = typer.main.get_command(app)
    options = {
        option
        for parameter in command.params
        for option in getattr(parameter, "opts", ())
    }

    assert "--catalog-json" in options
