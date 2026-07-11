from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from fpv_vggt_lab.public_demo import PublicDemoConfig, build_public_demo
from tests.test_public_demo import _synthetic_sources
from tests.test_public_demo_point_cloud import _authorized_config


def _resize_authorized_sample(scene: Path, point_count: int) -> None:
    viewer = scene / "viewer"
    positions = viewer / "points_preview.bin"
    colors = viewer / "points_preview_colors.bin"
    positions.write_bytes(positions.read_bytes()[: point_count * 3 * 4])
    colors.write_bytes(colors.read_bytes()[: point_count * 3])
    meta_path = viewer / "scene_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["reconstruction"]["point_count_source"] = point_count
    meta["reconstruction"]["point_count_viewer"] = point_count
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def _default_config(scene: Path, archive: Path, output: Path) -> PublicDemoConfig:
    return PublicDemoConfig(
        scene_root=scene,
        archive_root=archive,
        output_root=output,
        slug="relative-geometry-study",
        public_title="Relative Geometry Study",
        path_sample_count=16,
        max_density_cells=80,
        generated_at="2026-07-11T13:00:00Z",
    )


def test_builder_copies_webgl_asset_and_pages_load_it_before_page_scripts(
    tmp_path: Path,
) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    output = tmp_path / "docs"

    build_public_demo(_authorized_config(scene, archive, output))

    shared = output / "assets" / "point-cloud-webgl.js"
    gallery = (output / "index.html").read_text(encoding="utf-8")
    scene_html = (
        output / "scenes" / "relative-geometry-study" / "index.html"
    ).read_text(encoding="utf-8")
    assert shared.is_file()
    assert gallery.index('src="assets/point-cloud-webgl.js"') < gallery.index(
        'src="assets/gallery.js"'
    )
    assert scene_html.index('src="../../assets/point-cloud-webgl.js"') < scene_html.index(
        'src="../../assets/scene.js"'
    )


def test_scene_and_gallery_emit_layered_point_cloud_surfaces(tmp_path: Path) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    output = tmp_path / "docs"

    build_public_demo(_authorized_config(scene, archive, output))

    gallery = (output / "index.html").read_text(encoding="utf-8")
    scene_html = (
        output / "scenes" / "relative-geometry-study" / "index.html"
    ).read_text(encoding="utf-8")
    styles = (output / "assets" / "site.css").read_text(encoding="utf-8")
    assert 'id="hero-point-cloud-canvas"' in gallery
    assert 'id="hero-canvas"' in gallery
    assert 'id="point-cloud-canvas"' in scene_html
    assert 'id="scene-canvas"' in scene_html
    assert 'data-layer="points"' in scene_html
    assert 'data-layer="density"' in scene_html
    assert 'id="point-cloud-readout"' in scene_html
    assert "#hero-point-cloud-canvas" in styles
    assert "#point-cloud-canvas" in styles
    assert "pointer-events: none" in styles


def test_shared_webgl_asset_exposes_validating_dependency_free_renderer() -> None:
    asset = (
        Path(__file__).parents[1]
        / "src"
        / "fpv_vggt_lab"
        / "public_demo_assets"
        / "point-cloud-webgl.js"
    )

    assert asset.is_file()
    source = asset.read_text(encoding="utf-8")
    for required in (
        "window.FpvPointCloudWebGL",
        "loadPointCloud",
        "createPointCloudRenderer",
        "drawPointCloud",
        "new URL(",
        'cache: "force-cache"',
        '"float32_le"',
        '"uint8"',
        "response.ok",
        "byteLength",
        "bufferData",
        "gl.POINTS",
        "webglcontextlost",
        "dispose",
    ):
        assert required in source
    assert not re.search(r"https?://|//cdn\.|\bimport\s+", source, re.IGNORECASE)


def test_shared_webgl_loader_rejects_same_length_sha256_mismatch() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the browser-loader behavior test")
    asset = (
        Path(__file__).parents[1]
        / "src"
        / "fpv_vggt_lab"
        / "public_demo_assets"
        / "point-cloud-webgl.js"
    )
    harness = f"""
      const fs = require("fs");
      const vm = require("vm");
      const {{ createHash, webcrypto }} = require("crypto");
      const source = fs.readFileSync({json.dumps(str(asset))}, "utf8");
      const positions = new Uint8Array(12).buffer;
      const colors = new Uint8Array([1, 2, 3]).buffer;
      const positionHash = createHash("sha256").update(Buffer.from(positions)).digest("hex");
      const browser = {{
        document: {{ baseURI: "https://public.example/study/index.html" }},
        location: {{ href: "https://public.example/study/index.html" }},
        crypto: webcrypto,
      }};
      const context = {{
        window: browser,
        URL,
        Float32Array,
        Uint8Array,
        Uint32Array,
        ArrayBuffer,
        fetch: async (url) => ({{
          ok: true,
          status: 200,
          arrayBuffer: async () => String(url).includes("points")
            ? positions.slice(0)
            : colors.slice(0),
        }}),
      }};
      vm.runInNewContext(source, context);
      const contract = {{
        scale_status: "relative_only",
        point_count: 1,
        source_point_count: 1,
        positions: {{
          path: "geometry/points.bin",
          dtype: "float32_le",
          components: 3,
          bytes: 12,
          sha256: positionHash,
        }},
        colors: {{
          path: "geometry/colors.bin",
          dtype: "uint8",
          components: 3,
          bytes: 3,
          sha256: "0".repeat(64),
        }},
        display_transform: {{ center: [0, 0, 0], scale: 1 }},
      }};
      (async () => {{
        try {{
          await browser.FpvPointCloudWebGL.loadPointCloud("scenes/study/scene.json", contract);
          console.error("same-length corrupt buffer was accepted");
          process.exitCode = 2;
        }} catch (error) {{
          if (!String(error.message).includes("SHA-256 mismatch")) throw error;
          console.log("rejected same-length corrupt buffer");
        }}
      }})().catch((error) => {{ console.error(error); process.exitCode = 1; }});
    """

    result = subprocess.run(
        [node, "-e", harness],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    assert "rejected same-length corrupt buffer" in result.stdout


def test_page_scripts_consume_contract_and_activate_density_fallback() -> None:
    assets = (
        Path(__file__).parents[1] / "src" / "fpv_vggt_lab" / "public_demo_assets"
    )
    scene_js = (assets / "scene.js").read_text(encoding="utf-8")
    gallery_js = (assets / "gallery.js").read_text(encoding="utf-8")

    for source in (scene_js, gallery_js):
        assert "FpvPointCloudWebGL" in source
        assert ".point_cloud" in source
        assert "loadPointCloud" in source
        assert "createPointCloudRenderer" in source
        assert "density" in source.lower()
        assert "fallback" in source.lower()
        assert "catch" in source
    assert 'new Set(["points", "raw", "rts"])' in scene_js
    assert "pointCloudRenderer.drawPointCloud" in scene_js
    assert "pointCloudRenderer.drawPointCloud" in gallery_js


def test_authorized_build_copy_is_contract_driven_and_truthful(tmp_path: Path) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    _resize_authorized_sample(scene, 7_777)
    output = tmp_path / "docs"

    build_public_demo(_authorized_config(scene, archive, output))

    gallery = (output / "index.html").read_text(encoding="utf-8")
    scene_html = (
        output / "scenes" / "relative-geometry-study" / "index.html"
    ).read_text(encoding="utf-8")
    payload = json.loads(
        (output / "scenes" / "relative-geometry-study" / "scene.json").read_text(
            encoding="utf-8"
        )
    )
    combined = f"{gallery}\n{scene_html}".lower()
    assert payload["point_cloud"]["point_count"] == 7_777
    assert "7,777 points" in gallery
    assert "7,777 displayed / 7,777 source points · relative_only" in scene_html
    assert "authorized colored vggt omega point sample" in combined
    assert "video, source frames, recognizable reprojections, raw npz/ply, full arrays, and machine paths remain withheld" in combined
    assert "no raw point cloud" not in combined
    assert "only voxels" not in combined


def test_default_build_stays_density_only_without_real_cloud_claim(tmp_path: Path) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    output = tmp_path / "docs"

    build_public_demo(_default_config(scene, archive, output))

    gallery = (output / "index.html").read_text(encoding="utf-8")
    scene_html = (
        output / "scenes" / "relative-geometry-study" / "index.html"
    ).read_text(encoding="utf-8")
    payload = json.loads(
        (output / "scenes" / "relative-geometry-study" / "scene.json").read_text(
            encoding="utf-8"
        )
    )
    combined = f"{gallery}\n{scene_html}".lower()
    assert "point_cloud" not in payload
    assert "density fallback" in combined
    assert "authorized colored vggt omega" not in combined
    assert "displayed /" not in combined


def test_generated_frontend_has_no_external_runtime_dependency(tmp_path: Path) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    output = tmp_path / "docs"

    build_public_demo(_authorized_config(scene, archive, output))

    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output.rglob("*")
        if path.suffix in {".css", ".html", ".js", ".json"}
    )
    assert not re.search(r"https?://|//cdn\.|@import\s+url", text, re.IGNORECASE)
