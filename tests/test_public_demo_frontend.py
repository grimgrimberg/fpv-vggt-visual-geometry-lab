from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from fpv_vggt_lab.public_demo import PublicDemoConfig, build_public_demo
from tests.test_public_demo import _synthetic_sources
from tests.test_public_demo_point_cloud import ATTRIBUTION, AUTHORIZATION, _authorized_config


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
        'cache: "no-cache"',
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


def test_shared_webgl_loader_revalidates_and_loads_matching_binary_assets() -> None:
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
      const digest = (buffer) => createHash("sha256").update(Buffer.from(buffer)).digest("hex");
      const requests = [];
      const browser = {{
        document: {{ baseURI: "https://public.example/index.html" }},
        location: {{ href: "https://public.example/index.html" }},
        crypto: webcrypto,
      }};
      const context = {{
        window: browser,
        URL,
        Float32Array,
        Uint8Array,
        Uint32Array,
        ArrayBuffer,
        fetch: async (url, options) => {{
          requests.push({{ url: String(url), cache: options?.cache }});
          return {{
            ok: true,
            status: 200,
            arrayBuffer: async () => String(url).includes("positions")
              ? positions.slice(0)
              : colors.slice(0),
          }};
        }},
      }};
      vm.runInNewContext(source, context);
      const contract = {{
        scale_status: "relative_only",
        point_count: 1,
        source_point_count: 1,
        positions: {{
          path: "geometry/positions.bin", dtype: "float32_le", components: 3,
          bytes: 12, sha256: digest(positions),
        }},
        colors: {{
          path: "geometry/colors.bin", dtype: "uint8", components: 3,
          bytes: 3, sha256: digest(colors),
        }},
        display_transform: {{ center: [0, 0, 0], scale: 1 }},
      }};
      (async () => {{
        const cloud = await browser.FpvPointCloudWebGL.loadPointCloud(
          "scenes/study/scene.json",
          contract,
        );
        if (cloud.pointCount !== 1 || requests.length !== 2) throw new Error("load failed");
        if (requests.some((request) => request.cache !== "no-cache")) {{
          throw new Error(`binary cache policy was ${{JSON.stringify(requests)}}`);
        }}
        const expectedPrefix = "https://public.example/scenes/study/geometry/";
        if (requests.some((request) => !request.url.startsWith(expectedPrefix))) {{
          throw new Error(`asset escaped scene directory: ${{JSON.stringify(requests)}}`);
        }}
        console.log("revalidated matching binary assets");
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
    assert "revalidated matching binary assets" in result.stdout


def test_shared_webgl_loader_rejects_noncanonical_or_escaping_asset_paths() -> None:
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
    invalid_paths = [
        "%2e%2e/positions.bin",
        "%2E%2E/positions.bin",
        "geometry/%2e/positions.bin",
        "geometry/%2e%2e/positions.bin",
        "https://public.example/scenes/study/geometry/positions.bin",
        "//public.example/scenes/study/geometry/positions.bin",
        r"geometry\positions.bin",
        "geometry/%5cpositions.bin",
        "geometry/positions.bin?alias=1",
        "geometry/positions.bin#alias",
        "geometry/%ZZpositions.bin",
    ]
    harness = f"""
      const fs = require("fs");
      const vm = require("vm");
      const {{ createHash, webcrypto }} = require("crypto");
      const source = fs.readFileSync({json.dumps(str(asset))}, "utf8");
      const positions = new Uint8Array(12).buffer;
      const colors = new Uint8Array([1, 2, 3]).buffer;
      const digest = (buffer) => createHash("sha256").update(Buffer.from(buffer)).digest("hex");
      const browser = {{
        document: {{ baseURI: "https://public.example/index.html" }},
        location: {{ href: "https://public.example/index.html" }},
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
          arrayBuffer: async () => String(url).includes("colors")
            ? colors.slice(0)
            : positions.slice(0),
        }}),
      }};
      vm.runInNewContext(source, context);
      const baseContract = {{
        scale_status: "relative_only",
        point_count: 1,
        source_point_count: 1,
        positions: {{
          path: "geometry/positions.bin", dtype: "float32_le", components: 3,
          bytes: 12, sha256: digest(positions),
        }},
        colors: {{
          path: "geometry/colors.bin", dtype: "uint8", components: 3,
          bytes: 3, sha256: digest(colors),
        }},
        display_transform: {{ center: [0, 0, 0], scale: 1 }},
      }};
      const invalidPaths = {json.dumps(invalid_paths)};
      (async () => {{
        await browser.FpvPointCloudWebGL.loadPointCloud(
          "scenes/study/scene.json",
          baseContract,
        );
        for (const path of invalidPaths) {{
          const contract = {{
            ...baseContract,
            positions: {{ ...baseContract.positions, path }},
          }};
          try {{
            await browser.FpvPointCloudWebGL.loadPointCloud(
              "scenes/study/scene.json",
              contract,
            );
          }} catch (error) {{
            if (!String(error.message).startsWith("Point-cloud contract error: positions.path")) {{
              throw error;
            }}
            continue;
          }}
          throw new Error(`accepted invalid path: ${{path}}`);
        }}
        console.log("rejected noncanonical asset paths");
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
    assert "rejected noncanonical asset paths" in result.stdout


def test_shared_webgl_renderer_cleans_partial_gl_allocations_on_setup_failure() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the WebGL cleanup behavior test")
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
      const source = fs.readFileSync({json.dumps(str(asset))}, "utf8");
      const browser = {{}};
      vm.runInNewContext(source, {{
        window: browser,
        Float32Array,
        Uint8Array,
        Uint32Array,
        ArrayBuffer,
      }});
      const cloud = {{
        pointCount: 1,
        sourcePointCount: 1,
        displayTransform: {{ center: [0, 0, 0], scale: 1 }},
        positions: new Float32Array([0, 0, 0]),
        colors: new Uint8Array([1, 2, 3]),
      }};

      function makeGl(mode) {{
        let shaderId = 0;
        let bufferId = 0;
        const state = {{
          shaders: [], programs: [], buffers: [],
          deletedShaders: new Set(), deletedPrograms: new Set(), deletedBuffers: new Set(),
        }};
        const gl = {{
          VERTEX_SHADER: 1, FRAGMENT_SHADER: 2, COMPILE_STATUS: 3, LINK_STATUS: 4,
          ARRAY_BUFFER: 5, STATIC_DRAW: 6,
          createShader: (type) => {{
            const shader = {{ type, id: ++shaderId }};
            state.shaders.push(shader);
            return shader;
          }},
          shaderSource: () => {{}},
          compileShader: () => {{}},
          getShaderParameter: (shader) => !(mode === "compile" && shader.type === 2),
          getShaderInfoLog: () => "injected compile failure",
          deleteShader: (shader) => state.deletedShaders.add(shader),
          createProgram: () => {{
            const program = {{ id: 1 }};
            state.programs.push(program);
            return program;
          }},
          attachShader: () => {{}},
          linkProgram: () => {{}},
          getProgramParameter: () => mode !== "link",
          getProgramInfoLog: () => "injected link failure",
          deleteProgram: (program) => state.deletedPrograms.add(program),
          createBuffer: () => {{
            bufferId += 1;
            if (mode === "buffer" && bufferId === 2) return null;
            const buffer = {{ id: bufferId }};
            state.buffers.push(buffer);
            return buffer;
          }},
          bindBuffer: () => {{}},
          bufferData: () => {{}},
          deleteBuffer: (buffer) => state.deletedBuffers.add(buffer),
          getAttribLocation: () => 0,
          getUniformLocation: () => mode === "location" ? null : {{}},
        }};
        return {{ gl, state }};
      }}

      const expectations = {{
        compile: {{ shaders: 2, programs: 0, buffers: 0 }},
        link: {{ shaders: 2, programs: 1, buffers: 0 }},
        buffer: {{ shaders: 2, programs: 1, buffers: 1 }},
        location: {{ shaders: 2, programs: 1, buffers: 2 }},
      }};
      for (const [mode, expected] of Object.entries(expectations)) {{
        const {{ gl, state }} = makeGl(mode);
        const canvas = {{ getContext: () => gl }};
        let failed = false;
        try {{
          browser.FpvPointCloudWebGL.createPointCloudRenderer(canvas, cloud);
        }} catch (_error) {{
          failed = true;
        }}
        if (!failed) throw new Error(`${{mode}} setup did not fail`);
        const actual = {{
          shaders: state.deletedShaders.size,
          programs: state.deletedPrograms.size,
          buffers: state.deletedBuffers.size,
        }};
        if (JSON.stringify(actual) !== JSON.stringify(expected)) {{
          throw new Error(`${{mode}} cleanup was ${{JSON.stringify(actual)}}`);
        }}
      }}
      console.log("cleaned partial GL allocations");
    """

    result = subprocess.run(
        [node, "-e", harness],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    assert "cleaned partial GL allocations" in result.stdout


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


def test_point_cloud_provenance_is_visible_only_for_authorized_contracts(
    tmp_path: Path,
) -> None:
    scene, archive = _synthetic_sources(tmp_path)
    authorized_output = tmp_path / "authorized"
    default_output = tmp_path / "default"

    build_public_demo(_authorized_config(scene, archive, authorized_output))
    build_public_demo(_default_config(scene, archive, default_output))

    authorized_payload = json.loads(
        (
            authorized_output
            / "scenes"
            / "relative-geometry-study"
            / "scene.json"
        ).read_text(encoding="utf-8")
    )
    default_payload = json.loads(
        (
            default_output
            / "scenes"
            / "relative-geometry-study"
            / "scene.json"
        ).read_text(encoding="utf-8")
    )
    gallery = (authorized_output / "index.html").read_text(encoding="utf-8")
    scene_html = (
        authorized_output / "scenes" / "relative-geometry-study" / "index.html"
    ).read_text(encoding="utf-8")
    gallery_js = (authorized_output / "assets" / "gallery.js").read_text(encoding="utf-8")
    scene_js = (authorized_output / "assets" / "scene.js").read_text(encoding="utf-8")
    default_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in default_output.rglob("*")
        if path.suffix in {".html", ".json"}
    )

    assert authorized_payload["point_cloud"]["attribution"] == ATTRIBUTION
    assert (
        authorized_payload["point_cloud"]["authorization_provenance"] == AUTHORIZATION
    )
    assert "point_cloud" not in default_payload
    for required in (
        'id="gallery-point-cloud-credit"',
        'id="gallery-point-cloud-attribution"',
        'id="gallery-point-cloud-authorization"',
    ):
        assert required in gallery
    for required in (
        'id="scene-point-cloud-credit"',
        'id="scene-point-cloud-attribution"',
        'id="scene-point-cloud-authorization"',
    ):
        assert required in scene_html
    for source in (gallery_js, scene_js):
        assert ".attribution" in source
        assert ".authorization_provenance" in source
        assert ".textContent" in source
        assert "credit.hidden = false" in source
    assert ATTRIBUTION not in default_text
    assert AUTHORIZATION not in default_text


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
