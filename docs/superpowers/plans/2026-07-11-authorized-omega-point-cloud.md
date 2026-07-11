# Authorized VGGT Omega Point Cloud Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the public scene's coarse-only reconstruction view with an explicitly authorized, real 120,000-point colored VGGT Omega sample while preserving the relative-only and media-withholding boundaries.

**Architecture:** Extend the fail-closed Python public-demo builder with a narrow point-publication opt-in, exact binary validation/copying, shared path/point display-transform metadata, and exact-path audit allowlisting. Add a dependency-free WebGL point renderer shared by the landing hero and scene, keeping Canvas for analytical overlays and falling back to the existing density abstraction if binary loading or WebGL initialization fails.

**Tech Stack:** Python 3.11+, NumPy, pytest, Typer, dependency-free HTML/CSS/Canvas JavaScript, Node syntax checking, GitHub Pages.

---

## File map

- `src/fpv_vggt_lab/public_demo.py`: publication configuration, validation, binary copy, scene contract, audit allowlist, and CLI flags.
- `src/fpv_vggt_lab/public_demo_assets/point-cloud-webgl.js`: shared typed-array validation and WebGL point renderer.
- `src/fpv_vggt_lab/public_demo_assets/scene.js`: typed-array loading, shared display transform, colored point rendering, status, and fallback.
- `src/fpv_vggt_lab/public_demo_assets/scene.html`: point layer control, accessible labels, authorization/withholding copy, and point-count readout.
- `src/fpv_vggt_lab/public_demo_assets/gallery.js`: real-point hero initialization and density fallback.
- `src/fpv_vggt_lab/public_demo_assets/gallery.html`: real Omega hero/proof copy and shared WebGL asset loading.
- `src/fpv_vggt_lab/public_demo_assets/site.css`: loading/ready/fallback state styling only if existing styles are insufficient.
- `tests/test_public_demo_point_cloud.py`: opt-in, validation, asset-integrity, contract, and audit behavior.
- `tests/test_public_demo.py`: presentation contract assertions updated for the real-point layer without weakening default private behavior.
- `docs/adr/ADR-005-authorized-derived-point-cloud-publication.md`: narrow exception, provenance, reversibility boundary, and exclusions.

### Task 1: Lock the authorized publication contract with failing tests

**Files:**
- Create: `tests/test_public_demo_point_cloud.py`
- Modify: `tests/test_public_demo.py`

- [ ] **Step 1: Add synthetic RGB data and an authorized-build helper**

Add a helper that writes one RGB8 row per synthetic XYZ row and constructs this explicit configuration:

```python
def _authorized_config(scene: Path, archive: Path, output: Path) -> PublicDemoConfig:
    return PublicDemoConfig(
        scene_root=scene,
        archive_root=archive,
        output_root=output,
        slug="relative-geometry-study",
        public_title="Relative Geometry Study",
        publish_real_point_cloud=True,
        point_cloud_authorization=(
            "Dataset maintainer approval reported by the project owner on 2026-07-11"
        ),
        point_cloud_attribution="Itamar Weiss / FPV Drone Strikes Open Dataset",
        generated_at="2026-07-11T12:00:00Z",
    )
```

- [ ] **Step 2: Add focused failing tests**

Cover these behaviors with separate test functions:

```python
def test_authorized_build_copies_exact_colored_point_sample_and_contract(tmp_path: Path) -> None:
    ...
    assert payload["point_cloud"]["point_count"] == 9000
    assert payload["point_cloud"]["scale_status"] == "relative_only"
    assert payload["publication_boundary"]["real_point_sample_published"] is True
    assert published_positions.read_bytes() == source_positions.read_bytes()
    assert published_colors.read_bytes() == source_colors.read_bytes()


def test_real_point_publication_requires_authorization_and_attribution(tmp_path: Path) -> None:
    ...
    with pytest.raises(ValueError, match="authorization"):
        PublicDemoConfig(..., publish_real_point_cloud=True)


def test_real_point_publication_rejects_mismatched_color_count(tmp_path: Path) -> None:
    ...
    with pytest.raises(PublicDemoError, match="RGB"):
        build_public_demo(_authorized_config(scene, archive, output))


def test_public_audit_allows_only_declared_point_binary_paths(tmp_path: Path) -> None:
    ...
    assert allowed["status"] == "passed"
    assert undeclared["status"] == "failed"
```

Also assert that the existing default build emits no `.bin` files and continues to set `real_point_sample_published` false.

- [ ] **Step 3: Run the new tests and verify RED**

Run:

```powershell
python -m pytest tests/test_public_demo_point_cloud.py tests/test_public_demo.py -q
```

Expected: failures because `PublicDemoConfig` lacks the point-publication fields and no `point_cloud` contract/assets exist.

### Task 2: Implement the fail-closed point-publication builder

**Files:**
- Modify: `src/fpv_vggt_lab/public_demo.py`
- Create: `docs/adr/ADR-005-authorized-derived-point-cloud-publication.md`

- [ ] **Step 1: Add narrow opt-in configuration**

Add these dataclass fields and reject incomplete authorization in `__post_init__`:

```python
publish_real_point_cloud: bool = False
point_cloud_authorization: str | None = None
point_cloud_attribution: str | None = None

if self.publish_real_point_cloud:
    if not (self.point_cloud_authorization or "").strip():
        raise ValueError("point-cloud publication requires authorization provenance")
    if not (self.point_cloud_attribution or "").strip():
        raise ValueError("point-cloud publication requires attribution")
```

- [ ] **Step 2: Extract one shared display transform**

Introduce a helper returning the raw-path quantile center and scalar span. Use it in `_public_paths` and include the same values in the point-cloud contract:

```python
def _display_transform(raw_path: np.ndarray) -> tuple[np.ndarray, float]:
    low = np.quantile(raw_path, 0.01, axis=0)
    high = np.quantile(raw_path, 0.99, axis=0)
    return (low + high) * 0.5, max(float(np.max(high - low)) * 0.5, 1e-9)
```

- [ ] **Step 3: Validate and copy only the two authorized assets**

Validate finite XYZ float32 rows, exact RGB8 count, and the source count from `scene_meta.json`. Copy exact bytes to:

```text
scenes/<slug>/vggt_omega_points.f32.bin
scenes/<slug>/vggt_omega_colors.rgb8.bin
```

The `point_cloud` contract must include relative URLs, `point_count`, `source_point_count`, dtype/components, SHA-256, byte counts, `scale_status`, `display_transform`, attribution, and authorization provenance.

- [ ] **Step 4: Narrow the audit allowlist**

Extend `audit_public_demo` with `authorized_binary_paths: Iterable[str] = ()`. A `.bin` file passes only when its exact POSIX relative path is in that set; `.ply`, `.glb`, arrays, images, media, and every undeclared `.bin` remain forbidden.

- [ ] **Step 5: Update boundary metadata and CLI**

Add CLI options:

```python
publish_real_point_cloud: bool = typer.Option(False, "--publish-real-point-cloud"),
point_cloud_authorization: str | None = typer.Option(None, "--point-cloud-authorization"),
point_cloud_attribution: str | None = typer.Option(None, "--point-cloud-attribution"),
```

Set `real_point_sample_published` according to the opt-in. Keep video, source frames, recognizable reprojections, raw NPZ/PLY, full backend arrays, and absolute paths false.

Bump the public scene schema from `1.0.0` to `1.1.0`, and add `point-cloud-webgl.js` to the builder's exact static-asset copy list so both generated pages remain dependency-free.

- [ ] **Step 6: Document the narrow authorization exception**

ADR-005 must state that the project owner reported Itamar's permission on 2026-07-11, the exact two browser-sample assets are authorized, public Git history is durable, scale remains relative-only, and all media/full-array exclusions remain active.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_public_demo_point_cloud.py tests/test_public_demo.py -q
```

Expected: all focused tests pass with zero failures.

### Task 3: Render the real colored Omega sample with graceful fallback

**Files:**
- Create: `src/fpv_vggt_lab/public_demo_assets/point-cloud-webgl.js`
- Modify: `src/fpv_vggt_lab/public_demo_assets/scene.js`
- Modify: `src/fpv_vggt_lab/public_demo_assets/scene.html`
- Modify: `src/fpv_vggt_lab/public_demo_assets/gallery.js`
- Modify: `src/fpv_vggt_lab/public_demo_assets/gallery.html`
- Modify: `src/fpv_vggt_lab/public_demo_assets/site.css`
- Modify: `tests/test_public_demo.py`

- [ ] **Step 1: Add failing presentation assertions**

Require the generated assets to contain:

```python
for required in (
    'data-layer="points"',
    'id="point-cloud-readout"',
    "loadPointCloud",
    "drawPointCloud",
    "point-cloud-webgl.js",
    "vggt_omega_points",
    "density fallback",
    "120,000",
    "1,000,000",
):
    assert required in combined
```

Keep the existing assertions that there are no external runtimes or published source media.

- [ ] **Step 2: Run the presentation test and verify RED**

Run:

```powershell
python -m pytest tests/test_public_demo.py::test_public_site_is_offline_accessible_and_presentation_complete -q
```

Expected: failure because the point layer, loader, renderer, and copy do not exist.

- [ ] **Step 3: Add a shared typed-array loader and WebGL renderer**

Create `point-cloud-webgl.js` as a small browser global used by both pages. Load the two URLs from `state.data.point_cloud`, validate byte lengths against contract counts, compile a point vertex/fragment shader, and keep failure isolated:

```javascript
async function loadPointCloud(contract) {
  const [positionResponse, colorResponse] = await Promise.all([
    fetch(contract.positions.path, { cache: "force-cache" }),
    fetch(contract.colors.path, { cache: "force-cache" }),
  ]);
  if (!positionResponse.ok || !colorResponse.ok) throw new Error("authorized point sample request failed");
  const points = new Float32Array(await positionResponse.arrayBuffer());
  const colors = new Uint8Array(await colorResponse.arrayBuffer());
  if (points.length !== contract.point_count * 3 || colors.length !== contract.point_count * 3) {
    throw new Error("authorized point sample length mismatch");
  }
  return { points, colors };
}
```

- [ ] **Step 4: Render all aligned colored points**

Apply `(point - center) / scale` from the contract in the vertex shader, then reproduce the existing yaw/pitch/perspective projection. Draw all 120,000 points using source RGB, subtle alpha blending, and the current orbit/zoom state. Layer Canvas paths, grid, camera proxy, and diagnostics above the transparent WebGL canvas. Render density only when selected or when point loading/WebGL fails.

- [ ] **Step 5: Update the scene hierarchy and copy**

Make `Points` the first active layer, retain `Density fallback` as a separate toggle, and show:

```text
Real VGGT Omega point sample
120,000 loaded / 1,000,000 source points · relative_only
```

Update the media notice to state that the authorized colored point sample is published while original video, frames, recognizable reprojections, raw NPZ/PLY, and machine paths remain withheld.

Update the landing hero to render the same authorized sample behind its path overlay. Replace the old `voxels` proof/readout with `120,000 points`, and remove every stale claim that no real point cloud is included.

- [ ] **Step 6: Verify frontend behavior**

Run:

```powershell
node --check src/fpv_vggt_lab/public_demo_assets/scene.js
node --check src/fpv_vggt_lab/public_demo_assets/gallery.js
node --check src/fpv_vggt_lab/public_demo_assets/point-cloud-webgl.js
python -m pytest tests/test_public_demo.py tests/test_public_demo_point_cloud.py -q
```

Expected: JavaScript syntax exit code 0 and all focused tests pass.

### Task 4: Build, publish, and verify GitHub Pages

**Files:**
- Generated on deployment branch: `assets/*`, `scenes/engineering-vehicle-relative-geometry/*`, `build-manifest.json`

- [ ] **Step 1: Run the complete source verification gate**

Run:

```powershell
python -m pytest -q
python -m ruff check src tests
python -m compileall -q src tests
node --check src/fpv_vggt_lab/public_demo_assets/scene.js
node --check src/fpv_vggt_lab/public_demo_assets/gallery.js
node --check src/fpv_vggt_lab/public_demo_assets/point-cloud-webgl.js
```

Expected: every command exits 0.

- [ ] **Step 2: Build the authorized real-data site**

From the source worktree, run the public-demo module with the real scene/archive paths, the existing slug/title, `--publish-real-point-cloud`, the approval provenance, and Itamar attribution. Use the isolated `codex/pages-site` worktree as `--output-root`.

- [ ] **Step 3: Audit generated files and exact sizes**

Verify the manifest contains exactly two `.bin` files, their SHA-256 values match the local source assets, positions are 1,440,000 bytes, colors are 360,000 bytes, and every publication-boundary exclusion remains false except `real_point_sample_published`.

- [ ] **Step 4: Run a local HTTP/browser smoke check**

Serve the Pages worktree with `python -m http.server` on a free localhost port. Verify the scene loads the point assets, reports 120,000/1,000,000, renders colored geometry, supports orbit/playback/layer toggles, has no console errors, and falls back to density when the point URL is deliberately blocked in an isolated test copy.

- [ ] **Step 5: Commit and push source and Pages branches**

Commit the builder/tests/spec/ADR to `codex/pages-wow-layer`. Commit the generated static site and two authorized binaries to `codex/pages-site`. Push both branches without force.

- [ ] **Step 6: Verify the live deployment**

Poll the GitHub Pages deployment until the scene, both binary asset URLs, and build manifest return HTTP 200. Open the live scene, confirm the point-count/status copy and visible colored reconstruction, and verify zero browser-console errors before reporting completion.
