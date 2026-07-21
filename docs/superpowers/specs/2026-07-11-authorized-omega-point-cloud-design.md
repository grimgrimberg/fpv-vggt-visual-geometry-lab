# Authorized VGGT Omega Point Cloud Publication Design

## Decision

Publish the authorized, real 120,000-point colored VGGT Omega browser sample on the existing GitHub Pages research scene. Preserve the full 1,000,000-point Omega reconstruction as source provenance, but do not make the 16.2 MB full-fidelity GLB the default presentation payload in this iteration.

The project owner reports that dataset maintainer Itamar approved publication of the real point cloud on 2026-07-11. This narrow authorization applies to point geometry and color only. The original video, extracted frames, recognizable reprojections, raw prediction arrays, and machine-local paths remain excluded.

## Visual thesis

A dark, instrument-like research scene whose dominant visual is the real colored reconstruction, with scale-free paths and diagnostics reading as quiet analytical overlays rather than competing dashboard chrome.

## Content plan

1. The scene canvas displays the real colored point sample, path layers, and animated camera-pose proxy.
2. The top-line and layer rail state exactly what is displayed: 120,000 sampled points from a 1,000,000-point relative reconstruction.
3. The evidence inspector retains method I/O, provenance, failure classification, calibration state, and the narrowed publication boundary.
4. The trajectory and connectivity docks remain supporting evidence below the reconstruction.

## Interaction thesis

- Orbit, zoom, reset, and synchronized pose playback remain the main spatial interaction.
- Point geometry loads asynchronously with an explicit loading/ready/fallback state; the coarse density layer remains available if binary loading or WebGL initialization fails.
- A dependency-free WebGL point layer renders all 120,000 published sample points, while Canvas remains responsible for paths, grids, diagnostics, and the camera proxy.

## Public scene contract

The public builder must require an explicit opt-in before copying real point assets. When enabled it will:

- validate `points_preview.bin` as finite little-endian float32 XYZ rows;
- validate `points_preview_colors.bin` as exactly one RGB8 triplet per point;
- compute one scale-free similarity transform from the raw camera path and apply it to both paths and point positions at display time;
- copy the exact browser-sample point positions and source RGB values to two narrowly allowlisted binary files under the scene directory, preserving the source-coordinate sample separately from its display transform;
- record count, source count, dtype, component count, byte count, SHA-256, scale state, attribution, and authorization provenance in `scene.json` and `build-manifest.json`;
- retain coarse normalized density as a fallback layer;
- mark `real_point_sample_published=true`, while keeping video, frames, reprojections, full backend arrays, raw NPZ/PLY, and absolute paths false.

The viewer must label the result `relative_only`; it must never display meters, metric speed, calibrated pose, or body attitude.

## Publication audit

The existing fail-closed audit remains the default. It may accept only the two generated scene point assets whose exact relative paths appear in the authorized scene contract. Other `.bin`, `.ply`, `.glb`, media, array, and image files remain forbidden.

## Performance boundary

The default payload is the existing 120,000-point sample: 1,440,000 bytes of XYZ float32 plus 360,000 bytes of RGB8. The scene and gallery share one small dependency-free WebGL renderer so the published count is the rendered count. The full one-million-point GLB is deferred to a later lazy-loaded or object-storage iteration.

## Acceptance criteria

- A build without explicit authorization still publishes only the coarse abstraction.
- An authorized build publishes the two validated, hashed point assets and no other real-media artifact.
- The landing hero and public scene visibly render colored Omega geometry aligned with every trajectory layer.
- UI copy states `120,000 displayed / 1,000,000 source points` and `relative_only`.
- If binary loading fails, the density fallback remains usable and the page reports the fallback state.
- Unit, audit, JavaScript syntax, full test-suite, local-browser, and live GitHub Pages checks pass before the upgrade is reported complete.
