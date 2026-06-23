# Methodology

## Overview

The project is an offline review pipeline for relative visual geometry:

1. Ingest a pinned dataset catalog snapshot.
2. Select a small local subset explicitly.
3. Download selected media into ignored local cache paths.
4. Audit local media cache readiness.
5. Propose flight segments using cheap quality and motion diagnostics.
6. Require human acceptance of segment boundaries.
7. Sample a fixed number of frames deterministically.
8. Import a VGGT prediction bundle.
9. Validate frame alignment and geometry shapes.
10. Compute reconstruction reliability.
11. Extract scale-free path descriptors.
12. Render local review artifacts and optional side-by-side MP4 exports.

## Safe Feature Coverage Registry

The review workflow tracks these safe feature surfaces as non-operational
diagnostics:

- `review focus queue`: prioritize clips for human inspection using readiness,
  bundle-validity, heatmap, and reliability status, not targets or locations.
- `pose-jump diagnostics`: flag discontinuities in VGGT-relative camera centers
  so descriptors and smoothing can be gated before they imply stable geometry.
- `heatmap qa`: report whether image-space overlay layers were generated,
  degraded, or skipped, with failures recorded as review diagnostics.
- `bundle provenance`: keep source tool, source URL or repository, version,
  frame alignment, and cloud-return audit status adjacent to review evidence.
- `run landing page`: use the run folder's `summary.json`, `NEXT_STEPS.md`,
  review report, and comparison HTML as the local entry point for a completed or
  blocked review run.
- `synthetic geo`: reserve any calibrated ground-plane or map-style demo for
  synthetic or explicitly permission-cleared calibrated inputs, never for real
  historical strike media.

All six surfaces inherit the same warning block: no geolocation, no meters,
relative VGGT frame, and local-only media.

## Local Media Audit

`fpv media audit` is a workflow-readiness check for already fetched local media.
It verifies:

- local file presence,
- checksum consistency with the media inventory,
- byte count consistency,
- basic OpenCV decode probes,
- sane recorded metadata such as positive FPS, frame count, and dimensions.

The audit does not establish authenticity, location, scale, target identity, or
operational usefulness. A `needs_review` result means refresh the cache, inspect
the local file, or choose another clip before spending annotation or GPU time.
Frame-quality warnings are tiered: isolated dark or low-blur sampled frames are
kept as warnings but may still allow cloud handoff; color-dominant edit-artifact
warnings remain blocking until reviewed.

## Segment Proposal

V1 segment proposal may use:

- duration and frame-count checks,
- black/title/end-card likelihood,
- freeze or near-duplicate detection,
- blur and brightness statistics,
- simple motion continuity diagnostics.

It must not use object detection, target detection, geolocation, map data, or
tactical scoring.

Draft segments have `status="proposed"`. Downstream analysis requires
`status="accepted"`.

## Frame Sampling

V1 uses deterministic fixed-count sampling across an accepted segment. The
default target is 64 frames, with smaller values such as 32 supported for low
memory or quick tests.

Each sampled frame records:

- `video_id`,
- `segment_id`,
- source frame index,
- timestamp in seconds,
- local path,
- resize metadata.

## Image-Space Heatmaps

Optional heatmaps are generated from sampled frames after segment acceptance and
frame sampling. The default layers are frame difference, optical flow magnitude,
blur/focus signal, and visibility change. They are PNG overlays plus a
`heatmaps.json` manifest for local review pages.

`fpv viz render` may load the manifest with `--heatmap-manifest`, while `fpv
review run-three --heatmaps` and `fpv run --heatmaps` generate the layers inside
the review artifact folder. If heatmap generation fails for a clip, the workflow
should fail soft: keep the review page and comparison outputs where possible,
omit the unavailable overlay, and record the reason in diagnostics.
When heatmaps are requested, run summaries record `heatmaps: done` when all
requested clip overlays were created and `heatmaps: failed_soft` when one or
more overlays degraded.

Heatmaps are image-space diagnostic aids only. They do not establish physical
scale, location, target identity, or operational usefulness. Do not describe
them as map or ground projection, meters or coordinates, approach corridor
analysis, or tactical guidance.

## Reconstruction Reliability

Reliability is a diagnostic score, not a truth claim. It should combine available
signals such as:

- VGGT confidence fields when present,
- valid pose coverage,
- pose jumps or outliers,
- point-cloud availability,
- frame quality warnings,
- frame alignment completeness.

Outputs should include:

- `reliability_score`,
- `reliability_label`: `good`, `mixed`, `poor`, or `failed`,
- `failure_flags`,
- `safe_to_use_for_descriptors`.

Bad bundles should fail soft: emit warnings and partial review artifacts where
possible, but gate descriptors and stretch diagnostics.

## Scale-Free Descriptors

Allowed v1 descriptors include:

- normalized path length,
- displacement ratio,
- turn-angle summary,
- pose-jump count,
- valid pose count,
- sampled frame count.

Do not compute target-relative features, map features, speed in meters/second,
altitude, standoff distance, dive angle, or approach corridor.

## Relative Pose Smoothing

A 6DoF smoother may be added as a reliability-gated stretch diagnostic. It must
be described as VGGT-frame relative pose smoothing, not physical drone dynamics
or prediction. If a clip is not reliable enough for smoothing, the three-clip
review should still produce the clip summary, review HTML, and comparison page;
the per-clip run report records smoothing as skipped with the reason.

## Review Visualization

Review pages may show local video playback, RGB point clouds when the bundle
provides color, confidence/depth filters, camera frustums, orbit/zoom controls,
and a retrospective VGGT-frame 6DoF pose glyph plus side-by-side drone-state
animation. When smoothing succeeds, the page may expose a raw-vs-smoothed path
toggle; raw VGGT remains the default and the smoothed path is diagnostic only.
When heatmap overlays are available, the page may expose an image-space layer
selector and opacity control. These are diagnostic review views only.
Missing per-point RGB, confidence, or depth must be labeled in the bundle
quality panel and handled with conservative fallbacks rather than silently
implying physical range, scale, or calibrated drone dynamics.
