# Safety Scope

## Allowed

- Offline catalog ingestion.
- Explicit local media download for selected clips.
- Local media caching.
- Segment proposal using non-operational quality signals.
- Human segment acceptance and annotation.
- Deterministic frame sampling.
- Imported VGGT prediction bundles.
- Relative camera path and point-cloud visualization in VGGT coordinates.
- Reconstruction reliability diagnostics.
- Conservative scale-free path descriptors.
- Image-space heatmap diagnostics such as frame difference, optical flow
  magnitude, blur/focus signal, and visibility change, limited to local review
  overlays.
- Local-only HTML review artifacts.
- Future retrospective TTVE benchmarking under strict constraints.

## Forbidden

- Live detection or live tracking.
- Target detection for operational use.
- Target vulnerability analysis.
- Drone guidance, control, route optimization, or maneuver recommendation.
- Next-maneuver prediction.
- Precise geolocation, coordinate inference, or map-based corridor analysis.
- Launch-point or target-coordinate inference.
- True speed, true standoff distance, or true dive-angle claims.
- Attack simulation or generative FPV attack-trajectory modeling.
- Tactical recommendations.
- Treating heatmaps as geolocation, ground or map projection, physical units,
  approach corridor analysis, target inference, or tactical recommendation.
- Redistribution of third-party videos, thumbnails, extracted real frames, or
  real-video point clouds.

## Dataset Descriptions

Dataset descriptions are source metadata. Store them as `source_description` and
display them only in provenance or catalog context.

Do not use them to derive target classes, normalize place names, geocode
locations, rank tactical interest, or produce new operational labels.

## Real-Media Artifacts

These are local-only by default:

- downloaded videos,
- thumbnails,
- sampled real frames,
- rendered review HTML embedding real frames,
- rendered clips or GIFs,
- point clouds from real footage,
- heatmap PNGs or heatmap manifests from real footage,
- screenshots of real-media review pages.

Publishable examples must use synthetic or permission-cleared media.

Before sharing, packaging, or reviewing the repository state, run:

```powershell
fpv review audit-local-only --root . --report outputs/reviews/local_only_audit.json
```

The audit scans for common media-derived file extensions outside approved
local-only roots such as `data/media`, `data/frames`, `data/vggt`, `.tmp`, and
`outputs`. It is a placement audit, not a license review.
