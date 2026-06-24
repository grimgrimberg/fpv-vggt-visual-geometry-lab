# ADR-005: H100 Full-Dataset Processing Run

## Status

Accepted.

## Context

The project needs a mostly unattended, high-quality RunPod H100 workflow for the
latest version of the FPV dataset. The live dataset `main` branch was verified on
2026-06-24 at commit `08a630abafb381df482b39c4b06db4d5631ad77d`, whose README
reports 161 MP4 files and whose manifest contains 161 rows.

The goal is to make the H100 run close to "zero-shot" for the operator:
prepare once locally, launch one cloud command, wait for completion or useful
partial completion, then import one return package locally.

The project safety and data rules still apply. Real media and real-media-derived
artifacts are local-only by default. The workflow must not implement
geolocation, route or approach inference, maneuver prediction, target analysis,
true speed, standoff distance, dive-angle estimation, tactical guidance, or
VGGT fine-tuning on this dataset.

## Decision

Use an H100 full-dataset processing run with frozen VGGT or VGGT-compatible
inference, adaptive tiered frame sampling, selected-best bundle import, and
small downstream diagnostic models only.

The user-facing workflow is:

```powershell
fpv h100 prepare --dataset latest --workdir outputs/h100/latest_full_run
```

Upload the generated RunPod job ZIP, then run on the pod:

```bash
bash run_all.sh
```

Return `h100_return.zip`, then import locally:

```powershell
fpv h100 import-return `
  --source outputs/h100_returns/h100_return.zip `
  --workdir outputs/h100/latest_full_run
```

The H100 processing unit is an accepted flight segment, not a whole edited
video. H100 inputs may include human-accepted segments and strict
auto-accepted segments. Strict auto-acceptance must use only non-operational
quality gates and must carry visible provenance such as
`accepted_by = "auto_strict"`. Ambiguous clips remain `needs_human_review`.

Default cloud upload uses sampled frame packs, not full videos. Frame packs must
preserve source frame indices, timestamps, segment bounds, resize metadata,
checksums, and source provenance so returned bundles align with local review
video playback.

The default H100 frame strategy is adaptive tiered sampling:

- smoke tier: small input set to validate environment and runner,
- scout tier: lower frame count across all included segments,
- main tier: high-quality frame count for scout-passing segments,
- high-detail tier: promoted clips with strong scout/main reliability.

Cloud runs keep tiered VGGT bundles for audit, but local import installs one
selected-best validated bundle per segment by default. The selected bundle is
the highest-detail tier that passes strict validation and improves or preserves
reliability relative to lower-detail tiers. Highest frame count is not selected
unconditionally.

Production-grade VGGT output means:

- valid repo-owned VGGT prediction bundle schema,
- complete frame alignment,
- finite camera and point arrays,
- normalizable quaternions,
- secret-free provenance,
- point RGB/confidence/depth preservation when available,
- tier comparison,
- reconstruction reliability,
- explicit fallback and failure reasons.

The first H100 production pass trains diagnostic models only:

- reconstruction suitability model,
- failure-mode classifier,
- tier selection model,
- segment QA ranker.

Dataset titles and descriptions are provenance-only source metadata. They may
travel with job and return artifacts for audit and reproducibility. Diagnostic
models may use only non-semantic completeness features such as metadata
presence, text length, row hashes, or catalog availability flags. They must not
use place names, target names/classes, semantic text embeddings, geocoded
strings, route terms, approach terms, or event labels as predictive features.

RunPod execution should use a pinned container image plus runtime environment
checks. Live dependency installation is allowed only as a fallback/debug path
and should be warned about in production logs.

The cloud returns data, features, diagnostic models, reports, logs, and
provenance by default. Local import renders real-media review HTML, comparison
HTML, run landing pages, heatmap overlays, and side-by-side MP4 exports.

The run must never fail as crash-only. It should return a complete partial
package whenever possible, including clip statuses and reasons. Local import may
accept partial returns as `done_partial` when every clip has a status and reason
and the package validates.

## Consequences

The H100 run prioritizes quality, detail, completeness, reproducibility, and
explainability over fastest runtime.

The implementation needs new or expanded components for:

- latest dataset snapshot refresh and pinning,
- strict local segmentation and strict auto-acceptance,
- H100 frame-pack creation,
- adaptive tier config,
- pinned container/environment checks,
- tiered cloud runner,
- selected-best bundle selection,
- return package validation,
- diagnostic feature tables,
- diagnostic model training,
- local import and review regeneration.

The workflow avoids wasting H100 time on raw edit-boundary segmentation and
keeps real-media visual rendering local. It also preserves auditability when
high-detail tiers fail or produce worse reconstruction than lower-detail tiers.

This decision intentionally excludes VGGT fine-tuning and operational
geospatial/kinematic/maneuver inference. If a future project chooses to explore
those outside this repository's safety scope, it must not be hidden behind this
H100 pipeline or represented as part of this project.
