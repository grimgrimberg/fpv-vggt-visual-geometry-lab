# Context

This project analyzes already-published FPV footage as offline historical media.
Its domain language must keep visual-geometry review separate from operational
drone analysis.

## Glossary

### Dataset catalog

The remote repository metadata that lists dataset video identity, dates,
descriptions, thumbnails, and download links. The catalog is not a license grant
for third-party media.

### Source description

The original description text supplied by the dataset catalog. It is provenance
metadata only and must not become an analytic target label, a map feature, or an
operational category.

### Local media cache

Downloaded media files held under ignored local paths for analysis. Local media
cache contents are not redistributed by this repository.

### Flight segment

The portion of a clip selected for relative visual-geometry analysis. V1 may
propose draft segments automatically, but downstream analysis requires a human
accepted segment.

### Strict auto-accepted flight segment

A flight segment promoted by conservative non-operational quality gates rather
than direct human review. The gates may use edit-boundary removal, black/title/end
card detection, freeze/replay checks, blur/brightness sanity, motion continuity,
visual texture, duration, and frame extraction success. They must not use target
class, location, object detection, tactical semantics, or map context.

Strict auto-accepted segments carry provenance such as `accepted_by = "auto_strict"`. Ambiguous clips remain `needs_human_review`. H100 processing
may include human accepted and strict auto-accepted segments, but the distinction
must remain visible in manifests and review artifacts.
The segmentation strategy for an H100 run is local strict segmentation first and
H100 post-VGGT quality feedback second. Local CPU segmentation may use cheap
video/frame diagnostics to avoid spending H100 time on raw edit-boundary
selection. H100 outputs may later flag segment-quality problems through
reconstruction failures, pose jumps, confidence collapse, or low point count,
but v1 should report those as resampling or review prompts rather than silently
moving segment bounds.

### VGGT prediction bundle

The repo-owned imported artifact format for VGGT outputs. It stores relative
camera data, point-cloud data, frame alignment, and provenance without requiring
VGGT to be installed.

### Relative camera path

The camera-center trajectory reconstructed in the VGGT coordinate frame. It is
scale ambiguous and rotation ambiguous by default. It is not a real-world path
in meters.

### Reconstruction reliability

A transparent diagnostic estimate of whether the imported relative
reconstruction is usable for review or conservative descriptors. It is not a
truth claim about the event.

### Scale-free path descriptor

A relative path-shape metric that avoids meters, geolocation, target-relative
features, and physical speed claims.

### TTVE

Time-to-visible-terminal-event. In this project it is future-only retrospective
benchmarking on historical edited clips, not live prediction or operational
warning.

### Relative pose smoother

An optional reliability-gated diagnostic that smooths imported VGGT-frame poses.
It is not drone dynamics, guidance, control, or future trajectory prediction.

### H100 full-dataset processing run

A mostly unattended, resumable cloud batch run over a pinned latest dataset
snapshot and accepted local flight segments. It uses an H100 to run frozen VGGT
or VGGT-compatible inference, normalize repo-owned VGGT prediction bundles,
extract reconstruction features, train small downstream diagnostic models, and
return detailed local review artifacts.

The run should be close to "zero-shot" for the operator: prepare once, launch on
RunPod, let it run until completion or a useful soft failure, then import one
return package locally. "Accuracy" in this context means best-effort diagnostic
fidelity: input audits, frame/bundle alignment, VGGT confidence, reconstruction
reliability, provenance, and validation checks. It does not mean true
geolocation, physical scale, target identity, speed, standoff distance, or
operational prediction.

The H100 run prioritizes quality, detail, and completeness over fastest runtime.
It should prefer richer frame sampling, higher-resolution inference after a
smoke test, full optional VGGT arrays when available, strict per-clip validation,
and explicit failed-soft records over silently skipping difficult clips.
The default H100 frame strategy is adaptive tiered sampling. A run may start
with a small smoke tier, process all accepted segments through a scout tier,
then promote reliable clips to main and high-detail tiers. A single operator
command can still run the full sequence, but internal tiers protect quality and
cost by avoiding high-detail inference on clips that fail basic reconstruction
or segment-quality gates.

The unit of H100 processing is an accepted flight segment, not a whole edited
video. Whole videos may contain title cards, black frames, edits, replays, or
non-flight portions that would degrade VGGT. Each uploaded cloud item should be
identified by `video_id`, `segment_id`, accepted start/end bounds, sampled
frames, frame manifest, and source provenance.
The H100 cloud upload unit is a sampled frame pack by default, not a source
video. Frame packs must be dense and high-resolution enough for smooth playback
alignment and detailed VGGT reconstruction. They must preserve exact source
frame indices, timestamps, segment bounds, resize metadata, checksums, and
source provenance so returned bundles can be synchronized back to the local
video review page.

H100 cloud runs keep tiered VGGT bundles for audit, but local import installs one
selected-best validated bundle per segment by default. The return package should
preserve scout, main, and high-detail tier diagnostics so the selected bundle is
explainable. Production-grade VGGT output means validated arrays, complete
provenance, frame alignment, point/RGB/confidence/depth availability when
possible, tier comparison, reliability scoring, and clear fallback reasons. It
does not mean choosing the highest frame-count tier unconditionally.

A selected-best VGGT bundle must pass hard validation and quality selection.
Hard validation covers required files, metadata schema, finite camera arrays,
frame count and timestamp alignment, normalizable quaternions, secret-free
provenance, finite point clouds when present, and optional point RGB/confidence/
depth arrays aligned to point count. Quality selection considers valid-pose
coverage, pose confidence, pose jumps, point count, optional point attributes,
reconstruction reliability, frame-quality warnings, and tier-to-tier improvement.
The selected tier should be the highest-detail tier that passes strict validation
and improves or preserves reliability relative to lower-detail tiers.

The first H100 production pass trains diagnostic models only: reconstruction
suitability, failure-mode classification, tier selection, and segment QA
ranking. It must not train or scaffold geolocation, route or approach inference,
maneuver prediction, target analysis, or real-world kinematic estimation. Dataset
titles and descriptions may be preserved as isolated source metadata for
provenance, but they must not be converted into geocodes, coordinates, map
features, target classes, or operational labels.

Source metadata is provenance-only for the H100 run. Raw titles/descriptions,
source URLs, dataset commit, and manifest row hashes may travel with job and
return artifacts for audit and reproducibility. Diagnostic models may use only
non-semantic completeness features such as metadata presence, text length, row
hashes, or catalog availability flags. They must not use place names, target
names/classes, semantic text embeddings, geocoded strings, route terms, approach
terms, or event labels as predictive features.

The H100 zero-shot operator workflow is `fpv h100 prepare`, upload the generated
RunPod job ZIP, run `bash run_all.sh` in the pod, download `h100_return.zip`, and
finish with `fpv h100 import-return`. The prepare command owns latest dataset
snapshotting, local strict segmentation, frame-pack creation, tier config,
provenance-only metadata packaging, and preflight validation. The import command
owns return validation, selected-best bundle installation, feature/model import,
and local review artifact generation.

H100 RunPod execution prefers a pinned container image plus runtime environment
checks over live dependency installation during the expensive run. The job should
record image tag or digest, CUDA/PyTorch/VGGT versions, GPU model, VRAM, disk,
and package versions before processing. A fallback setup path may exist for
manual debugging, but production runs should warn if dependencies were installed
live instead of coming from the pinned image.

H100 runs must never fail as crash-only. They should return a complete partial
package whenever possible, including run logs, environment snapshot, dataset
snapshot, clip statuses, tier report, selected-bundle report, failures, and any
valid completed bundles. Each clip should have a status and reason such as
`done`, `selected_lower_tier`, `failed_soft`, `needs_human_review`,
`needs_resample`, `blocked_environment`, or `blocked_input`. Local import may
accept partial returns but must label the run accordingly.

H100 cloud returns data, features, diagnostic models, reports, logs, and
provenance by default. Local import renders real-media review HTML, comparison
HTML, run landing pages, heatmap overlays, and side-by-side MP4 exports. Cloud
rendering of real-media visuals is not the default because local rendering keeps
third-party media-derived artifacts under local-only paths and preserves local
path alignment.

An H100 run is done only if the latest dataset snapshot is pinned, all selected
media are locally audited or skipped with reasons, strict segmentation assigns
every clip a status, included segments have frame packs, the RunPod environment
check passes, all eligible tiers complete or record reasons, each segment has a
selected-best decision or failure reason, the return package validates locally,
selected bundles import, feature tables are written, diagnostic models train or
skip with reasons, local review artifacts render, and the local-only audit
passes. A run may be `done_partial` when some clips fail if every clip still has
a status and reason and the return package imports.

