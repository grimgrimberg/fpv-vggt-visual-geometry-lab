# AGENTS.md

## Project identity

This repository is `fpv-vggt-visual-geometry-lab`.

It is an offline historical-media analysis project for visual-geometry reconstruction and ML benchmarking on already-published FPV footage.

The core goal is to:
1. isolate real FPV-flight segments from edited clips,
2. extract frames,
3. run or import VGGT predictions,
4. reconstruct relative camera poses and point clouds,
5. visualize relative camera paths,
6. score reconstruction reliability,
7. extract scale-free path descriptors,
8. optionally train small downstream models for time-to-visible-terminal-event.

For v1, keep TTVE out of scope. Treat it as future retrospective research only.

## Safety scope

Allowed:
- offline video ingestion
- local media caching
- frame extraction
- segment annotation
- quality features
- optical flow
- VGGT inference/import
- relative camera-pose visualization
- point-cloud visualization
- reconstruction diagnostics
- scale-free, non-geolocated path features
- time-to-visible-terminal-event modeling
- dashboard for human review
- local HTML review artifacts
- reconstruction-suitability shortlisting using non-operational quality signals only

Forbidden:
- live detection
- live tracking
- target detection for operational use
- target vulnerability analysis
- drone guidance/control
- route optimization
- next-maneuver prediction
- precise geolocation
- map-based approach-corridor analysis
- launch-point inference
- target-coordinate inference
- true speed in meters/second
- true standoff distance
- true dive angle unless explicitly calibrated and clearly marked as qualitative
- attack simulation
- generative FPV attack-trajectory modeling
- tactical recommendations

Dataset descriptions can be retained only as source metadata. Do not convert
them into analytic labels, target classes, map inputs, or tactical categories.

## Terminology

Use:
- "relative camera path"
- "VGGT coordinate frame"
- "scale-free path descriptor"
- "time-to-visible-terminal-event"
- "TTVE"
- "reconstruction reliability"

Do not use unqualified:
- "actual speed"
- "actual standoff distance"
- "actual approach corridor"
- "true attack trajectory"
- "next maneuver"
- "targeting"

VGGT camera centers are relative and scale ambiguous by default. Do not label them as meters.

## Data rules

Never commit:
- videos
- thumbnails
- extracted frames
- dense optical flow
- VGGT point clouds from real videos
- rendered clips from real videos unless explicitly intended as local-only output
- rendered HTML review pages that embed real frames or real point clouds
- screenshots of real-media review pages

Ignored local paths:
- `data/media/`
- `data/frames/`
- `data/features/`
- `data/vggt/`
- `data/geometry/`
- `data/models/`
- `data/annotations/`
- `outputs/`

The dataset repo’s metadata/catalog can be used, but referenced media are third-party materials and must not be redistributed by this project.

Pin dataset catalog snapshots by commit, manifest date, or retrieval date. The
source dataset is expected to change over time.

## VGGT rules

Do not vendor VGGT.
Do not require VGGT for core tests.
Provide:
- an import path for externally generated VGGT predictions
- a repo-owned VGGT prediction bundle format
- a VGGT adapter only if `vggt` is installed
- mocks for tests

V1 is import-first. Hugging Face, Colab, RunPod, or local VGGT runs are allowed
only as optional ways to produce an importable bundle.

Always document:
- global scale ambiguity
- global rotation ambiguity
- pose outliers
- low-texture failures
- title-card/freeze/replay contamination

## ML rules

Train only small downstream models on precomputed features.

Do not fine-tune VGGT on this dataset.
Do not train large video foundation models.
Do not train generative trajectory models.

Splits must be grouped by `video_id`, never random frame-row splits.

Always include:
- timestamp-only baseline
- reconstruction-reliability-stratified metrics
- warning when performance is dominated by clip-position leakage

TTVE modeling is not part of v1. If added later, frame it as retrospective
benchmarking on historical edited footage, not prediction for live use.

## Testing

Automated tests must use synthetic videos and mocked VGGT outputs only.

Required checks:
- `pytest`
- `fpv --help`
- synthetic end-to-end pipeline

Tests must not download real media, call VGGT, require CUDA, use Hugging Face,
or require Colab/RunPod.

## V1 workflow rules

The v1 workflow is:
1. ingest a pinned dataset catalog snapshot,
2. explicitly fetch a small local subset,
3. propose segment boundaries using cheap non-operational diagnostics,
4. require human acceptance before downstream analysis,
5. sample frames deterministically,
6. import and validate a VGGT prediction bundle,
7. compute reconstruction reliability and conservative descriptors,
8. render local-only HTML review artifacts.

The high-level `fpv run` command should orchestrate the happy path, but it must
stop with clear next steps when human segment review or a VGGT bundle is needed.
Each run must write:
- `run.log`
- `summary.json`
- `NEXT_STEPS.md`
- artifact paths
- stage statuses: `done`, `needs_human_review`, `needs_vggt_bundle`,
  `failed_soft`, or `blocked`
- a warning block: no geolocation, no meters, relative VGGT frame, local-only media
- an environment snapshot without secrets

Failure should produce useful review diagnostics where possible. Descriptor
extraction, comparison summaries, and 6DoF smoothing must be gated by
reconstruction reliability.

## V1 stretch diagnostics

A reliability-gated 6DoF relative pose smoother may be added as a stretch
diagnostic. It must be labeled experimental, scale ambiguous, and not physically
calibrated. Do not describe it as drone dynamics, control, maneuver prediction,
or future trajectory prediction.

## Engineering style

Use Python 3.11+.
Use Typer for CLI.
Use Pydantic for schemas/config validation.
Use Parquet for intermediate tables.
Keep modules small and typed.
Prefer explicit warnings over silent assumptions.
