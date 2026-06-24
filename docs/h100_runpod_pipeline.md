# H100 RunPod Pipeline

## Purpose

This document is the operator and implementation guide for a production-grade
H100 full-dataset processing run. It expands the small manual RunPod workflow in
`docs/runpod_vggt.md` into a full latest-dataset batch process.

The H100 run is intended to be close to zero-shot:

1. prepare locally,
2. upload one ZIP to RunPod,
3. run one shell command,
4. download one return ZIP,
5. import locally,
6. review local artifacts.

## What The H100 Run Is

The H100 run is a batch processing workflow that:

- pins the latest dataset snapshot,
- audits media availability,
- performs strict local segment scouting,
- builds tiered frame packs,
- runs frozen VGGT or VGGT-compatible inference,
- validates tiered prediction bundles,
- selects the best validated tier per segment,
- extracts safe diagnostic features,
- trains small diagnostic models,
- returns a complete package,
- renders review artifacts locally after import.

## What The H100 Run Is Not

The H100 run is not:

- VGGT fine-tuning,
- geolocation,
- coordinate inference,
- route or approach analysis,
- maneuver prediction,
- true speed/standoff/dive-angle estimation,
- target analysis,
- tactical guidance,
- redistribution of media.

## One-Page Operator Flow

Local:

```powershell
fpv h100 prepare `
  --dataset latest `
  --workdir outputs/h100/latest_full_run `
  --auto-segment strict `
  --frame-scout 32 `
  --frame-main 96 `
  --frame-high-detail 128 `
  --resize-scout 768 `
  --resize-main 1024 `
  --resize-high-detail 1024
```

Upload:

```text
outputs/h100/latest_full_run/runpod_job.zip
```

RunPod:

```bash
mkdir -p /workspace/fpv-h100
cd /workspace/fpv-h100
unzip /workspace/runpod_job.zip
bash run_all.sh
```

Download:

```text
/workspace/fpv-h100/h100_return.zip
```

Local import:

```powershell
fpv h100 import-return `
  --source outputs/h100_returns/h100_return.zip `
  --workdir outputs/h100/latest_full_run `
  --vggt-root data/vggt `
  --review-output outputs/reviews/h100_latest_full_run
```

Open:

```text
outputs/reviews/h100_latest_full_run/index.html
```

## Local Preparation Details

### Dataset Snapshot

The preparation command must refresh the live dataset. The planning baseline was
verified on 2026-06-24:

- commit: `08a630abafb381df482b39c4b06db4d5631ad77d`
- README count: `161 MP4 files`
- manifest rows: `161`

The implementation must refresh this at runtime and write:

```text
dataset_snapshot.json
dataset_snapshot.parquet
dataset_manifest.tsv
dataset_readme.md
media_notice.md
```

`dataset_snapshot.json` should include:

```json
{
  "dataset_repo": "https://github.com/itamarwe/fpv-drone-strikes-lebanon-dataset",
  "dataset_branch": "main",
  "dataset_commit": "<sha>",
  "dataset_commit_date": "<iso8601>",
  "retrieved_at": "<iso8601>",
  "manifest_name": "2026-06-21_fpv_renamed_from_first_frames_manifest.tsv",
  "row_count": 161,
  "mp4_url_count": 161,
  "thumbnail_url_count": 161,
  "status": "done"
}
```

Counts should be dynamic. A future dataset update should not break the command
just because the row count is no longer `161`.

### Source Metadata Policy

Raw dataset titles and descriptions are preserved only for provenance.

Allowed fields:

- `source_description_raw`
- `source_title_raw`
- `video_url`
- `thumbnail_url`
- `dataset_commit`
- `manifest_row_hash`
- `metadata_present`
- `metadata_text_length`

Forbidden derived fields:

- coordinates,
- place geocodes,
- target classes,
- target labels,
- route labels,
- approach labels,
- maneuver labels,
- title embeddings,
- description embeddings.

### Media Audit

Preparation should classify every dataset row:

```text
local_ready
downloadable
missing_local
unavailable_remote
decode_failed
checksum_mismatch
skipped_by_policy
```

Recommended checks:

- local path exists,
- byte size,
- checksum,
- OpenCV decode,
- duration,
- FPS,
- frame count,
- width and height,
- sane first/middle/last frame decode.

### Strict Local Segment Scout

The scout should produce `segment_decisions.parquet` with one or more candidate
segments per video and one selected decision.

Allowed diagnostics:

- black frame ratio,
- title-card likelihood from visual features only,
- end-card likelihood from visual features only,
- duplicate/freeze ratio,
- frame-difference continuity,
- optical-flow magnitude continuity,
- blur/focus score,
- brightness/contrast,
- texture score,
- duration,
- successful frame extraction ratio.

Forbidden diagnostics:

- object detection,
- target detection,
- OCR-derived labels,
- place-name parsing,
- geolocation,
- map context,
- tactical semantics.

Decision examples:

```json
{
  "video_id": "2026-06-17_example",
  "segment_id": "segment-001",
  "status": "accepted",
  "accepted_by": "auto_strict",
  "start_sec": 1.42,
  "end_sec": 11.86,
  "confidence": 0.91,
  "reasons": [
    "removed initial title-card-like frames",
    "motion continuity passed",
    "freeze ratio below threshold",
    "texture score passed"
  ],
  "warnings": []
}
```

Ambiguous example:

```json
{
  "video_id": "2026-06-17_example",
  "segment_id": "segment-001",
  "status": "needs_human_review",
  "accepted_by": null,
  "reasons": [
    "multiple edit boundaries detected",
    "long freeze region inside candidate"
  ]
}
```

### Frame-Pack Generation

Frame packs are the default upload unit.

Each tier item:

```text
frame_packs/<tier>/<video_id>/<segment_id>/
  frames.json
  frames/
    frame_000000.jpg
    frame_000001.jpg
  segment_provenance.json
  source_catalog_row.json
  checksums.json
```

`frames.json` must preserve:

- `video_id`,
- `segment_id`,
- `source_video`,
- `segment_start_sec`,
- `segment_end_sec`,
- `frame_index`,
- `timestamp_sec`,
- frame relative path,
- original width and height,
- resized width and height,
- resize long edge,
- checksum.

The cloud ZIP should not include raw MP4 files by default.

## RunPod Pod Setup

### Hardware

Preferred GPU:

- NVIDIA H100.

Minimum for production:

- CUDA visible,
- enough VRAM for high-detail VGGT tier,
- enough disk for frame packs, tier outputs, and return ZIP.

### Container

Use a pinned container image where possible. Record:

- image name,
- image tag,
- image digest if available,
- CUDA version,
- PyTorch version,
- VGGT source and commit/version,
- Python version,
- package list.

Live dependency installation should be fallback/debug only. If the job installs
dependencies during production execution, `environment.json` should include:

```json
{
  "live_dependency_install": true,
  "warning": "production run used live install instead of pinned image"
}
```

### Suggested Pod Layout

```text
/workspace/fpv-h100/
  run_all.sh
  job.yaml
  job_manifest.json
  frame_packs/
  scripts/
  tiers/
  selected/
  features/
  models/
  model_reports/
  logs/
  h100_return.zip
```

## Cloud Stage Order

`run_all.sh` should run these stages in order:

```bash
python scripts/00_env_check.py
python scripts/10_validate_inputs.py
python scripts/20_run_vggt_tier.py --tier smoke
python scripts/20_run_vggt_tier.py --tier scout
python scripts/30_validate_tiers.py --tier scout
python scripts/20_run_vggt_tier.py --tier main
python scripts/30_validate_tiers.py --tier main
python scripts/20_run_vggt_tier.py --tier high_detail
python scripts/30_validate_tiers.py --tier high_detail
python scripts/40_select_best_tier.py
python scripts/50_extract_features.py
python scripts/60_train_diagnostics.py
python scripts/70_package_return.py
```

Every script should append to `run.log` and update `cloud_summary.json`.

## Resumability

The cloud run should be resumable:

- skip environment check only if current environment hash matches,
- skip tier inference when a tier bundle validates,
- rerun invalid tiers only when configured,
- preserve previous failure reports,
- always run packaging at the end if possible.

Status files:

```text
state/
  env_check.done.json
  tier_scout.done.json
  tier_main.done.json
  tier_high_detail.done.json
  selected.done.json
```

## Tier Promotion Policy

Scout is attempted for all included accepted segments.

Main requires:

- scout hard validation passed,
- minimum valid pose coverage,
- no severe segment contamination flags,
- enough segment duration and texture.

High-detail requires:

- main hard validation passed,
- main reliability not failed,
- high-detail frame count meaningful for segment duration,
- no evidence that extra frames are mostly redundant or degraded.

## Selected-Best Policy

For each segment:

1. discard invalid tiers,
2. score remaining tiers,
3. compare reliability and failure flags,
4. prefer higher detail only when reliability is preserved or improved,
5. select lower tier when high-detail degrades reconstruction,
6. record reason.

Example:

```json
{
  "video_id": "example",
  "segment_id": "segment-001",
  "selected_tier": "main",
  "available_tiers": ["scout", "main", "high_detail"],
  "reason": "high_detail introduced pose jumps; main preserved reliability",
  "status": "selected_lower_tier"
}
```

## Diagnostic Model Training

The cloud run trains only small diagnostic models.

Models:

- reconstruction suitability,
- failure-mode classifier,
- tier selection,
- segment QA ranker.

Training rules:

- split by `video_id`,
- reject forbidden columns,
- save feature lists,
- save metrics,
- save model provenance,
- mark skipped models with reason.

Forbidden feature families:

- geolocation,
- semantic source text,
- target labels,
- route/approach terms,
- maneuver labels,
- true physical kinematics.

## Return Package

The cloud job must create `h100_return.zip` even for partial completion whenever
possible. See `docs/h100_return_bundle_format.md` for the exact contract.

Minimum return files:

```text
cloud_summary.json
run.log
environment.json
dataset_snapshot.json
clip_status.parquet
tier_report.parquet
selected_bundle_report.parquet
failures.json
```

## Local Import

The local import command should:

- validate ZIP structure,
- validate `cloud_summary.json`,
- validate selected bundles,
- check forbidden feature columns,
- install selected bundles into `data/vggt`,
- copy feature/model reports into ignored output roots,
- render local review artifacts,
- write `import_report.json`,
- run or recommend local-only audit.

## Troubleshooting

### H100 Out Of Memory

Expected action:

- finish current clip status as `failed_soft` or `selected_lower_tier`,
- continue with lower tier if available,
- do not crash the whole run.

### Dependency Failure

Expected action:

- mark run `blocked_environment`,
- write environment logs,
- package partial return if possible.

### Bad Segment

Expected action:

- mark clip `needs_resample` or `needs_human_review`,
- keep lower-tier diagnostics if valid,
- do not silently shift segment bounds on cloud.

### Return Import Failure

Expected action:

- do not copy partial bundles into `data/vggt`,
- write import report,
- preserve return ZIP for diagnosis.

## Operator Checklist

Before launch:

- confirm latest dataset snapshot count,
- confirm local disk space,
- confirm RunPod storage space,
- confirm pinned image,
- confirm no raw source videos in ZIP unless explicitly intended,
- confirm `runpod_job.zip` SHA256.

After cloud run:

- download `h100_return.zip`,
- preserve `run.log`,
- run dry import,
- inspect partial statuses,
- run local import,
- open review landing page,
- run local-only audit before sharing anything.
