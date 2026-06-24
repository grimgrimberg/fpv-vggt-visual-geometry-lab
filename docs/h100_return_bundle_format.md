# H100 Return Bundle Format

## Purpose

This document defines the file contract for `h100_return.zip`, the output of an
H100 full-dataset processing run. The local importer should validate this
contract before installing selected VGGT bundles or rendering review artifacts.

The return bundle is local-only by default because it may contain real-media
derived point clouds, camera paths, features, logs, and model artifacts.

## Top-Level Layout

```text
h100_return.zip
  RETURN_MANIFEST.json
  cloud_summary.json
  run.log
  environment.json
  dataset_snapshot.json
  failures.json
  clip_status.parquet
  tier_report.parquet
  selected_bundle_report.parquet
  selected/
    <video_id>/<segment_id>/
      metadata.json
      cameras.npz
      points.npz
      validation.json
      quality.json
  tiers/
    scout/<video_id>/<segment_id>/
    main/<video_id>/<segment_id>/
    high_detail/<video_id>/<segment_id>/
  features/
    segment_features.parquet
    frame_features.parquet
    bundle_quality.parquet
    reconstruction_summary.parquet
    tier_features.parquet
    source_metadata_completeness.parquet
  models/
    reconstruction_suitability.joblib
    failure_mode_classifier.joblib
    tier_selection.joblib
    segment_qa_ranker.joblib
  model_reports/
    reconstruction_suitability.json
    failure_mode_classifier.json
    tier_selection.json
    segment_qa_ranker.json
```

Only `RETURN_MANIFEST.json`, `cloud_summary.json`, `run.log`,
`environment.json`, `dataset_snapshot.json`, `failures.json`,
`clip_status.parquet`, `tier_report.parquet`, and
`selected_bundle_report.parquet` are required for a complete partial return.
Selected bundles, features, and models may be absent when a run is blocked
early, but their absence must be explained by statuses and reasons.

## RETURN_MANIFEST.json

Required fields:

```json
{
  "schema_version": "h100-return-v1",
  "created_at": "2026-06-24T00:00:00Z",
  "run_id": "h100_latest_full_run",
  "status": "done_partial",
  "dataset_commit": "08a630abafb381df482b39c4b06db4d5631ad77d",
  "dataset_row_count": 161,
  "selected_bundle_count": 120,
  "clip_count": 161,
  "clip_status_path": "clip_status.parquet",
  "tier_report_path": "tier_report.parquet",
  "selected_bundle_report_path": "selected_bundle_report.parquet",
  "warnings": [
    "no geolocation",
    "no meters",
    "relative VGGT frame",
    "local-only media"
  ]
}
```

Allowed `status` values:

- `done`
- `done_partial`
- `failed_soft`
- `blocked_environment`
- `blocked_input`

Validation:

- `schema_version` must equal `h100-return-v1`,
- timestamps must parse as ISO-8601,
- counts must be non-negative,
- referenced top-level paths must exist,
- warnings must include the safety labels.

## cloud_summary.json

Purpose:

- human-readable and machine-readable run summary,
- stage status,
- high-level counts,
- package provenance.

Required fields:

```json
{
  "status": "done_partial",
  "run_id": "h100_latest_full_run",
  "started_at": "...",
  "finished_at": "...",
  "stage_status": {
    "environment": "done",
    "input_validation": "done",
    "scout": "done",
    "main": "done_partial",
    "high_detail": "done_partial",
    "selection": "done",
    "features": "done",
    "models": "done",
    "packaging": "done"
  },
  "counts": {
    "dataset_rows": 161,
    "included_segments": 120,
    "selected_bundles": 118,
    "failed_soft": 2,
    "needs_human_review": 41
  },
  "warnings": [
    "no geolocation",
    "no meters",
    "relative VGGT frame",
    "local-only media"
  ]
}
```

## environment.json

Required fields:

```json
{
  "python": "3.11.x",
  "platform": "Linux",
  "gpu_name": "NVIDIA H100 ...",
  "cuda_available": true,
  "cuda_version": "...",
  "torch_version": "...",
  "vggt_source": "facebookresearch/vggt",
  "vggt_version_or_commit": "...",
  "container_image": "...",
  "container_digest": "...",
  "live_dependency_install": false,
  "disk_free_gb_at_start": 0,
  "disk_free_gb_at_finish": 0
}
```

Validation:

- `cuda_available` must be true for any attempted VGGT tier,
- `gpu_name` must be present,
- live dependency install should trigger a warning,
- no secrets or credentials may appear.

## clip_status.parquet

One row per dataset video or accepted segment, depending on stage. Required
columns:

```text
dataset_commit: string
video_id: string
segment_id: string
source_row_hash: string
source_metadata_present: bool
source_metadata_length: int
media_status: string
segment_status: string
accepted_by: string|null
included_in_h100: bool
final_status: string
final_reason: string
selected_tier: string|null
selected_bundle_path: string|null
warnings: list<string>
```

Allowed `final_status` values:

- `done`
- `selected_lower_tier`
- `failed_soft`
- `needs_human_review`
- `needs_resample`
- `blocked_environment`
- `blocked_input`
- `skipped_by_policy`

Validation:

- every row must have `video_id`,
- every included row must have `segment_id`,
- every row must have `final_status` and `final_reason`,
- included rows with `done` or `selected_lower_tier` must reference selected
  bundle paths.

## tier_report.parquet

One row per attempted tier per segment.

Required columns:

```text
video_id: string
segment_id: string
tier: string
frame_count: int
resize_long_edge: int
status: string
reason: string
bundle_path: string|null
hard_valid: bool
validation_errors: list<string>
validation_warnings: list<string>
reliability_label: string|null
reliability_score: float|null
valid_pose_fraction: float|null
mean_pose_confidence: float|null
pose_jump_count: int|null
point_count: int|null
has_point_rgb: bool
has_point_confidence: bool
has_point_depth: bool
started_at: string|null
finished_at: string|null
runtime_sec: float|null
```

Allowed `tier` values:

- `smoke`
- `scout`
- `main`
- `high_detail`

Allowed `status` values:

- `done`
- `failed_soft`
- `skipped_not_promoted`
- `skipped_existing_valid`
- `blocked_environment`
- `blocked_input`

Validation:

- `hard_valid` true requires no validation errors,
- `done` requires bundle path,
- skipped rows require reason.

## selected_bundle_report.parquet

One row per included segment.

Required columns:

```text
video_id: string
segment_id: string
selected_tier: string|null
selected_bundle_path: string|null
selection_status: string
selection_reason: string
available_tiers: list<string>
rejected_tiers: list<string>
selected_reliability_score: float|null
selected_reliability_label: string|null
selected_pose_jump_count: int|null
selected_point_count: int|null
```

Allowed `selection_status` values:

- `selected`
- `selected_lower_tier`
- `no_valid_tier`
- `needs_resample`
- `needs_human_review`
- `blocked_input`

Validation:

- `selected` and `selected_lower_tier` require selected bundle path,
- `no_valid_tier` requires a non-empty reason,
- high-detail selection requires explanation if lower tiers were available.

## Selected Bundle Format

Each selected bundle must match the repo-owned VGGT prediction bundle format.

Required:

```text
metadata.json
cameras.npz
validation.json
quality.json
```

Optional:

```text
points.npz
```

### metadata.json

Required fields:

```json
{
  "schema_version": "vggt-bundle-v1",
  "video_id": "...",
  "segment_id": "segment-001",
  "source_tool": "runpod",
  "source_url_or_repo": "facebookresearch/vggt",
  "source_commit_or_version": "...",
  "generated_at": "...",
  "frame_indices": [0, 1, 2],
  "frame_timestamps_sec": [0.0, 0.5, 1.0],
  "coordinate_frame": "VGGT relative camera frame",
  "warnings": []
}
```

Must not include:

- API keys,
- tokens,
- signed URLs with secrets,
- RunPod credentials,
- account identifiers not needed for reproducibility.

### cameras.npz

Required arrays:

```text
camera_centers: float, shape (N, 3)
quaternions_xyzw: float, shape (N, 4)
valid_pose_mask: bool, shape (N,)
pose_confidence: float, shape (N,)
```

Validation:

- all arrays finite except boolean mask,
- `N` matches metadata frame count,
- quaternions are finite and normalizable,
- confidence values are finite.

### points.npz

Required if present:

```text
points: float, shape (P, 3)
```

Optional arrays:

```text
point_colors_rgb: shape (P, 3)
point_confidence: shape (P,)
point_depth: shape (P,)
```

Validation:

- all present arrays finite,
- optional arrays align to point count,
- RGB values are interpretable as either `0..1` float or `0..255` integer.

## features/

Feature tables are diagnostic only.

Allowed:

- frame quality metrics,
- media decode metrics,
- segment quality metrics,
- VGGT validation metrics,
- reconstruction reliability,
- scale-free descriptors,
- tier comparison metrics,
- source metadata completeness flags.

Forbidden:

- coordinates,
- geocoded strings,
- place-name features,
- target class features,
- route/approach labels,
- maneuver labels,
- real-world speed/standoff/dive-angle estimates,
- semantic embeddings of titles/descriptions.

The importer must reject feature tables with forbidden column names or metadata
declaring forbidden feature families.

## models/

Allowed model names:

- `reconstruction_suitability.joblib`
- `failure_mode_classifier.joblib`
- `tier_selection.joblib`
- `segment_qa_ranker.joblib`

Each model must have a matching report in `model_reports/`.

Model report required fields:

```json
{
  "model_name": "reconstruction_suitability",
  "status": "done",
  "trained_at": "...",
  "training_rows": 0,
  "split_policy": "grouped_by_video_id",
  "feature_columns": [],
  "forbidden_feature_check": "passed",
  "metrics": {},
  "warnings": [
    "diagnostic model only",
    "not an operational predictor"
  ]
}
```

Allowed model statuses:

- `done`
- `skipped_insufficient_data`
- `skipped_policy`
- `failed_soft`

## failures.json

Purpose:

- preserve every non-crash failure as structured information.

Format:

```json
{
  "failures": [
    {
      "scope": "clip",
      "video_id": "...",
      "segment_id": "segment-001",
      "stage": "high_detail",
      "status": "failed_soft",
      "reason": "CUDA out of memory",
      "recoverable": true,
      "next_step": "selected lower tier"
    }
  ]
}
```

## Importer Validation Checklist

The local importer must validate:

- ZIP opens,
- `RETURN_MANIFEST.json` exists,
- schema version matches,
- required reports exist,
- every clip has status and reason,
- selected bundles validate,
- selected paths do not escape extraction root,
- no secret-looking strings appear in provenance,
- feature policy passes,
- model reports exist for model files,
- local destination paths are under configured ignored roots.

## Status Semantics

### done

All included segments completed, selected bundles validate, feature/model
artifacts validate, and local import can proceed normally.

### done_partial

Some clips failed or were skipped, but every clip has status and reason, all
returned selected bundles validate, and local import can proceed for completed
clips.

### failed_soft

The return package is usable for diagnosis, but no selected bundles or models
should be installed without review.

### blocked_environment

The cloud environment was not suitable. Example: no CUDA, wrong GPU, missing
model dependency, or insufficient disk.

### blocked_input

Inputs were invalid before VGGT inference. Example: corrupted frame pack,
missing manifest, bad checksum.

## Local-Only Rules

Never commit or publish:

- returned selected bundles from real media,
- tier bundles from real media,
- point clouds from real media,
- cloud logs containing real-media local paths if sensitive,
- generated review HTML embedding real frames,
- generated side-by-side MP4s from real videos.

Run before sharing:

```powershell
fpv review audit-local-only --root . --report outputs/reviews/local_only_audit.json
```
