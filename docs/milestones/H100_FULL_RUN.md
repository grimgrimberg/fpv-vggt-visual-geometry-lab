# Milestone H100: Full-Dataset Processing Run

## Goal

Build a production-grade, mostly unattended RunPod H100 workflow for the latest
dataset snapshot. The run should prepare local inputs once, execute a single
cloud command, return a complete data package, import selected VGGT bundles
locally, train safe diagnostic models, and render local review artifacts.

The milestone implements the decisions in
`docs/adr/ADR-005-H100-full-dataset-processing-run.md`.

## Live Dataset Baseline

The dataset is pinned at preparation time. The planning baseline verified on
2026-06-24 is:

- repository: `https://github.com/itamarwe/fpv-drone-strikes-lebanon-dataset`
- branch: `main`
- commit: `08a630abafb381df482b39c4b06db4d5631ad77d`
- commit date: `2026-06-23T18:04:23Z`
- README-reported media count: `161 MP4 files`
- manifest: `2026-06-21_fpv_renamed_from_first_frames_manifest.tsv`
- manifest rows: `161`
- README MP4 URLs: `161` unique URLs

Implementation must not hard-code `161`. It must refresh and pin the dataset at
run preparation time, then write the exact observed count to the run folder.

## Safety Boundary

Allowed in this milestone:

- latest dataset snapshot and manifest ingestion,
- media availability and local cache audits,
- strict non-operational segment scouting,
- strict auto-acceptance with provenance,
- deterministic frame-pack generation,
- frozen VGGT or VGGT-compatible inference on RunPod H100,
- tiered VGGT bundle generation,
- selected-best bundle import,
- reconstruction reliability,
- scale-free descriptors,
- diagnostic feature tables,
- diagnostic model training,
- local review artifact rendering.

Forbidden in this milestone:

- VGGT fine-tuning on this dataset,
- geolocation or coordinate inference,
- map projection for real videos,
- route or approach inference,
- maneuver prediction,
- target identification or target vulnerability analysis,
- true speed, standoff distance, altitude, or dive-angle estimation,
- tactical recommendations,
- semantic title/description embeddings as model features,
- publishing real videos, real frames, real review HTML, or real point clouds.

## Terminology

- `H100 full-dataset processing run`: one mostly unattended cloud batch over a
  pinned dataset snapshot and accepted local flight segments.
- `Strict auto-accepted flight segment`: a segment promoted by conservative
  non-operational quality gates, with `accepted_by = "auto_strict"` provenance.
- `Selected-best VGGT bundle`: the highest-detail tier that passes hard
  validation and improves or preserves reliability relative to lower tiers.
- `Done partial`: a successful partial return where every clip has a status and
  reason and all returned artifacts validate.

## User-Facing Commands

### Prepare

```powershell
fpv h100 prepare `
  --dataset latest `
  --workdir outputs/h100/latest_full_run `
  --media-dir data/media `
  --media-inventory data/media/media_inventory.parquet `
  --annotations data/annotations/segments.jsonl `
  --frames-root data/frames `
  --frame-scout 32 `
  --frame-main 96 `
  --frame-high-detail 128 `
  --resize-scout 768 `
  --resize-main 1024 `
  --resize-high-detail 1024 `
  --auto-segment strict `
  --metadata-policy provenance-only
```

Expected result:

```text
outputs/h100/latest_full_run/
  run.log
  summary.json
  NEXT_STEPS.md
  dataset_snapshot.json
  dataset_snapshot.parquet
  media_availability.parquet
  segment_decisions.parquet
  frame_pack_manifest.parquet
  runpod_job/
  runpod_job.zip
```

### Run On RunPod

```bash
mkdir -p /workspace/fpv-h100
cd /workspace/fpv-h100
python -m zipfile -e /workspace/runpod_job.zip .
bash run_all.sh
```

Expected cloud result:

```text
h100_return.zip
cloud_summary.json
run.log
environment.json
```

### Import Return

```powershell
fpv h100 import-return `
  --source outputs/h100_returns/h100_return.zip `
  --workdir outputs/h100/latest_full_run `
  --vggt-root data/vggt `
  --review-output outputs/reviews/h100_latest_full_run
```

Expected local result:

```text
outputs/reviews/h100_latest_full_run/
  index.html
  summary.json
  NEXT_STEPS.md
  run.log
  import_report.json
  selected_bundle_report.parquet
  feature_tables/
  model_reports/
  review_artifacts/
```

## Pipeline Stages

### Stage 0: Source Worktree Preflight

Inputs:

- current project checkout,
- local ignored paths,
- network access for dataset metadata refresh.

Actions:

- check Python version and package versions,
- check `ffmpeg` or OpenCV decode capability,
- check free disk space,
- check no secrets are present in RunPod config,
- check local-only audit baseline if outputs already exist,
- record current git commit and dirty-state summary.

Status values:

- `done`
- `failed_soft`
- `blocked_environment`

### Stage 1: Latest Dataset Snapshot

Actions:

- fetch GitHub `main` commit metadata,
- fetch README,
- fetch `MEDIA_NOTICE.md`,
- fetch manifest TSV,
- parse README video table,
- parse MP4 URLs and thumbnail URLs,
- cross-check README MP4 URL stems against manifest `target_stem`,
- write row hashes,
- write source metadata with raw title/description preserved as provenance only.

Outputs:

```text
dataset_snapshot.json
dataset_snapshot.parquet
dataset_manifest.tsv
dataset_readme.md
media_notice.md
```

Required fields:

- `dataset_repo`
- `dataset_branch`
- `dataset_commit`
- `dataset_commit_date`
- `retrieved_at`
- `manifest_name`
- `manifest_sha256`
- `readme_sha256`
- `media_notice_sha256`
- `row_count`
- `mp4_url_count`
- `thumbnail_url_count`
- `video_id`
- `date`
- `target_stem`
- `source_description_raw`
- `video_url`
- `thumbnail_url`
- `manifest_confidence`
- `manifest_notes`
- `manifest_row_hash`

Acceptance:

- row count equals unique MP4 URL count,
- every manifest stem has a URL,
- every URL has one manifest row,
- no rows have blank `video_id` or `video_url`.

### Stage 2: Media Availability And Local Cache Audit

Actions:

- optionally perform HTTP HEAD or ranged GET checks,
- check local media cache,
- verify local file checksum when already cached,
- probe duration, FPS, frame count, width, height,
- record unavailable or skipped media with reasons.

Outputs:

```text
media_availability.parquet
media_inventory_update.parquet
media_audit_report.json
```

Clip status values:

- `local_ready`
- `downloadable`
- `missing_local`
- `unavailable_remote`
- `decode_failed`
- `checksum_mismatch`
- `skipped_by_policy`

Acceptance:

- every dataset row has a media status,
- locally selected clips have successful decode probes,
- unavailable clips are not packaged for H100.

### Stage 3: Strict Local Segment Scout

Actions:

- use local CPU frame/video diagnostics,
- detect and remove likely title cards, end cards, black frames, freeze regions,
  and replay/edit contamination,
- compute blur, brightness, texture, motion continuity, near-duplicate rate,
  duration sanity, and frame extraction stability,
- propose one or more flight-segment candidates,
- strict-auto-accept only candidates passing conservative gates.

Allowed signals:

- pixel brightness and contrast,
- blur and focus statistics,
- color-dominant edit artifacts,
- frame difference,
- optical flow magnitude,
- freeze or near-duplicate detection,
- local frame extraction success,
- duration and frame-count sanity.

Forbidden signals:

- object detection,
- target class,
- place-name parsing,
- geolocation,
- map context,
- tactical semantics.

Outputs:

```text
segment_candidates.parquet
segment_decisions.parquet
segment_contact_sheets/
```

Decision values:

- `accepted_human`
- `accepted_auto_strict`
- `needs_human_review`
- `rejected`
- `blocked_input`

Acceptance:

- every video has a segment decision,
- every auto-accepted segment has threshold evidence,
- ambiguous clips are not silently accepted.

### Stage 4: Adaptive Frame-Pack Generation

Actions:

- generate deterministic frame packs for accepted segments,
- preserve original source frame index and timestamp,
- resize by tier without changing aspect ratio,
- write per-tier manifests,
- compute checksums for every frame and manifest.

Default tiers:

```text
smoke:       8 frames,   resize 512 or 768, first 1-3 clips only
scout:      32 frames,  resize 768
main:       96 frames,  resize 1024
high_detail:128 frames, resize 1024
```

Promotion policy:

- all included clips get scout,
- main requires scout hard validation and minimum reliability,
- high-detail requires main reliability and enough segment duration/texture.

Outputs:

```text
frame_packs/
  <tier>/<video_id>/<segment_id>/frames.json
  <tier>/<video_id>/<segment_id>/frames/*.jpg
frame_pack_manifest.parquet
```

Acceptance:

- frame count matches tier config or explicit short-clip reason,
- timestamps are strictly nondecreasing,
- frame indices are strictly increasing unless source decode reports otherwise,
- frame paths are relative in cloud package,
- checksums are present.

### Stage 5: RunPod Job Package

Actions:

- write job manifest,
- copy frame packs,
- include cloud runner scripts,
- include environment check script,
- include selected config,
- include README with exact RunPod commands,
- zip package and compute SHA256.

Outputs:

```text
runpod_job/
  README.md
  run_all.sh
  job.yaml
  job_manifest.json
  scripts/
  frame_packs/
runpod_job.zip
```

Acceptance:

- no raw MP4s by default,
- no absolute local paths,
- no credentials,
- no unapproved real-media artifacts outside frame packs,
- zip hash recorded in `summary.json`.

### Stage 6: RunPod Environment Check

Cloud actions:

- verify H100 or expected GPU class,
- verify CUDA availability,
- verify PyTorch CUDA,
- verify VGGT import,
- verify model/checkpoint availability,
- verify disk space,
- verify package versions,
- verify no missing frame packs.

Outputs:

```text
environment.json
environment_check.log
```

Acceptance:

- hard fail before expensive inference if GPU/model/deps are wrong,
- production run warns if dependencies were installed live.

### Stage 7: Tiered VGGT Inference

Cloud actions:

- run smoke tier,
- run scout tier,
- validate scout outputs,
- promote eligible clips to main,
- validate main outputs,
- promote eligible clips to high-detail,
- validate high-detail outputs,
- skip completed valid tiers on rerun.

Outputs:

```text
tiers/<tier>/<video_id>/<segment_id>/
  metadata.json
  cameras.npz
  points.npz
  validation.json
  quality.json
predictions/<tier>/...
```

Acceptance:

- each attempted tier has status and reason,
- invalid bundles are preserved for diagnostics only if safe and clearly marked,
- valid bundles conform to repo-owned format.

### Stage 8: Selected-Best Bundle Selection

Cloud actions:

- compare tiers per segment,
- reject tiers failing hard validation,
- score reliability and quality,
- select highest-detail tier that improves or preserves reliability,
- write selected bundle tree.

Outputs:

```text
selected/<video_id>/<segment_id>/
selected_bundle_report.parquet
tier_report.parquet
```

Acceptance:

- every included segment has selected tier or failure reason,
- high-detail is not selected merely because it has more frames,
- selected bundle validates independently.

### Stage 9: Feature Tables

Cloud actions:

- extract frame quality features,
- extract bundle quality features,
- extract reconstruction reliability features,
- extract conservative scale-free descriptors,
- extract tier comparison features,
- include only non-semantic source metadata completeness features.

Outputs:

```text
features/
  segment_features.parquet
  frame_features.parquet
  bundle_quality.parquet
  reconstruction_summary.parquet
  tier_features.parquet
  source_metadata_completeness.parquet
```

Forbidden feature columns:

- coordinates,
- geocoded place names,
- target classes,
- target-relative features,
- route or approach labels,
- maneuver labels,
- true speed/standoff/dive-angle,
- title/description embeddings.

### Stage 10: Diagnostic Model Training

Models:

- reconstruction suitability model,
- failure-mode classifier,
- tier selection model,
- segment QA ranker.

Rules:

- split by `video_id`,
- include timestamp-only and simple quality baselines where applicable,
- record feature list for every model,
- reject forbidden feature names,
- write metrics stratified by reconstruction reliability where meaningful,
- train small models only.

Outputs:

```text
models/
  reconstruction_suitability.joblib
  failure_mode_classifier.joblib
  tier_selection.joblib
  segment_qa_ranker.joblib
model_reports/
  *.json
```

Acceptance:

- models train or skip with reason,
- no forbidden feature columns,
- every model has report and provenance,
- metrics are clearly diagnostic, not operational claims.

### Stage 11: Return Package

Cloud actions:

- package selected bundles,
- package tier reports and features,
- package diagnostic models and reports,
- package logs and environment,
- package partial results even on failure.

Output:

```text
h100_return.zip
```

Acceptance:

- return package validates against `docs/h100_return_bundle_format.md`,
- every clip has a status and reason,
- partial returns are accepted when complete enough to diagnose.

### Stage 12: Local Import And Rendering

Local actions:

- validate return package,
- dry-run selected bundle install,
- install selected bundles to `data/vggt`,
- import feature tables and model reports under ignored outputs,
- regenerate local review pages,
- render run landing page,
- render comparison page,
- optionally render side-by-side MP4s locally,
- run local-only audit.

Outputs:

```text
outputs/reviews/h100_latest_full_run/
  index.html
  comparison.html
  review_artifacts/
  import_report.json
  summary.json
  NEXT_STEPS.md
```

Acceptance:

- imported selected bundles validate locally,
- feature/model reports are linked from landing page,
- local-only audit passes.

## File-By-File Implementation Plan

### New CLI Surface

- `src/fpv_vggt_lab/cli.py`
  - add `h100_app = typer.Typer(...)`
  - add `fpv h100 prepare`
  - add `fpv h100 import-return`
  - add `fpv h100 inspect-return`
  - add `fpv h100 summarize`

### Dataset Snapshot

- `src/fpv_vggt_lab/dataset_snapshot.py`
  - fetch GitHub repo metadata,
  - fetch README, media notice, manifest,
  - parse README media table,
  - parse manifest TSV,
  - cross-check counts/stems,
  - write JSON/Parquet outputs.

### Strict Segmentation

- `src/fpv_vggt_lab/segment_scout.py`
  - detect edit/title/end-card regions,
  - compute frame diagnostics,
  - propose segment candidates,
  - strict auto-accept candidates,
  - write decisions and contact sheets.

### H100 Package Builder

- `src/fpv_vggt_lab/h100_package.py`
  - build frame packs,
  - write tier config,
  - write job manifest,
  - write cloud scripts,
  - zip and hash the package.

### H100 Cloud Scripts

Generated into `runpod_job/scripts/`:

- `00_env_check.py`
- `10_validate_inputs.py`
- `20_run_vggt_tier.py`
- `30_validate_bundle.py`
- `40_select_best_tier.py`
- `50_extract_features.py`
- `60_train_diagnostics.py`
- `70_package_return.py`

Top-level generated:

- `run_all.sh`
- `job.yaml`
- `README.md`

### Return Importer

- `src/fpv_vggt_lab/h100_return.py`
  - validate return ZIP,
  - parse reports,
  - validate selected bundles,
  - import selected bundles,
  - import feature/model reports,
  - trigger local review rendering.

### Feature Tables And Models

- `src/fpv_vggt_lab/features.py`
  - create safe diagnostic feature tables.
- `src/fpv_vggt_lab/diagnostic_models.py`
  - train small diagnostic models.
- `src/fpv_vggt_lab/feature_policy.py`
  - forbid unsafe feature columns and semantic source metadata.

### Schemas

- `src/fpv_vggt_lab/schemas.py`
  - add H100 config schema,
  - add dataset snapshot schema,
  - add segment decision schema,
  - add tier report schema,
  - add selected bundle report schema,
  - add model report schema.

### Docs

- `docs/h100_runpod_pipeline.md`
- `docs/h100_return_bundle_format.md`
- `docs/milestones/H100_FULL_RUN.md`
- `docs/adr/ADR-005-H100-full-dataset-processing-run.md`

## Test Plan

Tests must use synthetic videos, mocked VGGT outputs, and small fake H100 return
packages. They must not download real media, require CUDA, call real VGGT, or
contact RunPod.

Required tests:

- dataset snapshot parser with saved fixture README/manifest,
- stale local catalog detection,
- README MP4 URL count equals manifest row count,
- strict segment scout on synthetic title/flight/end-card video,
- strict segment scout rejects ambiguous edited synthetic clip,
- frame pack builder preserves timestamps and frame indices,
- frame pack ZIP excludes raw MP4 and absolute paths,
- generated `run_all.sh` exists and references ordered stages,
- fake tier reports select lower tier when high-detail reliability is worse,
- return package validator accepts complete partial package,
- return importer rejects forbidden feature columns,
- diagnostic model trainer splits by `video_id`,
- local import produces landing page links,
- local-only audit ignores approved H100 output roots and flags misplaced media.

Recommended commands:

```powershell
ruff check src tests
pytest tests/test_dataset_snapshot.py -q
pytest tests/test_segment_scout.py -q
pytest tests/test_h100_package.py -q
pytest tests/test_h100_return.py -q
pytest tests/test_diagnostic_models.py -q
fpv h100 prepare --help
fpv h100 import-return --help
```

Full guard:

```powershell
ruff check src tests
pytest
fpv review audit-local-only --root . --report outputs/reviews/local_only_audit.json
```

## Acceptance Criteria

The milestone is complete when:

- `fpv h100 prepare --help` works,
- `fpv h100 import-return --help` works,
- a synthetic full-run fixture produces `runpod_job.zip`,
- the generated job contains no raw MP4s by default,
- a fake H100 return imports locally,
- selected-best tier logic is tested,
- forbidden feature policy is tested,
- local review landing page links imported reports,
- docs contain exact operator commands,
- full guard passes.

## Open Decisions Before Coding

- Exact pinned container image or image-building method.
- Whether to support 192-frame high-detail tier later.
- Whether to generate contact sheets for every auto-accepted segment or only
  borderline segments.
- Whether local media download should be all-at-once or batch/resumable by date.
- Where model artifacts should live under ignored outputs.

## Risks

- The latest dataset may change while a run is prepared. Mitigation: pin commit
  and manifest hashes in `dataset_snapshot.json`.
- Some video URLs may fail or decode differently. Mitigation: local audit and
  per-clip status.
- Strict auto-acceptance may reject too many clips. Mitigation: preserve
  `needs_human_review` list and make thresholds configurable.
- More frames may degrade VGGT on edited or low-texture clips. Mitigation:
  adaptive tiers and selected-best logic.
- H100 dependency drift could waste cost. Mitigation: pinned image and early
  environment check.
- Large return packages may be slow to move. Mitigation: return selected bundles
  plus compressed tier reports; keep raw predictions optional.
