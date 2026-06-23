# VGGT Notes

## V1 Position

V1 is import-first. The core project must work without VGGT installed and tests
must use mocked VGGT outputs.

VGGT predictions can come from:

- Hugging Face,
- Colab,
- RunPod,
- local VGGT installation,
- tests or synthetic mocks.

Each source must be converted into the repo-owned VGGT prediction bundle.

## Prediction Bundle

The v1 bundle directory contains:

- `metadata.json`
- `cameras.npz`
- `points.npz` when point-cloud coordinates are available

Required metadata:

- `schema_version`
- `video_id`
- `segment_id`
- `source_tool`
- `source_url_or_repo` when known
- `source_commit_or_version` when known
- `export_notes` when useful
- `generated_at`
- `frame_indices`
- `frame_timestamps_sec`
- `warnings`
- `coordinate_frame`

`source_tool` should be one of:

- `huggingface-space`
- `colab`
- `runpod`
- `local-vggt`
- `mock`

Do not include API keys, private tokens, signed URLs with secrets, or account
details.

Validation treats secret-looking provenance text as an error. If a bundle source
URL, version field, or export note contains strings such as `token`, `api_key`,
`secret`, or `sk-...`, remove the sensitive value and replace it with a
non-secret provenance note before import.

## Hugging Face Conversion Notes

For a Hugging Face VGGT Space or notebook export, convert its output into the
repo-owned bundle instead of coupling the pipeline to the Space response shape.

Minimum conversion contract:

1. Save frame-aligned camera centers as `cameras.npz` key `camera_centers` with
   shape `(frame_count, 3)`.
2. Save orientations as `quaternions_xyzw` with shape `(frame_count, 4)`. If the
   source does not expose usable orientations, write identity quaternions and add
   a metadata warning.
3. Save `valid_pose_mask` and `pose_confidence` arrays with shape
   `(frame_count,)`.
4. Save point cloud coordinates as `points.npz` key `points` with shape
   `(point_count, 3)` when available.
5. Fill `metadata.json` with source tool, public repo/Space URL when known,
   generation date, frame indices, frame timestamps, and limitations.

Optional `points.npz` arrays:

- `point_colors_rgb` with shape `(point_count, 3)`, RGB order, either `0..1`
  float or `0..255` integer values.
- `point_confidence` with shape `(point_count,)` when the source exposes a
  per-point or depth-map confidence aligned to the exported points.
- `point_depth` with shape `(point_count,)` when the source exposes relative
  depth aligned to the exported points.

If optional arrays are missing, review HTML still renders the point cloud. It
labels missing RGB/confidence/depth explicitly and uses relative distance from
the first camera center as a depth-filter fallback. That fallback is a display
diagnostic only; it is not physical range.

`frame_indices` must be non-empty, non-negative, and strictly increasing.
`frame_timestamps_sec` must align one-to-one with `frame_indices`, be
non-negative, and stay in frame order. Do not collapse, sort independently, or
deduplicate frames after VGGT; regenerate the bundle from the original frame
manifest instead.

All floating point arrays in `cameras.npz` and `points.npz` must contain only
finite values. The validator rejects `NaN` or `Inf` in camera centers,
quaternions, pose confidence, point coordinates, and optional point attributes
before the bundle can be used for reliability scoring, HTML review,
descriptors, video export, or smoothing.

Run:

```powershell
fpv vggt validate --bundle <bundle_dir> --report <bundle_dir>/validation.json
fpv vggt inspect --bundle <bundle_dir>
```

Validation warnings are acceptable for known limitations. Validation errors must
be fixed before the bundle can drive review HTML, descriptors, or smoothing.
The inspect command prints the array shapes, optional point RGB/confidence/depth
presence, and the exact review-page fallback behavior in plain English.

## Local Handoff Commands

Create an upload package from sampled frames:

```powershell
fpv vggt export-request `
  --frame-manifest data/frames/<video_id>/segment-001/frames.json `
  --output outputs/vggt_requests/<video_id>__segment-001
```

For a multi-clip GPU run on RunPod, Colab, or another manual cloud environment,
run the local readiness audit before creating or uploading the portable job
package:

```powershell
fpv review audit-readiness `
  --media-inventory data/media/media_inventory.parquet `
  --annotations data/annotations/segments.jsonl `
  --frames-root data/frames `
  --video-id <clip-a> `
  --video-id <clip-b> `
  --video-id <clip-c> `
  --segment-id segment-001 `
  --min-expected-frames 64 `
  --report outputs/data_readiness/readiness_report.json
```

The readiness audit checks local media availability, decode probes, accepted
segment bounds, frame manifest completeness, missing sampled frames, and
simple visual-quality warnings. It is a workflow gate only; it does not infer
location, scale, target identity, or operational suitability. If it returns
`needs_review`, tighten segment bounds or resample before spending GPU time.

After readiness passes, prefer the portable job package:

```powershell
fpv vggt cloud-job `
  --frame-manifest data/frames/<clip-a>/segment-001/frames.json `
  --frame-manifest data/frames/<clip-b>/segment-001/frames.json `
  --frame-manifest data/frames/<clip-c>/segment-001/frames.json `
  --output outputs/cloud_vggt_job
```

The package contains copied sampled frames, relative frame manifests, a
standalone `run_vggt_job.py`, `job_manifest.json`, and
`cloud_vggt_job.zip`. It intentionally avoids absolute local media paths and
does not include secrets. Run the script only in a GPU environment you choose.
It writes `cloud_run.log`, `cloud_summary.json`, normalized `bundles/`, and
`bundles.zip`. Return `bundles.zip`, `cloud_summary.json`, and `cloud_run.log`
to this workstation.
The packaged runner validates normalized bundles before marking a clip done,
including frame-identity metadata, shape checks, and finite-value checks for
cameras, confidence, point coordinates, and optional point attributes.

For a first Google Colab T4 run, use the notebook and checklist in
`docs/colab_vggt_t4_runner.ipynb` and `docs/colab_vggt_t4.md`. Start with a
small package, such as one clip with 4-8 sampled frames and
`--resized-long-edge 512`, then scale up after `bundles.zip` imports locally.
For RunPod or full workstation migration, use `docs/runpod_vggt.md`; it keeps
the GPU step limited to `cloud_vggt_job.zip`, dry-run import validation, and
local-only artifact checks.

Before installing returned cloud bundles, run a dry-run import:

```powershell
fpv vggt import-cloud-job `
  --source <returned-bundles.zip> `
  --output-root data/vggt `
  --report outputs/reviews/cloud_bundle_import_dry_run.json `
  --expected-from-run outputs/reviews/three_clip_run/summary.json `
  --dry-run
```

The dry run validates the returned bundle tree, reports `would_import_count`,
and does not create or overwrite anything under `data/vggt`.
`--expected-from-run` restricts the return package to the video/segment ids from
that run summary, marks missing expected bundles as failures, and skips unrelated
valid bundles instead of installing them. When an adjacent `cloud_summary.json`
contains keyed clip entries, the report also includes `cloud_summary_audit`; a
failed expected cloud clip downgrades the import report to `failed_soft` so the
return package can be reviewed before installation. Direct non-dry-run import
also refuses to copy bundles while `cloud_summary_audit` needs review. A
destination that already exists without `--overwrite` is still reported as a
feasibility error.

Then import and validate returned cloud bundles with:

```powershell
fpv vggt import-cloud-job `
  --source <returned-bundles.zip> `
  --output-root data/vggt `
  --report outputs/reviews/cloud_bundle_import.json
```

The importer accepts either `bundles.zip` or an extracted `bundles/` directory,
validates each bundle before copying, writes valid bundles to
`data/vggt/<video_id>/<segment_id>`, and records adjacent `cloud_summary.json`
and `cloud_run.log` paths when present. Use `--overwrite` only when
intentionally replacing an existing local bundle.

After running VGGT externally and converting its camera/point outputs to
`cameras.npz` and optional `points.npz`, normalize it into this repo:

```powershell
fpv vggt import-bundle `
  --source <converted_bundle_dir> `
  --frame-manifest data/frames/<video_id>/segment-001/frames.json `
  --output data/vggt/<video_id>/segment-001 `
  --source-tool huggingface-space `
  --source-url-or-repo https://huggingface.co/spaces/facebook/vggt `
  --source-commit-or-version manual-export `
  --export-notes "Converted from external VGGT outputs."
```

Then validate:

```powershell
fpv vggt validate --bundle data/vggt/<video_id>/segment-001 --report data/vggt/<video_id>/segment-001/validation.json
```

If the external VGGT tool gives you the official `predictions.npz` file, import
it directly:

```powershell
fpv vggt import-predictions `
  --predictions <predictions.npz> `
  --frame-manifest data/frames/<video_id>/segment-001/frames.json `
  --output data/vggt/<video_id>/segment-001 `
  --source-tool huggingface-space `
  --source-url-or-repo https://huggingface.co/spaces/facebook/vggt `
  --source-commit-or-version manual-export `
  --export-notes "Imported from official VGGT predictions.npz."
```

The converter expects `extrinsic` and will use `world_points_from_depth`,
`point_map`, `points`, or `world_points` when present. When aligned arrays are
available, it also preserves `point_colors_rgb`, `point_confidence`, and
`point_depth`. Camera centers are derived from OpenCV-style camera-from-world
extrinsics and remain relative and scale-ambiguous.

If you are on RunPod, Colab, or another GPU environment with the official VGGT
package installed, run VGGT and normalize the bundle in one step:

```powershell
fpv vggt run-installed `
  --frame-manifest data/frames/<video_id>/segment-001/frames.json `
  --predictions-output outputs/vggt_predictions/<video_id>__segment-001/predictions.npz `
  --bundle-output data/vggt/<video_id>/segment-001 `
  --source-url-or-repo https://github.com/facebookresearch/vggt
```

This command is optional. It imports `vggt` lazily and refuses CPU inference by
default because the model is large.

## Coordinate Limits

VGGT camera centers are relative and scale ambiguous by default. Global rotation
is also arbitrary. Do not label coordinates as meters or use them for
geolocation.

## Failure Modes

Document and surface:

- low texture,
- blur,
- title-card or replay contamination,
- pose outliers,
- incomplete poses,
- sparse or missing points,
- source/export version uncertainty.
