# RunPod VGGT Workflow

This workflow runs only the VGGT inference step on RunPod or another CUDA GPU
machine. Keep catalog ingestion, media fetch, segment review, frame sampling,
bundle import, review HTML, and comparison artifacts local unless you
explicitly choose otherwise.

Safety labels for every artifact in this workflow:

- no geolocation
- no meters
- relative VGGT frame
- local-only media-derived frames

## 1. Prove Local Readiness

Before uploading anything, verify the local cache and sampled frames:

```powershell
fpv run `
  --workdir outputs/reviews/three_clip_run `
  --catalog data/catalog/catalog.parquet `
  --media-dir data/media `
  --media-inventory data/media/media_inventory.parquet `
  --annotations data/annotations/segments.jsonl `
  --frames-root data/frames `
  --vggt-root data/vggt `
  --cloud-job-output outputs/cloud_vggt_job `
  --frames 64 `
  --resized-long-edge 768 `
  --smooth `
  --video-id <clip-a-video-id> `
  --video-id <clip-b-video-id> `
  --video-id <clip-c-video-id>
```

For a cloud handoff, `needs_vggt_bundle` is a valid status when:

- `media_audit_report.json` is `ready`,
- `readiness_report.json` is `ready_for_cloud`,
- `summary.json` lists `cloud_job_package`,
- `summary.json` lists `cloud_job_package_bytes`,
- `summary.json` lists `cloud_job_package_sha256`.

This proves the selected local files decode, checksums match the local media
inventory, accepted segments exist, sampled frames exist, and basic frame
quality checks passed. It does not prove authenticity, location, scale, target
identity, or operational suitability.

## 2. Move Only The Cloud Package

Move this ZIP to the GPU machine:

```text
outputs/cloud_vggt_job/cloud_vggt_job.zip
```

Do not move the whole repository unless you intentionally need to. The ZIP
contains only sampled frames, relative manifests, the packaged runner,
`job_manifest.json`, and a README. It should not contain source videos, absolute
local media paths, or credentials.

On the cloud machine, verify the uploaded file when possible:

```bash
sha256sum cloud_vggt_job.zip
```

Compare the hash to `cloud_job_package_sha256` in the local
`outputs/reviews/three_clip_run/summary.json`.

## 3. Run On RunPod

Choose a CUDA PyTorch image or another image where Python, pip, and CUDA are
available. A 64-frame package gives a smoother path animation and MP4 export.
A T4-class GPU may require fewer frames such as 24 or 32 for a first debug run.

Inside the pod:

```bash
mkdir -p /workspace/fpv-vggt-job
cd /workspace/fpv-vggt-job
unzip /workspace/cloud_vggt_job.zip
python run_vggt_job.py
```

The runner installs `facebookresearch/vggt` if it is missing, requires CUDA,
runs `facebook/VGGT-1B`, writes normalized repo-owned bundles, validates them,
and creates:

- `bundles.zip`
- `cloud_summary.json`
- `cloud_run.log`

The runner is resumable. If a valid normalized bundle already exists for a clip,
a rerun skips that clip.

## 4. Return Only The Result Files

Copy these files back to the local workstation:

```text
bundles.zip
cloud_summary.json
cloud_run.log
```

Keep them under an ignored local path such as `outputs/cloud_returns/`.

Do not publish returned bundles or logs from real media. A bundle may contain
real-media-derived point clouds and camera paths.

## 5. Dry-Run Import Locally

Validate the returned ZIP before installing it:

```powershell
fpv vggt import-cloud-job `
  --source outputs/cloud_returns/bundles.zip `
  --output-root data/vggt `
  --report outputs/reviews/cloud_bundle_import_dry_run.json `
  --expected-from-run outputs/reviews/three_clip_run/summary.json `
  --dry-run
```

Continue only if the dry run reports `done`. The dry run verifies expected
video/segment ids, validates bundle shape/provenance/frame alignment, reports
missing expected bundles, and skips unrelated bundles. If `cloud_summary.json`
contains keyed clip entries and an expected clip is marked failed or invalid,
the report includes `cloud_summary_audit.status = "needs_review"` and returns
`failed_soft` even if a bundle directory is present. Direct import refuses to
install bundles while this audit needs review; inspect the report and rerun or
repair the cloud job first.

## 6. Import And Resume The Local Review

After a successful dry run:

```powershell
fpv run `
  --workdir outputs/reviews/three_clip_run `
  --catalog data/catalog/catalog.parquet `
  --media-dir data/media `
  --media-inventory data/media/media_inventory.parquet `
  --annotations data/annotations/segments.jsonl `
  --frames-root data/frames `
  --vggt-root data/vggt `
  --cloud-job-output outputs/cloud_vggt_job `
  --frames 64 `
  --resized-long-edge 768 `
  --cloud-return outputs/cloud_returns/bundles.zip `
  --cloud-import-report outputs/reviews/cloud_bundle_import.json `
  --smooth `
  --export-video `
  --video-id <clip-a-video-id> `
  --video-id <clip-b-video-id> `
  --video-id <clip-c-video-id>
```

This imports valid bundles, reruns the expected-bundle audit, and renders local
review artifacts only after validation passes.

## 7. Migrating The Project

For a full workstation migration, copy the repository and then decide whether
to copy local-only artifacts separately.

Safe to share or version:

- source code,
- docs,
- tests,
- synthetic fixtures,
- catalog metadata snapshots when license-compatible.

Local-only by default:

- `data/media/`
- `data/frames/`
- `data/vggt/`
- `data/geometry/`
- `outputs/`
- real review HTML,
- real screenshots,
- returned VGGT bundles.

Before sharing a migrated copy, run:

```powershell
fpv review audit-local-only --root . --report outputs/reviews/local_only_audit.json
```

Treat any finding outside approved ignored roots as a packaging blocker.

## Troubleshooting

If RunPod runs out of GPU memory:

- resample fewer frames,
- lower `--resized-long-edge`,
- run one clip per package,
- restart the pod before rerunning,
- import partial successful bundles only after the dry-run report is clear.

If dependency installation fails, keep `cloud_run.log`, restart with a clean
CUDA PyTorch image, and rerun the same extracted package.

If the local import fails, do not edit returned bundles by hand first. Inspect
`outputs/reviews/cloud_bundle_import_dry_run.json`, fix the cloud export or
source package, rerun the cloud job, and dry-run import again.
