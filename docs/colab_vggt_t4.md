# Colab T4 VGGT Workflow

This workflow runs VGGT inference on Google Colab using this repo's portable
cloud job package. Keep the main project workflow local; use Colab only for the
CUDA inference step.

Warnings for every run:

- no geolocation
- no meters
- relative VGGT frame
- local-only media-derived frames

## 1. Start Small Locally

For a first T4 test, sample a small accepted segment. Eight frames is a good
smoke test; increase only after the full handoff works.

```bash
fpv frames sample-accepted \
  --media-inventory data/media/media_inventory.parquet \
  --annotations data/annotations/segments.jsonl \
  --video-id <video_id> \
  --segment-id segment-001 \
  --output data/frames/<video_id>/segment-001 \
  --count 8 \
  --resized-long-edge 512
```

Create the upload package:

```bash
fpv vggt cloud-job \
  --frame-manifest data/frames/<video_id>/segment-001/frames.json \
  --output outputs/cloud_vggt_job
```

Upload this file to Colab:

```text
outputs/cloud_vggt_job/cloud_vggt_job.zip
```

## 2. Run In Colab

Open `docs/colab_vggt_t4_runner.ipynb` in Colab.

If you are using the VS Code Colab extension or already placed the ZIP in
Google Drive, prefer the cleaner Drive-based notebook:
`docs/colab_drive_vggt_runner.ipynb`. It defaults to `cloud_vggt_job.zip` and
copies returned files back to `My Drive/fpv_vggt_returns/cloud_vggt_job/`.

In Colab:

1. Select `Runtime > Change runtime type`.
2. Choose `T4 GPU`.
3. Run the notebook cells in order.
4. Upload `cloud_vggt_job.zip` when prompted.
5. Wait for VGGT and the model download to finish.
6. Download:
   - `bundles.zip`
   - `cloud_summary.json`
   - `cloud_run.log`

The notebook runs the packaged `run_vggt_job.py` directly. That script installs
`facebookresearch/vggt` if missing, runs `facebook/VGGT-1B`, writes normalized
repo-owned bundles, and validates them before creating `bundles.zip`.

### VS Code Colab Extension Upload Fallback

The VS Code Colab extension may not expose Colab's browser file panel, and
`google.colab.files.upload()` widgets can be unavailable or stale. In that case,
upload `cloud_vggt_job.zip` to Google Drive first, then use this cell instead of
the notebook upload-widget cell:

```python
import json
import shutil
import zipfile
from pathlib import Path

from google.colab import drive

drive.mount("/content/drive")

WORK = Path("/content/fpv_vggt_job")
DRIVE_ZIP = Path("/content/drive/MyDrive/cloud_vggt_job.zip")

if not DRIVE_ZIP.exists():
    matches = sorted(Path("/content/drive/MyDrive").rglob("cloud_vggt_job.zip"))
    if not matches:
        raise FileNotFoundError(
            "Put cloud_vggt_job.zip in Google Drive, then rerun this cell."
        )
    DRIVE_ZIP = matches[0]

if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(DRIVE_ZIP) as archive:
    archive.extractall(WORK)

candidates = sorted(WORK.rglob("run_vggt_job.py"))
if not candidates:
    raise FileNotFoundError("run_vggt_job.py was not found inside the ZIP.")

JOB_DIR = candidates[0].parent
manifest = json.loads((JOB_DIR / "job_manifest.json").read_text())

print("Job dir:", JOB_DIR)
print("Clip count:", len(manifest["clips"]))
for clip in manifest["clips"]:
    print("-", clip["clip_id"], "frames=", clip["frame_count"])
```

Then continue with the VGGT run cell:

```python
run([sys.executable, "run_vggt_job.py"], cwd=JOB_DIR)
```

## 3. Import Back Locally

Place the returned `bundles.zip` somewhere local. You can import it directly:

```bash
fpv vggt import-cloud-job \
  --source <returned-bundles.zip> \
  --output-root data/vggt \
  --report outputs/reviews/cloud_bundle_import.json
```

If the direct import succeeds, continue the regular review flow without passing
`--cloud-return` again:

```bash
fpv run \
  --workdir outputs/reviews/three_clip_run \
  --video-id <video_id>
```

Alternatively, let `fpv run` import the returned cloud bundle in one step:

```bash
fpv run \
  --workdir outputs/reviews/three_clip_run \
  --video-id <video_id> \
  --cloud-return <returned-bundles.zip> \
  --cloud-import-report outputs/reviews/cloud_bundle_import.json
```

## T4 Troubleshooting

If Colab reports CUDA out of memory:

- retry one clip at a time,
- reduce `--count` to 4 or 6,
- keep `--resized-long-edge 512`,
- restart the Colab runtime before rerunning,
- only increase frames after a valid `bundles.zip` imports locally.

If Colab disconnects after VGGT finishes, check the notebook file browser first.
The return files are written next to `run_vggt_job.py` inside the extracted job
directory.

If dependency installation fails, restart the runtime and rerun from the first
cell. The packaged runner is resumable and skips clips with already valid
normalized bundles.
