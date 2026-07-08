# Colab T4 VGGT Workflow

This workflow runs VGGT inference on Google Colab using this repo's portable
cloud job package. Keep the main project workflow local; use Colab only for the
CUDA inference step.

Warnings for every run:

- no geolocation
- no meters
- relative VGGT frame
- local-only media-derived frames

## 1. Create A T4 Stable-Window Package Locally

For the useful T4 test, do not send a full edited segment. First create stable
windows locally, then sample about 2 FPS inside each selected window. This keeps
Colab focused on a few cleaner image sets instead of wasting memory on blur,
cuts, overlays, and terminal chaos.

Recommended high-quality T4 preset for one or a few clips:

```powershell
fpv windows colab-t4 `
  --media-inventory data/media/media_inventory.parquet `
  --annotations data/annotations/segments.jsonl `
  --frames-root data/frames `
  --output-dir outputs/colab/t4_window_run `
  --video-id <video_id> `
  --window-sec 8 `
  --stride-sec 3 `
  --target-fps 2 `
  --max-frames 20 `
  --candidate-limit 2 `
  --resized-long-edge 1024
```

Upload this file to Colab:

```text
outputs/colab/t4_window_run/cloud_vggt_job/cloud_vggt_job.zip
```

This preset usually creates up to two windows per input clip, about 16 frames
per 8 second window. `--max-frames 20` is the T4 memory cap; raise it only after
one complete Colab return imports cleanly. If T4 runs out of memory, lower
`--max-frames` to 12 or 16, or use `--resized-long-edge 768`.

For a tiny smoke test only, the old direct frame-package path still works:

```powershell
fpv frames sample-accepted `
  --media-inventory data/media/media_inventory.parquet `
  --annotations data/annotations/segments.jsonl `
  --video-id <video_id> `
  --segment-id segment-001 `
  --output data/frames/<video_id>/segment-001 `
  --count 8 `
  --resized-long-edge 512

fpv vggt cloud-job `
  --frame-manifest data/frames/<video_id>/segment-001/frames.json `
  --output outputs/cloud_vggt_job
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
4. Upload `cloud_vggt_job.zip` when prompted, or place it in Google Drive for the Drive notebook.
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
- lower `--max-frames` to 12 or 16 for `fpv windows colab-t4`,
- lower `--resized-long-edge` from 1024 to 768,
- reduce `--candidate-limit` to 1,
- restart the Colab runtime before rerunning,
- only increase frames after a valid `bundles.zip` imports locally.

If Colab disconnects after VGGT finishes, check the notebook file browser first.
The return files are written next to `run_vggt_job.py` inside the extracted job
directory.

If dependency installation fails, restart the runtime and rerun from the first
cell. The packaged runner is resumable and skips clips with already valid
normalized bundles.
