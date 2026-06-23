# fpv-vggt-visual-geometry-lab

Offline visual-geometry review for already-published FPV footage.

This project ingests the `itamarwe/fpv-drone-strikes-lebanon-dataset` catalog,
downloads only explicitly selected local media, proposes and accepts flight
segments, samples frames, imports VGGT prediction bundles, renders local review
HTML, and computes reconstruction diagnostics plus conservative scale-free path
descriptors.

The project does not perform live detection, geolocation, target detection,
approach-corridor mapping, route optimization, guidance, real-world speed or
distance estimation, next-maneuver prediction, or attack simulation.

## Dataset

The source dataset is a catalog repository:

- GitHub: https://github.com/itamarwe/fpv-drone-strikes-lebanon-dataset
- Observed on 2026-06-22 as a README catalog with 157 MP4 rows.
- Metadata, manifests, and documentation are CC0.
- Referenced videos and thumbnails are third-party media and are not licensed by
  that repository.

Real media and media-derived artifacts are local-only by default.

## Credit

Dataset catalog credit goes to
[itamarwe/fpv-drone-strikes-lebanon-dataset](https://github.com/itamarwe/fpv-drone-strikes-lebanon-dataset).
This project is an independent offline visual-geometry review tool built around
that catalog. It does not redistribute the referenced third-party media.

## V1 scope

The smallest useful v1 proves the workflow on three accepted real clips:

1. ingest a pinned catalog snapshot,
2. fetch explicitly selected videos locally,
3. propose segment boundaries with cheap non-operational diagnostics,
4. require human segment acceptance,
5. sample frames deterministically,
6. import a repo-owned VGGT prediction bundle,
7. score reconstruction reliability,
8. extract conservative scale-free descriptors,
9. render polished local HTML review artifacts,
10. render a small three-clip comparison summary.

Milestone 1 is smaller: it builds the package skeleton and synthetic/mocked core
only. It does not download real media, run VGGT, ingest the real catalog, or
render the final dashboard/review HTML.

## Local setup

Use Python 3.11 or newer.

```powershell
python -m pip install -e ".[dev]"
pytest
ruff check src tests
```

## Commands

```powershell
fpv --help
fpv catalog sync --output data/catalog
fpv catalog shortlist --catalog data/catalog/catalog.parquet --limit 5
fpv media fetch --catalog data/catalog/catalog.parquet --video-id <video_id>
fpv media audit --inventory data/media/media_inventory.parquet --video-id <video_id> --report outputs/media_audit/<video_id>.json
fpv segment propose --media-inventory data/media/media_inventory.parquet --video-id <video_id>
fpv segment contact-sheet --media-inventory data/media/media_inventory.parquet --annotations data/annotations/segments.jsonl --video-id <video_id> --segment-id segment-001 --output outputs/segment_review/<video_id>__segment-001.jpg
fpv segment accept --video-id <video_id> --segment-id segment-001 --start 0 --end 2.0
fpv segment edit --video-id <video_id> --segment-id segment-001 --start 0.25 --end 1.75 --notes "Tightened after contact sheet review." --confidence high
fpv frames sample-accepted --media-inventory data/media/media_inventory.parquet --annotations data/annotations/segments.jsonl --video-id <video_id> --segment-id segment-001 --output data/frames/<video_id>/segment-001 --count 64
fpv review audit-readiness --media-inventory data/media/media_inventory.parquet --annotations data/annotations/segments.jsonl --frames-root data/frames --video-id <video-a> --video-id <video-b> --video-id <video-c> --segment-id segment-001 --min-expected-frames 64 --report outputs/data_readiness/readiness_report.json
fpv vggt export-request --frame-manifest data/frames/<video_id>/segment-001/frames.json --output outputs/vggt_requests/<video_id>__segment-001
fpv vggt cloud-job --frame-manifest data/frames/<video-a>/segment-001/frames.json --frame-manifest data/frames/<video-b>/segment-001/frames.json --frame-manifest data/frames/<video-c>/segment-001/frames.json --output outputs/cloud_vggt_job
fpv vggt import-cloud-job --source <returned-bundles.zip> --output-root data/vggt --report outputs/reviews/cloud_bundle_import_dry_run.json --expected-from-run outputs/reviews/three_clip_run/summary.json --dry-run
fpv vggt import-cloud-job --source <returned-bundles.zip> --output-root data/vggt --report outputs/reviews/cloud_bundle_import.json
fpv vggt import-bundle --source <converted_bundle_dir> --frame-manifest data/frames/<video_id>/segment-001/frames.json --output data/vggt/<video_id>/segment-001 --source-tool huggingface-space --source-url-or-repo https://huggingface.co/spaces/facebook/vggt
fpv vggt import-predictions --predictions <predictions.npz> --frame-manifest data/frames/<video_id>/segment-001/frames.json --output data/vggt/<video_id>/segment-001 --source-url-or-repo https://huggingface.co/spaces/facebook/vggt
fpv vggt run-installed --frame-manifest data/frames/<video_id>/segment-001/frames.json --predictions-output outputs/vggt_predictions/<video_id>__segment-001/predictions.npz --bundle-output data/vggt/<video_id>/segment-001
fpv synthetic-video create --output .tmp/synthetic.mp4 --frames 24
fpv frames sample --video .tmp/synthetic.mp4 --output .tmp/frames --count 6
fpv heatmap generate --frame-manifest .tmp/frames/frames.json --output .tmp/heatmaps
fpv vggt mock --frame-manifest .tmp/frames/frames.json --output .tmp/vggt_bundle
fpv vggt validate --bundle .tmp/vggt_bundle --report .tmp/validation.json
fpv vggt inspect --bundle .tmp/vggt_bundle
fpv reconstruct summarize --bundle .tmp/vggt_bundle --output .tmp/summary.json
fpv smooth poses --bundle .tmp/vggt_bundle --summary .tmp/summary.json --output .tmp/smoothed.npz --metadata-output .tmp/smoothed.json
fpv viz render --frame-manifest .tmp/frames/frames.json --bundle .tmp/vggt_bundle --summary .tmp/summary.json --output .tmp/review.html --smoothed-poses .tmp/smoothed.npz
fpv viz render --frame-manifest .tmp/frames/frames.json --bundle .tmp/vggt_bundle --summary .tmp/summary.json --heatmap-manifest .tmp/heatmaps/heatmaps.json --output .tmp/review-with-heatmaps.html
fpv viz export-video --frame-manifest .tmp/frames/frames.json --bundle .tmp/vggt_bundle --summary .tmp/summary.json --output .tmp/side_by_side.mp4 --fps 8
fpv compare render --summary .tmp/summary-a.json --summary .tmp/summary-b.json --summary .tmp/summary-c.json --output .tmp/compare.html
fpv review audit-local-only --root . --report outputs/reviews/local_only_audit.json
fpv review audit-three --frame-manifest <frames-a.json> --bundle <bundle-a> --frame-manifest <frames-b.json> --bundle <bundle-b> --frame-manifest <frames-c.json> --bundle <bundle-c> --output outputs/reviews/real_three_clip_audit.json
fpv review run-three --frame-manifest <frames-a.json> --bundle <bundle-a> --frame-manifest <frames-b.json> --bundle <bundle-b> --frame-manifest <frames-c.json> --bundle <bundle-c> --output-dir outputs/reviews/three_clip_run --smooth --export-video --heatmaps
fpv run --workdir outputs/reviews/three_clip_run --frame-manifest <frames-a.json> --bundle <bundle-a> --frame-manifest <frames-b.json> --bundle <bundle-b> --frame-manifest <frames-c.json> --bundle <bundle-c> --smooth --export-video --heatmaps
fpv run --workdir outputs/reviews/three_clip_run --catalog data/catalog/catalog.parquet --media-inventory data/media/media_inventory.parquet --annotations data/annotations/segments.jsonl --video-id <video-a> --video-id <video-b> --video-id <video-c> --fetch-media --frames 64 --smooth
fpv run --workdir outputs/reviews/three_clip_run --catalog data/catalog/catalog.parquet --media-inventory data/media/media_inventory.parquet --annotations data/annotations/segments.jsonl --video-id <video-a> --video-id <video-b> --video-id <video-c> --cloud-return <returned-bundles.zip> --cloud-import-report outputs/reviews/cloud_bundle_import.json --frames 64 --smooth --export-video --heatmaps
fpv run --synthetic --workdir .tmp/run-demo
pytest
```

`fpv run` writes a debuggable run folder with logs, structured summary, next
steps, artifact paths, warning text, and environment details. For real review
inputs it audits first. With `--video-id`, it can fetch explicitly selected
local media, enforce accepted segment annotations, sample frames, create the
cloud VGGT job package when bundles are missing, import a returned
`bundles.zip` with `--cloud-return`, and render review artifacts once bundles
validate. If bundles are missing, it writes `needs_vggt_bundle` status plus the
next cloud-job/return command instead of failing as a dead end.
The default sampled-frame count for `fpv run` is 64 for smoother review
animation; pass a smaller `--frames` value for quick GPU tests. `--export-video`
adds local side-by-side MP4 renders next to the HTML review pages.
Use `fpv vggt inspect --bundle <bundle_dir>` when you want a plain-English
check of camera arrays, point-cloud arrays, RGB/confidence/depth availability,
and the review-page filter behavior. Review pages include a bundle-quality
panel, a side-by-side VGGT-frame 6DoF drone animation, and, when smoothing
succeeds, a raw-vs-smoothed diagnostic path toggle.
Review pages can also overlay optional image-space heatmap layers such as frame
difference, optical flow magnitude, blur/focus signal, and visibility change
when a heatmap manifest is provided or `--heatmaps` is requested during review
runs. These overlays are local-only diagnostic aids. They are not geolocation,
map or ground projection, meters, target inference, approach corridor analysis,
or tactical guidance.
The run `summary.json` includes a compact `data_checks` block with
`media_audit` and `frame_readiness` status/counts, plus links to the detailed
JSON reports. Treat these as workflow-readiness gates, not truth, location,
scale, target, or operational claims.

After fetching local media, run `fpv media audit` when you want to check whether
the cached video is workflow-ready before segment review. It verifies the local
file exists, the inventory checksum still matches, OpenCV can decode probe
frames, and recorded metadata is sane. This is a cache/data-quality gate only,
not an authenticity, geolocation, target, scale, or operational claim.

Before uploading sampled frames to RunPod, Colab, or another GPU environment,
run `fpv review audit-readiness`. It checks local media availability, decode
probes, accepted segment bounds, sampled frame manifests, missing frame files,
and conservative visual-quality warnings such as very dark, low-contrast, or
color-dominant edit-artifact frames. A `needs_review` result means tighten the
segment or resample before spending cloud GPU time.

## Safety defaults

- Treat dataset descriptions as source metadata only.
- Do not label VGGT coordinates as meters.
- Do not commit downloaded videos, thumbnails, extracted real frames, point
  clouds from real videos, heatmap PNGs or heatmap manifests from real footage,
  or rendered real-media review pages.
- Automated tests use synthetic videos and mocked VGGT only.
- VGGT is import-first; Hugging Face, Colab, RunPod, or local VGGT are optional
  bundle sources.
- Segment proposals are not analysis-ready until explicitly accepted.
- The three-clip comparison command expects local summaries. Real three-clip
  smoke runs also require user-provided imported VGGT bundles.
- `fpv vggt export-request` creates a local-only frame package for manual VGGT
  execution. `fpv vggt import-bundle` normalizes already-converted VGGT arrays
  into the repo-owned bundle format with provenance.
- `fpv vggt cloud-job` creates a portable local-only GPU job package with
  copied frames, relative manifests, a standalone `run_vggt_job.py`, and clear
  warnings. It is intended for manual RunPod/Colab use; do not upload it unless
  you explicitly choose to run VGGT externally. The cloud runner writes
  `cloud_run.log`, `cloud_summary.json`, and `bundles.zip` for return/import.
- `fpv vggt import-cloud-job` validates a returned cloud `bundles.zip` or
  `bundles/` tree and installs valid bundles under
  `data/vggt/<video_id>/<segment_id>` with a JSON report. Use `--dry-run`
  and `--expected-from-run <run-summary.json>` first to validate the return
  package, prove the expected run bundles are present, skip unrelated bundles,
  and see what would be installed without changing `data/vggt`.
- `fpv vggt import-predictions` converts official VGGT-style `predictions.npz`
  exports into the repo-owned bundle format.
- `fpv vggt run-installed` is optional and intended for cloud/GPU environments
  where `facebookresearch/vggt` is installed. It refuses CPU inference by
  default.
- `fpv review audit-local-only` scans for common media-derived artifacts outside
  approved local-only roots before sharing or packaging the repository.

See `docs/safety_scope.md` and `docs/methodology.md` for details.
For a RunPod-style GPU handoff and workstation migration checklist, see
`docs/runpod_vggt.md`.
