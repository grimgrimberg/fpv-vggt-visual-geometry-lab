# Hugging Face FPV Hezbo Handoff

This project has a private Hugging Face dataset used as a control plane for
RunPod and Hugging Face handoffs:

- Canonical dataset: https://huggingface.co/datasets/Grimster/FPV_Hezbo
- Verified on: 2026-07-09
- Authenticated account used for verification: `Grimster`
- Hub status: private
- Last modified on the Hub: 2026-07-08

The dataset is not a public media dataset and must not become one. It should
carry metadata needed to reproduce offline processing runs: pinned catalog
snapshots, manifests, package handoff notes, and run-control files. Referenced
MP4 files remain third-party media and should be downloaded only into ignored
local or pod-local caches.

The intended RunPod pattern is:

1. Read the pinned catalog snapshot from `data/catalog/latest`.
2. Download referenced MP4s into a pod-local cache.
3. Build frame packs from selected accepted/stable windows.
4. Run frozen VGGT or VGGT-Omega inference.
5. Return only structured reconstruction bundles, manifests, logs, and quality
   reports that remain local/ignored unless explicitly reviewed for release.

Do not use the Hugging Face dataset as evidence that training works. It is a
handoff and provenance layer. Model quality must be judged from window-level
labels, frozen-model reconstruction results, and grouped-by-video validation.

There is also a private dataset at:

- https://huggingface.co/datasets/Grimster420/FPV_Hezbo

Use `Grimster/FPV_Hezbo` as the canonical repo for this project unless a future
cleanup intentionally migrates ownership.
