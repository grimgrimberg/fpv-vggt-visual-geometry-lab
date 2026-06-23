# Annotation Guide

## Segment Status

Segments move through these states:

- `proposed`: produced by automated quality/motion heuristics.
- `accepted`: reviewed and approved for downstream analysis.
- `rejected`: unsuitable or out of scope.

Only accepted segments may be used for VGGT import, reliability summaries,
descriptors, comparison pages, or stretch smoothing.

## Flight Segment Boundary

The flight segment is the usable continuous FPV camera-motion portion selected
for relative visual-geometry review.

Exclude or mark:

- title cards,
- intro/outro slates,
- freeze frames,
- replay sections,
- heavily edited cuts,
- very low-quality stretches,
- unrelated overlays when they dominate the frame.

If the visible terminal event is included, mark that explicitly. Do not infer
anything beyond visible footage.

## Required Fields

- `video_id`
- `segment_id`
- `start_sec`
- `end_sec`
- `status`
- `include_terminal_event`
- `excluded_ranges_sec`
- `annotation_confidence`
- `annotation_notes`

## Review Principle

Acceptance means "usable for relative reconstruction review", not "the dataset
description is correct" and not "the reconstruction is physically accurate".

## Contact Sheets

Use contact sheets to review proposed boundaries without creating analysis-ready
artifacts:

```powershell
fpv segment contact-sheet `
  --media-inventory data/media/media_inventory.parquet `
  --annotations data/annotations/segments.jsonl `
  --video-id <video_id> `
  --segment-id segment-001 `
  --output outputs/segment_review/<video_id>__segment-001.jpg
```

Contact sheets from real footage are media-derived local-only artifacts. Keep
them ignored and do not publish them.

## Editing Accepted Bounds

After contact-sheet review, tighten an existing annotation with:

```powershell
fpv segment edit `
  --annotations data/annotations/segments.jsonl `
  --video-id <video_id> `
  --segment-id segment-001 `
  --start <sec> `
  --end <sec> `
  --notes "Tightened after contact sheet review." `
  --confidence high
```

Editing preserves the existing status and diagnostics. Use `fpv segment accept`
to mark a segment accepted, and `fpv segment reject` to reject it.
