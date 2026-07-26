# COLMAP / SfM diagnostic

This report is retrospective and does not execute a retry.

## Outcome

- Status: `completed_with_warnings`
- Successful reconstruction retained: `true`
- Maximum registered-frame count observed in logs: `19`
- Bundle-adjustment log events: `133`
- Bounded fallback eligible: `false`

## Classification

- `insufficient_matches`

## Bounded fallback recommendations

- `sequential_matching`
- `lightglue_pair_graph`
- `reduce_window`

## Evidence

### insufficient_matches

- line 5510: I20260711 02:42:47.658698 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
- line 9847: I20260711 02:42:47.963223 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
- line 14965: I20260711 02:42:48.699395 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
- line 19817: I20260711 02:42:53.687651 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
- line 21624: I20260711 02:42:53.897897 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
