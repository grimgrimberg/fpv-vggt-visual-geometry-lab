# COLMAP / SfM diagnostic

This report is retrospective and does not execute a retry.

## Outcome

- Status: `completed_with_warnings`
- Successful reconstruction retained: `true`
- Maximum registered-frame count observed in logs: `102`
- Bundle-adjustment log events: `527`
- Bounded fallback eligible: `false`

## Classification

- `insufficient_matches`

## Bounded fallback recommendations

- `sequential_matching`
- `lightglue_pair_graph`
- `reduce_window`

## Evidence

### insufficient_matches

- line 8541: I20260711 02:33:23.361277 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
- line 10141: I20260711 02:33:42.092127 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
- line 22752: I20260711 02:34:55.147800 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
- line 35332: I20260711 02:37:38.427875 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
- line 37296: I20260711 02:38:26.630460 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
