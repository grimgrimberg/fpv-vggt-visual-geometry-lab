# COLMAP / SfM diagnostic

This report is retrospective and does not execute a retry.

## Outcome

- Status: `completed_with_warnings`
- Successful reconstruction retained: `true`
- Maximum registered-frame count observed in logs: `183`
- Bundle-adjustment log events: `77`
- Bounded fallback eligible: `false`

## Classification

- `insufficient_matches`
- `ba_numerical_instability`

## Bounded fallback recommendations

- `sequential_matching`
- `lightglue_pair_graph`
- `reduce_window`
- `alternate_sparse_linear_solver`
- `disable_focal_refinement`

## Evidence

### ba_numerical_instability

- line 5013: W20260711 03:02:22.430435 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 5014: W20260711 03:02:22.435540 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 5015: W20260711 03:02:22.436228 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 5016: W20260711 03:02:22.441263 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 5017: W20260711 03:02:22.441952 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 5019: W20260711 03:02:22.635212 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 5020: W20260711 03:02:22.655136 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 5021: W20260711 03:02:22.661683 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 5022: W20260711 03:02:22.681295 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 5023: W20260711 03:02:22.700894 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 5024: W20260711 03:02:22.720519 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 5025: W20260711 03:02:22.740089 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 5026: W20260711 03:02:22.746647 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 5027: W20260711 03:02:22.779308 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 5028: W20260711 03:02:22.785873 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 5029: W20260711 03:02:22.805772 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 5030: W20260711 03:02:22.812334 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 5031: W20260711 03:02:22.845022 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 5032: W20260711 03:02:22.864638 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 5033: W20260711 03:02:22.884216 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.

### insufficient_matches

- line 5571: I20260711 03:06:22.167526 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
- line 5575: I20260711 03:06:22.176052 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
- line 5579: I20260711 03:06:22.184461 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
- line 5583: I20260711 03:06:22.195123 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
- line 5587: I20260711 03:06:22.203916 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
