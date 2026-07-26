# COLMAP / SfM diagnostic

This report is retrospective and does not execute a retry.

## Outcome

- Status: `completed_with_warnings`
- Successful reconstruction retained: `true`
- Maximum registered-frame count observed in logs: `194`
- Bundle-adjustment log events: `49`
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

- line 2039: W20260711 03:48:17.377618 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 2040: W20260711 03:48:17.377860 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 2063: W20260711 03:48:21.252488 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 2064: W20260711 03:48:21.268373 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 2065: W20260711 03:48:21.345085 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 2066: W20260711 03:48:21.391767 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 2353: W20260711 03:48:54.202391 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 2354: W20260711 03:48:54.300213 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 2355: W20260711 03:48:55.268724 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 2356: W20260711 03:48:55.340348 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 2357: W20260711 03:48:55.411493 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 2358: W20260711 03:48:56.143727 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 2359: W20260711 03:48:56.218075 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 2360: W20260711 03:48:56.251603 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 2361: W20260711 03:48:56.326088 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 2362: W20260711 03:48:56.400455 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 2363: W20260711 03:48:56.469705 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 2364: W20260711 03:48:56.498976 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 2365: W20260711 03:48:56.611139 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 2366: W20260711 03:48:56.678073 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.

### insufficient_matches

- line 2752: I20260711 03:51:09.927572 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
- line 2756: I20260711 03:51:09.930514 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
- line 2760: I20260711 03:51:09.933276 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
- line 2764: I20260711 03:51:09.936040 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
- line 2768: I20260711 03:51:09.938752 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
