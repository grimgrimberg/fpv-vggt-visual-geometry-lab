# COLMAP / SfM diagnostic

This report is retrospective and does not execute a retry.

## Outcome

- Status: `completed_with_warnings`
- Successful reconstruction retained: `true`
- Maximum registered-frame count observed in logs: `190`
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

- line 7395: W20260711 03:43:39.826807 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 7396: W20260711 03:43:39.827148 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 7397: W20260711 03:43:39.828654 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 7436: W20260711 03:44:19.551012 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 7437: W20260711 03:44:19.663196 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 7438: W20260711 03:44:19.914452 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 7439: W20260711 03:44:20.165392 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 7440: W20260711 03:44:20.415770 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 7441: W20260711 03:44:20.529930 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 7442: W20260711 03:44:20.787718 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 7443: W20260711 03:44:21.044990 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 7444: W20260711 03:44:21.450145 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 7445: W20260711 03:44:21.560923 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 7446: W20260711 03:44:21.828428 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 7447: W20260711 03:44:21.938689 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 7448: W20260711 03:44:22.349667 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 7449: W20260711 03:44:22.617887 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 7450: W20260711 03:44:22.736126 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 7451: W20260711 03:44:23.018402 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 7452: W20260711 03:44:23.293309 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.

### insufficient_matches

- line 7602: I20260711 03:46:22.557362 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
- line 7606: I20260711 03:46:22.561581 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
- line 7610: I20260711 03:46:22.565324 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
- line 7614: I20260711 03:46:22.569074 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
- line 7618: I20260711 03:46:22.572786 127739432001536 incremental_pipeline.cc:394] => No good initial image pair found.
