# COLMAP / SfM diagnostic

This report is retrospective and does not execute a retry.

## Outcome

- Status: `completed_with_warnings`
- Successful reconstruction retained: `true`
- Maximum registered-frame count observed in logs: `191`
- Bundle-adjustment log events: `65`
- Bounded fallback eligible: `false`

## Classification

- `ba_numerical_instability`

## Bounded fallback recommendations

- `alternate_sparse_linear_solver`
- `disable_focal_refinement`

## Evidence

### ba_numerical_instability

- line 5046: W20260711 02:41:11.268957 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 5047: W20260711 02:41:11.297412 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 5048: W20260711 02:41:11.357144 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 5049: W20260711 02:41:11.450964 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 5050: W20260711 02:41:11.477777 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 5051: W20260711 02:41:11.540971 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 5052: W20260711 02:41:11.603519 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 5053: W20260711 02:41:11.659743 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 5054: W20260711 02:41:11.688515 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 5055: W20260711 02:41:11.784114 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 5056: W20260711 02:41:11.810532 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 5057: W20260711 02:41:11.868613 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 5058: W20260711 02:41:11.927771 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 5059: W20260711 02:41:11.983943 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 5060: W20260711 02:41:12.008970 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 5061: W20260711 02:41:12.106195 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 5062: W20260711 02:41:12.164707 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 5063: W20260711 02:41:12.191406 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 5064: W20260711 02:41:12.696216 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
