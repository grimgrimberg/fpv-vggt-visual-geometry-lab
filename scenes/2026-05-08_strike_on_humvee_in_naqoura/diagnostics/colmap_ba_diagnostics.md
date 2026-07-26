# COLMAP / SfM diagnostic

This report is retrospective and does not execute a retry.

## Outcome

- Status: `completed_with_warnings`
- Successful reconstruction retained: `true`
- Maximum registered-frame count observed in logs: `188`
- Bundle-adjustment log events: `63`
- Bounded fallback eligible: `false`

## Classification

- `ba_numerical_instability`

## Bounded fallback recommendations

- `alternate_sparse_linear_solver`
- `disable_focal_refinement`

## Evidence

### ba_numerical_instability

- line 1436: W20260711 03:30:54.896882 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1437: W20260711 03:30:54.897369 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1726: W20260711 03:32:43.721775 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1727: W20260711 03:32:43.723060 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1728: W20260711 03:32:43.724285 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1729: W20260711 03:32:43.724443 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1730: W20260711 03:32:43.725665 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1731: W20260711 03:32:43.726888 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1732: W20260711 03:32:43.728112 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1740: W20260711 03:32:47.523586 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 1741: W20260711 03:32:49.034343 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 1742: W20260711 03:32:49.088075 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 1743: W20260711 03:32:49.142307 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 1744: W20260711 03:32:49.162532 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 1745: W20260711 03:32:50.544556 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 1746: W20260711 03:32:50.598909 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 1747: W20260711 03:32:50.622082 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 1748: W20260711 03:32:50.710047 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 1749: W20260711 03:32:50.733488 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
- line 1750: W20260711 03:32:50.783230 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: CHOLMOD warning: Matrix not positive definite.
