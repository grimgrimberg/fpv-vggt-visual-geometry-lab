# COLMAP / SfM diagnostic

This report is retrospective and does not execute a retry.

## Outcome

- Status: `completed_with_warnings`
- Successful reconstruction retained: `true`
- Maximum registered-frame count observed in logs: `199`
- Bundle-adjustment log events: `275`
- Bounded fallback eligible: `false`

## Classification

- `ba_numerical_instability`

## Bounded fallback recommendations

- `alternate_sparse_linear_solver`
- `disable_focal_refinement`

## Evidence

### ba_numerical_instability

- line 1430: W20260710 21:51:44.085922 130370942226432 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1431: W20260710 21:51:44.440107 130370942226432 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1432: W20260710 21:51:44.462726 130370942226432 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1723: W20260710 21:55:58.979606 130370942226432 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1724: W20260710 21:55:59.035663 130370942226432 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1725: W20260710 21:55:59.045134 130370942226432 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1726: W20260710 21:55:59.089524 130370942226432 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1727: W20260710 21:55:59.098994 130370942226432 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1728: W20260710 21:55:59.143189 130370942226432 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1729: W20260710 21:55:59.152663 130370942226432 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1730: W20260710 21:55:59.179694 130370942226432 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1731: W20260710 21:55:59.206539 130370942226432 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1732: W20260710 21:55:59.233391 130370942226432 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1733: W20260710 21:55:59.242901 130370942226432 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1734: W20260710 21:55:59.270388 130370942226432 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1735: W20260710 21:55:59.477295 130370942226432 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1736: W20260710 21:56:02.298123 130370942226432 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1737: W20260710 21:56:02.496701 130370942226432 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1738: W20260710 21:56:02.594741 130370942226432 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 1739: W20260710 21:56:02.988236 130370942226432 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.

