# COLMAP / SfM diagnostic

This report is retrospective and does not execute a retry.

## Outcome

- Status: `completed_with_warnings`
- Successful reconstruction retained: `true`
- Maximum registered-frame count observed in logs: `186`
- Bundle-adjustment log events: `49`
- Bounded fallback eligible: `false`

## Classification

- `ba_numerical_instability`

## Bounded fallback recommendations

- `alternate_sparse_linear_solver`
- `disable_focal_refinement`

## Evidence

### ba_numerical_instability

- line 222: W20260711 02:48:50.239708 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 223: W20260711 02:48:50.259709 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 224: W20260711 02:48:50.264317 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 225: W20260711 02:48:50.283919 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 226: W20260711 02:48:50.288559 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 227: W20260711 02:48:50.300650 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 228: W20260711 02:48:50.305249 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 229: W20260711 02:48:50.324797 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 230: W20260711 02:48:50.329413 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 231: W20260711 02:48:50.356523 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 232: W20260711 02:48:50.368587 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 233: W20260711 02:48:50.373197 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 234: W20260711 02:48:50.377819 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 235: W20260711 02:48:50.397440 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 236: W20260711 02:48:50.409540 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 244: W20260711 02:48:50.647812 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 245: W20260711 02:48:50.663763 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 246: W20260711 02:48:50.689391 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 247: W20260711 02:48:50.695652 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
- line 248: W20260711 02:48:50.721283 127739432001536 levenberg_marquardt_strategy.cc:123] Linear solver failure. Failed to compute a step: Eigen failure. Unable to perform dense Cholesky factorization.
