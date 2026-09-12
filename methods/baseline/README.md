# baseline
Split-conformal ordinary least squares: fits OLS on three quarters of the train split and calibrates a constant absolute-residual half width on the remaining quarter with the finite-sample correction. Writes no `center`, so NCIW anchors on the interval midpoint.
- paper: none
- source: original
- fast: ignored
- knobs: none
- threading: single threaded; phase workers equal to cores
- deviations: none
