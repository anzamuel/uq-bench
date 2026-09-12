# pcs
Runs PCS-OOB from the authors' repository: screens the paper's nine-model zoo on a 75/25 split, refits the top model on 1000 bootstraps, forms per-point out-of-bag quantile intervals, and calibrates a multiplicative gamma against out-of-bag coverage. Writes the bootstrap-ensemble median as `center`.
- paper: https://arxiv.org/abs/2505.08784
- source: the `PCS_OOB` and `PCS_UQ` classes are carried as `pcs_oob.py` and `pcs_uq.py`, vendored from `aagarwal1996/PCS_UQ` at `b004ebc`, verifiable with `diff -b` per their headers, because the upstream packaging omits `src/PCS` from built wheels
- fast: bootstraps drop from 1000 to 10
- knobs: `PCS_BOOTSTRAPS` overrides the bootstrap count outright, default 1000
- threading: every zoo model and the hidden BLAS pools pinned to one thread, one core per cell; phase workers equal to cores
- deviations: global numpy state is seeded before fit because the class ignores its seed argument for bootstrap draws, matching the paper's own driver; every zoo model and the BLAS pools run single threaded because threaded reductions sum in nondeterministic order; scikit-learn is capped below 1.7 for celer compatibility; the repository's calibration quirks are kept for parity, gamma floored at 1 and no finite-sample correction
