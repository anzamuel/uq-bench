# clear
Runs the paper's variant c benchmark pipeline: the vendored PCS_UQ screens the nine-model zoo and bootstraps the top model for epistemic uncertainty, a QRF on residuals with the PCS median as feature gives aleatoric, and the clear package calibrates lambda and gamma on the 20 percent validation split over the 4011-point grid, standard mode with validation doubling as calibration. Splits inside the method use sklearn train_test_split at the given seed, matching the authors' driver. Writes the PCS median as `center`.
- paper: https://arxiv.org/abs/2507.08150
- source: `clear-uq` from git, `Unco3892/clear` pinned to `9b3c0ea`; the repo's vendored `PCS_UQ` class is carried as `pcs_uq.py`, same commit, verifiable with `diff -b` per its header, because its own packaging is broken upstream, see Unco3892/clear issue 5
- fast: bootstrap counts drop from 100 to 5
- knobs: `CLEAR_BOOTSTRAPS` overrides both stages' bootstrap counts outright, default 100; `CLEAR_JOBS` bounds the aleatoric worker pool, default 4
- threading: forest models single threaded, the aleatoric stage fans out over a `CLEAR_JOBS` loky pool; phase workers sized as cores divided by `CLEAR_JOBS` and bounded by memory
- deviations: forest and boosting models run single threaded for byte determinism; only the standard variant is implemented; the aleatoric stage keeps the authors' hardcoded seed 777, so it is faithful to their runs rather than to the seed argument
