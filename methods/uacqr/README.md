# uacqr
Runs UACQR-P from the authors' implementation with the fast quantile-random-forest recipe: B single-tree quantile forests on bootstrap draws form the percentile ensemble, a full B-tree forest is the base model, and the calibrated rank is the exact conformal order statistic on a 50/50 fit and calibration split, the authors' 40/40/20 proportions. Intervals can be legitimately infinite when the rank hits the sentinel column, which the paper discloses and our metrics handle. Writes the base forest's 0.5 quantile as `center`.
- paper: https://arxiv.org/abs/2306.08693
- source: `uacqr.py` and `helper.py` vendored from `rrross/UACQR` at `884718a`, verifiable with `diff -b` per their headers, because the repository has no packaging at all
- fast: bootstraps drop from 100 to 10
- knobs: `UACQR_BOOTSTRAPS` overrides the bootstrap count outright, default 100
- threading: the vendored forests keep their internal `n_jobs` of -1, empirically byte-stable; phase workers about half the cores to offset the internal pools
- deviations: alpha is assigned after construction because the class hardcodes 0.1 and ignores its argument; the non-randomized cutoff variant is used, matching the paper's appendix table; `min_samples_leaf` is fixed at 10 instead of the authors' cross-validated tuning, the package default of 1 makes single-tree members interpolate and degenerates the percentile ensemble; the `center` output is our addition, the method defines no point predictor; one helper import of the unpackaged oqr repo is stubbed, it feeds only evaluation paths the benchmark never calls
