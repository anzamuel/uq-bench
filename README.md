# uq-bench
Work in progress. Interfaces, method set, and results may still change.

A reproducible benchmark for uncertainty quantification on tabular regression. Every method runs under one protocol and gets scored by the same metrics, so differences come from the method and not from the setup. We built it to evaluate [CTabPFN](https://github.com/anzamuel/ctabpfn) [11], conformal intervals on TabPFN's native quantiles, against CLEAR, PCS-UQ, UACQR, native TabPFN, and a split-conformal baseline.

## Design
- Every method sees the same train-test splits and seeds. The current grid is 12 real datasets, 10 seeds, 80/20 splits, and target coverage 0.9.
- A method is a small adapter in `methods/<name>/`, a standalone uv project with its own pinned environment. `methods/README.md` defines the contract.
- A method takes a split and a seed and returns interval bounds. Same seed and machine give identical bytes, and `tests/` checks this.
- Results land in one tidy `results.csv`, one row per method, dataset, seed, and coverage.
- Metrics cover coverage (PICP), width (MPIW, NIW, NCIW), and interval quality (interval score AISL, pinball loss).
- Versioned models carry the generation in the folder name, for example `ctabpfn-q-v3.5`, so old rows stay reproducible.
- One command reruns the full grid and resumes from `results.csv`.

## Run
```bash
./setup.sh                 # uv sync and pre-commit hooks
uv run python benchmark.py # every phase of the grid, appends to results.csv
```
Datasets download from https://huggingface.co/datasets/anzamuel/uq-bench on first use. The TabPFN methods need a `TABPFN_TOKEN` from a Prior Labs account.

## References
Method folders: `baseline` [1, 2], `uacqr` [5], `pcs` [6], `clear` [7], `tabpfn-v3` [8, 9], `tabpfn-v3.5` [8], `ctabpfn-a-*` [3, 10, 11], `ctabpfn-m-*` [1, 2, 11], `ctabpfn-q-*` [4, 11]. TabPFN-3.5 has no published reference yet.

1. V. Vovk, A. Gammerman, G. Shafer. Algorithmic Learning in a Random World. Springer, 2005.
2. J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani, L. Wasserman. Distribution-Free Predictive Inference for Regression. Journal of the American Statistical Association, 113(523):1094-1111, 2018.
3. Y. Romano, E. Patterson, E. J. Candès. Conformalized Quantile Regression. Advances in Neural Information Processing Systems 32, 2019.
4. V. Chernozhukov, K. Wüthrich, Y. Zhu. Distributional Conformal Prediction. Proceedings of the National Academy of Sciences, 118(48), 2021. [arXiv:1909.07889](https://arxiv.org/abs/1909.07889)
5. R. Rossellini, R. F. Barber, R. Willett. Integrating Uncertainty Awareness into Conformalized Quantile Regression. International Conference on Artificial Intelligence and Statistics, 2024. [arXiv:2306.08693](https://arxiv.org/abs/2306.08693)
6. A. Agarwal, F. Xiao, R. Barter, O. Ronen, B. Fan, B. Yu. PCS-UQ: Uncertainty Quantification via the Predictability-Computability-Stability Framework. arXiv preprint, 2025. [arXiv:2505.08784](https://arxiv.org/abs/2505.08784)
7. I. Azizi, J. Bodik, J. Heiss, B. Yu. CLEAR: Calibrated Learning for Epistemic and Aleatoric Risk. arXiv preprint, 2026. [arXiv:2507.08150](https://arxiv.org/abs/2507.08150)
8. N. Hollmann, S. Müller, L. Purucker, A. Krishnakumar, M. Körfer, S. B. Hoo, R. T. Schirrmeister, F. Hutter. Accurate Predictions on Small Data with a Tabular Foundation Model. Nature, 637:319-326, 2025. [doi:10.1038/s41586-024-08328-7](https://doi.org/10.1038/s41586-024-08328-7)
9. L. Grinsztajn, K. Flöge, O. Key, F. Birkel, P. Jund, B. Roof, et al. TabPFN-3: Technical Report. arXiv preprint, 2026. [arXiv:2605.13986](https://arxiv.org/abs/2605.13986)
10. F. D. van Leeuwen. Conformal Prediction for Tabular Prior-Data Fitted Networks with Missing Data. OpenReview preprint, 2025. [openreview:TnZcC7GXI5](https://openreview.net/forum?id=TnZcC7GXI5)
11. S. Anzalone, J. Heiss. CTabPFN: Conformal Uncertainty Quantification with TabPFN. Semester project, ETH Zurich, 2026. [paper](https://github.com/anzamuel/ctabpfn/blob/main/paper/ctabpfn.pdf)

## Authors
Samuel Anzalone and Jakob Heiss.

## License
Apache 2.0, see [LICENSE](LICENSE). Vendored code in `methods/clear`, `methods/pcs`, and `methods/uacqr` keeps its upstream license, see the LICENSE or NOTICE file in each folder.
