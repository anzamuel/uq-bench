"""
Split-conformal linear regression baseline.

Carves a quarter of the train split as a calibration set with the given seed, fits ordinary least squares on the rest, and calibrates a constant absolute-residual interval on it. Reads the split from argv[1], writes lower and upper test bounds to argv[2], and takes the target coverage as argv[3] and the seed as argv[4].
"""

import sys

import numpy as np
from numpy.typing import NDArray

type Array = NDArray[np.float64]


def _design_matrix(x: Array) -> Array:
    return np.column_stack([np.ones(len(x)), x])


def main() -> None:
    """Carve a calibration set, fit OLS, and write conformal test bounds."""
    split_path, preds_path = sys.argv[1], sys.argv[2]
    coverage, seed = float(sys.argv[3]), int(sys.argv[4])
    data = np.load(split_path)
    idx = np.random.default_rng(seed).permutation(len(data["y_train"]))
    n_cal = int(len(idx) * 0.25)
    cal, fit = idx[:n_cal], idx[n_cal:]
    X_fit, y_fit = data["X_train"][fit], data["y_train"][fit]
    X_cal, y_cal = data["X_train"][cal], data["y_train"][cal]
    beta, *_ = np.linalg.lstsq(_design_matrix(X_fit), y_fit, rcond=None)
    residuals = np.abs(y_cal - _design_matrix(X_cal) @ beta)
    level = min(np.ceil((n_cal + 1) * coverage) / n_cal, 1.0)
    q = np.quantile(residuals, level, method="higher")
    center = _design_matrix(data["X_test"]) @ beta
    np.savez(preds_path, lower=center - q, upper=center + q)


if __name__ == "__main__":
    main()
