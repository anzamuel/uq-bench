# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy>=2.5.0"]
# ///
"""
Split-conformal linear regression baseline.

Fits ordinary least squares on the train split and calibrates a constant absolute-residual interval on the cal split. Reads the split from argv[1], writes lower and upper test bounds to argv[2], and takes the target coverage as argv[3].
"""

import sys

import numpy as np
from numpy.typing import NDArray

type Array = NDArray[np.float64]


def _design_matrix(x: Array) -> Array:
    return np.column_stack([np.ones(len(x)), x])


def main() -> None:
    """Fit OLS, calibrate an absolute-residual interval, and write test bounds."""
    split_path, preds_path, coverage = sys.argv[1], sys.argv[2], float(sys.argv[3])
    data = np.load(split_path)
    beta, *_ = np.linalg.lstsq(
        _design_matrix(data["X_train"]), data["y_train"], rcond=None
    )
    residuals = np.abs(data["y_cal"] - _design_matrix(data["X_cal"]) @ beta)
    n_cal = len(residuals)
    level = min(np.ceil((n_cal + 1) * coverage) / n_cal, 1.0)
    q = np.quantile(residuals, level, method="higher")
    center = _design_matrix(data["X_test"]) @ beta
    np.savez(preds_path, lower=center - q, upper=center + q)


if __name__ == "__main__":
    main()
