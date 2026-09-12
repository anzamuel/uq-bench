"""
TabPFN adapter over the tabpfn package.

Fits the pretrained TabPFN foundation model and reads the interval directly off its native predictive-distribution quantiles at alpha/2 and 1-alpha/2, with no conformal layer on top, so coverage reflects the model's own calibration. Runs on CPU by default for reproducible output, TABPFN_DEVICE overrides. Reads the split from argv[1], writes lower, upper, and center test bounds to argv[2], and takes the target coverage as argv[3] and the seed as argv[4].
"""

import os
import sys

import numpy as np
from tabpfn import TabPFNRegressor


def main() -> None:
    """Fit TabPFN on the train part and write native-quantile test intervals."""
    split_path, preds_path = sys.argv[1], sys.argv[2]
    coverage, seed = float(sys.argv[3]), int(sys.argv[4])
    alpha = 1.0 - coverage
    data = np.load(split_path)
    model = TabPFNRegressor(
        device=os.environ.get("TABPFN_DEVICE", "cpu"), random_state=seed
    )
    model.fit(data["X_train"], data["y_train"])
    lower, center, upper = model.predict(
        data["X_test"],
        output_type="quantiles",
        quantiles=[alpha / 2.0, 0.5, 1.0 - alpha / 2.0],
    )
    np.savez(preds_path, lower=lower, upper=upper, center=center)


if __name__ == "__main__":
    main()
