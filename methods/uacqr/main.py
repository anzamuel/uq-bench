"""
UACQR-P adapter over the vendored authors' implementation.

Runs UACQR-P with the fast quantile-random-forest recipe: B single-tree ensembles form the percentile family, a full B-tree forest anchors the base interval, and the exact conformal order statistic picks the calibrated rank on a 50/50 split of the incoming train part, matching the authors' 40/40/20 proportions; cutoffs are the non-randomized variant. The vendored class hardcodes its miscoverage level, so the adapter assigns alpha after construction. Writes the base forest's 0.5 quantile as center, an addition of ours since the method defines no point predictor; UQ_BENCH_FAST drops B from 100 to 10 and UACQR_BOOTSTRAPS overrides it outright, see methods/README.md.
"""

import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from uacqr import uacqr


def main() -> None:
    """Fit UACQR-P on the train part and write calibrated test intervals."""
    split_path, preds_path = sys.argv[1], sys.argv[2]
    coverage, seed = float(sys.argv[3]), int(sys.argv[4])
    alpha = 1.0 - coverage
    b = int(
        os.environ.get(
            "UACQR_BOOTSTRAPS", "10" if os.environ.get("UQ_BENCH_FAST") else "100"
        )
    )
    data = np.load(split_path)
    x_fit, x_cal, y_fit, y_cal = train_test_split(
        data["X_train"], data["y_train"], test_size=0.5, random_state=seed
    )
    model = uacqr(
        model_params={"min_samples_leaf": 10},  # leaf size 1 degenerates the P ensemble
        B=b,
        q_lower=100.0 * alpha / 2.0,
        q_upper=100.0 * (1.0 - alpha / 2.0),
        model_type="rfqr",
        random_state=seed,
    )
    model.alpha = alpha  # the class hardcodes 0.1
    model.fit(x_fit, y_fit)
    model.calibrate(x_cal, y_cal)
    bounds = model.predict(data["X_test"])
    model.cqr_base_model.set_params(q=0.5)
    center = model.cqr_base_model.predict(data["X_test"])
    np.savez(
        preds_path,
        lower=bounds[("UACQR-P", "lower")].to_numpy(),
        upper=bounds[("UACQR-P", "upper")].to_numpy(),
        center=center,
    )


if __name__ == "__main__":
    main()
