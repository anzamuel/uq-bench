"""
CLEAR adapter over the authors' source checkout, variant c of the paper's benchmark.

Reproduces the standard pcs_qrf pipeline end to end: the incoming train split is carved 75/25 with sklearn at the given seed, the repo's vendored PCS_UQ fits the paper's nine-model zoo with top_k=1 for the epistemic component, and the clear package fits a QRF on residuals with the PCS median as feature, calibrates lambda and gamma on the validation part over the paper's 4011-point grid, and predicts with both components passed externally. Forest models run single threaded for byte determinism and the aleatoric stage keeps the authors' hardcoded seed 777; UQ_BENCH_FAST drops the bootstrap counts from 100 to 5, CLEAR_BOOTSTRAPS overrides them outright, and CLEAR_JOBS bounds the worker pool, see methods/README.md.
"""

import os
import sys

import numpy as np
from celer import ElasticNetCV, LassoCV
from clear.clear import CLEAR
from pcs_uq import PCS_UQ
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


def _estimators() -> dict:
    return {
        "OLS": LinearRegression(),
        "Ridge": RidgeCV(),
        "Lasso": LassoCV(cv=3, n_jobs=1),
        "ElasticNet": ElasticNetCV(cv=3, n_jobs=1),
        "RandomForest": RandomForestRegressor(
            min_samples_leaf=5,
            max_features=0.33,
            n_estimators=100,
            random_state=42,
            n_jobs=1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            min_samples_leaf=5,
            max_features=0.33,
            n_estimators=100,
            random_state=42,
            n_jobs=1,
        ),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42, n_jobs=1),
        "MLP": MLPRegressor(random_state=42, hidden_layer_sizes=(64,)),
    }


def main() -> None:
    """Fit the variant c pipeline on the train part and write calibrated test intervals."""
    split_path, preds_path = sys.argv[1], sys.argv[2]
    coverage, seed = float(sys.argv[3]), int(sys.argv[4])
    n_boot = int(
        os.environ.get(
            "CLEAR_BOOTSTRAPS", "5" if os.environ.get("UQ_BENCH_FAST") else "100"
        )
    )
    n_jobs = int(
        os.environ.get("CLEAR_JOBS", "4")
    )  # every worker holds its own forests
    data = np.load(split_path)
    X_test = data["X_test"]
    x_tr, x_val, y_tr, y_val = train_test_split(
        data["X_train"], data["y_train"], test_size=0.25, random_state=seed
    )
    pcs = PCS_UQ(
        models=_estimators(),
        num_bootstraps=n_boot,
        alpha=1.0 - coverage,
        seed=seed,
        top_k=1,
    )
    pcs.fit(x_tr, y_tr, X_calib=x_val, y_calib=y_val)
    raw = {
        name: pcs.get_intervals(x)
        for name, x in (("train", x_tr), ("val", x_val), ("test", X_test))
    }
    model = CLEAR(
        desired_coverage=coverage,
        lambdas=np.concatenate((np.linspace(0, 0.09, 10), np.logspace(-1, 2, 4001))),
        n_bootstraps=n_boot,
        random_state=777,
        n_jobs=n_jobs,
    )
    model.fit_aleatoric(
        x_tr,
        y_tr,
        quantile_model="rf",
        model_params={
            "n_estimators": 100,
            "random_state": 777,
            "min_samples_leaf": 10,
            "n_jobs": 1,
        },
        fit_on_residuals=True,
        epistemic_preds=raw["train"][:, 1],
    )
    al_med_val, al_lo_val, al_up_val = model.predict_aleatoric(
        x_val, epistemic_preds=raw["val"][:, 1]
    )
    al_med_test, al_lo_test, al_up_test = model.predict_aleatoric(
        X_test, epistemic_preds=raw["test"][:, 1]
    )
    model.calibrate(
        y_val,
        median_epistemic=raw["val"][:, 1],
        aleatoric_median=al_med_val,
        aleatoric_lower=al_lo_val,
        aleatoric_upper=al_up_val,
        epistemic_lower=raw["val"][:, 0],
        epistemic_upper=raw["val"][:, 2],
        verbose=False,
    )
    lower, upper = model.predict(
        X_test,
        external_epistemic={
            "median": raw["test"][:, 1],
            "lower": raw["test"][:, 0],
            "upper": raw["test"][:, 2],
        },
        external_aleatoric={
            "median": al_med_test,
            "lower": al_lo_test,
            "upper": al_up_test,
        },
    )
    np.savez(preds_path, lower=lower, upper=upper, center=raw["test"][:, 1])


if __name__ == "__main__":
    main()
