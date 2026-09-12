# ruff: noqa: E402
"""
PCS-OOB adapter over the vendored pcs_uq classes.

Wraps PCS_OOB with the paper's regression zoo and settings, B=1000 bootstraps and top_k=1, seeding global numpy state before fit because the class draws its bootstraps from it instead of its seed argument. The zoo matches experiments/configs/regression_consts.py including celer's cross-validated linear models, with every model held to one thread because threaded predicts and BLAS pools sum in nondeterministic order; the test pass replicates predict's ensemble quantile math once, yielding bounds and center together. Reads the split from argv[1], writes lower, upper, and center test bounds to argv[2], and takes the target coverage as argv[3] and the seed as argv[4]; UQ_BENCH_FAST drops B to 10 and PCS_BOOTSTRAPS overrides it outright, see methods/README.md.
"""

import os
import sys

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_var] = "1"  # pin hidden blas pools before numeric imports

import numpy as np
from celer import ElasticNetCV, LassoCV
from pcs_oob import PCS_OOB
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


def _models() -> dict:
    return {
        "OLS": LinearRegression(n_jobs=-1),
        "Ridge": RidgeCV(),
        "Lasso": LassoCV(cv=3, n_jobs=-1),
        "ElasticNet": ElasticNetCV(cv=3, n_jobs=-1),
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
    """Fit PCS_OOB on the train part and write calibrated test intervals."""
    split_path, preds_path = sys.argv[1], sys.argv[2]
    coverage, seed = float(sys.argv[3]), int(sys.argv[4])
    default_b = "10" if os.environ.get("UQ_BENCH_FAST") else "1000"
    num_bootstraps = int(os.environ.get("PCS_BOOTSTRAPS", default_b))
    alpha = 1.0 - coverage
    data = np.load(split_path)
    np.random.seed(seed)  # pcs_oob bootstraps from global state
    model = PCS_OOB(
        models=_models(),
        num_bootstraps=num_bootstraps,
        alpha=alpha,
        top_k=1,
        seed=seed,
        load_models=False,
        save_path=None,
    )
    model.fit(data["X_train"], data["y_train"])
    preds = np.column_stack(
        [
            bootstrap_model.predict(data["X_test"])
            for name in model.top_k_models
            for bootstrap_model in model.bootstrap_models[name]
        ]
    )
    preds.sort(axis=1)  # mirror predict's arithmetic exactly
    lower_raw = np.nanquantile(preds, alpha / 2.0, axis=1)
    median = np.nanquantile(preds, 0.5, axis=1)
    upper_raw = np.nanquantile(preds, 1.0 - alpha / 2.0, axis=1)
    np.savez(
        preds_path,
        lower=median - model.gamma * (median - lower_raw),
        upper=median + model.gamma * (upper_raw - median),
        center=median,
    )


if __name__ == "__main__":
    main()
