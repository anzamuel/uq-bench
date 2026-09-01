"""
PCS-OOB adapter over the local pcs_uq checkout.

Wraps PCS_OOB with the paper's regression zoo and settings, B=1000 bootstraps and top_k=1, seeding global numpy state before fit because the class draws its bootstraps from it instead of its seed argument. The zoo swaps celer's LassoCV and ElasticNetCV for the scikit-learn ones to avoid the compiled dependency; everything else matches experiments/configs/regression_consts.py. Reads the split from argv[1], writes lower, upper, and center test bounds to argv[2], and takes the target coverage as argv[3] and the seed as argv[4]; PCS_BOOTSTRAPS overrides B for quick runs.
"""

import os
import sys

import numpy as np
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from src.PCS.regression.pcs_oob import PCS_OOB
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
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            min_samples_leaf=5,
            max_features=0.33,
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42, n_jobs=-1),
        "MLP": MLPRegressor(random_state=42, hidden_layer_sizes=(64,)),
    }


def _median(model: PCS_OOB, X: np.ndarray) -> np.ndarray:
    preds = np.column_stack(
        [
            bootstrap_model.predict(X)
            for name in model.top_k_models
            for bootstrap_model in model.bootstrap_models[name]
        ]
    )
    return np.quantile(preds, 0.5, axis=1)


def main() -> None:
    """Fit PCS_OOB on the train part and write calibrated test intervals."""
    split_path, preds_path = sys.argv[1], sys.argv[2]
    coverage, seed = float(sys.argv[3]), int(sys.argv[4])
    num_bootstraps = int(os.environ.get("PCS_BOOTSTRAPS", "1000"))
    data = np.load(split_path)
    np.random.seed(seed)  # pcs_oob bootstraps from global state
    model = PCS_OOB(
        models=_models(),
        num_bootstraps=num_bootstraps,
        alpha=1.0 - coverage,
        top_k=1,
        seed=seed,
        load_models=False,
        save_path=None,
    )
    model.fit(data["X_train"], data["y_train"])
    pred = model.predict(data["X_test"])
    np.savez(
        preds_path,
        lower=pred[:, 0],
        upper=pred[:, 1],
        center=_median(model, data["X_test"]),
    )


if __name__ == "__main__":
    main()
