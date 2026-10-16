"""Week 4 Improved Model: scorer factory and pre-registered estimator selection.

The Week 2 baseline code (`baseline.py`) is NOT modified. Week 4 can use either
  * Ridge (same as Week 2/3), or
  * RandomForestRegressor (small, shallow, fixed settings; no hyperparameter search)
chosen per rubric dimension by a rule fixed BEFORE any Week 4 result was seen:

  use random_forest for a dimension only if its leave-one-group-out CV MAE on TRAIN+VAL
  is at least `min_relative_gain` (default 10%) lower than Ridge's; otherwise keep Ridge.

The test split is never used for this choice. With very small data a forest rarely clears the
bar, which is intended: the simpler, more explainable model is the default.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline

from .baseline import DimensionModel


class ForestDimensionModel:
    """Same interface as DimensionModel. Fixed, conservative settings (no tuning)."""

    def __init__(self, dimension, features, max_score, seed: int = 42):
        self.dimension, self.features, self.max_score = dimension, features, max_score
        self.alpha, self.selection = None, []
        self.model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                   RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=2,
                                                         random_state=seed, n_jobs=1))

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train[self.features].to_numpy(float), y_train)
        return self

    def predict(self, X):
        return np.clip(self.model.predict(X[self.features].to_numpy(float)), 0, self.max_score)

    def coefficients(self) -> dict:  # importances, reported in the same slot as Ridge coefficients
        return {f: float(v) for f, v in zip(self.features, self.model[-1].feature_importances_)} | {"_kind": "importance"}


def make_dimension_model(estimator: str, dimension, features, max_score, alpha_grid=(0.1, 1.0, 10.0), seed=42):
    if estimator == "ridge":
        return DimensionModel(dimension, features, max_score, alpha_grid)
    if estimator == "random_forest":
        return ForestDimensionModel(dimension, features, max_score, seed)
    raise ValueError(estimator)


def select_estimators(dev, feats: dict, dims, maxes: dict, cv_alpha: float = 1.0,
                      min_relative_gain: float = 0.10, seed: int = 42) -> dict:
    """Pre-registered rule above. `dev` = TRAIN+VAL rows only, with a 'group' column."""
    groups = dev["group"].to_numpy()
    out = {}
    for d in dims:
        maes = {}
        for est in ("ridge", "random_forest"):
            oof = np.zeros(len(dev))
            for tr, te in LeaveOneGroupOut().split(dev, groups=groups):
                A, B = dev.iloc[tr], dev.iloc[te]
                m = make_dimension_model(est, d, feats[d], maxes[d], [cv_alpha], seed).fit(A, A[d].to_numpy(float))
                oof[te] = m.predict(B)
            maes[est] = float(np.mean(np.abs(oof - dev[d].to_numpy(float))))
        gain = (maes["ridge"] - maes["random_forest"]) / maes["ridge"] if maes["ridge"] > 0 else 0.0
        chosen = "random_forest" if gain >= min_relative_gain else "ridge"
        out[d] = {"cv_MAE": maes, "relative_gain_of_forest": gain, "chosen": chosen,
                  "rule": f"forest only if CV MAE (train+val) is >= {min_relative_gain:.0%} lower than Ridge"}
    return out
