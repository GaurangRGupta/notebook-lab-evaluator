"""Week 2 baseline: one linear (Ridge) regressor per rubric dimension.

Why linear regression rather than logistic regression:
  the Week 1 proposal frames scoring as regression against human marks; the targets
  are ordinal integer marks (0-8, 0-8, 0-4) and the annotated set is small, so a
  9-class logistic model would be unstable and would ignore the ordering of marks.
Predictions are clipped to [0, max] (and rounded only for agreement metrics).
Missing features (e.g. no expected outputs for an assignment) are median-imputed
with statistics learned on the training split only.
"""
from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(alpha: float):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), Ridge(alpha=alpha))


class DimensionModel:
    def __init__(self, dimension: str, features: list[str], max_score: float, alpha_grid=(0.1, 1.0, 10.0)):
        self.dimension, self.features, self.max_score = dimension, features, max_score
        self.alpha_grid = list(alpha_grid)
        self.alpha = None
        self.model = None
        self.selection = []

    def _clip(self, p):
        return np.clip(p, 0, self.max_score)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Alpha chosen by validation MAE when a validation set exists, else the middle of the grid."""
        Xt = X_train[self.features].to_numpy(float)
        if X_val is not None and len(X_val):
            Xv = X_val[self.features].to_numpy(float)
            for a in self.alpha_grid:
                m = make_model(a).fit(Xt, y_train)
                mae = float(np.mean(np.abs(self._clip(m.predict(Xv)) - y_val)))
                self.selection.append({"alpha": a, "val_MAE": mae})
            self.alpha = min(self.selection, key=lambda r: (r["val_MAE"], r["alpha"]))["alpha"]
        else:
            self.alpha = self.alpha_grid[len(self.alpha_grid) // 2]
        self.model = make_model(self.alpha).fit(Xt, y_train)
        return self

    def predict(self, X):
        return self._clip(self.model.predict(X[self.features].to_numpy(float)))

    def coefficients(self) -> dict:
        ridge = self.model[-1]
        return {f: float(c) for f, c in zip(self.features, ridge.coef_)} | {"_intercept": float(ridge.intercept_)}


class MeanBaseline:
    """Reference floor: always predicts the training mean. A useful model must beat it."""

    def __init__(self, max_score):
        self.max_score = max_score
        self.m = DummyRegressor(strategy="mean")

    def fit(self, y_train):
        self.m.fit(np.zeros((len(y_train), 1)), y_train)
        return self

    def predict(self, n):
        return np.clip(self.m.predict(np.zeros((n, 1))), 0, self.max_score)
