"""Two non-deep baselines (target-series only)."""
import numpy as np
from sklearn.linear_model import Ridge


def seasonal_naive(X, horizon, period=24):
    """Forecast = repeat the last `period` hours (yesterday's pattern). 0 parameters."""
    last = X[:, -period:]
    reps = int(np.ceil(horizon / period))
    return np.tile(last, (1, reps))[:, :horizon]


class LinearBaseline:
    """Ridge regression: past `lookback` target values -> next `horizon` values (DLinear-style)."""
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, X, Y):
        self.model.fit(X, Y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    @property
    def n_params(self):
        return int(self.model.coef_.size + self.model.intercept_.size)
