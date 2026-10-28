"""Return forecasting models with safe fallbacks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import ic, rank_ic, rmse


@dataclass
class ReturnForecast:
    forecast: pd.Series
    model_name: str
    weights: float = 1.0
    metrics: dict[str, float] | None = None


class BaselineMean:
    name = "baseline_mean"

    def fit(self, returns: pd.Series):
        self.mean_ = returns.mean() if len(returns) else 0.0
        return self

    def predict(self, horizon: int = 1) -> float:
        return float(self.mean_)


class RidgeReturn:
    name = "ridge"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        try:
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
            Xs = self.scaler.fit_transform(X.fillna(0.0))
            self.model = Ridge(alpha=1.0)
            self.model.fit(Xs, y.fillna(0.0))
            self.feature_names_ = list(X.columns)
        except Exception:
            self.model = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            if getattr(self, "model", None) is None:
                return np.zeros(len(X))
            Xs = self.scaler.transform(X.fillna(0.0))
            return self.model.predict(Xs)
        except Exception:
            return np.zeros(len(X))


class GBMReturn:
    name = "gboost"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        try:
            from sklearn.ensemble import GradientBoostingRegressor

            self.model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=12345)
            self.model.fit(X.fillna(0.0), y.fillna(0.0))
        except Exception:
            self.model = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            if getattr(self, "model", None) is None:
                return np.zeros(len(X))
            return self.model.predict(X.fillna(0.0))
        except Exception:
            return np.zeros(len(X))


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    mask = y_true.notna() & y_pred.notna()
    if not mask.sum():
        return {"rmse": float("nan"), "ic": float("nan"), "rank_ic": float("nan")}
    return {
        "rmse": rmse(y_true[mask], y_pred[mask]),
        "ic": ic(y_true[mask], y_pred[mask]),
        "rank_ic": rank_ic(y_true[mask], y_pred[mask]),
    }


def build_ensemble(forecasts: dict[str, pd.Series], val_metrics: dict[str, dict[str, float]]) -> pd.Series:
    """Inverse-error weighting with fallback to baseline if no skill."""
    if not forecasts:
        return pd.Series(dtype=float)
    weights = {}
    for name, pred in forecasts.items():
        m = val_metrics.get(name, {})
        r = m.get("rmse", float("inf"))
        if not np.isfinite(r) or r <= 0:
            weights[name] = 0.0
        else:
            weights[name] = 1.0 / (r * r)
    # require positive skill vs baseline if present
    base_rmse = val_metrics.get("baseline_mean", {}).get("rmse", float("inf"))
    # drop models worse than baseline
    for name in list(weights.keys()):
        if name == "baseline_mean":
            continue
        r = val_metrics.get(name, {}).get("rmse", float("inf"))
        if not np.isfinite(r) or r >= base_rmse - 1e-12:  # must beat or equal? require better
            weights[name] = 0.0
    wsum = sum(weights.values())
    if wsum <= 0:
        # fallback to baseline
        if "baseline_mean" in forecasts:
            return forecasts["baseline_mean"] * 1.0
        return pd.Series(0.0, index=list(forecasts.values())[0].index)
    ens = pd.Series(0.0, index=list(forecasts.values())[0].index)
    for name, pred in forecasts.items():
        ens = ens + weights[name] / wsum * pred
    return ens
