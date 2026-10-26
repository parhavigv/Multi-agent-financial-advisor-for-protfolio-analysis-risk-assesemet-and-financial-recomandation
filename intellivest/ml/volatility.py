"""Volatility forecasting models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import qlike, rmse


class EWMA:
    name = "ewma"

    def fit(self, returns: pd.Series, halflife: int = 32):
        self.halflife = halflife
        self.returns_ = returns
        return self

    def predict(self, periods: int = 252) -> float:
        r = self.returns_.dropna().fillna(0.0)
        if r.empty:
            return 0.01
        var = (r**2).ewm(halflife=self.halflife, adjust=False).mean().iloc[-1]
        return float(np.sqrt(max(var, 1e-18) * periods))


class HAR:
    name = "har"

    def fit(self, rv: pd.Series):
        self.rv_ = rv.dropna()
        return self

    def predict(self, periods: int = 252) -> float:
        rv = self.rv_
        if len(rv) < 22:
            # fallback
            return float(rv.mean() if len(rv) else 0.01)
        # simple HAR(1,5,22)
        try:
            df = pd.DataFrame({"rv": rv})
            df["rv1"] = df["rv"].shift(1)
            df["rv5"] = df["rv"].rolling(5).mean().shift(1)
            df["rv22"] = df["rv"].rolling(22).mean().shift(1)
            df = df.dropna()
            if df.empty:
                return float(rv.iloc[-1])
            X = df[["rv1", "rv5", "rv22"]]
            y = df["rv"]
            try:
                from sklearn.linear_model import Ridge

                m = Ridge(alpha=1.0)
                m.fit(X, y)
                x0 = pd.DataFrame(
                    {
                        "rv1": [rv.iloc[-1]],
                        "rv5": [rv.iloc[-5:].mean()],
                        "rv22": [rv.iloc[-22:].mean()],
                    }
                )
                pred = float(m.predict(x0)[0])
                return max(pred, 1e-6)
            except Exception:
                return float((rv.iloc[-1] + rv.iloc[-5:].mean() + rv.iloc[-22:].mean()) / 3.0)
        except Exception:
            return float(rv.iloc[-1])


class GARCH11:
    name = "garch11"

    def fit(self, returns: pd.Series):
        self.returns_ = returns.dropna().fillna(0.0)
        return self

    def predict(self, periods: int = 252) -> float:
        r = self.returns_
        if len(r) < 50:
            return float(r.std(ddof=0) * np.sqrt(periods)) if len(r) else 0.01
        try:
            # simple GARCH(1,1) approximation via recursion (numerically stable)
            omega = 1e-6
            alpha = 0.05
            beta = 0.9
            var = float(np.var(r))
            for ret in r:
                var = omega + alpha * (ret * ret) + beta * var
            sigma = np.sqrt(max(var, 1e-18)) * np.sqrt(periods)
            return float(sigma)
        except Exception:
            return float(r.std(ddof=0) * np.sqrt(periods))


def eval_vol(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    mask = y_true.notna() & y_pred.notna()
    if not mask.sum():
        return {"rmse": float("nan"), "qlike": float("nan")}
    return {"rmse": rmse(y_true[mask], y_pred[mask]), "qlike": qlike(y_true[mask], y_pred[mask])}
