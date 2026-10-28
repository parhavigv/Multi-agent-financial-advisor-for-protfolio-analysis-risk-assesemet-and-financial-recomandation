"""Volatility forecast models: EWMA, log-HAR and GARCH(1,1).

All three consume the causal, one-day-lagged features built in
:mod:`intellivest.forecasting.data`, so a prediction at date ``t`` never sees a
return after ``t-1``. GARCH parameters are estimated with the ``arch`` package
on each training fold; the variance recursion is then applied forward, which is
the standard GARCH(1,1) filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ForecastModel, PanelData


class EWMAVolatility(ForecastModel):
    """Exponentially weighted moving average volatility (riskmetrics-style)."""

    name = "ewma"
    kind = "volatility"
    uses_features = True
    FEATURE = "ewma_vol"

    def fit(self, train: PanelData) -> "EWMAVolatility":
        return self

    def predict(self, predict: PanelData) -> pd.Series:
        if self.FEATURE not in predict.X.columns:
            return pd.Series(dtype=float, index=predict.X.index)
        values = predict.X[self.FEATURE].astype(float).clip(lower=1e-8)
        return pd.Series(values, index=predict.X.index, name="prediction")

    def coefficients(self) -> dict[str, float]:
        return {}


class HARVolatility(ForecastModel):
    """Log-HAR: regress log forward variance on log daily/weekly/monthly RV."""

    name = "har"
    kind = "volatility"
    uses_features = True
    FEATURES = ("rv_1", "rv_5", "rv_22")

    def __init__(self, horizon: int = 21, ridge_alpha: float = 1.0) -> None:
        self.horizon = int(horizon)
        self.ridge_alpha = float(ridge_alpha)
        self.model = None
        self.intercept_: float = 0.0
        self.coef_: dict[str, float] = {}

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        cols = []
        for name in self.FEATURES:
            if name in X.columns:
                cols.append(np.log(X[name].astype(float).clip(lower=1e-12).to_numpy()))
            else:
                cols.append(np.zeros(len(X)))
        return np.column_stack(cols)

    def fit(self, train: PanelData) -> "HARVolatility":
        from sklearn.linear_model import Ridge

        if train.y.empty:
            return self
        X = self._design(train.X)
        # Target in log annualised variance units; the constant absorbs the
        # difference between daily RV and the h-day annualised target.
        y = np.log(np.maximum(train.y.astype(float).to_numpy() ** 2, 1e-12))
        mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if mask.sum() < 30:
            return self
        model = Ridge(alpha=self.ridge_alpha)
        model.fit(X[mask], y[mask])
        self.model = model
        self.intercept_ = float(model.intercept_)
        self.coef_ = {name: float(c) for name, c in zip(self.FEATURES, model.coef_)}
        return self

    def predict(self, predict: PanelData) -> pd.Series:
        if self.model is None or predict.X.empty:
            return pd.Series(dtype=float, index=predict.X.index)
        X = self._design(predict.X)
        log_var = self.model.predict(X)
        # Convert annualised variance back to annualised volatility.
        vol = np.exp(0.5 * np.clip(log_var, -30.0, 10.0))
        return pd.Series(vol, index=predict.X.index, name="prediction")

    def coefficients(self) -> dict[str, float]:
        return {"intercept": float(self.intercept_), **self.coef_}


class GARCH11Volatility(ForecastModel):
    """GARCH(1,1) estimated with ``arch`` and filtered forward causally.

    Parameters are estimated once per training fold. The conditional variance
    is then filtered forward using the lagged returns available at each
    prediction date, and the h-step variance forecast is the average of the
    analytic GARCH term structure over the horizon.
    """

    name = "garch11"
    kind = "volatility"
    uses_features = True
    RETURN_FEATURE = "ret_lag"

    def __init__(self, horizon: int = 21, periods: int = 252) -> None:
        self.horizon = int(horizon)
        self.periods = int(periods)
        self.params_: dict[str, tuple[float, float, float]] = {}
        self.feature_names_: list[str] = []
        self.history_: dict[str, pd.Series] = {}
        self.fallback_ewma: dict[str, float] = {}

    def fit(self, train: PanelData) -> "GARCH11Volatility":
        self.feature_names_ = list(train.X.columns)
        if train.X.empty or self.RETURN_FEATURE not in train.X.columns:
            return self
        self.params_ = {}
        self.history_ = {}
        for ticker, block in train.X.groupby(level="ticker"):
            series = block[self.RETURN_FEATURE].astype(float).dropna()
            # groupby on a MultiIndex keeps it; reduce to a plain date index so
            # it can be merged with the prediction block later.
            series.index = pd.DatetimeIndex(series.index.get_level_values("date"))
            series = series[~series.index.duplicated(keep="last")].sort_index()
            self.history_[str(ticker)] = series
            rets = series.to_numpy()
            if len(rets) < 100 or float(np.nanvar(rets)) <= 0:
                continue
            try:
                from arch import arch_model

                # Scale to percent for numerical stability; convert back later.
                am = arch_model(rets * 100.0, mean="Zero", vol="Garch", p=1, q=1)
                res = am.fit(disp="off", show_warning=False)
                omega = float(res.params.get("omega", np.nan))
                alpha = float(res.params.get("alpha[1]", np.nan))
                beta = float(res.params.get("beta[1]", np.nan))
                if not all(np.isfinite([omega, alpha, beta])):
                    continue
                if alpha + beta >= 0.999:
                    beta = 0.999 - alpha
                self.params_[str(ticker)] = (omega, alpha, max(beta, 0.0))
            except Exception:
                continue
        return self

    def _variance_path(
        self, rets: np.ndarray, params: tuple[float, float, float]
    ) -> np.ndarray:
        """Conditional variance after observing each return (in percent^2)."""
        omega, alpha, beta = params
        path = np.empty(len(rets), dtype=float)
        var = float(np.nanvar(rets)) if len(rets) > 1 else max(omega, 1e-8)
        var = max(var, 1e-8)
        for i, ret in enumerate(rets):
            if np.isfinite(ret):
                var = omega + alpha * ret * ret + beta * var
            var = max(var, 1e-10)
            path[i] = var
        return path

    def _forecast_from(self, var_pct: float, params: tuple[float, float, float]) -> float:
        omega, alpha, beta = params
        persistence = min(max(alpha + beta, 0.0), 0.999)
        long_run = omega / max(1.0 - persistence, 1e-8)
        path = [
            long_run + (var_pct - long_run) * (persistence ** k)
            for k in range(1, self.horizon + 1)
        ]
        avg_var_pct = float(np.mean(path))
        avg_var = avg_var_pct / 10_000.0  # percent^2 -> decimal^2
        return float(np.sqrt(max(avg_var, 1e-18) * self.periods))

    def predict(self, predict: PanelData) -> pd.Series:
        if self.RETURN_FEATURE not in predict.X.columns:
            return pd.Series(dtype=float, index=predict.X.index)
        out = pd.Series(np.nan, index=predict.X.index, name="prediction", dtype=float)
        for ticker in predict.X.index.get_level_values("ticker").unique():
            block = predict.X.xs(ticker, level="ticker")
            if str(ticker) not in self.params_:
                # Documented fallback: EWMA when GARCH did not converge.
                if "ewma_vol" in block.columns:
                    out.loc[(slice(None), ticker)] = block["ewma_vol"].to_numpy()
                continue
            params = self.params_[str(ticker)]
            history = self.history_.get(str(ticker), pd.Series(dtype=float))
            observed = block[self.RETURN_FEATURE].astype(float)
            # Merge on the date index so in-sample predictions do not duplicate
            # the training history and forward predictions extend it cleanly.
            combined = pd.concat([history, observed])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            combined = combined.dropna()
            variances = self._variance_path(combined.to_numpy(), params)
            positions = {date: i for i, date in enumerate(combined.index)}
            for date in pd.DatetimeIndex(block.index):
                idx = positions.get(date)
                if idx is None or idx < 30:
                    continue
                out.loc[(date, ticker)] = self._forecast_from(variances[idx], params)
        return out

    def coefficients(self) -> dict[str, float]:
        flat: dict[str, float] = {}
        for ticker, (omega, alpha, beta) in self.params_.items():
            flat[f"{ticker}:omega"] = omega
            flat[f"{ticker}:alpha"] = alpha
            flat[f"{ticker}:beta"] = beta
        return flat


VOLATILITY_MODELS = (EWMAVolatility, HARVolatility, GARCH11Volatility)

__all__ = [
    "EWMAVolatility",
    "GARCH11Volatility",
    "HARVolatility",
    "VOLATILITY_MODELS",
]
