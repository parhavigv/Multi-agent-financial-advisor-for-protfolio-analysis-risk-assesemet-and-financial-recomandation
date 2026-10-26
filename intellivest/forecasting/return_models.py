"""Return-forecast models: three baselines and three candidates.

Baselines
---------
* ``shrunk_mean``  -  James-Stein-style shrink of per-ticker historical means.
* ``momentum_12_1``  -  the canonical 12-1 cross-sectional momentum signal,
  calibrated to return units on the training window.
* ``one_over_n``  -  no cross-sectional view (the forecast implied by holding the
  equal-weighted portfolio); all assets receive the training grand mean.

Candidates
----------
* ``ridge``  -  L2-regularised linear model on causal cross-sectional features.
* ``elastic_net``  -  L1/L2 mixture, useful when many features are redundant.
* ``lightgbm``  -  shallow gradient boosting, captures non-linear interactions.

Hyper-parameters are chosen by an **inner** time split of each training fold
(with the same purge as the outer CV), never on the validation or test split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import ForecastModel, PanelData


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _split_train_inner(
    train: PanelData, horizon: int, val_fraction: float = 0.2
) -> tuple[PanelData, PanelData]:
    """Last ``val_fraction`` of training dates, with a purge of ``horizon``."""
    dates = train.dates
    if len(dates) < 120:
        return train, train
    cut = int(len(dates) * (1.0 - val_fraction))
    cut = max(60, min(cut, len(dates) - 30))
    inner_train_dates = dates[: max(1, cut - horizon)]
    inner_val_dates = dates[cut:]
    return _subset(train, inner_train_dates), _subset(train, inner_val_dates)


def _subset(panel: PanelData, dates: pd.DatetimeIndex) -> PanelData:
    mask = panel.X.index.get_level_values("date").isin(pd.DatetimeIndex(dates))
    return PanelData(X=panel.X.loc[mask], y=panel.y.loc[mask], tickers=panel.tickers)


def _clip_targets(y: pd.Series) -> tuple[pd.Series, tuple[float, float]]:
    """Winsorise the training target at 1%/99% to limit outlier leverage."""
    low, high = float(y.quantile(0.01)), float(y.quantile(0.99))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return y, (-np.inf, np.inf)
    return y.clip(low, high), (low, high)


class _LinearForecast(ForecastModel):
    """Shared plumbing for standardised linear models."""

    kind = "return"
    uses_features = True

    def __init__(self, horizon: int = 21) -> None:
        self.horizon = int(horizon)
        self.scaler = None
        self.model = None
        self.bounds: tuple[float, float] = (-np.inf, np.inf)
        self.feature_names_: list[str] = []
        self.alpha_: float | None = None

    # subclasses implement _new_model(alpha) and optionally _grid()
    def _new_model(self, alpha: float):  # pragma: no cover - abstract
        raise NotImplementedError

    def _grid(self) -> list[float]:
        return [1.0]

    def fit(self, train: PanelData) -> "_LinearForecast":
        from sklearn.preprocessing import StandardScaler

        X = train.X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        y_clipped, self.bounds = _clip_targets(train.y.astype(float))
        y = y_clipped.to_numpy(dtype=float)
        self.feature_names_ = list(train.X.columns)

        inner_train, inner_val = _split_train_inner(train, horizon=self.horizon)
        if len(inner_val.X) > 0 and len(inner_train.X) > 50:
            Xtr = inner_train.X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
            ytr = inner_train.y.astype(float).to_numpy()
            Xva = inner_val.X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
            yva = inner_val.y.astype(float).to_numpy()
            best = (np.inf, self._grid()[0])
            for alpha in self._grid():
                try:
                    scaler = StandardScaler().fit(Xtr)
                    model = self._new_model(alpha).fit(scaler.transform(Xtr), ytr)
                    pred = model.predict(scaler.transform(Xva))
                    err = float(np.sqrt(np.mean((yva - pred) ** 2)))
                except Exception:
                    continue
                if np.isfinite(err) and err < best[0]:
                    best = (err, alpha)
            self.alpha_ = best[1]
        else:
            self.alpha_ = self._grid()[0]

        self.scaler = StandardScaler().fit(X)
        self.model = self._new_model(self.alpha_).fit(self.scaler.transform(X), y)
        return self

    def predict(self, predict: PanelData) -> pd.Series:
        if self.model is None or self.scaler is None or predict.X.empty:
            return pd.Series(dtype=float, index=predict.X.index)
        X = predict.X.reindex(columns=self.feature_names_, fill_value=0.0)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        out = self.model.predict(self.scaler.transform(X))
        low, high = self.bounds
        if np.isfinite(low) and np.isfinite(high):
            out = np.clip(out, low, high)
        return pd.Series(out, index=predict.X.index, name="prediction")

    def coefficients(self) -> dict[str, float]:
        try:
            coefs = np.asarray(self.model.coef_, dtype=float).ravel()
            return {name: float(c) for name, c in zip(self.feature_names_, coefs)}
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


class ShrunkMeanForecast(ForecastModel):
    """Per-ticker mean shrunk toward the cross-sectional mean."""

    name = "shrunk_mean"
    kind = "return"
    uses_features = False

    def __init__(self, shrinkage: float = 0.5) -> None:
        self.shrinkage = float(shrinkage)
        self.means_: pd.Series = pd.Series(dtype=float)
        self.grand_mean_: float = 0.0

    def fit(self, train: PanelData) -> "ShrunkMeanForecast":
        if train.y.empty:
            self.means_ = pd.Series(dtype=float)
            self.grand_mean_ = 0.0
            return self
        per_asset = train.y.groupby(level="ticker").mean()
        self.grand_mean_ = float(per_asset.mean())
        self.means_ = self.grand_mean_ + (1.0 - self.shrinkage) * (
            per_asset - self.grand_mean_
        )
        return self

    def predict(self, predict: PanelData) -> pd.Series:
        tickers = predict.X.index.get_level_values("ticker")
        values = np.array(
            [float(self.means_.get(t, self.grand_mean_)) for t in tickers], dtype=float
        )
        return pd.Series(values, index=predict.X.index, name="prediction")

    def coefficients(self) -> dict[str, float]:
        return {str(k): float(v) for k, v in self.means_.items()}


class MomentumForecast(ForecastModel):
    """12-1 momentum, calibrated to return units with a pooled OLS slope."""

    name = "momentum_12_1"
    kind = "return"
    uses_features = True
    SIGNAL = "cs_mom_12_1"

    def __init__(self) -> None:
        self.intercept_: float = 0.0
        self.slope_: float = 0.0

    def fit(self, train: PanelData) -> "MomentumForecast":
        if train.y.empty or self.SIGNAL not in train.X.columns:
            return self
        x = train.X[self.SIGNAL].to_numpy(dtype=float)
        y = train.y.to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 30:
            self.intercept_ = float(np.nanmean(y)) if len(y) else 0.0
            self.slope_ = 0.0
            return self
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        self.slope_, self.intercept_ = float(slope), float(intercept)
        return self

    def predict(self, predict: PanelData) -> pd.Series:
        if self.SIGNAL not in predict.X.columns:
            return pd.Series(0.0, index=predict.X.index)
        x = predict.X[self.SIGNAL].to_numpy(dtype=float)
        return pd.Series(
            self.intercept_ + self.slope_ * x, index=predict.X.index, name="prediction"
        )

    def coefficients(self) -> dict[str, float]:
        return {"intercept": float(self.intercept_), self.SIGNAL: float(self.slope_)}


class OneOverNForecast(ForecastModel):
    """No cross-sectional view: every asset gets the equal-weight expectation.

    This is the forecast analogue of a 1/N portfolio. Its rank IC is undefined
    by construction (all predictions are identical) and is reported as 0.0 with
    an explanatory note rather than silently dropped.
    """

    name = "one_over_n"
    kind = "return"
    uses_features = False

    def __init__(self) -> None:
        self.mean_: float = 0.0

    def fit(self, train: PanelData) -> "OneOverNForecast":
        self.mean_ = float(train.y.mean()) if not train.y.empty else 0.0
        return self

    def predict(self, predict: PanelData) -> pd.Series:
        return pd.Series(self.mean_, index=predict.X.index, name="prediction")

    def coefficients(self) -> dict[str, float]:
        return {"mean": float(self.mean_)}


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


class RidgeForecast(_LinearForecast):
    name = "ridge"

    def _new_model(self, alpha: float):
        from sklearn.linear_model import Ridge

        return Ridge(alpha=float(alpha))

    def _grid(self) -> list[float]:
        return [0.1, 1.0, 10.0, 100.0, 1000.0]


class ElasticNetForecast(_LinearForecast):
    name = "elastic_net"

    def __init__(self, horizon: int = 21, l1_ratio: float = 0.5) -> None:
        super().__init__(horizon=horizon)
        self.l1_ratio_ = float(l1_ratio)

    def _new_model(self, alpha: float):
        from sklearn.linear_model import ElasticNet

        # ``random`` coordinate descent with a looser tolerance keeps the inner
        # grid affordable; the near-unregularised alphas converge far too slowly
        # to be useful on a 20k-row panel and are omitted.
        return ElasticNet(
            alpha=float(alpha),
            l1_ratio=self.l1_ratio_,
            max_iter=2000,
            tol=1e-3,
            selection="random",
            random_state=12345,
        )

    def _grid(self) -> list[float]:
        return [1e-3, 1e-2, 1e-1]


class LightGBMForecast(ForecastModel):
    name = "lightgbm"
    kind = "return"
    uses_features = True

    def __init__(
        self,
        *,
        horizon: int = 21,
        n_estimators: int = 400,
        learning_rate: float = 0.03,
        num_leaves: int = 7,
        random_state: int = 12345,
    ) -> None:
        self.horizon = int(horizon)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.num_leaves = int(num_leaves)
        self.random_state = int(random_state)
        self.model = None
        self.feature_names_: list[str] = []

    def fit(self, train: PanelData) -> "LightGBMForecast":
        try:
            from lightgbm import LGBMRegressor, early_stopping
        except Exception as exc:  # pragma: no cover - dependency guard
            raise ImportError(f"lightgbm is required: {exc}") from exc

        X = train.X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = train.y.astype(float)
        self.feature_names_ = list(X.columns)

        inner_train, inner_val = _split_train_inner(train, horizon=self.horizon)
        Xtr = inner_train.X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ytr = inner_train.y.astype(float)
        fit_kwargs: dict[str, object] = {}
        if len(inner_val.X) > 0 and len(Xtr) > 50:
            Xva = inner_val.X.reindex(columns=self.feature_names_, fill_value=0.0)
            Xva = Xva.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            yva = inner_val.y.astype(float)
            fit_kwargs = {
                "eval_set": [(Xva, yva)],
                "callbacks": [early_stopping(50, verbose=False)],
            }
        model = LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=3,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=self.random_state,
            deterministic=True,
            force_col_wise=True,
            verbose=-1,
        )
        try:
            model.fit(X, y, **fit_kwargs)
        except TypeError:
            model.fit(X, y)
        self.model = model
        return self

    def predict(self, predict: PanelData) -> pd.Series:
        if self.model is None or predict.X.empty:
            return pd.Series(dtype=float, index=predict.X.index)
        X = predict.X.reindex(columns=self.feature_names_, fill_value=0.0)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out = self.model.predict(X)
        return pd.Series(out, index=predict.X.index, name="prediction")

    def coefficients(self) -> dict[str, float]:
        try:
            importances = np.asarray(self.model.feature_importances_, dtype=float)
            return {name: float(v) for name, v in zip(self.feature_names_, importances)}
        except Exception:
            return {}


BASELINES = (ShrunkMeanForecast, MomentumForecast, OneOverNForecast)
CANDIDATES = (RidgeForecast, ElasticNetForecast, LightGBMForecast)


__all__ = [
    "BASELINES",
    "CANDIDATES",
    "ElasticNetForecast",
    "LightGBMForecast",
    "MomentumForecast",
    "OneOverNForecast",
    "RidgeForecast",
    "ShrunkMeanForecast",
]
