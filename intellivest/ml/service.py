"""ML forecast service: the production bridge to the validated models.

The research pipeline in :mod:`intellivest.forecasting` establishes *which*
models survive a purged walk-forward evaluation. This service applies the same
models at a single decision date, with a cheap **inner** time split so that a
candidate is only used when it beats the baseline on data the decision date has
not seen.

Design notes
------------
* Horizon is 21 trading days, matching the evaluation report.
* Candidates are deliberately small: Ridge for returns, log-HAR for
  volatility. ElasticNet and LightGBM are evaluated offline but are too slow
  (and, in the current sample, too unstable) for a per-snapshot call.
* The panel builder drops the last ``horizon`` dates because forward targets
  are unknown. Predictions therefore come from the *full* causal feature matrix
  at the snapshot date, not from the truncated panel.
* If anything fails, or history is too short, the service falls back to the
  audited simple estimators in :mod:`intellivest.quant.forecast` and records the
  reason. A failure must degrade the forecast, never fabricate one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from intellivest.forecasting.base import PanelData
from intellivest.forecasting.data import build_features, build_panel
from intellivest.forecasting.metrics import return_metrics, volatility_metrics
from intellivest.forecasting.regime import infer_regime
from intellivest.forecasting.return_models import RidgeForecast, ShrunkMeanForecast
from intellivest.forecasting.vol_models import EWMAVolatility, HARVolatility

HORIZON = 21
# A candidate must beat its baseline by this much on the inner validation
# window before it is allowed to drive production numbers.
RETURN_EDGE = 0.01       # rank IC
VOL_EDGE = 0.01          # QLIKE (absolute; lower is better)
MIN_DATES = 140
SANE_RETURN = (-0.35, 0.35)
SANE_VOL = (1e-4, 2.0)


@dataclass
class MLForecastResult:
    expected_returns: pd.Series
    volatility_forecast: pd.Series
    regime: str
    regime_confidence: float
    weights: dict[str, float]
    validation_metrics: dict[str, dict[str, float]]
    model_identifiers: list[str]
    rejected: list[dict[str, str]]
    attribution: list[dict[str, object]]
    disagreement: float
    uncertainty: float
    horizon: int = HORIZON
    forecast_units: str = "annualized_return"
    fallbacks: list[str] = field(default_factory=list)


def _finite_metrics(values: dict[str, float]) -> dict[str, float]:
    """Keep only finite numbers: the trace is serialised to JSON."""
    return {
        key: float(value)
        for key, value in values.items()
        if isinstance(value, (int, float)) and np.isfinite(value)
    }


def _inner_split(
    panel: PanelData, horizon: int, val_fraction: float = 0.25
) -> tuple[PanelData, PanelData] | None:
    """Last ``val_fraction`` of dates as validation, purged by ``horizon``."""
    dates = panel.dates
    n = len(dates)
    if n < MIN_DATES:
        return None
    cut = max(MIN_DATES // 2, int(n * (1.0 - val_fraction)))
    train_cut = cut - horizon - 1
    if train_cut < 60:
        return None
    positions = np.arange(n)
    train = panel.take_dates(positions < train_cut)
    validation = panel.take_dates(positions >= cut)
    if train.X.empty or validation.X.empty:
        return None
    return train, validation


def _latest_cross_section(features: pd.DataFrame) -> PanelData | None:
    """One-row-per-ticker panel at the most recent feature date."""
    if features.empty:
        return None
    dates = features.index.get_level_values("date")
    latest = dates.max()
    mask = dates == latest
    X = features.loc[mask]
    if X.empty:
        return None
    y = pd.Series(np.nan, index=X.index, dtype=float)
    return PanelData(X=X, y=y, tickers=sorted(X.index.get_level_values("ticker").unique()))


def _to_ticker_index(series: pd.Series) -> pd.Series:
    """Drop the date level from ``(date, ticker)`` predictions."""
    out = series.copy()
    if isinstance(out.index, pd.MultiIndex):
        out.index = pd.Index(
            out.index.get_level_values("ticker"), name="ticker"
        )
    return out


class MLForecastService:
    def __init__(self, seed: int = 12345):
        self.seed = seed

    # ------------------------------------------------------------------ api
    def forecast(self, snapshot, settings) -> MLForecastResult:
        returns = snapshot.returns.dropna(how="all").fillna(0.0)
        prices = snapshot.prices
        tickers = [str(t) for t in returns.columns]
        rejected: list[dict[str, str]] = []
        metrics: dict[str, dict[str, float]] = {}
        fallbacks: list[str] = []

        # Regime: causal HMM when possible, snapshot label otherwise.
        try:
            benchmark = returns["SPX"] if "SPX" in returns else returns.iloc[:, 0]
            regime, regime_confidence = infer_regime(
                benchmark, random_state=self.seed
            )
        except Exception as exc:  # pragma: no cover - defensive
            rejected.append({"model": "regime_hmm", "reason": str(exc)})
            regime, regime_confidence = snapshot.regime, 0.5

        expected_returns: pd.Series
        vol_forecast: pd.Series
        attribution: list[dict[str, object]] = []
        descriptions: list[str] = []
        weights: dict[str, float] = {}

        try:
            features = build_features(returns, prices)
            latest = _latest_cross_section(features)
            if latest is None:
                raise ValueError("no usable feature date")
            if len(latest.X) < 3:
                raise ValueError("fewer than three assets in the latest cross-section")

            return_panel = build_panel(
                returns, prices, kind="return", horizon=HORIZON, features=features
            ).to_panel()
            vol_panel = build_panel(
                returns, prices, kind="volatility", horizon=HORIZON, features=features
            ).to_panel()

            # -- returns ---------------------------------------------------
            return_name, return_model = self._select_return_model(
                return_panel, metrics, rejected
            )
            raw_return = _to_ticker_index(return_model.predict(latest)).astype(float)
            annualised = (1.0 + raw_return.clip(-0.5, 0.5)) ** (
                float(settings.trading_days) / HORIZON
            ) - 1.0
            expected_returns = annualised.clip(*SANE_RETURN)
            descriptions.append(f"returns:{return_name}")
            weights[f"returns:{return_name}"] = 1.0
            if return_name != "shrunk_mean":
                attribution = self._attribution(return_model)

            # -- volatility ------------------------------------------------
            vol_name, vol_model = self._select_vol_model(vol_panel, metrics, rejected)
            raw_vol = _to_ticker_index(vol_model.predict(latest)).astype(float)
            vol_forecast = raw_vol.clip(*SANE_VOL)
            descriptions.append(f"volatility:{vol_name}")
            weights[f"volatility:{vol_name}"] = 1.0
        except Exception as exc:
            # Never fabricate: fall back to the audited simple estimators.
            rejected.append({"model": "forecasting_panel", "reason": str(exc)})
            fallbacks.append(
                "insufficient or unusable history; used the simple quantitative "
                "forecast (shrunk mean + EWMA)"
            )
            expected_returns, vol_forecast = self._simple_fallback(
                returns, settings
            )
            descriptions = ["returns:shrunk_mean_simple", "volatility:ewma_simple"]
            weights = {name: 1.0 for name in descriptions}

        if expected_returns.index.has_duplicates:
            expected_returns = expected_returns.groupby(level=0).last()
        # Align to the snapshot universe; never introduce a ticker that the
        # optimizer does not know about.
        expected_returns = expected_returns.reindex(tickers).fillna(0.0)
        vol_forecast = vol_forecast.reindex(tickers)
        fallback_vol = returns.std(ddof=1) * np.sqrt(settings.trading_days)
        vol_forecast = vol_forecast.fillna(fallback_vol).fillna(1e-3)

        disagreement = float(np.std(expected_returns.to_numpy(dtype=float)))
        uncertainty = float(np.mean(vol_forecast.to_numpy(dtype=float)))

        return MLForecastResult(
            expected_returns=expected_returns,
            volatility_forecast=vol_forecast,
            regime=regime,
            regime_confidence=float(regime_confidence),
            weights=weights,
            validation_metrics=metrics,
            model_identifiers=descriptions,
            rejected=rejected,
            attribution=attribution,
            disagreement=disagreement,
            uncertainty=uncertainty,
            horizon=HORIZON,
            fallbacks=fallbacks,
        )

    # ------------------------------------------------------------- selection
    def _select_return_model(
        self,
        panel: PanelData,
        metrics: dict[str, dict[str, float]],
        rejected: list[dict[str, str]],
    ) -> tuple[str, object]:
        split = _inner_split(panel, HORIZON)
        if split is None:
            raise ValueError("not enough history for an inner return split")
        train, validation = split

        baseline: object = ShrunkMeanForecast().fit(train)
        base_pred = baseline.predict(validation)
        base_metrics = _finite_metrics(
            return_metrics(validation.y, base_pred, horizon=HORIZON)
        )
        metrics["returns:shrunk_mean"] = base_metrics
        base_ic = base_metrics.get("rank_ic", float("-inf"))

        candidate: object = RidgeForecast(horizon=HORIZON).fit(train)
        cand_pred = candidate.predict(validation)
        cand_metrics = _finite_metrics(
            return_metrics(validation.y, cand_pred, horizon=HORIZON)
        )
        metrics["returns:ridge"] = cand_metrics
        cand_ic = cand_metrics.get("rank_ic", float("-inf"))
        dispersion = cand_metrics.get("dispersion", 0.0)

        # In-sample IC of the same candidate. A model that only wins because it
        # memorised the inner training window is exactly the instability that
        # flagged Ridge in the walk-forward report.
        train_metrics = _finite_metrics(
            return_metrics(train.y, candidate.predict(train), horizon=HORIZON)
        )
        train_ic = train_metrics.get("rank_ic", float("nan"))
        gap = float(train_ic - cand_ic) if np.isfinite(train_ic) else float("nan")

        if not np.isfinite(cand_ic) or dispersion <= 1e-12:
            rejected.append(
                {"model": "ridge", "reason": "degenerate inner-validation forecast"}
            )
        elif cand_ic <= 0.0:
            rejected.append(
                {
                    "model": "ridge",
                    "reason": f"inner validation rank IC {cand_ic:.3f} is not positive",
                }
            )
        elif np.isfinite(gap) and gap > 0.10:
            rejected.append(
                {
                    "model": "ridge",
                    "reason": (
                        f"inner train-to-validation rank IC gap {gap:.3f} is too "
                        "large (limit 0.10)"
                    ),
                }
            )
        elif cand_ic - base_ic >= RETURN_EDGE:
            # Refit the winner on everything available at this date.
            return "ridge", RidgeForecast(horizon=HORIZON).fit(panel)
        else:
            rejected.append(
                {
                    "model": "ridge",
                    "reason": (
                        f"inner validation rank IC {cand_ic:.3f} does not beat "
                        f"baseline {base_ic:.3f} by {RETURN_EDGE:.2f}"
                    ),
                }
            )
        return "shrunk_mean", ShrunkMeanForecast().fit(panel)

    def _select_vol_model(
        self,
        panel: PanelData,
        metrics: dict[str, dict[str, float]],
        rejected: list[dict[str, str]],
    ) -> tuple[str, object]:
        split = _inner_split(panel, HORIZON)
        if split is None:
            raise ValueError("not enough history for an inner volatility split")
        train, validation = split

        baseline: object = EWMAVolatility().fit(train)
        base_metrics = _finite_metrics(
            volatility_metrics(validation.y, baseline.predict(validation))
        )
        metrics["volatility:ewma"] = base_metrics
        base_qlike = base_metrics.get("qlike", float("inf"))

        candidate: object = HARVolatility(horizon=HORIZON).fit(train)
        cand_metrics = _finite_metrics(
            volatility_metrics(validation.y, candidate.predict(validation))
        )
        metrics["volatility:har"] = cand_metrics
        cand_qlike = cand_metrics.get("qlike", float("inf"))

        if np.isfinite(cand_qlike) and cand_qlike < base_qlike - VOL_EDGE:
            return "har", HARVolatility(horizon=HORIZON).fit(panel)
        rejected.append(
            {
                "model": "har",
                "reason": (
                    f"inner validation QLIKE {cand_qlike:.4f} does not improve on "
                    f"EWMA {base_qlike:.4f} by {VOL_EDGE:.2f}"
                ),
            }
        )
        return "ewma", EWMAVolatility().fit(panel)

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _attribution(model: object, limit: int = 10) -> list[dict[str, object]]:
        try:
            coefs = model.coefficients()  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive
            return []
        rows = [
            {
                "factor": str(name),
                "importance": float(abs(value)),
                "coef": float(value),
                "method": "ridge_coefficients",
            }
            for name, value in coefs.items()
            if np.isfinite(value)
        ]
        rows.sort(key=lambda row: row["importance"], reverse=True)
        return rows[:limit]

    @staticmethod
    def _simple_fallback(returns: pd.DataFrame, settings) -> tuple[pd.Series, pd.Series]:
        try:
            from intellivest.quant.forecast import expected_returns, expected_volatility

            return (
                expected_returns(returns, periods=settings.trading_days),
                expected_volatility(returns, periods=settings.trading_days),
            )
        except Exception:  # pragma: no cover - last resort
            mean = returns.mean() * settings.trading_days
            vol = returns.std(ddof=1) * np.sqrt(settings.trading_days)
            return mean.fillna(0.0), vol.fillna(1e-3)
