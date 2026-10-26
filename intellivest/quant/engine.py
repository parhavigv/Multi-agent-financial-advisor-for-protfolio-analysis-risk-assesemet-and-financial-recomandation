"""Quantitative feature engine.

Computes every number the agents and the optimizer are allowed to use. This is
the boundary that keeps LLM reasoning from inventing financial quantities: the
agents receive a frozen ``QuantFeatures`` object built from data truncated at
``as_of``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping

import numpy as np
import pandas as pd

from intellivest.config import Settings
from intellivest.contracts import RiskMetrics
from intellivest.data.market import MarketSnapshot
from intellivest.data.universe import sector_of
from intellivest.quant import forecast, risk
from intellivest.quant.covariance import estimate_covariance

try:
    from intellivest.ml.service import MLForecastService
except Exception:  # pragma: no cover
    MLForecastService = None  # type: ignore

try:
    from intellivest.ml.service import MLForecastService
except Exception:  # pragma: no cover
    MLForecastService = None  # type: ignore


@dataclass(frozen=True)
class QuantFeatures:
    as_of: date
    tickers: list[str]
    covariance: pd.DataFrame
    expected_returns: pd.Series
    momentum: pd.Series
    volatility_forecast: pd.Series
    realized_volatility: pd.Series
    per_asset_cvar: pd.Series
    per_asset_drawdown: pd.Series
    betas: pd.Series
    correlation: pd.DataFrame
    regime: str
    regime_confidence: float
    benchmark_metrics: RiskMetrics
    covariance_method: str

    def metrics_for(self, weights: Mapping[str, float], returns: pd.DataFrame, settings: Settings) -> RiskMetrics:
        return risk.risk_metrics(
            weights,
            returns,
            returns["SPX"] if "SPX" in returns else returns.iloc[:, 0],
            risk_free=settings.risk_free_rate,
            periods=settings.trading_days,
        )


class QuantEngine:
    def __init__(self, settings: Settings, covariance_method: str = "ledoit_wolf", use_ml: bool = False) -> None:
        self.settings = settings
        self.covariance_method = covariance_method
        self.use_ml = use_ml
        self._ml_service = MLForecastService(settings.seed) if (use_ml and MLForecastService is not None) else None
        self.last_ml_forecast = None

    def build(self, snapshot: MarketSnapshot) -> QuantFeatures:
        returns = snapshot.returns.dropna(how="all").fillna(0.0)
        tickers = list(returns.columns)
        cov = estimate_covariance(
            returns,
            method=self.covariance_method,
            periods=self.settings.trading_days,
        )
        expected = forecast.expected_returns(returns, periods=self.settings.trading_days)
        mom = forecast.momentum(returns)
        vol_forecast = forecast.expected_volatility(returns, periods=self.settings.trading_days)
        realized_vol = forecast.realized_volatility(returns, periods=self.settings.trading_days)
        per_asset_cvar = pd.Series(
            {t: risk.historical_cvar(returns[t], 0.95) for t in tickers}
        )
        per_asset_dd = pd.Series({t: risk.max_drawdown(returns[t]) for t in tickers})
        bench = returns["SPX"] if "SPX" in returns else returns.iloc[:, 0]
        betas = pd.Series({t: risk.beta(returns[t], bench) for t in tickers})
        correlation = returns.corr().fillna(0.0)

        equal = pd.Series(1.0 / len(tickers), index=tickers)
        benchmark_metrics = risk.risk_metrics(
            equal.to_dict(),
            returns,
            bench,
            risk_free=self.settings.risk_free_rate,
            periods=self.settings.trading_days,
        )

        regime_label, regime_confidence = risk.detect_regime(bench, self.settings.trading_days)
        return QuantFeatures(
            as_of=snapshot.as_of,
            tickers=tickers,
            covariance=cov,
            expected_returns=expected,
            momentum=mom,
            volatility_forecast=vol_forecast,
            realized_volatility=realized_vol,
            per_asset_cvar=per_asset_cvar,
            per_asset_drawdown=per_asset_dd,
            betas=betas,
            correlation=correlation,
            regime=snapshot.regime if snapshot.regime else regime_label,
            regime_confidence=regime_confidence,
            benchmark_metrics=benchmark_metrics,
            covariance_method=self.covariance_method,
        )

    def sector_exposure(self, weights: Mapping[str, float]) -> dict[str, float]:
        totals: dict[str, float] = {}
        for ticker, weight in weights.items():
            sector = sector_of(ticker)
            totals[sector] = totals.get(sector, 0.0) + weight
        return totals

    def apply_tilts(
        self,
        features: QuantFeatures,
        tilts: Mapping[str, float],
        *,
        max_adjustment: float = 0.02,
    ) -> pd.Series:
        """Blend agent conviction into expected returns with a hard bound.

        ``max_adjustment`` caps the total annualised return adjustment so that
        narrative conviction can never dominate the quantitative estimate.
        """
        base = features.expected_returns.copy()
        for ticker, tilt in tilts.items():
            if ticker in base.index:
                bounded = float(np.clip(tilt, -1.0, 1.0)) * max_adjustment
                base.loc[ticker] = float(base.loc[ticker] + bounded)
        return base


__all__ = ["QuantFeatures", "QuantEngine"]
