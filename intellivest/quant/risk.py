"""Risk measurement.

Conventions
-----------
* Returns are simple daily returns.
* ``var_95`` / ``cvar_95`` are reported as **positive loss magnitudes**
  (0.021 == a 2.1% one-day loss threshold). This avoids sign confusion when
  the numbers are rendered into prose or compared against suitability limits.
* ``max_drawdown`` is negative (0.0 == no drawdown, -0.25 == a 25% peak-to-
  trough loss).
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

from intellivest.contracts import RiskMetrics


def as_array(values: np.ndarray | pd.Series | list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def portfolio_returns(weights: Mapping[str, float], returns: pd.DataFrame) -> pd.Series:
    """Weighted daily return series, aligned to the columns present in both."""
    common = [ticker for ticker in weights if ticker in returns.columns]
    if not common:
        raise ValueError("no overlap between weights and return columns")
    matrix = returns[common].to_numpy(dtype=float)
    vector = np.array([weights[t] for t in common], dtype=float)
    total = vector.sum()
    if total <= 0:
        raise ValueError("weights must have positive total")
    return pd.Series(matrix @ vector, index=returns.index, name="portfolio")


def cumulative(series: pd.Series) -> pd.Series:
    return (1.0 + series.fillna(0.0)).cumprod()


def annualized_return(series: pd.Series, periods: int = 252) -> float:
    values = as_array(series)
    if values.size == 0:
        return 0.0
    growth = float(np.prod(1.0 + values))
    if growth <= 0:
        return -1.0
    return growth ** (periods / values.size) - 1.0


def annualized_volatility(series: pd.Series, periods: int = 252) -> float:
    values = as_array(series)
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1) * math.sqrt(periods))


def sharpe_ratio(series: pd.Series, risk_free: float, periods: int = 252) -> float:
    volatility = annualized_volatility(series, periods)
    if volatility < 1e-9:
        return 0.0
    return (annualized_return(series, periods) - risk_free) / volatility


def downside_deviation(series: pd.Series, risk_free: float, periods: int = 252) -> float:
    values = as_array(series)
    if values.size == 0:
        return 0.0
    hurdle = (1.0 + risk_free) ** (1.0 / periods) - 1.0
    downside = np.minimum(values - hurdle, 0.0)
    return float(math.sqrt(np.mean(downside**2)) * math.sqrt(periods))


def sortino_ratio(series: pd.Series, risk_free: float, periods: int = 252) -> float:
    deviation = downside_deviation(series, risk_free, periods)
    if deviation < 1e-9:
        return 0.0
    return (annualized_return(series, periods) - risk_free) / deviation


def drawdown_series(series: pd.Series) -> pd.Series:
    equity = cumulative(series)
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(drawdown_series(series).min())


def historical_var(series: pd.Series, level: float = 0.95) -> float:
    """One-day historical VaR as a positive loss magnitude."""
    values = as_array(series)
    if values.size == 0:
        return 0.0
    quantile = float(np.quantile(values, 1.0 - level))
    return max(0.0, -quantile)


def historical_cvar(series: pd.Series, level: float = 0.95) -> float:
    """Expected shortfall beyond the VaR threshold, positive loss magnitude."""
    values = as_array(series)
    if values.size == 0:
        return 0.0
    quantile = float(np.quantile(values, 1.0 - level))
    tail = values[values <= quantile]
    if tail.size == 0:
        return max(0.0, -quantile)
    return max(0.0, -float(tail.mean()))


def parametric_var(series: pd.Series, level: float = 0.95) -> float:
    values = as_array(series)
    if values.size < 2:
        return 0.0
    mean = float(np.mean(values))
    sigma = float(np.std(values, ddof=1))
    z = {0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263}.get(round(level, 2), 1.6449)
    return max(0.0, -(mean - z * sigma))


def beta(portfolio: pd.Series, benchmark: pd.Series) -> float:
    frame = pd.concat([portfolio, benchmark], axis=1, join="inner").dropna()
    if len(frame) < 3:
        return 0.0
    cov = np.cov(frame.iloc[:, 0], frame.iloc[:, 1], ddof=1)
    market_var = cov[1, 1]
    return float(cov[0, 1] / market_var) if market_var > 1e-12 else 0.0


def tracking_error(portfolio: pd.Series, benchmark: pd.Series, periods: int = 252) -> float:
    frame = pd.concat([portfolio, benchmark], axis=1, join="inner").dropna()
    if len(frame) < 3:
        return 0.0
    active = frame.iloc[:, 0] - frame.iloc[:, 1]
    return float(np.std(active, ddof=1) * math.sqrt(periods))


def herfindahl(weights: Mapping[str, float]) -> float:
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    return float(sum((w / total) ** 2 for w in weights.values() if w > 0))


def effective_n(weights: Mapping[str, float]) -> float:
    hhi = herfindahl(weights)
    return float(1.0 / hhi) if hhi > 0 else 0.0


def risk_metrics(
    weights: Mapping[str, float],
    returns: pd.DataFrame,
    benchmark: pd.Series,
    *,
    risk_free: float = 0.0,
    periods: int = 252,
) -> RiskMetrics:
    series = portfolio_returns(weights, returns)
    return RiskMetrics(
        annual_return=annualized_return(series, periods),
        annual_volatility=annualized_volatility(series, periods),
        sharpe=sharpe_ratio(series, risk_free, periods),
        sortino=sortino_ratio(series, risk_free, periods),
        var_95=historical_var(series, 0.95),
        cvar_95=historical_cvar(series, 0.95),
        max_drawdown=max_drawdown(series),
        beta=beta(series, benchmark),
        effective_n=effective_n(weights),
        herfindahl=herfindahl(weights),
        tracking_error=tracking_error(series, benchmark, periods),
    )


def detect_regime(benchmark: pd.Series, periods: int = 252) -> tuple[str, float]:
    """Volatility-and-trend regime label with a stability window.

    Uses only realised data up to the caller's ``as_of`` because the caller
    passes an already-truncated series.
    """
    values = as_array(benchmark)
    if values.size < 63:
        return "elevated", 0.4
    short_vol = float(np.std(values[-21:], ddof=1) * math.sqrt(periods))
    long_vol = float(np.std(values[-252:], ddof=1) * math.sqrt(periods)) if values.size >= 252 else short_vol
    trend = float(np.prod(1.0 + values[-63:]) - 1.0)
    ratio = short_vol / long_vol if long_vol > 1e-9 else 1.0
    if ratio > 1.35 or trend < -0.10:
        label = "stress"
    elif ratio > 1.10 or trend < -0.03:
        label = "elevated"
    else:
        label = "calm"
    confidence = float(np.clip(abs(ratio - 1.0) / 0.5 + abs(trend), 0.3, 0.95))
    return label, confidence


__all__ = [
    "annualized_return",
    "annualized_volatility",
    "beta",
    "cumulative",
    "detect_regime",
    "downside_deviation",
    "drawdown_series",
    "effective_n",
    "herfindahl",
    "historical_cvar",
    "historical_var",
    "max_drawdown",
    "parametric_var",
    "portfolio_returns",
    "risk_metrics",
    "sharpe_ratio",
    "sortino_ratio",
    "tracking_error",
]
