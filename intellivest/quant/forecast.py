"""Return and volatility forecasts.

Deliberately simple and auditable: shrunk historical means, cross-sectional
momentum, and EWMA volatility. The survey's point is not that complex models
are unavailable, but that an advisory system needs *validated* numerical
inputs rather than numbers produced by prose. These estimators are easy to
backtest and easy to explain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_volatility(returns: pd.DataFrame, halflife: int = 32, periods: int = 252) -> pd.Series:
    """Annualised EWMA volatility per asset (last observation)."""
    clean = returns.dropna(how="all").fillna(0.0)
    variance = clean.pow(2).ewm(halflife=halflife, adjust=False).mean().iloc[-1]
    return np.sqrt(variance.clip(lower=0.0) * periods)


def shrunk_mean_returns(
    returns: pd.DataFrame, periods: int = 252, shrinkage: float = 0.50
) -> pd.Series:
    """James–Stein-style shrink of annualised sample means toward the cross-section."""
    clean = returns.dropna(how="all").fillna(0.0)
    means = clean.mean() * periods
    grand_mean = float(means.mean())
    return grand_mean + (1.0 - shrinkage) * (means - grand_mean)


def momentum(
    returns: pd.DataFrame, lookback: int = 252, skip: int = 21
) -> pd.Series:
    """12-1 momentum: cumulative return over ``lookback`` excluding ``skip`` days."""
    clean = returns.dropna(how="all").fillna(0.0)
    if len(clean) <= skip + 5:
        return pd.Series(0.0, index=clean.columns)
    window = clean.iloc[-(lookback + skip): -skip] if len(clean) > lookback + skip else clean.iloc[:-skip]
    return (1.0 + window).prod() - 1.0


def realized_volatility(
    returns: pd.DataFrame, window: int = 63, periods: int = 252
) -> pd.Series:
    clean = returns.dropna(how="all").fillna(0.0).iloc[-window:]
    return clean.std(ddof=1) * np.sqrt(periods)


def cross_sectional_zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    std = float(values.std(ddof=0))
    if std < 1e-12:
        return pd.Series(0.0, index=values.index)
    return (values - float(values.mean())) / std


def expected_returns(
    returns: pd.DataFrame,
    *,
    periods: int = 252,
    shrinkage: float = 0.50,
    momentum_weight: float = 0.35,
    momentum_scale: float = 0.10,
    cap: float = 0.35,
) -> pd.Series:
    """Blend of shrunk historical mean and momentum, bounded for sanity.

    The cap exists because unconstrained sample means produce absurd
    allocations; the survey documents estimation error as the central failure
    mode of classical optimization [5], [6].
    """
    base = shrunk_mean_returns(returns, periods, shrinkage)
    mom = momentum(returns)
    mom_component = cross_sectional_zscore(mom) * momentum_scale
    blended = (1.0 - momentum_weight) * base + momentum_weight * mom_component
    return blended.clip(-cap, cap)


def expected_volatility(returns: pd.DataFrame, periods: int = 252) -> pd.Series:
    return ewma_volatility(returns, periods=periods)


__all__ = [
    "cross_sectional_zscore",
    "ewma_volatility",
    "expected_returns",
    "expected_volatility",
    "momentum",
    "realized_volatility",
    "shrunk_mean_returns",
]
