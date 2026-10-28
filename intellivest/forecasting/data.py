"""Causal feature and target construction for the forecasting evaluation.

All features at date ``t`` use information available at the close of ``t`` and
are shifted one day so they can only be used to predict returns after ``t``.
Targets are forward-looking *by construction* and are never shifted into the
feature matrix; the purge in :mod:`intellivest.forecasting.cv` prevents a
training row whose target window overlaps the prediction window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import PanelData, wide_to_long

# 12-1 momentum is the canonical cross-sectional equity signal; the shorter
# windows are included because the universe is multi-asset (bonds/gold behave
# differently from equities).
MOMENTUM_WINDOWS = (5, 21, 63, 126)
VOLATILITY_WINDOWS = (21, 63)
DRAWDOWN_WINDOWS = (63, 252)


@dataclass(frozen=True)
class FeatureTargetPanel:
    features: pd.DataFrame        # MultiIndex (date, ticker) x feature columns
    target: pd.Series             # MultiIndex (date, ticker)
    target_wide: pd.DataFrame     # date x ticker
    feature_names: list[str]

    def to_panel(self) -> PanelData:
        return PanelData(
            X=self.features,
            y=self.target,
            tickers=sorted(self.features.index.get_level_values("ticker").unique()),
        )


def _cross_sectional_z(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _empty(index: pd.Index, columns: list[str], value: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(value, index=index, columns=columns)


def build_feature_blocks(
    returns: pd.DataFrame, prices: pd.DataFrame | None = None
) -> dict[str, pd.DataFrame]:
    """Wide (date x ticker) causal feature blocks, **before** the one-day lag.

    Value proxies are price-based (long-horizon reversal and distance from the
    long moving average) because the ETF universe has no fundamental data. They
    are labelled ``value_*`` and documented as proxies, not book-to-market.
    """
    clean = returns.dropna(how="all").fillna(0.0)
    tickers = list(clean.columns)
    blocks: dict[str, pd.DataFrame] = {}

    for window in MOMENTUM_WINDOWS:
        blocks[f"mom_{window}"] = (1.0 + clean).rolling(window).apply(np.prod, raw=True) - 1.0

    # 12-1 momentum: 12-month return skipping the most recent month.
    blocks["mom_12_1"] = (1.0 + clean).rolling(252).apply(np.prod, raw=True).shift(21) - 1.0

    for window in VOLATILITY_WINDOWS:
        blocks[f"vol_{window}"] = clean.rolling(window).std(ddof=0) * np.sqrt(252.0)

    blocks["vol_chg"] = blocks["vol_21"] - blocks["vol_63"]
    blocks["mom_1"] = clean.shift(1)

    # Volatility-model inputs (all lagged): daily realised variance at 1/5/22
    # day windows and an annualised EWMA estimate. HAR and EWMA consume these
    # directly, which keeps their predictions causal without extra plumbing.
    blocks["ret_lag"] = clean.shift(1)
    blocks["rv_1"] = clean.pow(2).shift(1)
    blocks["rv_5"] = clean.pow(2).rolling(5).mean().shift(1)
    blocks["rv_22"] = clean.pow(2).rolling(22).mean().shift(1)
    blocks["ewma_vol"] = (
        clean.pow(2).ewm(halflife=32, adjust=False).mean().shift(1) * 252.0
    ).pow(0.5)

    if prices is not None and not prices.empty:
        aligned = prices.reindex(index=clean.index, columns=tickers).ffill()
        min_periods = 20
        for window in DRAWDOWN_WINDOWS:
            rolling_max = aligned.rolling(window, min_periods=min_periods).max()
            blocks[f"dd_{window}"] = aligned / rolling_max.replace(0.0, np.nan) - 1.0
        for window in (63, 252):
            moving_avg = aligned.rolling(window, min_periods=min_periods).mean()
            blocks[f"ma_gap_{window}"] = aligned / moving_avg.replace(0.0, np.nan) - 1.0
    else:
        for window in DRAWDOWN_WINDOWS:
            blocks[f"dd_{window}"] = _empty(clean.index, tickers)
        for window in (63, 252):
            blocks[f"ma_gap_{window}"] = _empty(clean.index, tickers)

    # Price-based value proxies: cheap relative to own long history.
    blocks["value_rev_252"] = -blocks["mom_12_1"]
    blocks["value_ma_gap_252"] = -blocks["ma_gap_252"]

    # Cross-sectional context: a model that only sees raw values cannot know
    # whether 8% momentum is high or low *for this date*.
    for source in ("mom_12_1", "vol_63", "dd_252", "value_rev_252"):
        blocks[f"cs_{source}"] = _cross_sectional_z(blocks[source])

    return blocks


def build_features(returns: pd.DataFrame, prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long (date, ticker) feature matrix, shifted one day to stay causal."""
    blocks = build_feature_blocks(returns, prices)
    # Shift each wide block by one day *before* stacking: a feature observed at
    # the close of t is only usable for predictions made after t.
    lagged = {name: wide_to_long(frame.shift(1)) for name, frame in blocks.items()}
    features = pd.concat(lagged, axis=1)
    features.columns = [str(name) for name in features.columns]
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features


def forward_return(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Simple cumulative return from t+1 to t+h (NaN where unavailable)."""
    clean = returns.dropna(how="all").fillna(0.0)
    gross = (1.0 + clean).rolling(horizon).apply(np.prod, raw=True)
    return gross.shift(-horizon) - 1.0


def forward_volatility(returns: pd.DataFrame, horizon: int, periods: int = 252) -> pd.DataFrame:
    """Annualised realised volatility over t+1 .. t+h."""
    clean = returns.dropna(how="all").fillna(0.0)
    realised = clean.rolling(horizon).std(ddof=0) * np.sqrt(periods)
    return realised.shift(-horizon)


def build_panel(
    returns: pd.DataFrame,
    prices: pd.DataFrame | None,
    *,
    kind: str = "return",
    horizon: int = 21,
    periods: int = 252,
    features: pd.DataFrame | None = None,
) -> FeatureTargetPanel:
    """Assemble a leakage-safe feature/target panel for returns or volatility.

    ``features`` may be supplied when the caller already built the causal
    feature matrix (the evaluation builds one panel per target kind).
    """
    features = build_features(returns, prices) if features is None else features.copy()
    if kind == "return":
        target_wide = forward_return(returns, horizon)
    elif kind == "volatility":
        target_wide = forward_volatility(returns, horizon, periods)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown panel kind: {kind}")

    # Drop the warm-up period: before the 12-1 momentum window (252 + 21
    # trading days) is populated, most features are zero-filled placeholders and
    # would only teach the models noise.
    warmup = 252 + 21
    clean_dates = pd.DatetimeIndex(sorted(returns.dropna(how="all").index.unique()))
    if len(clean_dates) > warmup:
        cutoff = clean_dates[warmup]
        dates = features.index.get_level_values("date")
        features = features.loc[dates >= cutoff]

    target = wide_to_long(target_wide).rename("target")
    valid = features.index.intersection(target.index)
    features = features.loc[valid]
    target = target.loc[valid]
    finite = target.replace([np.inf, -np.inf], np.nan).notna()
    features = features.loc[finite]
    target = target.loc[finite]

    return FeatureTargetPanel(
        features=features,
        target=target,
        target_wide=target_wide,
        feature_names=list(features.columns),
    )


__all__ = [
    "FeatureTargetPanel",
    "MOMENTUM_WINDOWS",
    "VOLATILITY_WINDOWS",
    "DRAWDOWN_WINDOWS",
    "build_feature_blocks",
    "build_features",
    "build_panel",
    "forward_return",
    "forward_volatility",
]
