"""Causal, leakage-safe feature engineering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSet:
    features: pd.DataFrame
    feature_names: list[str]
    as_of: pd.Timestamp | None = None


def _rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    m = x.rolling(window, min_periods=max(5, window // 2)).mean()
    s = x.rolling(window, min_periods=max(5, window // 2)).std(ddof=0)
    s = s.replace(0, np.nan)
    return ((x - m) / s).fillna(0.0)


def build_causal_features(returns: pd.DataFrame, prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build features using only past information."""
    if returns.empty:
        return pd.DataFrame(index=returns.index)
    feats = pd.DataFrame(index=returns.index)

    # momentum (lagged)
    for w in [5, 21, 63, 126]:
        if len(returns) > w:
            feats[f"mom_{w}"] = (1.0 + returns).rolling(w).prod() - 1.0
        else:
            feats[f"mom_{w}"] = 0.0
    # reversal
    feats["mom_1"] = returns.shift(1).fillna(0.0)
    # volatility
    for w in [21, 63]:
        feats[f"vol_{w}"] = returns.rolling(w).std(ddof=0).fillna(0.0)
    # MA ratios
    if prices is not None and not prices.empty:
        for w in [21, 63]:
            ma = prices.rolling(w).mean()
            feats[f"ma_ratio_{w}"] = (prices / ma.replace(0, np.nan) - 1.0).fillna(0.0).mean(axis=1)
    # cross-sectional z of momentum
    if "mom_21" in feats:
        cs = feats["mom_21"]
        feats["mom21_cs_z"] = ((cs - cs.mean()) / cs.std(ddof=0).replace(0, np.nan)).fillna(0.0)
    # lagged vol of vol proxy
    if "vol_21" in feats:
        feats["vol_chg"] = feats["vol_21"].diff().fillna(0.0)

    feats = feats.fillna(0.0)
    # shift by 1 to ensure no lookahead
    feats = feats.shift(1).fillna(0.0)
    return feats
