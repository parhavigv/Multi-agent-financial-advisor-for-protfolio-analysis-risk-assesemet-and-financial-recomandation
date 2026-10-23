"""Attribution and counterfactual explanation.

Feature-level explanation answers "which model input mattered". An advisory
system also needs "which holding contributed this risk", "which agent moved the
decision", and "what would change if the investor were one risk class higher".
Those are computed here from the optimizer's actual inputs and outputs.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Mapping

import numpy as np
import pandas as pd


def variance_contributions(weights: Mapping[str, float], cov: pd.DataFrame) -> dict[str, float]:
    """Marginal risk contribution: w_i * (C w)_i / (w' C w), shares sum to 1."""
    tickers = [t for t in weights if t in cov.columns and weights[t] > 1e-9]
    if not tickers:
        return {}
    w = np.array([weights[t] for t in tickers], dtype=float)
    sigma = cov.loc[tickers, tickers].to_numpy(dtype=float)
    total = float(w @ sigma @ w)
    if total <= 1e-12:
        return {ticker: 0.0 for ticker in tickers}
    marginal = sigma @ w
    contributions = w * marginal / total
    return {ticker: round(float(value), 6) for ticker, value in zip(tickers, contributions)}


def variance_shapley(weights: Mapping[str, float], cov: pd.DataFrame) -> dict[str, float]:
    """Exact Shapley decomposition of portfolio variance.

    Value function ``v(S) = sum_{i,j in S} w_i w_j C_ij``. Shapley values sum to
    total portfolio variance, giving a contribution share that is fair even with
    correlated assets (unlike naive marginal attribution).
    """
    tickers = [t for t in weights if t in cov.columns and weights[t] > 1e-9]
    n = len(tickers)
    if n == 0:
        return {}
    if n > 12:
        return variance_contributions(weights, cov)
    w = np.array([weights[t] for t in tickers], dtype=float)
    sigma = cov.loc[tickers, tickers].to_numpy(dtype=float)
    full = np.zeros(n, dtype=float)

    def value(subset: tuple[int, ...]) -> float:
        if not subset:
            return 0.0
        index = list(subset)
        ww = w[index]
        return float(ww @ sigma[np.ix_(index, index)] @ ww)

    for i in range(n):
        others = [j for j in range(n) if j != i]
        for size in range(n):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for combo in combinations(others, size):
                gain = value(tuple(combo) + (i,)) - value(combo)
                full[i] += weight * gain
    total_variance = float(w @ sigma @ w)
    if total_variance <= 1e-12:
        return {ticker: 0.0 for ticker in tickers}
    shares = full / total_variance
    return {ticker: round(float(share), 6) for ticker, share in zip(tickers, shares)}


def return_contributions(weights: Mapping[str, float], expected: pd.Series) -> dict[str, float]:
    return {
        ticker: round(float(weight * float(expected.get(ticker, 0.0))), 6)
        for ticker, weight in weights.items()
        if abs(weight) > 1e-9
    }


def equity_share(weights: Mapping[str, float], asset_class_of) -> float:
    return float(sum(w for t, w in weights.items() if asset_class_of(t) == "equity"))


__all__ = [
    "equity_share",
    "return_contributions",
    "variance_contributions",
    "variance_shapley",
]
