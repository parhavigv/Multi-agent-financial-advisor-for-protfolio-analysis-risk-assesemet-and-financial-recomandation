"""Black-Litterman blending of the quantitative prior with agent views.

The quantitative engine supplies the prior expected returns and covariance.
Each coordinated agent view is an absolute view on one asset:

    Q_i = mu_i + tilt_i * view_shift

where ``view_shift`` is capped annualised return conviction and the aggregate
portfolio view scales it. Confidence enters through the view covariance
``Omega``: a confident view gets a smaller Omega and moves the posterior more.
The total absolute movement is capped so agent narrative can never dominate the
quantitative forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from intellivest.data.universe import asset_class_of


@dataclass(frozen=True)
class BlackLittermanResult:
    prior: pd.Series
    posterior: pd.Series
    views: pd.Series
    view_returns: pd.Series
    omega: pd.Series
    confidence: pd.Series
    total_influence: float
    influence_capped: bool
    n_views: int


def posterior_expected_returns(
    prior: pd.Series,
    covariance: pd.DataFrame,
    views: Mapping[str, float],
    *,
    confidence: Mapping[str, float] | None = None,
    market_score: float = 0.0,
    market_confidence: float = 0.5,
    portfolio_score: float = 0.0,
    tau: float = 0.05,
    view_shift: float = 0.03,
    max_total_influence: float = 0.12,
    market_overlay: float = 0.25,
    min_confidence: float = 0.05,
) -> BlackLittermanResult:
    """Return the Black-Litterman posterior expected returns.

    ``views`` maps tickers to coordinated tilts in [-1, 1]. Tickers outside the
    covariance universe are ignored. A market view is added as an equal-weight
    equity-portfolio view when ``market_score`` is non-zero.
    """
    tickers = [ticker for ticker in covariance.columns if ticker in prior.index]
    if not tickers:
        empty = pd.Series(dtype=float)
        return BlackLittermanResult(
            prior=prior.copy(),
            posterior=prior.copy(),
            views=empty,
            view_returns=empty,
            omega=empty,
            confidence=empty,
            total_influence=0.0,
            influence_capped=False,
            n_views=0,
        )

    mu = prior.reindex(tickers).astype(float).fillna(0.0)
    sigma = covariance.loc[tickers, tickers].astype(float).to_numpy()
    sigma = np.nan_to_num((sigma + sigma.T) / 2.0, nan=0.0, posinf=0.0, neginf=0.0)
    # A tiny ridge keeps the linear system solvable when an asset has no
    # variance in a short or degenerate sample.
    sigma = sigma + np.eye(len(tickers)) * 1e-12

    conviction = float(np.clip(1.0 + 0.35 * float(portfolio_score), 0.70, 1.10))
    shift = max(0.0, float(view_shift)) * conviction
    confidence = confidence or {}

    rows: list[np.ndarray] = []
    view_returns: list[float] = []
    labels: list[str] = []
    confidences: list[float] = []
    raw_views: dict[str, float] = {}

    for index, ticker in enumerate(tickers):
        tilt = float(views.get(ticker, 0.0))
        if abs(tilt) < 1e-8:
            continue
        tilt = float(np.clip(tilt, -1.0, 1.0))
        row = np.zeros(len(tickers))
        row[index] = 1.0
        rows.append(row)
        view_returns.append(float(mu.iloc[index] + tilt * shift))
        labels.append(f"asset:{ticker}")
        confidences.append(
            float(np.clip(float(confidence.get(ticker, 0.5)), min_confidence, 0.99))
        )
        raw_views[ticker] = round(tilt, 4)

    if abs(float(market_score)) > 1e-8:
        equity = [ticker for ticker in tickers if asset_class_of(ticker) == "equity"]
        if equity:
            row = np.zeros(len(tickers))
            for ticker in equity:
                row[tickers.index(ticker)] = 1.0 / len(equity)
            rows.append(row)
            view_returns.append(
                float(row @ mu.to_numpy())
                + float(market_overlay) * float(market_score) * shift
            )
            labels.append("market:equity")
            raw_views["MARKET"] = round(float(market_score), 4)
            confidences.append(
                float(np.clip(float(market_confidence), min_confidence, 0.99))
            )

    if not rows:
        empty = pd.Series(dtype=float)
        return BlackLittermanResult(
            prior=mu.copy(),
            posterior=mu.copy(),
            views=empty,
            view_returns=empty,
            omega=empty,
            confidence=empty,
            total_influence=0.0,
            influence_capped=False,
            n_views=0,
        )

    pick = np.vstack(rows)
    q = np.asarray(view_returns, dtype=float)
    view_conf = np.asarray(confidences, dtype=float)
    tau_sigma = float(tau) * sigma

    # He-Litterman view covariance: Omega_ii = P_i tau Sigma P_i' (1-c)/c.
    base_variance = float(tau) * np.einsum("ij,jk,ik->i", pick, sigma, pick)
    base_variance = np.clip(base_variance, 1e-12, None)
    omega = base_variance * ((1.0 - view_conf) / view_conf)
    omega = np.clip(omega, 0.01 * base_variance, 20.0 * base_variance)

    system = pick @ tau_sigma @ pick.T + np.diag(omega)
    rhs = q - pick @ mu.to_numpy()
    try:
        delta = tau_sigma @ pick.T @ np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        delta = tau_sigma @ pick.T @ np.linalg.pinv(system) @ rhs
    delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

    # Hard bounds: no single asset moves by more than the stated view shift,
    # and the sum of absolute movements cannot exceed the global cap.
    if shift > 0:
        delta = np.clip(delta, -shift, shift)
    else:
        delta = np.zeros_like(delta)
    influence = float(np.abs(delta).sum())
    cap = max(0.0, float(max_total_influence))
    capped = False
    if influence > cap and influence > 0:
        delta = delta * (cap / influence)
        influence = cap
        capped = True

    posterior = mu + pd.Series(delta, index=tickers)
    return BlackLittermanResult(
        prior=mu.copy(),
        posterior=posterior,
        views=pd.Series(raw_views, dtype=float),
        view_returns=pd.Series(view_returns, index=labels, dtype=float),
        omega=pd.Series(omega, index=labels, dtype=float),
        confidence=pd.Series(view_conf, index=labels, dtype=float),
        total_influence=round(float(influence), 8),
        influence_capped=capped,
        n_views=len(labels),
    )


__all__ = ["BlackLittermanResult", "posterior_expected_returns"]
