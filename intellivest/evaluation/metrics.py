"""Evaluation metrics for advisory configurations."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from intellivest.quant import risk


@dataclass
class PerformanceSummary:
    annual_return: float
    annual_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    cvar_95: float
    calmar: float
    turnover_per_rebalance: float
    annual_cost_bps: float
    hit_rate: float
    rebalances: int
    average_equity_weight: float
    suitability_breach_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def summarise(
    daily_returns: pd.Series,
    *,
    risk_free: float,
    periods: int = 252,
    turnovers: list[float] | None = None,
    equity_weights: list[float] | None = None,
    breaches: list[bool] | None = None,
    cost_bps: float = 0.0,
) -> PerformanceSummary:
    series = daily_returns.dropna()
    annual_return = risk.annualized_return(series, periods)
    annual_vol = risk.annualized_volatility(series, periods)
    drawdown = risk.max_drawdown(series)
    years = max(len(series) / periods, 1e-9)
    mean_turnover = float(np.mean(turnovers)) if turnovers else 0.0
    # Turnover is per rebalance, so annualise by the *realised* rebalance
    # frequency rather than assuming a fixed calendar.
    rebalances_per_year = len(turnovers) / years if turnovers else 0.0
    annual_cost = mean_turnover * cost_bps * rebalances_per_year
    hit_rate = float((series > 0).mean()) if len(series) else 0.0
    return PerformanceSummary(
        annual_return=round(annual_return, 6),
        annual_volatility=round(annual_vol, 6),
        sharpe=round(risk.sharpe_ratio(series, risk_free, periods), 4),
        sortino=round(risk.sortino_ratio(series, risk_free, periods), 4),
        max_drawdown=round(drawdown, 6),
        cvar_95=round(risk.historical_cvar(series, 0.95), 6),
        calmar=round(annual_return / abs(drawdown), 4) if drawdown < -1e-9 else 0.0,
        turnover_per_rebalance=round(mean_turnover, 4),
        annual_cost_bps=round(annual_cost, 2),
        hit_rate=round(hit_rate, 4),
        rebalances=len(turnovers or []),
        average_equity_weight=round(float(np.mean(equity_weights)), 4) if equity_weights else 0.0,
        suitability_breach_rate=round(float(np.mean(breaches)), 4) if breaches else 0.0,
    )
