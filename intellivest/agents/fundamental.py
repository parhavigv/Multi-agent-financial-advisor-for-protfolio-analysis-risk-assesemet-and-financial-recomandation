"""Fundamental analyst: value, quality and revision signals."""

from __future__ import annotations

import math

import pandas as pd

from intellivest.agents.base import AgentContext, AnalystAgent, directional_signal
from intellivest.contracts import AgentReport, Evidence, Signal


class FundamentalAnalysisAgent(AnalystAgent):
    name = "fundamental_analysis"
    role = "Fundamental value, quality and earnings-revision analyst"
    coordination_weight = 0.9

    def analyze(self, ctx: AgentContext) -> AgentReport:
        return self._with_llm(ctx, lambda: self._deterministic(ctx))

    def _deterministic(self, ctx: AgentContext) -> AgentReport:
        fundamentals = ctx.snapshot.fundamentals
        if fundamentals is None or fundamentals.empty:
            return self._report(
                ctx,
                summary=(
                    "Fundamental data unavailable for this data provider; the agent abstains "
                    "rather than fabricating a view."
                ),
                signals=[],
                evidence=[],
                confidence=0.0,
                degraded=True,
            )

        frame = fundamentals.reindex(ctx.features.tickers).dropna(how="all")
        if frame.empty:
            return self._report(ctx, summary="No fundamental coverage.", confidence=0.0, degraded=True)

        value = _z(frame.get("pe", pd.Series(dtype=float)) * -1) + _z(
            frame.get("pb", pd.Series(dtype=float)) * -1
        )
        quality = (
            _z(frame.get("roe", pd.Series(dtype=float)))
            + _z(frame.get("debt_to_equity", pd.Series(dtype=float)) * -1)
            + _z(frame.get("quality_score", pd.Series(dtype=float)))
        )
        growth = _z(frame.get("earnings_growth", pd.Series(dtype=float))) + _z(
            frame.get("analyst_revision", pd.Series(dtype=float))
        )
        composite = 0.34 * value.fillna(0) + 0.41 * quality.fillna(0) + 0.25 * growth.fillna(0)

        signals: list[Signal] = []
        for ticker, score in composite.items():
            row = frame.loc[ticker]
            signals.append(
                directional_signal(
                    name="fundamental_composite",
                    subject=ticker,
                    strength=math.tanh(float(score) / 1.4),
                    rationale=(
                        f"Value {float(value.get(ticker, 0.0)):+.2f}, quality "
                        f"{float(quality.get(ticker, 0.0)):+.2f}, growth "
                        f"{float(growth.get(ticker, 0.0)):+.2f} (cross-sectional z-scores)."
                    ),
                    confidence=0.6,
                )
            )

        leaders = composite.sort_values(ascending=False).head(3)
        summary = (
            "Strongest fundamental composite: "
            + ", ".join(f"{ticker} ({score:+.2f})" for ticker, score in leaders.items())
            + ". Valuation is penalised rather than rewarded when it coincides with weak quality."
        )
        evidence = [
            Evidence(
                source="Fundamentals (latest quarter)",
                as_of=ctx.snapshot.as_of,
                claim=(
                    f"{ticker} PE {float(frame.loc[ticker].get('pe', 0.0)):.1f}, "
                    f"ROE {float(frame.loc[ticker].get('roe', 0.0)):.1%}, "
                    f"revision {float(frame.loc[ticker].get('analyst_revision', 0.0)):+.2f}"
                ),
                value=float(composite.get(ticker, 0.0)),
            )
            for ticker in leaders.index
        ]
        return self._report(
            ctx,
            summary=summary,
            signals=signals,
            evidence=evidence,
            metrics={
                "mean_composite": float(composite.mean()),
                "dispersion_composite": float(composite.std(ddof=0)),
            },
            confidence=0.62,
        )


def _z(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    std = float(values.std(ddof=0))
    if not math.isfinite(std) or std < 1e-12:
        return pd.Series(0.0, index=values.index)
    return (values - float(values.mean())) / std


__all__ = ["FundamentalAnalysisAgent"]
