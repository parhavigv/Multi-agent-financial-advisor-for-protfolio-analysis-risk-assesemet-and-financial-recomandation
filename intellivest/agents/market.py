"""Market and regime analyst."""

from __future__ import annotations

from intellivest.agents.base import AgentContext, AnalystAgent, directional_signal
from intellivest.contracts import AgentReport, Evidence, Signal


class MarketAnalysisAgent(AnalystAgent):
    name = "market_analysis"
    role = "Market conditions, trend and regime analyst"
    coordination_weight = 1.0

    def analyze(self, ctx: AgentContext) -> AgentReport:
        return self._with_llm(ctx, lambda: self._deterministic(ctx))

    def _deterministic(self, ctx: AgentContext) -> AgentReport:
        features = ctx.features
        signals: list[Signal] = [
            directional_signal(
                name="market_regime",
                subject="MARKET",
                strength=_regime_score(features.regime, features.regime_confidence),
                rationale=(
                    f"Regime classified as {features.regime} with "
                    f"{features.regime_confidence:.0%} persistence over the last quarter."
                ),
                confidence=0.55 + 0.3 * features.regime_confidence,
            )
        ]

        for ticker in features.tickers:
            momentum = float(features.momentum.get(ticker, 0.0))
            realised = max(float(features.realized_volatility.get(ticker, 0.0)), 1e-6)
            forecast = float(features.volatility_forecast.get(ticker, 0.0))
            vol_pressure = max(0.0, forecast / realised - 1.15)
            score = 0.95 * _tanh(momentum / 0.18) - 0.35 * vol_pressure
            signals.append(
                directional_signal(
                    name="trend_and_volatility",
                    subject=ticker,
                    strength=score,
                    rationale=(
                        f"12-1 momentum {momentum:+.1%}; forecast volatility "
                        f"{forecast:.1%} vs realised {realised:.1%}."
                    ),
                    confidence=0.6,
                )
            )

        benchmark = features.benchmark_metrics
        summary = (
            f"{features.regime.capitalize()} regime. Equal-weight benchmark: "
            f"{benchmark.annual_return:.1%} annualised return, {benchmark.annual_volatility:.1%} "
            f"volatility, {benchmark.max_drawdown:.1%} max drawdown. Cross-sectional momentum "
            "is the primary driver of the per-asset signals."
        )
        evidence = [
            Evidence(
                source="QuantEngine.regime",
                as_of=ctx.snapshot.as_of,
                claim=f"Regime {features.regime} (persistence {features.regime_confidence:.2f})",
            ),
            Evidence(
                source="QuantEngine.momentum",
                as_of=ctx.snapshot.as_of,
                claim="12-1 cross-sectional momentum computed from prices truncated at as_of",
            ),
        ]
        return self._report(
            ctx,
            summary=summary,
            signals=signals,
            evidence=evidence,
            metrics={
                "benchmark_return": benchmark.annual_return,
                "benchmark_volatility": benchmark.annual_volatility,
                "benchmark_max_drawdown": benchmark.max_drawdown,
                "regime_confidence": features.regime_confidence,
            },
            confidence=0.65,
        )


def _regime_score(regime: str, confidence: float) -> float:
    base = {"calm": 0.45, "elevated": -0.10, "stress": -0.75}.get(regime, 0.0)
    return base * max(0.4, confidence)


def _tanh(value: float) -> float:
    import math

    return math.tanh(value)


__all__ = ["MarketAnalysisAgent"]
