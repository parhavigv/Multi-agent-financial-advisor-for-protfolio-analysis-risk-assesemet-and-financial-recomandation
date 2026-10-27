"""Risk-budget analyst.

Converts measured portfolio risk into bounded, directional signals that tell
the coordinator when to de-risk, and which positions carry the tail.
"""

from __future__ import annotations

from intellivest.agents.base import AgentContext, AnalystAgent, directional_signal
from intellivest.contracts import AgentReport, Evidence, Signal
from intellivest.personalization.suitability import policy_for, reference_allocation
from intellivest.quant import risk as qrisk


class RiskAnalysisAgent(AnalystAgent):
    name = "risk_analysis"
    role = "Portfolio risk-budget and tail-risk analyst"
    coordination_weight = 1.2

    def analyze(self, ctx: AgentContext) -> AgentReport:
        return self._with_llm(ctx, lambda: self._deterministic(ctx))

    def _deterministic(self, ctx: AgentContext) -> AgentReport:
        policy = policy_for(ctx.profile)
        weights = {t: w for t, w in ctx.current_weights.items() if w > 1e-9}
        if not weights:
            weights = reference_allocation(ctx.profile, ctx.features.tickers)
        metrics = ctx.features.metrics_for(weights, ctx.snapshot.returns, ctx.settings)
        volatility = metrics.annual_volatility
        cvar = metrics.cvar_95
        drawdown = abs(metrics.max_drawdown)

        vol_ratio = volatility / policy.max_volatility if policy.max_volatility else 1.0
        cvar_ratio = cvar / policy.max_cvar if policy.max_cvar else 1.0
        dd_ratio = drawdown / max(ctx.profile.loss_tolerance_pct, 1e-6)
        excess = max(vol_ratio, cvar_ratio, dd_ratio, 1.0) - 1.0
        # Above policy -> de-risk; comfortably inside -> mild risk-on allowance.
        headroom = 1.0 - max(vol_ratio, cvar_ratio, dd_ratio)
        score = -min(1.0, excess * 0.9) if excess > 0 else min(0.35, headroom * 0.6)

        signals: list[Signal] = [
            directional_signal(
                name="risk_budget",
                subject="PORTFOLIO",
                strength=score,
                rationale=(
                    f"Volatility {volatility:.1%} vs limit {policy.max_volatility:.1%}; "
                    f"CVaR95 {cvar:.2%} vs limit {policy.max_cvar:.2%}; "
                    f"drawdown {drawdown:.1%} vs tolerance {ctx.profile.loss_tolerance_pct:.1%}."
                ),
                confidence=0.8,
            )
        ]

        # Tail-risk ranking: penalise the assets contributing the most daily CVaR.
        cvar_series = ctx.features.per_asset_cvar
        median = float(cvar_series.median()) if not cvar_series.empty else 0.0
        risk_sensitivity = max(0.15, (4 - policy.risk_class) / 3.0)
        for ticker in ctx.features.tickers:
            asset_cvar = float(cvar_series.get(ticker, 0.0))
            relative = (asset_cvar / median - 1.0) if median > 1e-9 else 0.0
            signals.append(
                directional_signal(
                    name="tail_risk",
                    subject=ticker,
                    strength=-risk_sensitivity * max(-1.0, min(1.5, relative)),
                    rationale=(
                        f"Daily CVaR95 {asset_cvar:.2%} versus universe median {median:.2%}; "
                        f"risk-class sensitivity {risk_sensitivity:.2f}."
                    ),
                    confidence=0.65,
                )
            )

        summary = (
            f"Portfolio volatility {volatility:.1%}, CVaR95 {cvar:.2%} per day, max drawdown "
            f"{drawdown:.1%}. Risk class {policy.risk_class} permits volatility up to "
            f"{policy.max_volatility:.1%} and CVaR up to {policy.max_cvar:.2%}. "
            + ("Board is inside budget." if excess <= 0 else "Risk budget is breached; de-risking suggested.")
        )
        evidence = [
            Evidence(
                source="Suitability policy",
                as_of=ctx.snapshot.as_of,
                claim=f"Class {policy.risk_class}: vol<={policy.max_volatility:.1%}, "
                f"CVaR<={policy.max_cvar:.2%}, max single {policy.max_single:.0%}",
            ),
            Evidence(
                source="QuantEngine.risk",
                as_of=ctx.snapshot.as_of,
                claim=f"Realised portfolio vol {volatility:.1%}, CVaR {cvar:.2%}, drawdown {drawdown:.1%}",
            ),
        ]
        return self._report(
            ctx,
            summary=summary,
            signals=signals,
            evidence=evidence,
            metrics={
                "portfolio_volatility": volatility,
                "portfolio_cvar": cvar,
                "portfolio_drawdown": drawdown,
                "vol_ratio": vol_ratio,
                "cvar_ratio": cvar_ratio,
            },
            confidence=0.78,
        )


__all__ = ["RiskAnalysisAgent"]
