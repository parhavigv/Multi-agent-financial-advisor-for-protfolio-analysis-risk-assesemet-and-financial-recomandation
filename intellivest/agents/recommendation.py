"""Recommendation agent.

Turns validated numbers into an investor-facing recommendation. The LLM, when
configured, is allowed to *phrase* the result — never to choose weights, invent
performance, or override a suitability decision. A numeric-grounding guard
rejects prose containing figures absent from the fact set.
"""

from __future__ import annotations

from intellivest.agents.base import AgentContext
from intellivest.agents.llm import build_narrator, numeric_tokens
from intellivest.config import Settings
from intellivest.contracts import (
    OptimizationResult,
    Recommendation,
    RiskMetrics,
    SuitabilityReport,
)
from intellivest.data.universe import asset_class_weights

DISCLAIMERS = [
    "Research prototype. Not personalised investment advice and not a solicitation.",
    "Backtested or synthetic performance does not guarantee future results.",
    "Allocations must be reviewed by a qualified adviser before acting.",
]


class RecommendationAgent:
    name = "recommendation"
    role = "Recommendation synthesis and investor communication"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.narrator = build_narrator(settings)

    def build(
        self,
        ctx: AgentContext,
        optimization: OptimizationResult,
        suitability: SuitabilityReport,
        metrics: RiskMetrics,
        *,
        agreement: float,
        degraded: bool,
        max_drawdown_estimate: float,
    ) -> Recommendation:
        weights = optimization.weights
        current = {t: w for t, w in ctx.current_weights.items() if w > 1e-9}
        changes = {
            ticker: round(weights.get(ticker, 0.0) - current.get(ticker, 0.0), 6)
            for ticker in sorted(set(weights) | set(current))
            if abs(weights.get(ticker, 0.0) - current.get(ticker, 0.0)) > 1e-4
        }
        classes = asset_class_weights(weights)
        class_text = ", ".join(f"{name} {weight:.0%}" for name, weight in sorted(classes.items()))
        headline = (
            f"Risk class {suitability.risk_class} {ctx.profile.goal.replace('_', ' ')} portfolio "
            f"with {metrics.annual_volatility:.1%} expected volatility"
        )

        draft = (
            f"Recommended allocation: {class_text}. "
            f"Expected annual return {optimization.expected_return:.1%}, expected volatility "
            f"{optimization.expected_volatility:.1%}, Sharpe {optimization.expected_sharpe:.2f}, "
            f"one-day CVaR95 {optimization.expected_cvar_95:.2%}, estimated max drawdown "
            f"{max_drawdown_estimate:.1%}. Rebalancing turnover is {optimization.turnover:.0%}, "
            f"about {optimization.turnover * self.settings.transaction_cost_bps:.1f} bps in cost. "
            +
            (
                "The allocation satisfies every mandatory suitability rule after "
                f"{suitability.repair_iterations} remediation step(s). "
                if suitability.passed and suitability.repaired
                else "The allocation satisfies every mandatory suitability rule. "
                if suitability.passed
                else "Residual suitability breaches remain and human review is required: "
                + "; ".join(suitability.violations)
                + ". "
            )
            + f"Specialist-agent agreement is {agreement:.0%}. "
            + (
                "Some analyst reports ran without the configured language model and fell back to "
                "the deterministic analyst. "
                if degraded
                else ""
            )
            + "The quantitative engine, not the language layer, produced every number above."
        )

        allowed = numeric_tokens(
            optimization.expected_return,
            optimization.expected_volatility,
            optimization.expected_sharpe,
            optimization.expected_cvar_95,
            optimization.turnover,
            optimization.turnover * self.settings.transaction_cost_bps / 10_000.0,
            metrics.annual_volatility,
            metrics.cvar_95,
            abs(metrics.max_drawdown),
            max_drawdown_estimate,
            agreement,
            float(suitability.risk_class),
            float(suitability.repair_iterations),
            float(ctx.profile.horizon_years),
            *weights.values(),
            *classes.values(),
        )
        # A narrator may legitimately date the advice.
        allowed.add(ctx.snapshot.as_of.isoformat())
        system = (
            "You are the communication layer of a regulated advisory system. Rewrite the DRAFT "
            "in 4-6 factual sentences for the investor. Use only the numbers in the draft. "
            "Do not add predictions, guarantees, or new figures."
        )
        prompt = f"Investor goal: {ctx.profile.goal}. Horizon: {ctx.profile.horizon_years} years.\nDRAFT:{draft}"
        try:
            rationale = self.narrator.narrate(system, prompt, allowed_numbers=allowed)
        except Exception:
            rationale = draft

        confidence = _confidence(agreement, suitability, degraded, optimization.solver_status)
        disclaimers = list(DISCLAIMERS)
        if self.settings.llm_provider == "deterministic":
            disclaimers.append("Narrative generated deterministically; no external LLM was called.")

        return Recommendation(
            investor_id=ctx.profile.investor_id,
            as_of=ctx.snapshot.as_of,
            headline=headline,
            rationale=rationale,
            weights=weights,
            allocation_changes=changes,
            expected_return=optimization.expected_return,
            expected_volatility=optimization.expected_volatility,
            expected_sharpe=optimization.expected_sharpe,
            cvar_95=optimization.expected_cvar_95,
            max_drawdown_estimate=max_drawdown_estimate,
            turnover=optimization.turnover,
            estimated_cost_bps=optimization.turnover * self.settings.transaction_cost_bps,
            suitability=suitability,
            confidence=confidence,
            agent_agreement=agreement,
            degraded=degraded,
            disclaimers=disclaimers,
        )


def _confidence(
    agreement: float, suitability: SuitabilityReport, degraded: bool, solver_status: str
) -> float:
    score = 0.45 + 0.35 * agreement
    if suitability.passed:
        score += 0.10
    if suitability.repaired:
        score -= 0.03
    if degraded:
        score -= 0.05
    if not solver_status.startswith("optimal"):
        score -= 0.10
    return round(max(0.05, min(0.97, score)), 4)


__all__ = ["RecommendationAgent", "DISCLAIMERS"]
