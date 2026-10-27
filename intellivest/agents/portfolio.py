"""Portfolio-structure analyst.

This agent looks at what the investor already holds: concentration, drift from
a class-appropriate anchor, and the gap to a diversified allocation. It is the
only agent that can propose trades motivated purely by portfolio structure
rather than market views.
"""

from __future__ import annotations

from intellivest.agents.base import AgentContext, AnalystAgent, directional_signal
from intellivest.contracts import AgentReport, Evidence, Signal
from intellivest.personalization.suitability import policy_for, reference_allocation
from intellivest.quant import risk


class PortfolioAnalysisAgent(AnalystAgent):
    name = "portfolio_analysis"
    role = "Current-holdings, concentration and drift analyst"
    coordination_weight = 0.8

    def analyze(self, ctx: AgentContext) -> AgentReport:
        return self._with_llm(ctx, lambda: self._deterministic(ctx))

    def _deterministic(self, ctx: AgentContext) -> AgentReport:
        current = {t: w for t, w in ctx.current_weights.items() if w > 1e-9}
        reference = reference_allocation(ctx.profile, ctx.features.tickers)
        policy = policy_for(ctx.profile)
        signals: list[Signal] = []

        if not current:
            summary = (
                "No existing holdings. The agent abstains from drift signals and defers to the "
                "optimizer's diversified starting point."
            )
            return self._report(
                ctx,
                summary=summary,
                signals=[
                    directional_signal(
                        name="portfolio_state",
                        subject="PORTFOLIO",
                        strength=0.0,
                        rationale="Fresh portfolio; no legacy positions to rebalance.",
                        confidence=0.4,
                    )
                ],
                evidence=[],
                confidence=0.4,
            )

        effective = risk.effective_n(current)
        hhi = risk.herfindahl(current)
        largest = max(current.items(), key=lambda pair: pair[1])

        for ticker in ctx.features.tickers:
            held = float(current.get(ticker, 0.0))
            target = float(reference.get(ticker, 0.0))
            drift = held - target
            if abs(drift) < 0.05:
                continue
            # Overweight -> trim (bearish tilt on adding more); underweight -> add.
            strength = max(-1.0, min(1.0, -drift * 2.2))
            signals.append(
                directional_signal(
                    name="allocation_drift",
                    subject=ticker,
                    strength=strength,
                    rationale=(
                        f"Held {held:.1%} versus class-appropriate anchor {target:.1%} "
                        f"(drift {drift:+.1%})."
                    ),
                    confidence=0.65,
                )
            )

        concentration_penalty = max(0.0, (1.0 / max(effective, 1e-6)) - 1.0 / policy.min_effective_n)
        signals.append(
            directional_signal(
                name="concentration",
                subject="PORTFOLIO",
                strength=-min(0.9, concentration_penalty * 4.0),
                rationale=(
                    f"Effective holdings {effective:.1f} (HHI {hhi:.3f}); largest position "
                    f"{largest[0]} at {largest[1]:.1%}."
                ),
                confidence=0.7,
            )
        )

        summary = (
            f"Existing portfolio has {len(current)} positions and {effective:.1f} effective "
            f"holdings; largest is {largest[0]} at {largest[1]:.1%}. "
            f"Signal logic trims positions above the class-{policy.risk_class} anchor and tops up "
            "the underweight sleeves."
        )
        evidence = [
            Evidence(
                source="Investor holdings",
                as_of=ctx.snapshot.as_of,
                claim=f"{ticker} held at {weight:.1%} versus anchor {reference.get(ticker, 0.0):.1%}",
                value=weight,
            )
            for ticker, weight in sorted(current.items(), key=lambda pair: -pair[1])[:4]
        ]
        return self._report(
            ctx,
            summary=summary,
            signals=signals,
            evidence=evidence,
            metrics={"effective_n": effective, "herfindahl": hhi, "positions": float(len(current))},
            confidence=0.7,
        )


__all__ = ["PortfolioAnalysisAgent"]
