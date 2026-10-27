"""Agent base classes and grounding rules.

Every agent returns an :class:`AgentReport` with typed signals and dated
evidence. A report is *grounded* before it leaves the agent:

* evidence dated after ``as_of`` is rejected (no look-ahead),
* signal subjects must exist in the universe,
* confidence is clamped, metrics must be finite,
* an optional LLM path is validated and, on any failure, replaced by the
  deterministic analyst with ``degraded=True`` rather than propagating an
  unverified claim.

This is the mechanism that lets the system use LLM reasoning without letting it
contradict the numerical engine.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Iterable

from intellivest.config import Settings
from intellivest.contracts import AgentReport, Evidence, InvestorProfile, Signal
from intellivest.data.market import MarketSnapshot
from intellivest.data.universe import TICKERS, asset_class_of
from intellivest.personalization.suitability import policy_for
from intellivest.quant.engine import QuantFeatures


@dataclass(frozen=True)
class AgentContext:
    profile: InvestorProfile
    snapshot: MarketSnapshot
    features: QuantFeatures
    current_weights: dict[str, float]
    settings: Settings

    def fact_sheet(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Compact, numeric view handed to an LLM, when enabled.

        The sheet includes portfolio-level and policy facts so a risk or
        portfolio analyst can cite the quantities it actually reasoned about.
        When there are no current holdings the portfolio block uses an equal
        weight assumption and says so explicitly.
        """
        features = self.features
        per_asset = {}
        for ticker in features.tickers:
            per_asset[ticker] = {
                "expected_return": round(float(features.expected_returns.get(ticker, 0.0)), 4),
                "volatility": round(float(features.volatility_forecast.get(ticker, 0.0)), 4),
                "momentum_12_1": round(float(features.momentum.get(ticker, 0.0)), 4),
                "cvar_95_daily": round(float(features.per_asset_cvar.get(ticker, 0.0)), 4),
                "max_drawdown": round(float(features.per_asset_drawdown.get(ticker, 0.0)), 4),
                "beta": round(float(features.betas.get(ticker, 0.0)), 3),
            }

        current = {
            ticker: float(weight)
            for ticker, weight in self.current_weights.items()
            if weight > 1e-9
        }
        assumed_equal = not current
        weights = (
            current
            if current
            else {ticker: 1.0 / len(features.tickers) for ticker in features.tickers}
        )
        metrics = features.metrics_for(weights, self.snapshot.returns, self.settings)
        policy = policy_for(self.profile)
        benchmark = features.benchmark_metrics

        payload: dict[str, Any] = {
            "as_of": self.snapshot.as_of.isoformat(),
            "regime": features.regime,
            "regime_confidence": round(features.regime_confidence, 3),
            "risk_class": self.profile.suitability_class,
            "goal": self.profile.goal,
            "horizon_years": self.profile.horizon_years,
            "loss_tolerance_pct": self.profile.loss_tolerance_pct,
            "risk_free": round(float(features.risk_free), 4),
            "n_assets": len(features.tickers),
            "exclusions": self.profile.exclusions,
            "assets": per_asset,
            "portfolio": {
                "assumed_equal_weight": assumed_equal,
                "current_positions": len(current),
                "current_weights": {
                    ticker: round(weight, 4)
                    for ticker, weight in sorted(current.items())
                },
                "equity_weight": round(
                    sum(
                        weight
                        for ticker, weight in current.items()
                        if asset_class_of(ticker) == "equity"
                    ),
                    4,
                ),
                "annual_return": round(metrics.annual_return, 4),
                "annual_volatility": round(metrics.annual_volatility, 4),
                "sharpe": round(metrics.sharpe, 4),
                "cvar_95_daily": round(metrics.cvar_95, 6),
                "max_drawdown": round(metrics.max_drawdown, 4),
                "beta": round(metrics.beta, 4),
                "effective_n": round(metrics.effective_n, 4),
                "herfindahl": round(metrics.herfindahl, 4),
            },
            "policy": {
                "risk_class": policy.risk_class,
                "max_equity": round(policy.max_equity, 4),
                "max_volatility": round(policy.max_volatility, 4),
                "max_cvar": round(policy.max_cvar, 6),
                "min_defensive": round(policy.min_defensive, 4),
                "max_single": round(policy.max_single, 4),
                "min_effective_n": round(policy.min_effective_n, 2),
            },
            "benchmark": {
                "annual_return": round(benchmark.annual_return, 4),
                "annual_volatility": round(benchmark.annual_volatility, 4),
                "sharpe": round(benchmark.sharpe, 4),
                "sortino": round(benchmark.sortino, 4),
                "cvar_95_daily": round(benchmark.cvar_95, 6),
                "max_drawdown": round(benchmark.max_drawdown, 4),
                "beta": round(benchmark.beta, 4),
                "effective_n": round(benchmark.effective_n, 4),
            },
        }
        if extra:
            payload.update(extra)
        return payload


class AnalystAgent(ABC):
    name: str = "analyst"
    role: str = "Generic analyst"
    coordination_weight: float = 1.0

    @abstractmethod
    def analyze(self, ctx: AgentContext) -> AgentReport:
        """Produce a grounded report for the context."""

    # -- helpers ----------------------------------------------------------
    def _report(
        self,
        ctx: AgentContext,
        *,
        summary: str,
        signals: Iterable[Signal] | None = None,
        evidence: Iterable[Evidence] | None = None,
        metrics: dict[str, float] | None = None,
        confidence: float = 0.6,
        degraded: bool = False,
    ) -> AgentReport:
        cleaned_evidence = [
            item for item in (evidence or []) if item.as_of <= ctx.snapshot.as_of
        ]
        cleaned_signals = [s for s in (signals or []) if _valid_subject(s.subject)]
        clean_metrics = {
            key: round(float(value), 6)
            for key, value in (metrics or {}).items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        return AgentReport(
            agent=self.name,
            role=self.role,
            as_of=ctx.snapshot.as_of,
            summary=_truncate(summary, 1200),
            confidence=round(float(min(1.0, max(0.0, confidence))), 4),
            signals=cleaned_signals,
            evidence=cleaned_evidence,
            metrics=clean_metrics,
            degraded=degraded,
        )

    def _with_llm(
        self, ctx: AgentContext, deterministic: Callable[[], AgentReport]
    ) -> AgentReport:
        """Use a validated LLM report, otherwise the deterministic analyst.

        A deterministic fallback is marked ``degraded`` when an LLM was
        configured, so the decision trace distinguishes a normal deterministic
        run from a failed model call.
        """
        report = self._llm_report(ctx)
        if report is not None:
            return report
        result = deterministic()
        from intellivest.agents.llm import llm_config

        if llm_config(ctx.settings) is not None and not result.degraded:
            result = result.model_copy(update={"degraded": True})
        return result

    def _llm_report(self, ctx: AgentContext, extra: dict[str, Any] | None = None) -> AgentReport | None:
        """Optional structured LLM path. Returns ``None`` on any failure."""
        from intellivest.agents.llm import complete_json, llm_config
        from intellivest.agents.schema import build_prompt, validate_llm_report

        config = llm_config(ctx.settings)
        if config is None:
            return None
        try:
            facts = ctx.fact_sheet(extra)
            system, user = build_prompt(self.role, self.name, facts)
            payload = complete_json(config, system, user)
            report = validate_llm_report(
                payload,
                agent=self.name,
                role=self.role,
                as_of=ctx.snapshot.as_of,
                fact_sheet=facts,
            )
        except Exception:
            # Any schema, citation, grounding or transport failure falls back
            # to the deterministic analyst. The wrapper marks the report
            # degraded when an LLM was actually configured.
            return None
        return report if report.signals else None


def _valid_subject(subject: str) -> bool:
    return subject in TICKERS or subject in {"MARKET", "PORTFOLIO"}


def directional_signal(
    *,
    name: str,
    subject: str,
    strength: float,
    rationale: str,
    confidence: float = 0.6,
) -> Signal:
    """Build a schema-valid signal from a signed score.

    Scores inside the neutral band are zeroed so that ``neutral`` never
    carries material conviction (enforced by ``Signal`` validation).
    """
    score = max(-1.0, min(1.0, float(strength)))
    if abs(score) <= 0.12:
        direction = "neutral"
        score = 0.0
    elif score > 0:
        direction = "bullish"
    else:
        direction = "bearish"
    return Signal(
        name=name,
        subject=subject,
        direction=direction,
        strength=round(score, 4),
        rationale=_truncate(rationale, 400),
        confidence=round(max(0.0, min(1.0, confidence)), 4),
    )


def _truncate(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


__all__ = ["AgentContext", "AnalystAgent", "directional_signal"]
