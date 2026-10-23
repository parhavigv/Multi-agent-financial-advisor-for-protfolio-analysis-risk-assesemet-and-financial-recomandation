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

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable

from intellivest.config import Settings
from intellivest.contracts import AgentReport, Evidence, InvestorProfile, Signal
from intellivest.data.market import MarketSnapshot
from intellivest.data.universe import TICKERS
from intellivest.quant.engine import QuantFeatures


@dataclass(frozen=True)
class AgentContext:
    profile: InvestorProfile
    snapshot: MarketSnapshot
    features: QuantFeatures
    current_weights: dict[str, float]
    settings: Settings

    def fact_sheet(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Compact, numeric-only view handed to an LLM, when enabled."""
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
        payload: dict[str, Any] = {
            "as_of": self.snapshot.as_of.isoformat(),
            "regime": features.regime,
            "regime_confidence": round(features.regime_confidence, 3),
            "risk_class": self.profile.suitability_class,
            "goal": self.profile.goal,
            "horizon_years": self.profile.horizon_years,
            "loss_tolerance_pct": self.profile.loss_tolerance_pct,
            "exclusions": self.profile.exclusions,
            "assets": per_asset,
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

    def _llm_report(self, ctx: AgentContext, extra: dict[str, Any] | None = None) -> AgentReport | None:
        """Optional LLM reasoning path. Returns ``None`` on any failure."""
        from intellivest.agents.llm import LLMUnavailable, complete_json, llm_config

        config = llm_config(ctx.settings)
        if config is None:
            return None
        system = (
            f"You are the {self.role} in a regulated investment-advisory pipeline. "
            "Use ONLY the numbers in the fact sheet. Do not invent data, prices, or dates. "
            "Return JSON with keys: summary (string), confidence (0-1), "
            "signals (list of {name, subject, direction, strength, rationale, confidence}), "
            "evidence (list of {source, as_of, claim, value}). "
            "direction must be bullish|bearish|neutral; strength in [-1,1] and must match direction. "
            f"Valid subjects: {', '.join(TICKERS)} or MARKET or PORTFOLIO. "
            "Evidence as_of must be on or before the fact-sheet as_of."
        )
        user = json.dumps(ctx.fact_sheet(extra), default=str)
        try:
            payload = complete_json(config, system, user)
        except LLMUnavailable:
            return None
        except Exception:
            return None

        try:
            signals = [Signal(**item) for item in payload.get("signals", [])]
            evidence = [Evidence(**item) for item in payload.get("evidence", [])]
            return self._report(
                ctx,
                summary=str(payload.get("summary") or "LLM analysis unavailable."),
                signals=signals,
                evidence=evidence,
                metrics=payload.get("metrics", {}),
                confidence=float(payload.get("confidence", 0.5)),
                degraded=False,
            )
        except Exception:
            return None


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
