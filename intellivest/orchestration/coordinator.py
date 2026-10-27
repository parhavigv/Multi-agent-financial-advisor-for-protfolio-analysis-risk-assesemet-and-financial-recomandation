"""Agent coordination and conflict resolution.

The coordinator is deliberately not another LLM. It is a deterministic
aggregation layer that:

* weights each signal by agent credibility and stated confidence,
* measures dispersion across agents per asset,
* damps the aggregate tilt when agents disagree,
* records each agent's contribution for the audit trail.

Without this step, "multi-agent" would simply mean several opinions appended to
one report — the reliability problem the survey highlights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from intellivest.agents.base import AnalystAgent
from intellivest.contracts import AgentReport


@dataclass(frozen=True)
class CoordinatorDecision:
    tilts: dict[str, float]
    market_score: float
    portfolio_score: float
    agreement: float
    dispersion: float
    conflicts: list[str] = field(default_factory=list)
    per_agent_contribution: dict[str, float] = field(default_factory=dict)
    # Per-asset confidence and disagreement are consumed by the
    # Black-Litterman layer: confidence sets Omega, dispersion lowers it.
    view_confidence: dict[str, float] = field(default_factory=dict)
    view_dispersion: dict[str, float] = field(default_factory=dict)
    market_confidence: float = 0.5
    portfolio_confidence: float = 0.5

    @property
    def degraded(self) -> bool:
        return bool(self.conflicts)


class Coordinator:
    def __init__(self, agents: list[AnalystAgent]) -> None:
        self.weights = {agent.name: agent.coordination_weight for agent in agents}

    def coordinate(self, reports: list[AgentReport], tickers: list[str]) -> CoordinatorDecision:
        per_asset_scores: dict[str, dict[str, float]] = {ticker: {} for ticker in tickers}
        per_asset_confidence: dict[str, dict[str, float]] = {
            ticker: {} for ticker in tickers
        }
        market_num = market_den = 0.0
        market_conf_num = market_conf_den = 0.0
        portfolio_num = portfolio_den = 0.0
        portfolio_conf_num = portfolio_conf_den = 0.0
        contributions: dict[str, float] = {}

        for report in reports:
            agent_weight = self.weights.get(report.agent, 1.0)
            for signal in report.signals:
                credibility = agent_weight * (0.35 + 0.65 * signal.confidence)
                if signal.subject in per_asset_scores:
                    per_asset_scores[signal.subject][report.agent] = signal.strength
                    per_asset_confidence[signal.subject][report.agent] = signal.confidence
                    contributions[report.agent] = contributions.get(report.agent, 0.0) + abs(
                        credibility * signal.strength
                    )
                elif signal.subject == "MARKET":
                    market_num += credibility * signal.strength
                    market_den += credibility
                    market_conf_num += credibility * signal.confidence
                    market_conf_den += credibility
                elif signal.subject == "PORTFOLIO":
                    portfolio_num += credibility * signal.strength
                    portfolio_den += credibility
                    portfolio_conf_num += credibility * signal.confidence
                    portfolio_conf_den += credibility
                else:
                    contributions[report.agent] = contributions.get(report.agent, 0.0) + abs(
                        credibility * signal.strength
                    )

        tilts: dict[str, float] = {}
        view_confidence: dict[str, float] = {}
        view_dispersion: dict[str, float] = {}
        dispersions: list[float] = []
        conflicts: list[str] = []
        agreement_pairs: list[tuple[float, bool]] = []

        for ticker, scores in per_asset_scores.items():
            if not scores:
                continue
            values = np.array(list(scores.values()), dtype=float)
            weights = np.array(
                [
                    self.weights.get(agent, 1.0) * _confidence_lookup(reports, agent, ticker)
                    for agent in scores
                ],
                dtype=float,
            )
            total = float(weights.sum())
            mean = float((values * weights).sum() / total) if total > 0 else 0.0
            dispersion = float(np.sqrt(((values - mean) ** 2 * weights).sum() / total)) if total > 0 else 0.0
            dispersions.append(dispersion)

            # Explicit conflict rule: strong opposing views cancel and are logged.
            if values.max() - values.min() > 1.20:
                mean *= 0.40
                conflicts.append(f"{ticker}: opposing agent views cancelled")

            damping = 1.0 - 0.5 * min(1.0, dispersion)
            tilt = float(np.clip(mean * damping, -1.0, 1.0))
            tilts[ticker] = round(tilt, 4)

            # Confidence is averaged by credibility weight, then reduced by
            # disagreement. BL turns this into view uncertainty (Omega).
            confidence_weights = np.array(
                [self.weights.get(agent, 1.0) for agent in scores], dtype=float
            )
            confidences = np.array(
                [per_asset_confidence[ticker][agent] for agent in scores], dtype=float
            )
            confidence_total = float(confidence_weights.sum())
            base_confidence = (
                float((confidence_weights * confidences).sum() / confidence_total)
                if confidence_total > 0
                else 0.5
            )
            view_confidence[ticker] = round(
                float(np.clip(base_confidence * damping, 0.05, 0.99)), 4
            )
            view_dispersion[ticker] = round(dispersion, 4)

            for agent, value in scores.items():
                if abs(value) > 0.12:
                    agreement_pairs.append(
                        (self.weights.get(agent, 1.0), (value > 0) == (tilt > 0))
                    )

        agreement = (
            float(sum(w for w, ok in agreement_pairs if ok) / sum(w for w, _ in agreement_pairs))
            if agreement_pairs
            else 1.0
        )
        total_contribution = sum(contributions.values()) or 1.0
        contribution_share = {
            agent: round(value / total_contribution, 4) for agent, value in contributions.items()
        }

        return CoordinatorDecision(
            tilts=tilts,
            market_score=round(market_num / market_den, 4) if market_den > 0 else 0.0,
            portfolio_score=round(portfolio_num / portfolio_den, 4) if portfolio_den > 0 else 0.0,
            agreement=round(agreement, 4),
            dispersion=round(float(np.mean(dispersions)) if dispersions else 0.0, 4),
            conflicts=conflicts,
            per_agent_contribution=contribution_share,
            view_confidence=view_confidence,
            view_dispersion=view_dispersion,
            market_confidence=round(market_conf_num / market_conf_den, 4)
            if market_conf_den > 0
            else 0.5,
            portfolio_confidence=round(portfolio_conf_num / portfolio_conf_den, 4)
            if portfolio_conf_den > 0
            else 0.5,
        )


def _confidence_lookup(reports: list[AgentReport], agent: str, ticker: str) -> float:
    for report in reports:
        if report.agent != agent:
            continue
        for signal in report.signals:
            if signal.subject == ticker:
                return signal.confidence
        return report.confidence
    return 0.5


__all__ = ["Coordinator", "CoordinatorDecision"]
