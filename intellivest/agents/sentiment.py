"""News and sentiment analyst.

Scores finance-specific language rather than generic positivity, following the
Loughran–McDonald finding that general-purpose sentiment dictionaries misread
financial text. Every scored item retains its publication date and headline so
the evidence is auditable.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import timedelta

from intellivest.agents.base import AgentContext, AnalystAgent, directional_signal
from intellivest.contracts import AgentReport, Evidence, Signal
from intellivest.data.news import NEGATIVE_TERMS, NEUTRAL_TERMS, POSITIVE_TERMS

WINDOW_DAYS = 45
HALF_LIFE_DAYS = 18.0


class SentimentAnalysisAgent(AnalystAgent):
    name = "sentiment_analysis"
    role = "Financial news and disclosure sentiment analyst"
    coordination_weight = 0.7

    def analyze(self, ctx: AgentContext) -> AgentReport:
        return self._with_llm(ctx, lambda: self._deterministic(ctx))

    def _deterministic(self, ctx: AgentContext) -> AgentReport:
        as_of = ctx.snapshot.as_of
        cutoff = as_of - timedelta(days=WINDOW_DAYS)
        items = [item for item in ctx.snapshot.news if cutoff <= item.published <= as_of]
        if not items:
            return self._report(
                ctx,
                summary="No real news items inside the sentiment window; no textual view is asserted.",
                confidence=0.0,
                signals=[],
                evidence=[],
                metrics={"items_scored": 0.0, "news_available": 0.0},
            )

        weighted: dict[str, float] = defaultdict(float)
        mass: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        top_items: dict[str, tuple[float, str, object]] = {}
        macro_score = 0.0
        macro_mass = 0.0

        for item in items:
            age = (as_of - item.published).days
            weight = math.exp(-age / HALF_LIFE_DAYS)
            score = _score_headline(item.headline)
            if not item.tickers:
                macro_score += weight * score
                macro_mass += weight
                continue
            for ticker in item.tickers:
                weighted[ticker] += weight * score
                mass[ticker] += weight
                counts[ticker] += 1
                previous = top_items.get(ticker)
                if previous is None or weight * abs(score) > previous[0]:
                    top_items[ticker] = (weight * abs(score), item.headline, item.published)

        signals: list[Signal] = []
        for ticker in ctx.features.tickers:
            total_mass = mass.get(ticker, 0.0)
            if total_mass <= 0:
                signals.append(
                    directional_signal(
                        name="news_sentiment",
                        subject=ticker,
                        strength=0.0,
                        rationale="No tagged news in the window.",
                        confidence=0.2,
                    )
                )
                continue
            normalised = weighted[ticker] / math.sqrt(total_mass)
            strength = math.tanh(normalised * 1.1)
            item = top_items.get(ticker)
            rationale = (
                f"{counts[ticker]} tagged items over {WINDOW_DAYS} days; "
                f"recency-weighted score {normalised:+.2f}."
            )
            if item:
                rationale += f" Lead: “{item[1]}” ({item[2]})."
            signals.append(
                directional_signal(
                    name="news_sentiment",
                    subject=ticker,
                    strength=strength,
                    rationale=rationale,
                    confidence=min(0.85, 0.30 + 0.06 * counts[ticker]),
                )
            )

        if macro_mass > 0:
            macro = macro_score / math.sqrt(macro_mass)
            signals.append(
                directional_signal(
                    name="macro_sentiment",
                    subject="MARKET",
                    strength=math.tanh(macro * 0.9),
                    rationale=(
                        f"Macro commentary inside the window scores {macro:+.2f} "
                        "(recency weighted, finance lexicon)."
                    ),
                    confidence=min(0.8, 0.3 + 0.05 * len(items)),
                )
            )

        most_positive = max(
            ((t, float(weighted[t] / math.sqrt(mass[t]))) for t in mass if mass[t] > 0),
            key=lambda pair: pair[1],
            default=("none", 0.0),
        )
        most_negative = min(
            ((t, float(weighted[t] / math.sqrt(mass[t]))) for t in mass if mass[t] > 0),
            key=lambda pair: pair[1],
            default=("none", 0.0),
        )
        summary = (
            f"Scored {len(items)} items. Most positive tone: {most_positive[0]} "
            f"({most_positive[1]:+.2f}); most negative: {most_negative[0]} "
            f"({most_negative[1]:+.2f}). Sentiment is a bounded overlay, not a price forecast."
        )
        evidence = [
            Evidence(
                source="Market Wire",
                as_of=published,
                claim=headline,
            )
            for _, headline, published in list(top_items.values())[:5]
        ]
        return self._report(
            ctx,
            summary=summary,
            signals=signals,
            evidence=evidence,
            metrics={"items_scored": float(len(items))},
            confidence=min(0.8, 0.4 + 0.02 * len(items)),
        )


def _score_headline(text: str) -> float:
    lowered = text.lower()
    score = 0.0
    for term in POSITIVE_TERMS:
        if term in lowered:
            score += 1.0
    for term in NEGATIVE_TERMS:
        if term in lowered:
            score -= 1.0
    if score == 0.0:
        for term in NEUTRAL_TERMS:
            if term in lowered:
                return 0.0
    return max(-1.0, min(1.0, score))


__all__ = ["SentimentAnalysisAgent"]
