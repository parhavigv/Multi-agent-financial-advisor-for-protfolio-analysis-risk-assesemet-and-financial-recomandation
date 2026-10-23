"""Date-stamped synthetic news corpus.

Headlines are generated from finance-specific language (in the spirit of
Loughran–McDonald) so the sentiment analyst must actually read text rather
than receive a pre-computed score. Every item carries a publication date so
look-ahead can be rejected structurally.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

POSITIVE_TERMS: tuple[str, ...] = (
    "beats expectations",
    "raises guidance",
    "record profit",
    "margin expansion",
    "upgrade",
    "strong demand",
    "dividend increase",
    "buyback announced",
    "outperform",
    "accelerating growth",
    "order win",
    "debt reduction",
)

NEGATIVE_TERMS: tuple[str, ...] = (
    "misses estimates",
    "cuts guidance",
    "profit warning",
    "margin pressure",
    "downgrade",
    "weak demand",
    "dividend cut",
    "regulatory probe",
    "underperform",
    "slowing growth",
    "litigation risk",
    "debt concerns",
)

NEUTRAL_TERMS: tuple[str, ...] = (
    "in line with expectations",
    "announces board meeting",
    "files routine disclosure",
    "schedules investor call",
    "appoints new director",
    "maintains outlook",
)

MACRO_TEMPLATES: tuple[tuple[str, int], ...] = (
    ("Central bank holds policy rate; markets await guidance", 0),
    ("Inflation print comes in above consensus", -1),
    ("Inflation eases for a third consecutive month", 1),
    ("Bond yields rise sharply on strong payroll data", -1),
    ("Bond yields retreat as growth concerns mount", 1),
    ("Regulators announce tighter disclosure norms for lenders", -1),
    ("Global risk appetite improves as volatility subsides", 1),
    ("Geopolitical tension lifts safe-haven demand", -1),
    ("Liquidity conditions tighten across credit markets", -1),
    ("Corporate earnings season begins with cautious optimism", 1),
)

COMPANY_TEMPLATES: tuple[str, ...] = (
    "{name} {term}",
    "{name} shares move after {term}",
    "Analysts note {name} {term}",
    "{name}: {term}",
    "Sector update — {name} {term}",
)


@dataclass(frozen=True)
class NewsItem:
    published: date
    headline: str
    tickers: tuple[str, ...]
    source: str
    tone: int = 0  # generator-side hint, never consumed by the analyst


def generate_news(
    *,
    start: date,
    end: date,
    tickers: tuple[str, ...],
    names: dict[str, str],
    rng: np.random.Generator,
    per_week: int = 10,
) -> list[NewsItem]:
    """Generate a reproducible, date-stamped news stream."""
    items: list[NewsItem] = []
    cursor = start
    while cursor <= end:
        for _ in range(per_week):
            day = cursor + timedelta(days=int(rng.integers(0, 7)))
            if day > end:
                continue
            if rng.random() < 0.22:
                headline, tone = MACRO_TEMPLATES[int(rng.integers(0, len(MACRO_TEMPLATES)))]
                items.append(
                    NewsItem(
                        published=day,
                        headline=headline,
                        tickers=(),
                        source="Macro Wire",
                        tone=tone,
                    )
                )
                continue
            ticker = tickers[int(rng.integers(0, len(tickers)))]
            roll = rng.random()
            if roll < 0.40:
                term = POSITIVE_TERMS[int(rng.integers(0, len(POSITIVE_TERMS)))]
                tone = 1
            elif roll < 0.80:
                term = NEGATIVE_TERMS[int(rng.integers(0, len(NEGATIVE_TERMS)))]
                tone = -1
            else:
                term = NEUTRAL_TERMS[int(rng.integers(0, len(NEUTRAL_TERMS)))]
                tone = 0
            template = COMPANY_TEMPLATES[int(rng.integers(0, len(COMPANY_TEMPLATES)))]
            headline = template.format(name=names.get(ticker, ticker), term=term)
            items.append(
                NewsItem(
                    published=day,
                    headline=headline,
                    tickers=(ticker,),
                    source="Market Wire",
                    tone=tone,
                )
            )
        cursor += timedelta(days=7)
    items.sort(key=lambda item: item.published)
    return items
