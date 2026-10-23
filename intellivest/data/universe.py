"""Investable universe.

Metadata drives sector caps, exclusion rules and asset-class suitability
checks, so it lives in code (and can be replaced by an institution's own
security master).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Asset:
    ticker: str
    name: str
    asset_class: str
    sector: str
    region: str
    tags: tuple[str, ...] = ()

    @property
    def is_equity(self) -> bool:
        return self.asset_class == "equity"


UNIVERSE: tuple[Asset, ...] = (
    Asset("SPX", "US Large Cap Equity", "equity", "broad_equity", "US", ("core",)),
    Asset("TECH", "Technology Sector Equity", "equity", "technology", "US", ("growth",)),
    Asset("FIN", "Financials Sector Equity", "equity", "financials", "US", ("value",)),
    Asset("HLTH", "Healthcare Sector Equity", "equity", "healthcare", "US", ("defensive",)),
    Asset("ENRG", "Energy Sector Equity", "equity", "energy", "US", ("fossil", "cyclical")),
    Asset("STPL", "Consumer Staples Equity", "equity", "consumer_staples", "US", ("defensive",)),
    Asset("INDU", "Industrials Sector Equity", "equity", "industrials", "US", ("cyclical",)),
    Asset("UTIL", "Utilities Sector Equity", "equity", "utilities", "US", ("defensive",)),
    Asset("INTL", "Developed ex-US Equity", "equity", "international_equity", "Global", ("core",)),
    Asset("EMEQ", "Emerging Market Equity", "equity", "emerging_markets", "EM", ("growth",)),
    Asset("GOVT", "Long Duration Government Bonds", "fixed_income", "government_bonds", "US", ("duration",)),
    Asset("CRED", "Investment Grade Corporate Credit", "fixed_income", "corporate_bonds", "US", ("credit",)),
    Asset("GOLD", "Gold", "commodity", "commodities", "Global", ("hedge",)),
    Asset("REIT", "Real Estate Investment Trusts", "real_estate", "real_estate", "US", ("income",)),
)

TICKERS: tuple[str, ...] = tuple(asset.ticker for asset in UNIVERSE)
ASSET_BY_TICKER: dict[str, Asset] = {asset.ticker: asset for asset in UNIVERSE}
SECTORS: tuple[str, ...] = tuple(sorted({asset.sector for asset in UNIVERSE}))
ASSET_CLASSES: tuple[str, ...] = tuple(sorted({asset.asset_class for asset in UNIVERSE}))


def sector_of(ticker: str) -> str:
    return ASSET_BY_TICKER[ticker].sector


def asset_class_of(ticker: str) -> str:
    return ASSET_BY_TICKER[ticker].asset_class


def tags_of(ticker: str) -> tuple[str, ...]:
    return ASSET_BY_TICKER[ticker].tags


def excluded_tickers(exclusions: list[str]) -> set[str]:
    """Map investor exclusions (sector, asset class, or tag) to tickers."""
    blocked = {item.strip().lower() for item in exclusions if item.strip()}
    if not blocked:
        return set()
    result: set[str] = set()
    for asset in UNIVERSE:
        labels = {asset.sector, asset.asset_class, *asset.tags}
        if labels & blocked:
            result.add(asset.ticker)
    return result


def sector_weights(weights: dict[str, float]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for ticker, weight in weights.items():
        totals[sector_of(ticker)] = totals.get(sector_of(ticker), 0.0) + weight
    return totals


def asset_class_weights(weights: dict[str, float]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for ticker, weight in weights.items():
        cls = asset_class_of(ticker)
        totals[cls] = totals.get(cls, 0.0) + weight
    return totals


def describe_universe() -> list[dict[str, object]]:
    return [
        {
            "ticker": a.ticker,
            "name": a.name,
            "asset_class": a.asset_class,
            "sector": a.sector,
            "region": a.region,
            "tags": list(a.tags),
        }
        for a in UNIVERSE
    ]
