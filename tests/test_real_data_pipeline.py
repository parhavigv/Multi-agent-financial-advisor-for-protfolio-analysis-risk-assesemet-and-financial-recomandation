"""Tests for the real-data path (shaped like yfinance/FRED output, offline)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from intellivest.agents.base import AgentContext
from intellivest.agents.sentiment import SentimentAnalysisAgent
from intellivest.config import load_settings
from intellivest.contracts import InvestorProfile
from intellivest.data.real import TICKER_MAP, RealMarket
from intellivest.data.universe import TICKERS
from intellivest.quant.engine import QuantEngine

# Real yfinance column shape: ETF symbols, not project tickers.
ETF_SYMBOLS = [TICKER_MAP[t] for t in TICKERS]


def _write_prices(path, start="2019-01-01", end="2024-06-28") -> None:
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(7)
    shocks = rng.normal(0.0002, 0.009, size=(len(dates), len(ETF_SYMBOLS)))
    prices = (1.0 + pd.DataFrame(shocks, index=dates, columns=ETF_SYMBOLS)).cumprod() * 100.0
    prices.to_csv(path)


def _write_rates(path) -> None:
    dates = pd.bdate_range("2019-01-01", "2024-06-28")
    # Percent per annum, like FRED DGS3MO.
    rates = pd.Series(np.linspace(0.5, 5.4, len(dates)), index=dates, name="dgs3mo")
    rates.to_csv(path)


def _provider(tmp_path) -> RealMarket:
    settings = load_settings()
    prices_path = tmp_path / "prices.csv"
    rates_path = tmp_path / "risk_free.csv"
    _write_prices(prices_path)
    _write_rates(rates_path)
    provider = RealMarket(settings, cache_path=prices_path, rf_cache_path=rates_path)
    return provider


def test_real_provider_maps_etfs_and_never_uses_future_rows(tmp_path) -> None:
    provider = _provider(tmp_path)
    assert list(provider.history().columns) == list(TICKERS)

    as_of = date(2023, 6, 30)
    snapshot = provider.snapshot(as_of)
    assert snapshot.as_of == as_of
    assert snapshot.prices.index[-1].date() <= as_of
    assert snapshot.returns.index[-1].date() <= as_of
    assert snapshot.news == []


def test_risk_free_is_point_in_time(tmp_path) -> None:
    provider = _provider(tmp_path)
    early = provider.get_risk_free(date(2020, 1, 2))
    late = provider.get_risk_free(date(2024, 6, 28))
    assert 0.0 < early < late
    assert early < 0.02
    assert 0.05 < late < 0.06
    # A date before the series starts must fall back, never look forward.
    assert provider.get_risk_free(date(1990, 1, 1)) == provider.settings.risk_free_rate


def test_real_path_no_lookahead_under_truncation(tmp_path) -> None:
    provider = _provider(tmp_path)
    as_of = date(2022, 3, 31)
    base = provider.snapshot(as_of)

    # Mutate every price after as_of; the snapshot must be unchanged.
    tampered = provider.history()
    future = tampered.index > pd.Timestamp(as_of)
    tampered.loc[future] = tampered.loc[future] * 10.0
    provider._prices = tampered

    again = provider.snapshot(as_of)
    pd.testing.assert_frame_equal(base.prices, again.prices)
    pd.testing.assert_frame_equal(base.returns, again.returns)


def test_sentiment_abstains_when_no_real_news(tmp_path) -> None:
    settings = load_settings()
    provider = _provider(tmp_path)
    snapshot = provider.snapshot(date(2023, 6, 30))
    features = QuantEngine(settings).build(
        snapshot, risk_free=provider.get_risk_free(snapshot.as_of)
    )
    profile = InvestorProfile(
        investor_id="REAL-001",
        as_of=snapshot.as_of,
        age=34,
        horizon_years=12,
        risk_tolerance=3,
        loss_tolerance_pct=0.15,
        goal="balanced_growth",
        liquidity_need="low",
        current_holdings={},
    )
    ctx = AgentContext(
        profile=profile,
        snapshot=snapshot,
        features=features,
        current_weights={},
        settings=settings,
    )
    report = SentimentAnalysisAgent().analyze(ctx)
    assert report.confidence == 0.0
    assert report.signals == []
    assert report.metrics.get("news_available") == 0.0
