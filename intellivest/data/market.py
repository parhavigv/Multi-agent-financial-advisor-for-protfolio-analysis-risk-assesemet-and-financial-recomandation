"""Market data layer.

Two providers are supported:

* ``SyntheticMarket`` — a seeded regime-switching factor model. It produces a
  realistic correlation structure (equities co-move, bonds and gold hedge in
  stress) so risk, regime and optimizer code is exercised meaningfully, and it
  is fully reproducible.
* ``CsvMarket`` — loads ``data/prices.csv`` (date index, one column per
  ticker). Use this to run against real prices without touching any other
  module.

Both expose the same interface and, critically, ``snapshot(as_of)`` never
returns a row dated after ``as_of``. That is the structural defence against
look-ahead bias that the survey identified as a recurring evaluation flaw.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from intellivest.config import Settings
from intellivest.data.news import NewsItem, generate_news
from intellivest.data.universe import ASSET_BY_TICKER, TICKERS, UNIVERSE

REGIME_NAMES: tuple[str, str, str] = ("calm", "elevated", "stress")


def infer_market_regime(
    benchmark: pd.Series, trading_days: int = 252
) -> tuple[str, float]:
    """Regime label for a market snapshot.

    Prefers the causal 3-state HMM (:func:`intellivest.forecasting.regime.infer_regime`)
    and falls back to the legacy realised-volatility thresholds if the optional
    ``hmmlearn`` dependency is missing or the fit fails. The caller must pass a
    series that has already been truncated at ``as_of``.
    """
    series = pd.Series(benchmark).dropna().astype(float)
    if len(series) >= 120:
        try:
            from intellivest.forecasting.regime import infer_regime

            return infer_regime(series, trading_days=trading_days)
        except Exception:  # pragma: no cover - optional dependency guard
            pass
    if len(series) < 63:
        return "elevated", 0.4
    realised = float(series.iloc[-63:].std(ddof=1) * np.sqrt(trading_days))
    if not np.isfinite(realised):
        return "elevated", 0.4
    if realised > 0.28:
        return "stress", 0.7
    if realised > 0.16:
        return "elevated", 0.5
    return "calm", 0.7

# The macro backdrop is generated from a fixed seed so that every backtest
# configuration — and every robustness seed — faces the same regime path and
# market factor. ``Settings.seed`` then controls only the cross-sectional
# (asset-specific) draws, which is what an A/B/C comparison should isolate.
REGIME_SEED = 60

# Row-stochastic transition matrix over (calm, elevated, stress).
TRANSITION = np.array(
    [
        [0.975, 0.020, 0.005],
        [0.100, 0.860, 0.040],
        [0.030, 0.150, 0.820],
    ]
)

REGIME_MU = np.array([0.00050, 0.00005, -0.00120])
REGIME_SIGMA = np.array([0.0068, 0.0115, 0.0210])

# Per-asset, per-day probability of an idiosyncratic jump of 6-20% in either
# direction (~0.45 jumps per asset per year). Rare enough to leave the factor
# structure intact, frequent enough that concentration has a visible tail cost.
JUMP_PROBABILITY = 0.0018

BETAS: dict[str, float] = {
    "SPX": 1.00,
    "TECH": 1.25,
    "FIN": 1.10,
    "HLTH": 0.75,
    "ENRG": 0.95,
    "STPL": 0.60,
    "INDU": 1.05,
    "UTIL": 0.55,
    "INTL": 0.90,
    "EMEQ": 1.20,
    "GOVT": -0.18,
    "CRED": 0.28,
    "GOLD": -0.12,
    "REIT": 0.85,
}

IDIO_VOL: dict[str, float] = {
    "SPX": 0.0035,
    "TECH": 0.0075,
    "FIN": 0.0055,
    "HLTH": 0.0055,
    "ENRG": 0.0080,
    "STPL": 0.0045,
    "INDU": 0.0055,
    "UTIL": 0.0045,
    "INTL": 0.0050,
    "EMEQ": 0.0085,
    "GOVT": 0.0035,
    "CRED": 0.0028,
    "GOLD": 0.0080,
    "REIT": 0.0065,
}

START_PRICES: dict[str, float] = {
    "SPX": 100.0,
    "TECH": 100.0,
    "FIN": 100.0,
    "HLTH": 100.0,
    "ENRG": 100.0,
    "STPL": 100.0,
    "INDU": 100.0,
    "UTIL": 100.0,
    "INTL": 100.0,
    "EMEQ": 100.0,
    "GOVT": 100.0,
    "CRED": 100.0,
    "GOLD": 100.0,
    "REIT": 100.0,
}


@dataclass(frozen=True)
class MarketSnapshot:
    as_of: date
    prices: pd.DataFrame
    returns: pd.DataFrame
    benchmark_returns: pd.Series
    fundamentals: pd.DataFrame
    news: list[NewsItem]
    regime: str
    regime_probability: float

    @property
    def tickers(self) -> list[str]:
        return list(self.returns.columns)


class MarketDataProvider:
    """Common snapshot interface."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def snapshot(self, as_of: date, lookback_days: int | None = None) -> MarketSnapshot:
        raise NotImplementedError

    def history(self) -> pd.DataFrame:
        """Full price history (used only by the evaluation harness)."""
        raise NotImplementedError

    def get_risk_free(self, as_of: date | None = None) -> float:
        """Point-in-time rate on or before ``as_of`` (synthetic: constant)."""
        return float(self.settings.risk_free_rate)


class SyntheticMarket(MarketDataProvider):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._prices, self._regimes, self._fundamentals = self._generate(settings)
        names = {ticker: ASSET_BY_TICKER[ticker].name for ticker in TICKERS}
        self._news = generate_news(
            start=date.fromisoformat(settings.data_start),
            end=date.fromisoformat(settings.data_end),
            tickers=TICKERS,
            names=names,
            rng=np.random.default_rng(settings.seed + 11),
        )

    # -- generation -------------------------------------------------------
    def _generate(
        self, settings: Settings
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        rng = np.random.default_rng(settings.seed)
        macro_rng = np.random.default_rng(REGIME_SEED)
        index = pd.bdate_range(settings.data_start, settings.data_end)
        n_days = len(index)

        regimes = np.empty(n_days, dtype=int)
        regimes[0] = 0
        for t in range(1, n_days):
            regimes[t] = macro_rng.choice(3, p=TRANSITION[regimes[t - 1]])

        market = macro_rng.normal(REGIME_MU[regimes], REGIME_SIGMA[regimes])
        # Mild momentum: a fraction of yesterday's market shock carries over.
        for t in range(1, n_days):
            market[t] += 0.05 * market[t - 1]

        vol_scale = np.where(regimes == 2, 1.35, np.where(regimes == 1, 1.10, 1.0))
        returns = np.empty((n_days, len(TICKERS)))
        for j, ticker in enumerate(TICKERS):
            beta = BETAS[ticker]
            # Equity beta compresses slightly in stress (correlations rise).
            effective_beta = beta * np.where(regimes == 2, 1.10, 1.0)
            idio = rng.normal(0.0, IDIO_VOL[ticker], n_days) * vol_scale
            if ticker == "GOLD":
                idio += np.where(regimes == 2, 0.0009, 0.0)
            if ticker in {"GOVT", "CRED"}:
                idio += np.where(regimes == 2, 0.0006, 0.0)
            returns[:, j] = effective_beta * market + idio

        returns = np.clip(returns, -0.12, 0.12)

        # Idiosyncratic jump risk: rare single-name shocks (profit warnings,
        # litigation, fraud, regulatory action, takeover pops). The direction is
        # symmetric so the *expected* return is unchanged, but the tails are
        # fatter and a five-name portfolio absorbs far more of them than a
        # diversified book. Without this the factor model understates the real
        # cost of concentration.
        hits = rng.random((n_days, len(TICKERS))) < JUMP_PROBABILITY
        directions = rng.choice(np.array([-1.0, 1.0]), size=(n_days, len(TICKERS)))
        magnitudes = rng.uniform(0.06, 0.20, size=(n_days, len(TICKERS)))
        returns = returns + hits * directions * magnitudes
        returns = np.clip(returns, -0.35, 0.30)

        prices = pd.DataFrame(
            {ticker: START_PRICES[ticker] * np.cumprod(1.0 + returns[:, j])
             for j, ticker in enumerate(TICKERS)},
            index=index,
        )
        regime_series = pd.Series([REGIME_NAMES[r] for r in regimes], index=index, name="regime")
        fundamentals = self._generate_fundamentals(rng, index)
        return prices, regime_series, fundamentals

    def _generate_fundamentals(
        self, rng: np.random.Generator, index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        quarters = pd.date_range(index[0], index[-1], freq="QE")
        rows: list[dict[str, object]] = []
        for quarter in quarters:
            for ticker in TICKERS:
                asset = ASSET_BY_TICKER[ticker]
                drift = {"technology": 0.6, "energy": -0.2, "utilities": 0.1}.get(asset.sector, 0.0)
                rows.append(
                    {
                        "date": quarter,
                        "ticker": ticker,
                        "pe": float(np.clip(rng.normal(19 + 4 * drift, 5), 6, 60)),
                        "pb": float(np.clip(rng.normal(2.8, 0.9), 0.4, 9)),
                        "roe": float(np.clip(rng.normal(0.15 + 0.03 * drift, 0.05), -0.10, 0.45)),
                        "debt_to_equity": float(
                            np.clip(rng.normal(0.9 if not asset.is_equity else 1.2, 0.5), 0.0, 4.0)
                        ),
                        "earnings_growth": float(np.clip(rng.normal(0.08 + 0.04 * drift, 0.10), -0.35, 0.60)),
                        "dividend_yield": float(
                            np.clip(
                                rng.normal(0.035 if asset.asset_class == "fixed_income" else 0.018, 0.010),
                                0.0,
                                0.09,
                            )
                        ),
                        "analyst_revision": float(np.clip(rng.normal(0.0, 0.25), -1, 1)),
                        "quality_score": float(np.clip(rng.normal(0.55, 0.15), 0.05, 0.98)),
                    }
                )
        return pd.DataFrame(rows)

    # -- interface --------------------------------------------------------
    def history(self) -> pd.DataFrame:
        return self._prices.copy()

    def snapshot(self, as_of: date, lookback_days: int | None = None) -> MarketSnapshot:
        lookback = lookback_days or self.settings.lookback_days
        cutoff = pd.Timestamp(as_of)
        window = self._prices.loc[:cutoff]
        if window.empty:
            raise ValueError(f"no market data on or before {as_of}")
        window = window.iloc[-(lookback + 1):]
        returns = window.pct_change().dropna(how="all")
        benchmark = returns["SPX"] if "SPX" in returns else returns.iloc[:, 0]

        regime_series = self._regimes.loc[:cutoff]
        regime = str(regime_series.iloc[-1])
        # Probability of the current regime under a short memory window.
        recent = regime_series.iloc[-63:]
        regime_probability = float((recent == regime).mean())

        fundamentals = self._fundamentals.loc[
            (self._fundamentals["date"] <= cutoff) & (self._fundamentals["ticker"].isin(TICKERS))
        ]
        latest = fundamentals.sort_values("date").groupby("ticker").tail(1).set_index("ticker")

        news_window = [
            item
            for item in self._news
            if cutoff.date() - timedelta(days=90) <= item.published <= cutoff.date()
        ]
        return MarketSnapshot(
            as_of=cutoff.date(),
            prices=window,
            returns=returns,
            benchmark_returns=benchmark,
            fundamentals=latest,
            news=news_window,
            regime=regime,
            regime_probability=regime_probability,
        )


class CsvMarket(MarketDataProvider):
    """Load real prices from ``data/prices.csv`` (date index, ticker columns)."""

    def __init__(self, settings: Settings, path: Path | None = None) -> None:
        super().__init__(settings)
        path = path or (settings.data_dir / "prices.csv")
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Provide a CSV with a date index and ticker columns, "
                "or use the synthetic provider."
            )
        frame = pd.read_csv(path, parse_dates=True, index_col=0).sort_index()
        available = [t for t in TICKERS if t in frame.columns]
        if len(available) < 3:
            raise ValueError("prices.csv must contain at least three known tickers")
        self._prices = frame[available].astype(float).ffill().dropna(how="all")
        self._news: list[NewsItem] = []
        self._fundamentals = pd.DataFrame(
            {"date": [self._prices.index[0]] * len(available), "ticker": available}
        )

    def history(self) -> pd.DataFrame:
        return self._prices.copy()

    def snapshot(self, as_of: date, lookback_days: int | None = None) -> MarketSnapshot:
        lookback = lookback_days or self.settings.lookback_days
        cutoff = pd.Timestamp(as_of)
        window = self._prices.loc[:cutoff].iloc[-(lookback + 1):]
        if window.empty:
            raise ValueError(f"no market data on or before {as_of}")
        returns = window.pct_change().dropna(how="all")
        benchmark = returns["SPX"] if "SPX" in returns else returns.iloc[:, 0]
        prices_window = window
        regime, regime_probability = infer_market_regime(
            benchmark, self.settings.trading_days
        )
        fundamentals = pd.DataFrame(index=pd.Index(returns.columns, name="ticker"))
        return MarketSnapshot(
            as_of=cutoff.date(),
            prices=prices_window,
            returns=returns,
            benchmark_returns=benchmark,
            fundamentals=fundamentals,
            news=list(self._news),
            regime=regime,
            regime_probability=regime_probability,
        )


def build_provider(settings: Settings) -> MarketDataProvider:
    """Use real prices when present, otherwise the synthetic provider."""
    prices_path = settings.data_dir / "prices.csv"
    if prices_path.exists():
        # Prefer RealMarket if available (uses cached CSV; can also fetch)
        try:
            from intellivest.data.real import RealMarket

            return RealMarket(settings, cache_path=prices_path)
        except Exception:
            try:
                return CsvMarket(settings)
            except Exception:
                pass
    # Also allow opt-in via env
    import os

    if os.getenv("INTELLIVEST_USE_REAL", "0").strip() in {"1", "true", "True"}:
        try:
            from intellivest.data.real import RealMarket

            return RealMarket(settings)
        except Exception:
            pass
    return SyntheticMarket(settings)


__all__ = [
    "MarketDataProvider",
    "MarketSnapshot",
    "SyntheticMarket",
    "CsvMarket",
    "build_provider",
    "infer_market_regime",
    "REGIME_NAMES",
    "UNIVERSE",
]
