"""Real market data provider using yfinance + FRED (with local caching).

Prices
------
Project tickers are mapped onto liquid, long-history US ETFs so the existing
universe, sector map and suitability policy keep working unchanged:

    SPX->SPY  TECH->XLK  FIN->XLF  HLTH->XLV  ENRG->XLE  STPL->XLP
    INDU->XLI UTIL->XLU  INTL->EFA  EMEQ->EEM  GOVT->IEF  CRED->LQD
    GOLD->GLD  REIT->VNQ

Downloaded adjusted closes are cached to ``data/prices.csv`` (gitignored) so
backtests are reproducible without network access.

Risk-free rate
--------------
FRED ``DGS3MO`` (3-month Treasury bill, secondary market, percent per annum)
is cached to ``data/risk_free.csv``. ``get_risk_free(as_of)`` returns the most
recent observation **on or before** ``as_of`` — never a later one — which keeps
the rate point-in-time consistent with the price snapshots. If FRED is
unreachable and nothing is cached, the configured fallback is used and the
caller can see that via ``risk_free_source``.

News is intentionally empty: the sentiment agent must abstain (confidence 0)
rather than invent a textual signal when no real articles are available.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from intellivest.config import Settings
from intellivest.data.market import (
    MarketDataProvider,
    MarketSnapshot,
    infer_market_regime,
)
from intellivest.data.news import NewsItem
from intellivest.data.universe import TICKERS

# Project ticker -> real ETF proxy.
TICKER_MAP: dict[str, str] = {
    "SPX": "SPY",
    "TECH": "XLK",
    "FIN": "XLF",
    "HLTH": "XLV",
    "ENRG": "XLE",
    "STPL": "XLP",
    "INDU": "XLI",
    "UTIL": "XLU",
    "INTL": "EFA",
    "EMEQ": "EEM",
    "GOVT": "IEF",
    "CRED": "LQD",
    "GOLD": "GLD",
    "REIT": "VNQ",
}

# Front-month T-bill series on FRED, quoted in percent per annum.
FRED_SERIES = "DGS3MO"


class RealMarket(MarketDataProvider):
    """Live/real ETF prices with a point-in-time risk-free rate."""

    def __init__(
        self,
        settings: Settings,
        cache_path: Path | None = None,
        rf_cache_path: Path | None = None,
    ) -> None:
        super().__init__(settings)
        self.cache_path = cache_path or (settings.data_dir / "prices.csv")
        self.rf_cache_path = rf_cache_path or (settings.data_dir / "risk_free.csv")
        self._prices = self._load_or_fetch()
        self._fundamentals = self._empty_fundamentals()
        self._news: list[NewsItem] = []
        self._rf_series: pd.Series | None = None
        self.risk_free_source = "uninitialised"

    # -- prices -----------------------------------------------------------
    def _load_or_fetch(self) -> pd.DataFrame:
        if self.cache_path.exists():
            cached = self._read_prices(self.cache_path)
            if cached is not None and not cached.empty:
                return cached
        frame = self._fetch_yf()
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(self.cache_path)
        except OSError:
            pass
        return frame

    @staticmethod
    def _read_prices(path: Path) -> pd.DataFrame | None:
        try:
            frame = pd.read_csv(path, parse_dates=True, index_col=0).sort_index()
        except Exception:
            return None
        frame = frame.loc[:, ~frame.columns.duplicated()]
        # Accept both project tickers and raw ETF symbols.
        reverse = {etf: ticker for ticker, etf in TICKER_MAP.items()}
        frame = frame.rename(columns=lambda c: reverse.get(str(c), str(c)))
        available = [t for t in TICKERS if t in frame.columns]
        if not available:
            return None
        out = frame[available].astype(float).ffill()
        return out.dropna(how="all")

    def _fetch_yf(self) -> pd.DataFrame:
        try:
            import yfinance as yf
        except Exception as exc:  # pragma: no cover - dependency guard
            raise ImportError(f"yfinance is required for real data: {exc}") from exc

        symbols = [TICKER_MAP[t] for t in TICKERS if t in TICKER_MAP]
        try:
            raw = yf.download(
                symbols,
                start=self.settings.data_start,
                end=self.settings.data_end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download real prices ({exc}). Provide data/prices.csv or "
                "use the synthetic provider."
            ) from exc

        if raw is None or len(raw) == 0:
            raise RuntimeError("yfinance returned no rows for the requested window")

        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs(
                raw.columns.get_level_values(0)[0], axis=1, level=0
            )
        else:
            close = raw[["Close"]] if "Close" in raw.columns else raw

        close = close.reindex(columns=symbols)
        reverse = {etf: ticker for ticker, etf in TICKER_MAP.items()}
        out = close.rename(columns=lambda c: reverse.get(str(c), str(c)))
        out = out.reindex(columns=[t for t in TICKERS if t in out.columns])
        # Forward-fill holidays/gaps but never back-fill: a missing early print
        # must stay missing rather than borrow a future price.
        out = out.ffill().dropna(how="all")
        return out

    def _empty_fundamentals(self) -> pd.DataFrame:
        return pd.DataFrame(index=pd.Index(list(self._prices.columns), name="ticker"))

    # -- risk-free --------------------------------------------------------
    def _load_rf(self) -> pd.Series:
        if self._rf_series is not None:
            return self._rf_series

        series: pd.Series | None = None
        if self.rf_cache_path.exists():
            try:
                cached = pd.read_csv(
                    self.rf_cache_path, parse_dates=True, index_col=0
                ).iloc[:, 0]
                cached = pd.to_numeric(cached, errors="coerce").dropna()
                if len(cached):
                    series = self._as_decimal(cached.sort_index().astype(float))
            except Exception:
                series = None

        if series is None:
            series = self._fetch_fred_rf()
            if series is not None and len(series):
                series = self._as_decimal(series)
                try:
                    self.rf_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    series.rename("dgs3mo").to_csv(self.rf_cache_path)
                except OSError:
                    pass

        if series is None or series.empty:
            self._rf_series = pd.Series(dtype=float)
            self.risk_free_source = "fallback_settings"
        else:
            self._rf_series = series
            self.risk_free_source = f"fred:{FRED_SERIES}"
        return self._rf_series

    def _fetch_fred_rf(self) -> pd.Series | None:
        # pandas-datareader hits the public FRED CSV endpoint (no API key).
        try:
            from pandas_datareader import data as pdr

            frame = pdr.DataReader(
                FRED_SERIES, "fred", self.settings.data_start, self.settings.data_end
            )
            column = frame.iloc[:, 0] if hasattr(frame, "iloc") else None
            if column is not None:
                clean = pd.to_numeric(column, errors="coerce").dropna()
                if len(clean):
                    return clean.rename("dgs3mo")
        except Exception:
            pass

        # Optional fredapi path if the user has a key.
        try:
            import os

            import fredapi

            key = os.getenv("FRED_API_KEY")
            if key:
                fred = fredapi.Fred(api_key=key)
                raw = fred.get_series(
                    FRED_SERIES,
                    observation_start=self.settings.data_start,
                    observation_end=self.settings.data_end,
                )
                clean = pd.to_numeric(raw, errors="coerce").dropna()
                if len(clean):
                    return clean.rename("dgs3mo")
        except Exception:
            pass
        return None

    @staticmethod
    def _as_decimal(series: pd.Series) -> pd.Series:
        """FRED publishes percent (e.g. 4.25); convert to decimal if needed."""
        if series.empty:
            return series
        if float(series.abs().median()) > 0.30:  # clearly a percent quote
            return series / 100.0
        return series

    def risk_free_series(self) -> pd.Series:
        """Daily risk-free rate (decimal) indexed by observation date."""
        return self._load_rf().copy()

    def get_risk_free(self, as_of: date | None = None) -> float:
        """Most recent rate on or before ``as_of``; settings fallback if absent."""
        series = self._load_rf()
        if series.empty:
            return float(self.settings.risk_free_rate)
        if as_of is None:
            return float(series.iloc[-1])
        usable = series.loc[: pd.Timestamp(as_of)]
        if usable.empty:
            return float(self.settings.risk_free_rate)
        return float(usable.iloc[-1])

    # -- interface --------------------------------------------------------
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

        regime, regime_probability = infer_market_regime(
            benchmark, self.settings.trading_days
        )

        return MarketSnapshot(
            as_of=cutoff.date(),
            prices=window,
            returns=returns,
            benchmark_returns=benchmark,
            fundamentals=self._fundamentals,
            news=[],  # no real news feed wired in; sentiment must abstain
            regime=regime,
            regime_probability=regime_probability,
        )


def download_prices(settings: Settings, force: bool = False) -> Path:
    """Fetch/re-fetch ``data/prices.csv`` and the FRED rate series."""
    cache = settings.data_dir / "prices.csv"
    if force and cache.exists():
        cache.unlink()
    provider = RealMarket(settings)
    provider.risk_free_series()
    return cache


__all__ = ["RealMarket", "TICKER_MAP", "FRED_SERIES", "download_prices"]
