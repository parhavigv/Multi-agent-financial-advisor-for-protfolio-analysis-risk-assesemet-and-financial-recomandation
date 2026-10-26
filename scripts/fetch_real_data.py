"""Download real ETF prices and the FRED risk-free series.

Writes (both gitignored, reproducible from this script):

* ``data/prices.csv``    adjusted daily closes for the 14 universe tickers
* ``data/risk_free.csv`` FRED ``DGS3MO`` (3-month T-bill, percent per annum)

Usage::

    python scripts/fetch_real_data.py [--force]

After it completes, :func:`intellivest.data.market.build_provider` automatically
uses the real provider because ``data/prices.csv`` exists. Delete that file (or
set ``INTELLIVEST_USE_REAL=0`` with no cache) to return to synthetic data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from intellivest.config import load_settings
from intellivest.data.real import FRED_SERIES, TICKER_MAP, RealMarket


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the cached files first and re-download.",
    )
    args = parser.parse_args()

    settings = load_settings()
    prices_path = settings.data_dir / "prices.csv"
    rf_path = settings.data_dir / "risk_free.csv"

    if args.force:
        for path in (prices_path, rf_path):
            if path.exists():
                path.unlink()

    provider = RealMarket(
        settings, cache_path=prices_path, rf_cache_path=rf_path
    )
    prices = provider.history()
    rates = provider.risk_free_series()

    print(f"prices : {prices.shape[0]} rows x {prices.shape[1]} tickers")
    print(f"window : {prices.index.min().date()} .. {prices.index.max().date()}")
    print(f"mapped : {', '.join(f'{k}->{v}' for k, v in TICKER_MAP.items())}")
    print(f"cache  : {prices_path}")
    if rates.empty:
        print(f"rates  : UNAVAILABLE ({FRED_SERIES}); using settings fallback")
    else:
        print(
            f"rates  : {FRED_SERIES} {rates.index.min().date()} .. "
            f"{rates.index.max().date()} ({rates.iloc[-1]:.4%} latest)"
        )
        print(f"cache  : {rf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
