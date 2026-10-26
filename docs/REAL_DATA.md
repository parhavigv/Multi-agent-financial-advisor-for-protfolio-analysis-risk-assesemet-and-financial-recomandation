# Real data path

The system ships with two interchangeable market-data providers behind one
interface (`MarketDataProvider`): a deterministic synthetic regime-switching
model and `RealMarket`, which pulls real ETF prices and the real risk-free
rate. Both produce the same `MarketSnapshot`, so agents, the optimizer, the
suitability gate and the audit trace are identical in either mode.

## Quick start

```bash
# Download prices + risk-free series (both cached under data/, gitignored)
python scripts/fetch_real_data.py

# Force a fresh download
python scripts/fetch_real_data.py --force
```

Once `data/prices.csv` exists, `build_provider()` uses `RealMarket`
automatically. Delete that file to fall back to the deterministic synthetic
provider. `INTELLIVEST_USE_REAL=1` forces a fetch when no cache exists.

## Ticker mapping

Project tickers are proxies for liquid, long-history US ETFs so the existing
universe, sector map and suitability policy keep working unchanged:

| Project | ETF | Asset class / sector |
|---------|-----|----------------------|
| SPX  | SPY | US large-cap equity |
| TECH | XLK | Technology |
| FIN  | XLF | Financials |
| HLTH | XLV | Health care |
| ENRG | XLE | Energy |
| STPL | XLP | Consumer staples |
| INDU | XLI | Industrials |
| UTIL | XLU | Utilities |
| INTL | EFA | Developed ex-US equity |
| EMEQ | EEM | Emerging-market equity |
| GOVT | IEF | 7–10y US Treasuries |
| CRED | LQD | Investment-grade credit |
| GOLD | GLD | Gold |
| REIT | VNQ | US REITs |

Prices are **adjusted closes** (`auto_adjust=True`), forward-filled across
holidays/gaps but **never back-filled**, so a missing early print cannot
borrow a future price.

## Risk-free rate

The risk-free rate comes from FRED **DGS3MO** (3-month Treasury bill,
secondary market, quoted in percent per annum), cached to
`data/risk_free.csv`. **It is not hard-coded to 6.5%.**

`RealMarket.get_risk_free(as_of)` returns the most recent observation **on or
before** `as_of`. Every consumer uses that point-in-time value:

* `AdvisoryEngine.advise` records it in the trace (`risk_free_rate`,
  `risk_free_source`) and in the config digest;
* the optimizer's Sharpe/Sortino annotation uses it;
* `QuantFeatures` carries it, so agent risk metrics and the suitability gate
  use the same rate;
* `Backtester` samples it at each rebalance date and averages it across the
  traded sample.

If FRED is unreachable and there is no cache, the configured
`INTELLIVEST_RISK_FREE_RATE` (default 0.065) is used and the trace records
`risk_free_source="fallback_settings"` so the substitution is visible.

## News and sentiment

There is **no real news feed wired in**. `RealMarket.snapshot()` returns an
empty news list, and the sentiment agent then abstains: confidence `0.0`, no
signals, `news_available=0.0`. It never invents a textual signal. To add a
real feed, populate `MarketSnapshot.news` with dated `NewsItem`s; the agent
already enforces `published <= as_of`.

## No-lookahead guarantees

* `snapshot(as_of)` truncates the price history at `as_of` (both providers).
* The risk-free lookup truncates the FRED series at `as_of`.
* Tests cover the real path explicitly:
  * `test_real_provider_maps_etfs_and_never_uses_future_rows`
  * `test_risk_free_is_point_in_time`
  * `test_real_path_no_lookahead_under_truncation` — mutates all rows after
    `as_of` and asserts the snapshot is byte-identical.
* `tests/test_core.py` keeps the original synthetic no-lookahead test and pins
  the synthetic provider so offline runs stay deterministic.

## Evaluation protocol

Splits are fixed by time (`intellivest/evaluation/splits.py`):

| Split | Window | Use |
|-------|--------|-----|
| dev | start .. 2021-12-31 | design, feature/agent tuning |
| validation | 2022-01-01 .. 2023-12-31 | model/parameter selection |
| test | 2024-01-01 .. latest | one touch, at the very end |

Never tune on the test window. Any reported metric must come from a script run
on the split being described; synthetic and real results are labelled
separately.

## Limitations

* ETFs are **proxies**, not the underlying assets in an investor's account;
  tracking difference and expense ratios are ignored.
* `IEF` (7–10y Treasuries) is a duration proxy for the `GOVT` sleeve, not a
  T-bill.
* No corporate fundamentals are available, so the fundamental agent stays on
  its documented neutral path in real mode.
* The regime label in real mode is a realised-volatility heuristic, not a
  fitted HMM; it is descriptive, not predictive.
