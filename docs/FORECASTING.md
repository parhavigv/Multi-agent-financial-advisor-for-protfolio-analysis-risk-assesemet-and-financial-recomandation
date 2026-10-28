# Forecasting evaluation and advisory integration

This document describes the Phase-2 forecasting layer: what is estimated, how it
is validated, what the held-out test period actually showed, and how the
surviving models are used by the advisory pipeline. Every number quoted here
comes from `results/forecast_report.md` / `.json`, which are produced by
`scripts/run_forecast_evaluation.py`. Nothing is hand-entered.

The design goal is narrow and deliberate: **LLM agents coordinate and interpret;
all numerics come from models that were evaluated out of sample.** A forecast is
promoted only if it survives a purged walk-forward evaluation and a set of
stability checks. Otherwise the audited simple estimator is retained.

## Data and universe

- 14 liquid US ETFs mapped onto the project tickers (`SPX->SPY`, `TECH->XLK`,
  `FIN->XLF`, `HLTH->XLV`, `ENRG->XLE`, `STPL->XLP`, `INDU->XLI`, `UTIL->XLU`,
  `INTL->EFA`, `EMEQ->EEM`, `GOVT->IEF`, `CRED->LQD`, `GOLD->GLD`, `REIT->VNQ`).
- Adjusted closes cached in `data/prices.csv` (gitignored). The evaluation sample
  is **2019-02-05 to 2026-05-28**: 1,838 feature dates and 25,732 asset-date
  rows after the 273-day feature warm-up and forward-target drop.
- Forecast horizon: **21 trading days** (one month).
- The risk-free rate used elsewhere in the system is FRED `DGS3MO`, read
  point-in-time (latest observation on or before the decision date). It is not
  a forecast target.

Features are causal by construction: 5/21/63/126-day momentum, 21/63-day realised
volatility, 63/252-day drawdown, EWMA volatility, and two price-based value
proxies (`value_rev_252`, `value_ma_gap_252`). Every feature at date `t` uses
returns only through `t-1`. The ETF universe has no fundamentals, so no
fundamental value factor is claimed.

## Splits and cross-validation

| window | dates | used for |
|---|---|---|
| dev | start - 2021-12-31 | model fitting, first stability check |
| validation | 2022-01-01 - 2023-12-31 | hyperparameters, flags, **selection** |
| test | 2024-01-01 - latest | final report only |

Scored feature dates: dev 713, validation 480, test 603.

Cross-validation is a purged, embargoed, expanding walk-forward:

- minimum training window: 378 dates;
- purge: 21 dates (target horizon);
- embargo: 21 dates;
- step: 21 dates;
- hyperparameters are tuned only on an inner time split of each training fold,
  never on the validation or test window;
- a training row whose 21-day target overlaps the prediction block is removed;
- the last `horizon` feature dates before each split boundary are excluded from
  the scored set, because their targets reach into the next split.

## Models

Returns (cross-sectional, panel):

- baselines: `shrunk_mean` (James-Stein shrink of annualised means),
  `momentum_12_1`, `one_over_n`;
- candidates: Ridge, ElasticNet, LightGBM. Each is fitted on the stacked
  date x ticker panel with the causal features above.

Volatility (per asset):

- baseline: `ewma` (RiskMetrics-style, from the feature matrix);
- candidates: log-HAR and GARCH(1,1) (`arch`, parameters re-estimated per fold
  and filtered forward).

Regime:

- Gaussian HMM with 2 or 3 states chosen by BIC on the **dev window only**;
- labels are **filtered** probabilities `P(state_t | x_1..x_t)`, never Viterbi or
  smoothed paths (those use future observations);
- states are named by volatility (`calm`, `elevated`, `stress`), so labels mean
  the same thing across fits.

An LSTM was **not run**. The pre-registered rule was to train one only if HAR
beat EWMA on validation; HAR did not, so the optional model was left out rather
than tuned into existence.

## Leakage controls

1. **Causal features.** A unit test appends future data and asserts that feature
   rows at earlier dates do not change.
2. **Purge and embargo.** Training targets cannot overlap the prediction block.
3. **Boundary exclusion.** Targets that cross a split boundary are not scored.
4. **Selection-blind diagnostics.** Train-vs-validation flags use only folds
   whose prediction block ends at or before the validation boundary. Adding the
   test window therefore cannot change a flag or the selected model. This is
   enforced by
   `test_selection_diagnostics_are_invariant_to_including_test`.
5. **Shuffled-label control.** Ridge is rerun with training labels permuted
   across tickers within each date, averaged over seeds `(12345, 1, 2)`. Honest
   pipelines collapse to zero rank IC. On validation the control gives
   **0.0241** (t = 0.46; per-seed 0.0258 / 0.0229 / 0.0237), comfortably inside
   the `|IC| < 0.05` and `|t| < 2` threshold.
6. **HMM causality.** Regime labels are filtered, and the HMM is fitted on dev
   data only.
7. **Point-in-time data.** Snapshots never include a row dated after `as_of`;
   the risk-free rate is read on or before the decision date.

### Note on the final report

During hardening, an earlier test-inclusive run exposed a real contamination
path: the "train in-sample" diagnostic used the *last* fold, which on a
test-inclusive run is trained on test-period data, changing the stability flag
and promoting Ridge. The diagnostic is now restricted to folds ending at the
validation boundary, and the selection is verified to be identical with and
without the test window. The report was regenerated after that fix. No model,
threshold or rule was changed in response to test-period performance; the
dev/validation metrics, flags, control and selected models are identical in the
two runs, and the test column is reported as-is.

## Selection rule

A candidate replaces the baseline only if **all** of the following hold:

- it is not flagged for a large train/validation gap (return rank IC may fall by
  at most 0.10; error may rise by at most 150%; volatility QLIKE may rise by at
  most 0.30);
- for returns, its validation rank IC is positive and exceeds the best baseline
  by at least **0.01**;
- for volatility, its validation QLIKE improves on the designated baseline by at
  least **0.01** (absolute, since QLIKE can be negative);
- for return candidates, the dev-to-validation rank IC does not swing by more
  than **0.15**.

Diebold-Mariano statistics are computed per split against the selected
baseline; a positive statistic means the model's loss is higher (worse) than the
baseline.

## Results

### Returns

| model | dev IC | validation IC | test IC | flag |
|---|---|---|---|---|
| `shrunk_mean` (selected) | 0.0043 | 0.0613 | 0.0977 | ok |
| `momentum_12_1` | -0.0913 | -0.0001 | 0.0539 | ok |
| `one_over_n` | 0.0000 | 0.0000 | 0.0000 | ok |
| `ridge` | -0.0361 | 0.0782 | 0.0770 | flagged: train IC gap 0.102 |
| `elastic_net` | -0.3239 | 0.0455 | 0.0769 | flagged: dev/val swing 0.369 |
| `lightgbm` | -0.0686 | -0.0017 | 0.0672 | flagged: train IC gap 0.351 |

Validation RMSE is 0.0564 for `shrunk_mean`; test RMSE is 0.0436 with a
validation hit rate of 0.498. Ridge has a slightly higher validation IC than the
baseline but was flagged for instability and was therefore **not** selected. Its
positive test IC is reported as-is; switching to it after seeing the test result
would be test-set tuning and is explicitly not done.

### Volatility

| model | dev QLIKE | validation QLIKE | test QLIKE | decision |
|---|---|---|---|---|
| `ewma` (selected) | -2.9129 | -2.4874 | -2.7941 | baseline retained |
| `har` | -2.8400 | -2.4680 | -2.7909 | worse validation QLIKE; test DM 0.14 |
| `garch11` | -2.7411 | -1.8497 | -2.5427 | clearly worse (validation DM 6.29) |

QLIKE is a loss (lower is better). HAR is marginally *worse* than EWMA on both
validation and test, so it fails the pre-registered requirement that it improve
validation QLIKE by at least 0.01; the test DM difference (0.14) is not
significant either. The baseline is retained.

The test result is not uniformly better. EWMA's test QLIKE improves, but its
absolute volatility RMSE rises from 0.0531 to 0.0704 and its MAE rises from
0.0419 to 0.0440. HAR has a lower test RMSE (0.0649) but a slightly worse test
QLIKE (-2.7909 versus -2.7941). Because selection is fixed on validation QLIKE,
EWMA remains selected; switching to HAR after seeing test would be test-set
selection. The test report states both readings.

### Regimes

Three states are selected by BIC (-274.9), fitted through 2021-12-31. Realised
state frequencies:

| split | calm | elevated | stress |
|---|---|---|---|
| dev | 0.457 | 0.443 | 0.099 |
| validation | 0.356 | 0.498 | 0.146 |
| test | 0.481 | 0.466 | 0.053 |

For the selected return model, test rank IC is 0.109 (calm), 0.081 (elevated)
and 0.141 (stress); the stress bucket has only 32 dates and its t-statistic
(2.79) should be read as indicative, not precise.

## Advisory integration

`AdvisoryEngine(use_ml=True)` (CLI: `--ml`) enables the overlay:

- `MLForecastService` builds the same causal panel at the decision date and
  performs a **single inner time split** (last 25% of dates, purged by the
  horizon) to choose between the baseline and the candidate;
- Ridge is used only if its inner validation IC is positive, beats the baseline
  by 0.01, and its inner train-to-validation IC gap is at most 0.10;
- HAR is used only if its inner QLIKE improves on EWMA by 0.01;
- otherwise the service falls back to `shrunk_mean` + `ewma`, and records the
  rejection reason;
- 21-day return forecasts are annualised by compounding, clipped to
  `[-0.35, 0.35]`, and volatility is clipped to `[0.0001, 2.0]`;
- if history is insufficient or the panel fails, the service falls back to
  `intellivest.quant.forecast` and records a `fallbacks` entry in the trace;
- the forecast, model identifiers, validation metrics, rejections and regime are
  written to `DecisionTrace.ml_forecast`.

On real data at 2023-06-30 the inner split rejects Ridge (gap 0.220) and HAR
(no QLIKE margin), so the advisory service uses `shrunk_mean` + `ewma`; the trace records
both rejections. That is the intended behaviour: the integration does not
override the evaluation evidence.

Market providers (`CsvMarket`, `RealMarket`) use the same causal HMM for the
snapshot regime, with a realised-volatility threshold fallback if `hmmlearn` is
unavailable. The overlay is opt-in so that the default synthetic backtest and
all previously reported results remain reproducible.

## Reproduction

```bash
python scripts/fetch_real_data.py                    # prices + FRED DGS3MO
python scripts/run_forecast_evaluation.py            # dev + validation only
python scripts/run_forecast_evaluation.py --include-test --output results/forecast_report.md
python -m pytest -q                                  # includes the leakage tests
```

The final command writes `results/forecast_report.md` and `.json`. The report is
deterministic and versioned as the evidence for this phase.

## Limitations

- The universe is 14 ETFs; results are not evidence about single stocks, private
  assets or fundamental value.
- The horizon is fixed at 21 trading days; overlapping targets are handled with
  Newey-West t-statistics, but the effective sample is smaller than the row
  count suggests.
- Six return candidates were compared, so some validation IC could be luck; the
  stability flags and the single held-out window are the defence, not a
  multiple-testing correction.
- The evaluation measures forecast accuracy, not net-of-cost portfolio
  performance; transaction costs are handled in the backtester, not here.
- ETF data are survivorship-biased by construction (the funds exist today).
- The LSTM is an explicit non-result, not a hidden success.
