# Phase 4 evaluation protocol

Phase 4 asks a narrower question than "does the system produce a high
backtest return?":

> Once the comparison includes 1/N and 60/40 benchmarks, a fair
> suitability-gated quantitative baseline, transaction costs, five investor
> profiles and uncertainty intervals that respect serial dependence, do the
> coordinated agent views add measurable value?

The answer in the versioned run is **no clear alpha advantage**. The hybrid is
almost indistinguishable from `quant_only_gated`, the same optimizer and gate
with neutral views. The blocking suitability gate does reduce breaches and tail
risk, but it is not a guarantee against realised drawdowns.

## Strategies

| Config | Agent views | Optimizer | Suitability gate |
|---|---|---|---|
| `one_over_n` | none | equal weight | no |
| `sixty_forty` | none | 60% equal-weight equity / 40% equal-weight fixed income | no |
| `min_variance` | none | constrained minimum variance | no |
| `quant_only_no_gate` | neutral | constrained mean-variance on the quantitative prior | no |
| `quant_only_gated` | neutral | same optimizer as hybrid | yes |
| `agents_only` | direct consensus | none | no |
| `hybrid` | coordinated views | mean-variance on Black-Litterman posterior | yes |

`quant_only_gated` is the fair baseline: it differs from `hybrid` only in the
absence of agent views. `quant_only_no_gate` isolates the incremental effect of
the gate. `hybrid_shuffled` remains available as an opt-in placebo but is not
part of the Phase 4 primary panel.

All strategies start from cash, pay the same initial build cost, and respect
investor exclusions. The 60/40 sleeve definitions are fixed in
`intellivest/evaluation/backtest.py`. Agent decisions for `agents_only` and
`hybrid` are recomputed at every rebalance using that arm's prior holdings, so
the portfolio and risk analysts see the actual book rather than an empty one.

## Investor panel

The panel spans the binding suitability classes:

| Profile | Class | Horizon | Loss tolerance | Volatility limit |
|---|---:|---:|---:|---:|
| `P1-CAPITAL` | 1 | 3 years | 5% | 8.5% |
| `P2-INCOME` | 2 | 5 years | 10% | 11.0% |
| `P3-BALANCED` | 3 | 10 years | 15% | 14.5% |
| `P4-GROWTH` | 4 | 20 years | 20% | 18.5% |
| `P5-AGGRESSIVE` | 5 | 30 years | 30% | 24.0% |

## Data and timing

- Cached real ETF prices from `data/prices.csv` (14 US ETFs).
- Point-in-time FRED `DGS3MO` risk-free rate from `data/risk_free.csv`.
- Window `2018-01-01` to `2026-06-30`; decisions use only data on or before
  each rebalance date.
- 252-trading-day estimation window and 21-trading-day rebalance interval. The
  one-year lookback is deliberate: the older 756-day default started the first
  rebalance in 2021 and missed the 2020 crash.
- 12 bps transaction cost applied to turnover at each rebalance, including the
  initial build from cash.

## Data split

The forecast models use a fixed time split: dev through `2021-12-31`,
validation `2022-01-01` to `2023-12-31`, and test from `2024-01-01`. Selection
uses validation only; the test window is not used to choose a model.

The advisory panel is a walk-forward analysis, not a model-selection experiment.
The same split boundaries are applied to its return paths only to make the final
readout transparent. The strategies are not trained on dev or validation, and
their weights are not chosen on test.

| Window | Hybrid annual return | Hybrid Sharpe | Hybrid max drawdown |
|---|---:|---:|---:|
| Dev (through 2021-12-31) | 10.1% | 0.63 | -20.1% |
| Validation (2022-2023) | 1.6% | -0.21 | -16.0% |
| Test (from 2024-01-01) | 14.3% | 1.00 | -8.9% |

The test window was strong for almost every configuration, including 1/N and
60/40. The validation window was weak for almost every configuration. This is a
descriptive out-of-sample readout, not a tuned improvement, and it does not
overturn the full-panel null result.

## Regime windows

The windows are fixed before looking at strategy results and are descriptive,
not tuned buckets:

| Key | Window | Purpose |
|---|---|---|
| `2020_crash` | 2020-02-19 to 2020-04-30 | COVID crash |
| `2022_rates` | 2022-01-01 to 2022-12-31 | Rate shock |

## Bootstrap and paired tests

- Moving-block bootstrap with 21-day blocks and 1000 replicates.
- 95% percentile confidence intervals for Sharpe and max drawdown.
- Paired tests resample the **same** block indices for hybrid and each baseline.
- Reported p-values are two-sided percentile bootstrap p-values, not exact
  finite-sample tests. No multiple-testing correction is applied; the panel of
  30 hybrid-vs-baseline comparisons is descriptive.

## Profile-alignment metrics

For every profile and strategy:

- realised annualised volatility versus the policy volatility limit;
- realised max drawdown versus the investor's stated loss tolerance;
- mean cosine similarity of consecutive rebalance weight vectors;
- mean coordinator agreement, reported only for agent-driven strategies.

The volatility and drawdown ratios can exceed 1.0. A blocking gate checks the
contemporaneous risk estimate, not the ex-post path; realised drawdowns over a
multi-year sample can therefore breach a stated tolerance even when the gate
passes at each decision date. This is visible in the results rather than hidden.

## Versioned results

The report is generated at `results/evaluation.md` and
`results/evaluation.json`. The same run is also published as
`results/phase4_evaluation.md` and `results/phase4_evaluation.json`. The
aggregate table is:

| Config | Return | Vol | Sharpe | MaxDD | Turnover | Cost | Breach | Vol/limit | DD/limit | Stability | Agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `one_over_n` | 12.8% | 14.5% | 0.66 | -30.1% | 0.6% | 0.8 bps | 50% | 1.09x | 2.71x | 1.00 | n/a |
| `sixty_forty` | 10.0% | 11.6% | 0.58 | -23.8% | 0.6% | 0.8 bps | 47% | 0.86x | 2.14x | 1.00 | n/a |
| `min_variance` | 5.1% | 8.0% | 0.24 | -18.3% | 5.0% | 7.2 bps | 60% | 0.59x | 1.64x | 0.99 | n/a |
| `quant_only_no_gate` | 13.0% | 13.8% | 0.70 | -26.4% | 10.2% | 14.8 bps | 56% | 1.02x | 2.34x | 0.97 | n/a |
| `quant_only_gated` | 9.0% | 11.2% | 0.45 | -24.9% | 8.2% | 11.9 bps | 41% | 0.76x | 2.15x | 0.98 | n/a |
| `agents_only` | 7.4% | 13.3% | 0.33 | -27.9% | 76.1% | 110.5 bps | 74% | 0.91x | 2.26x | 0.26 | 0.78 |
| `hybrid` | 9.1% | 11.3% | 0.46 | -24.9% | 8.0% | 11.7 bps | 40% | 0.76x | 2.15x | 0.98 | 0.81 |

Key readings:

- **Gate effect.** The hybrid breach rate is 40% versus 56% without the gate.
  The gate is doing real work, but it does not eliminate breaches, especially
  for the most conservative profile whose 5% loss tolerance was already
  exceeded by the realised path.
- **Fair baseline.** Hybrid annual return is 9.1% versus 9.0% for
  `quant_only_gated`; volatility is 11.3% versus 11.2%; Sharpe is 0.46 versus
  0.45. The agent layer does not establish an advantage over the same numerical
  pipeline with neutral views.
- **Strategic benchmark.** Hybrid returns 9.1% versus 10.0% for 60/40, with a
  worse max drawdown (-24.9% versus -23.8%). Beating 60/40 is not the relevant
  test for the agent claim.
- **Agents without guardrails.** `agents_only` is the clearest failure mode in
  the panel: 7.4% return, 76.1% turnover per rebalance, 110.5 bps annual cost,
  and a 74% breach rate. Holdings-aware portfolio views make it churn, not
  improve.
- **Paired bootstrap.** 4 of 30 Sharpe differences and 2 of 30 max-drawdown
  differences have p < 0.05 before any multiple-testing correction. The
  directions are mixed; this is not evidence of a general hybrid edge.
- **Regimes.** In the 2020 crash window hybrid returned -6.2% versus -7.8% for
  60/40. In the 2022 rate shock hybrid returned -5.8% versus -9.2% for 60/40.
  Both are risk-reduction observations in a small number of episodes.
- **Test-window readout.** Hybrid annual return is 14.3% in test versus 1.6% in
  validation, with Sharpe 1.00 versus -0.21 and test max drawdown -8.9%. This is
  descriptive: the same strong test period lifted 1/N and 60/40 as well, and the
  full-panel result remains the claim of record.
- **No seed-robustness claim.** The real-data path is identical across seeds,
  so repeated seeds would not be independent trials. Phase 4 replaces the old
  five-seed standard deviations with the five-profile panel and moving-block
  intervals.

## Reproduction

```bash
# Full Phase 4 panel, bootstrap and paired tests
python scripts/run_evaluation.py

# Faster run without bootstrap
python scripts/run_evaluation.py --no-bootstrap

# Tests
python -m pytest -q
```

## Limitations

- Backtest only; not live performance or investment advice.
- Five profiles, one reference universe and one ETF proxy mapping.
- The bootstrap addresses serial dependence in returns, not model-selection
  uncertainty or regime uncertainty.
- No multiple-testing correction across 30 hybrid-vs-baseline comparisons.
- Real news is not wired in, so the sentiment agent abstains.
- The optional LLM path is exercised with deterministic/mock providers in the
  reproducible evaluation, not a live paid model.
