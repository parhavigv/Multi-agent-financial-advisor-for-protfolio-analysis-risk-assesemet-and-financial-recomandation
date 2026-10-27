# Agents, Black-Litterman views and the Phase 3 ablation

This document describes the Phase 3 design: the optional LLM analyst path, how
coordinated agent views enter the portfolio optimizer, and the placebo ablation
that tests whether the asset-level mapping adds value.

The short version: deterministic agents remain the default and the fallback.
When an LLM is enabled it may only **interpret** the fact sheet and propose
bounded signals; it never chooses weights, supplies performance metrics, or
writes evidence. Agent views enter a Black-Litterman posterior with explicit
uncertainty and hard influence caps. In the current real-ETF ablation the
suitability gate is clearly valuable, but the placebo is economically
indistinguishable from the true asset mapping. We report that null result.

## Pipeline

```text
profile
  -> point-in-time market snapshot
  -> quantitative features (expected returns, covariance, risk, regime)
  -> specialist agents
       deterministic analyst (default)
       optional LLM analyst (strict JSON, grounded)
  -> coordinator (credibility-weighted tilts, confidence, dispersion)
  -> Black-Litterman posterior
  -> constrained mean-variance optimizer
  -> blocking suitability gate + bounded repair
  -> recommendation narrator (deterministic or guarded LLM)
  -> hash-chained decision trace
```

The orchestrator is the only component that knows this ordering. The LLM is
never in the control path: a failed or invalid model call produces the same
deterministic report that would have been produced without an LLM, marked
`degraded=True` so the trace distinguishes the two cases.

## Optional LLM path

### Strict schema

The LLM returns an `LLMReportPayload` with exactly these fields:

- `summary`: 1-1200 characters.
- `confidence`: in `[0, 1]`.
- `summary_citations`: dotted fact-sheet paths used by the summary.
- `signals`: at least one `LLMSignalPayload`, at most 40. Each signal has
  `name`, `subject`, `direction`, `strength`, `rationale`, `confidence` and one
  or more `citations`.

Unknown keys, including `metrics` and `evidence`, are rejected. `subject` must
be one of the model-universe tickers, `MARKET` or `PORTFOLIO`. Direction and
strength must agree.

### Citations

Citations are exact dotted paths into the flattened fact sheet, for example:

```text
assets.SPX.momentum_12_1
portfolio.sharpe
portfolio.cvar_95_daily
policy.max_cvar
benchmark.annual_volatility
```

Validation rejects a path that does not exist, and rejects a citation that is
not a leaf field. Evidence is rebuilt from the cited values; the model cannot
invent an evidence object. Signal citations are also stored on
`Signal.citations` so the decision trace shows what the model actually used.

### Numeric grounding

Every number in the summary must be represented by one of the summary's cited
field values. Every number in a signal rationale must be represented by that
signal's cited values. The guard covers:

- raw decimals and rounded forms such as `0.16`, `16.0%`, `0.16x`;
- percentages, including sentence-final punctuation;
- basis points, years, days, months, weeks and standard deviations;
- currency amounts with `$`, `€` or `£` and thousands separators;
- ratios such as `60/40`;
- ISO dates and month-name dates;
- bare integers, so an invented `Sharpe 3` is rejected just like `Sharpe 3.14`;
- structural labels such as `12-1 momentum`, which are not quantities.

A citation to a field whose value is `0.1602` authorises representations of
that value, not arbitrary other numbers. If any check fails, the agent falls
back and the report is marked degraded.

### Failure handling

`AnalystAgent._with_llm` calls the optional model path, then the deterministic
analyst if anything fails. The fallback covers:

- transport, authentication and JSON errors;
- schema violations and unknown fields;
- unknown subjects or missing citations;
- citations to nonexistent fields;
- invented numbers in the summary or rationale;
- a model response with no signals.

The deterministic narrator applies the same numeric guard to the final
recommendation prose, using only numbers already computed by the pipeline.

## Coordinator

The coordinator is deterministic. It does not call an LLM. For each asset:

- each agent signal is weighted by agent credibility and signal confidence;
- the weighted mean strength becomes the raw tilt;
- dispersion across agents damps the tilt (`1 - 0.5 * dispersion`);
- strong opposing views trigger an explicit conflict and cancel further;
- per-asset confidence is the credibility-weighted mean confidence, also
  damped by dispersion;
- a market view and a portfolio view are aggregated separately.

The coordinator records `view_confidence`, `view_dispersion`,
`market_confidence`, `portfolio_confidence`, per-agent contribution and any
conflicts in the decision trace.

## Black-Litterman layer

The quantitative engine supplies prior expected returns `mu` and covariance
`Sigma`. Each coordinated asset tilt is an absolute view:

```text
P_i = e_i
Q_i = mu_i + tilt_i * view_shift * conviction
conviction = clip(1 + 0.35 * portfolio_score, 0.70, 1.10)
```

A non-zero market score adds an equal-weight equity-portfolio view:

```text
P_market = 1 / n_equity on equity assets, 0 elsewhere
Q_market = P_market mu + market_overlay * market_score * view_shift * conviction
```

Confidence becomes view uncertainty using the He-Litterman form:

```text
Omega_i = P_i (tau Sigma) P_i' * (1 - c_i) / c_i
```

with `c_i` clipped to `[0.05, 0.99]`. The posterior is the standard
Black-Litterman update:

```text
mu_post = mu + tau Sigma P' (P tau Sigma P' + Omega)^-1 (Q - P mu)
```

Two hard caps are applied after the update:

- each asset's posterior movement is clipped to `+/- view_shift`;
- the L1 sum of absolute movements is capped by `bl_max_total_influence`.

The trace records the prior, posterior, raw views, view returns, Omega,
confidence, dispersion, total influence, whether the cap bound, and the number
of views. The same function is used by the advisory path and the
counterfactual, so they cannot drift apart.

### Why mean-variance for every risk class

The advisory optimizer used to switch conservative profiles to minimum-CVaR.
That objective has no expected-return term, so it would silently ignore every
agent view. The advisory path therefore uses constrained mean-variance for all
risk classes. Risk class still enters through:

- risk aversion (`6.0, 4.5, 3.0, 2.0, 1.4` for classes 1-5);
- maximum single position and sector caps;
- the blocking suitability constraints and repair loop.

`min_cvar` remains in `quant.optimize` as a tail-risk alternative for callers
that explicitly want it, but it is not used by `AdvisoryEngine.optimise`.

## Phase 3 ablation

`intellivest-evaluate` runs four configurations on the same seeded market,
rebalance dates, lookback and transaction-cost assumption:

| Config | Agent views | Optimizer | Suitability gate |
|---|---|---|---|
| `quant_only` | neutral | mean-variance | no |
| `agents_only` | direct consensus | none | no |
| `hybrid` | true coordinated views | mean-variance on BL posterior | yes |
| `hybrid_shuffled` | permuted asset mapping | mean-variance on BL posterior | yes |

`hybrid_shuffled` preserves the distribution of tilts, confidence and
dispersion and keeps the market view; it only permutes which asset receives
each tilt. The permutation is seeded by the run seed and rebalance index, so it
is reproducible.

### Actual results

Five seeds, primary seed `20261023`, window `2018-01-01` to `2026-06-30`,
rebalance every 21 trading days, 12 bps transaction cost. The run uses the
cached real ETF prices in `data/prices.csv` and the point-in-time FRED
`DGS3MO` series in `data/risk_free.csv` (`RealMarket`). Values are means
across seeds; the only non-zero standard deviations were 0.02% for shuffled
annual return and 0.01% for shuffled max drawdown.

| Metric | quant_only | agents_only | hybrid | hybrid_shuffled |
|---|---:|---:|---:|---:|
| Annual return | 11.64% | 13.31% | 9.87% | 9.84% |
| Volatility | 12.32% | 16.43% | 10.81% | 10.77% |
| Sharpe | 0.69 | 0.62 | 0.62 | 0.62 |
| Sortino | 0.97 | 0.87 | 0.88 | 0.88 |
| Max drawdown | -15.91% | -17.92% | -16.01% | -16.07% |
| CVaR95 (1-day) | 1.78% | 2.40% | 1.52% | 1.51% |
| Calmar | 0.73 | 0.74 | 0.62 | 0.61 |
| Turnover / rebalance | 6.8% | 20.2% | 6.6% | 6.4% |
| Annual cost | 9.9 bps | 29.5 bps | 9.6 bps | 9.4 bps |
| Suitability breach rate | 41% | 70% | 0% | 0% |
| Average equity weight | 74% | 85% | 61% | 60% |

### Interpretation

- **The suitability gate is the measurable contribution.** Hybrid eliminates
  suitability breaches and reduces turnover and tail risk relative to
  unconstrained consensus. This is a risk-control result, not an alpha result.
- **The agents-on versus agents-off comparison is not a win on return.** Hybrid
  gives up annual return versus `quant_only` (9.87% versus 11.64%) while
  reducing volatility (10.81% versus 12.32%), CVaR (1.52% versus 1.78%) and
  suitability breaches (0% versus 41%). Sharpe is lower (0.62 versus 0.69).
- **The placebo is essentially indistinguishable.** Hybrid and
  `hybrid_shuffled` differ by 0.03 percentage points of annual return and 0.00
  on Sharpe. The placebo does **not** support a claim that the asset-level
  mapping of agent views adds value in this real-ETF backtest.
- **The seed panel is not five independent trials.** The seeds share the same
  macro regime path and differ only in cross-sectional draws, and the resulting
  metrics are almost identical. Do not read the standard deviations as
  confidence intervals.
- **No test-period tuning.** The evaluation uses the full configured window for
  this ablation; forecast-model selection remains a separate validation-only
  protocol documented in `docs/FORECASTING.md`.

The honest conclusion is that the system demonstrates a traceable,
suitability-aware decision process with guarded agent reasoning, not
outperformance from agent views. Further work would need a larger investor
panel, real point-in-time news, and an out-of-sample design that can distinguish
agent skill from noise.

## Reproduction

```bash
# Full ablation, five seeds, writes results/evaluation.{json,md}
intellivest-evaluate

# Primary seed only
intellivest-evaluate --skip-robustness

# Tests
pytest -q
```

The generated evaluation is versioned at `results/phase3_ablation.md` (with
the raw JSON at `results/phase3_ablation.json`). Other run artifacts and audit
logs are gitignored.

## Limitations

- The ablation uses real ETF price history through 2026-06-29 and a single
  reference investor profile. It is a backtest, not live performance, and it is
  not a large investor panel. Real news is not wired in, so the sentiment agent
  abstains rather than inventing a narrative.
- The LLM path is exercised with a mock provider in tests. A live provider is
  optional and not part of the reproducible evaluation.
- The placebo tests the asset mapping only. It does not test whether the market
  or portfolio-level views add value.
- The numeric guard is a whitelist of representations of cited values. It
  prevents invented magnitudes but cannot enforce that a correctly cited number
  is used in a semantically correct sentence; a human review and the
  deterministic narrator remain the final controls.
- The five-seed panel shares one macro path and should not be treated as a
  statistical significance test.
