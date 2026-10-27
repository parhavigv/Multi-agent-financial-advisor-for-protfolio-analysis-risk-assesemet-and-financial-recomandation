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

## Phase 4 evaluation (current)

Phase 3 introduced the Black-Litterman view layer and the shuffled-view
placebo. Phase 4 replaces its seed panel with the evaluation protocol that is
actually defensible for the claim:

- five investor profiles spanning binding suitability classes 1-5;
- 1/N and 60/40 strategic benchmarks;
- minimum variance and quant-only baselines;
- a fair `quant_only_gated` baseline that uses the same optimizer and gate as
  the hybrid but neutral views;
- 252-day estimation window so the sample covers the 2020 crash and 2022 rate
  shock;
- holdings-aware agent decisions for `agents_only` and `hybrid`, recomputed at
  each rebalance with the arm's actual prior portfolio;
- transaction costs and turnover;
- moving-block bootstrap intervals and paired hybrid-vs-baseline tests;
- realised profile-alignment diagnostics.

`hybrid_shuffled` is retained as an opt-in placebo but is not part of the
primary seven-arm panel.

The full protocol, per-profile tables, confidence intervals and limitations are
in [EVALUATION.md](EVALUATION.md). The versioned evidence is
`results/phase4_evaluation.md` and `results/phase4_evaluation.json`.

### What the current run shows

- Hybrid and `quant_only_gated` are essentially tied: 9.1% versus 9.0% annual
  return, 11.3% versus 11.2% volatility, and Sharpe 0.46 versus 0.45. The agent
  views do not establish an alpha advantage over the same numerical pipeline
  with neutral views.
- The blocking gate reduces the breach rate from 56% without a gate to 40% with
  the gate, but it does not eliminate breaches. The most conservative profile's
  5% loss tolerance was already exceeded by the realised multi-year path.
- Hybrid returned -6.2% in the fixed 2020 crash window versus -7.8% for 60/40,
  and -5.8% in the 2022 rate shock versus -9.2% for 60/40. These are
  risk-reduction observations in two episodes, not general outperformance.
- `agents_only` is the failure mode the guardrails are meant to contain: 7.4%
  return, 76.1% turnover per rebalance, 110.5 bps annual cost, and a 74% breach
  rate. Holdings-aware portfolio views make it churn.
- Of 30 paired Sharpe comparisons, 4 have p < 0.05 before multiple-testing
  correction; 2 of 30 max-drawdown comparisons do. The directions are mixed.
  This is descriptive evidence, not a significant-edge claim.

### Phase 3 placebo (historical)

The shuffled-view placebo remains useful: it preserves the distribution of
agent views and the market view while destroying which asset receives each
view. The historical Phase 3 result was that the placebo was indistinguishable
from the true mapping, which argued against an asset-mapping alpha claim. Those
files remain at `results/phase3_ablation.md` and `results/phase3_ablation.json`.

The old five-seed panel is **not** a robustness result: every seed shared the
same macro path, so the near-zero spread was not an independent sample. Phase 4
does not use it as evidence and does not report seed-robustness claims.

## Reproduction

```bash
# Phase 4 panel with bootstrap intervals and paired tests
python -c "from intellivest.cli import evaluate_main; raise SystemExit(evaluate_main([]))"

# Fast run without bootstrap
python -c "from intellivest.cli import evaluate_main; raise SystemExit(evaluate_main(['--no-bootstrap']))"

# Tests
python -m pytest -q
```

## Limitations

- Backtest only; not live performance or investment advice.
- Five profiles, one reference universe and one ETF proxy mapping.
- The bootstrap addresses serial dependence in returns, not model-selection or
  regime uncertainty.
- No multiple-testing correction across the 30 hybrid-vs-baseline comparisons.
- Real news is not wired in, so the sentiment agent abstains rather than
  inventing a narrative.
- The LLM path is exercised with mock/deterministic providers in the
  reproducible evaluation. A live provider is optional and not part of the
  versioned result.
- The placebo tests the asset mapping only; it does not test whether market or
  portfolio-level views add value.
- The numeric guard prevents invented magnitudes but cannot enforce that a
  correctly cited number is used in a semantically correct sentence.
