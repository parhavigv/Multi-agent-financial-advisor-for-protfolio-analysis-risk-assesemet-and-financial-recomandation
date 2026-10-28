# Audit and phase resolutions

## Data
- The Phase 0 baseline was synthetic-only (seeded regime-switching factor model). Phase 1 added `RealMarket`; when the cached real files exist, subsequent evaluations use real ETF prices and FRED data (see the Phase 1 resolution below).
- The synthetic provider is deterministic with seeds; the macro regime path is fixed across seeds. This is reproducibility, not an independent-sample robustness design.

## Agents
- Rule-based agents by default (deterministic analyst). LLM layer optional via env vars (disabled unless keys provided).

## Benchmarks
- Phase 0 lacked explicit 1/N and 60/40 benchmarks. Phase 4 adds both, plus constrained minimum variance, quant-only with and without the gate, agents-only, and hybrid. See the Phase 4 resolution below.

## Investor/profile
- Phase 3 used a single reference investor (`EVAL-001`); Phase 4 adds a five-profile panel spanning binding suitability classes 1-5. The CLI demo profile is separate from the evaluation panel.

## Risk-free
- `risk_free_rate` defaults to 0.065 (6.5%) only as an explicit fallback. Real
  evaluations use the point-in-time FRED `DGS3MO` observation on or before each
  decision date.
- Optimizer `expected_sharpe` and the post-optimization annotation now use the
  same supplied risk-free rate, so the trace cannot show two Sharpe
  conventions.

## Optimizer
- `Constraints.diversification` defaults to 1.0 in orchestrator/backtester paths.
- Script `scripts/quant_diversification_check.py` quantifies effect:
  - With diversification=1.0: see `results/baseline_synthetic/diversification_check.json` (hhi/effective_n/max_weight)
  - Effect depends on inputs/constraints; diversification term encourages spread but constraints dominate here.

## Baseline artifacts
- `results/baseline_synthetic/evaluation.json`, `evaluation.md`, `equity_curves.csv`, `diversification_check.json`

## Phase 1 resolution (real data)
- Added `RealMarket` (`intellivest/data/real.py`): 14 US ETFs mapped onto the
  project tickers, cached to `data/prices.csv`; FRED `DGS3MO` cached to
  `data/risk_free.csv`.
- The risk-free rate is now **point-in-time** (latest observation on or before
  each decision date) and is recorded in the decision trace. The 6.5%
  constant remains only as an explicit fallback when FRED is unreachable.
- Real news is not wired in, so the sentiment agent abstains
  (`confidence=0`, no signals) rather than inventing a textual view.
- No-lookahead tests cover the real-shaped path, including a truncation test
  that mutates all future rows and asserts an identical snapshot.
- Details: `docs/REAL_DATA.md`.

## Phase 2 resolution (forecasting)
- Purged, embargoed, expanding walk-forward with time-based dev/validation/test
  splits and validation-only model selection.
- Selected `shrunk_mean` for returns and `ewma` for volatility; ML candidates
  were rejected by the stability/margin rules.
- Held-out test evidence: return rank IC 0.0977, RMSE 0.0436; volatility QLIKE
  -2.7941. Shuffled-label control validation rank IC 0.0241 (t = 0.46),
  consistent with no leakage; HMM selected three states by BIC -274.9.
- Protocol and limitations: `docs/FORECASTING.md`; generated evidence:
  `results/forecast_report.md`.

## Phase 3 resolution (agents, Black-Litterman and placebo)
- The optional LLM path now uses a strict Pydantic schema. The model may only
  interpret the fact sheet: it cannot supply weights, metrics or evidence.
  Each signal cites exact dotted fact-sheet paths, and every number in the
  summary or rationale must be represented by a cited value. Any schema,
  citation, grounding or transport failure falls back to the deterministic
  agent and marks the report `degraded=True`.
- Numeric grounding covers decimals, percentages, basis points, units,
  currency, ratios, ISO/month dates and bare integers, including
  sentence-final punctuation.
- Ad-hoc expected-return tilts were replaced by a Black-Litterman posterior:
  `Q_i = mu_i + tilt_i * view_shift`,
  `Omega_i = P_i tau Sigma P_i' (1-c)/c`, per-asset movement caps and a global
  L1 influence cap. A market view is an equal-weight equity-portfolio view.
- The advisory optimizer now uses constrained mean-variance for every risk
  class. Minimum-CVaR has no expected-return term and would silently ignore
  the views; risk class still enters through risk aversion, caps and the
  suitability gate.
- `hybrid_shuffled` was added as a placebo: it permutes the asset-to-tilt
  mapping (and the associated confidence/dispersion) with a seeded RNG while
  keeping the market view and the distribution of views.

### Phase 3 ablation (five seeds, real ETF data)
- `quant_only`: annual return 11.64%, volatility 12.32%, Sharpe 0.69, CVaR95
  1.78%, suitability breach rate 41%.
- `agents_only`: 13.31%, 16.43%, 0.62, 2.40%, 70%.
- `hybrid`: 9.87%, 10.81%, 0.62, 1.52%, 0%.
- `hybrid_shuffled`: 9.84%, 10.77%, 0.62, 1.51%, 0%.
- The suitability gate is the measurable contribution: it removes breaches and
  reduces turnover and tail risk relative to unconstrained consensus.
- The placebo is essentially indistinguishable from the true mapping on return
  and Sharpe. This run does **not** support a claim that asset-level agent
  views add alpha. The null result is reported, not tuned away.
- The five seeds share one macro path and are not independent trials; the
  near-zero standard deviations should not be read as confidence intervals.
- Design and limitations: `docs/AGENTS.md`; versioned evidence:
  `results/phase3_ablation.md` and `results/phase3_ablation.json`.

## Phase 4 resolution (multi-profile evaluation)
- The evaluation now runs seven primary configurations: `one_over_n`,
  `sixty_forty`, `min_variance`, `quant_only_no_gate`, `quant_only_gated`,
  `agents_only`, and `hybrid`. `quant_only_gated` is the fair baseline: same
  optimizer and gate as the hybrid, neutral views.
- Five investor profiles cover binding suitability classes 1-5. Every strategy
  starts from cash, pays the same initial build cost, and honours exclusions.
- The estimation window is 252 trading days with a 21-day rebalance interval,
  so the real-data sample covers the 2020 crash and 2022 rate shock. The older
  756-day default missed the 2020 episode.
- Transaction costs and turnover are applied at every rebalance. Profile
  alignment reports realised volatility versus policy limit, realised drawdown
  versus stated loss tolerance, consecutive-weight stability and coordinator
  agreement.
- Uncertainty uses a moving-block bootstrap with 21-day blocks and 1000
  replicates. Paired tests resample the same block indices for hybrid and each
  baseline. The p-value is a two-sided percentile bootstrap p-value; no
  multiple-testing correction is applied.
- In the versioned real-ETF run, hybrid and `quant_only_gated` are essentially
  tied (9.1% versus 9.0% annual return; Sharpe 0.46 versus 0.45). The gate
  reduces the aggregate breach rate for the same optimizer from 56% without the
  gate to 41% with it, but does not eliminate breaches. `agents_only` is the
  failure mode the guardrails contain: 7.4%
  return, 76.1% turnover per rebalance, 110.5 bps annual cost, and a 74% breach
  rate. Of 30 paired Sharpe comparisons, 4 have p < 0.05 before correction;
  2 of 30 max-drawdown comparisons do. The directions are mixed.
- The real-data path is identical across seeds, so the old five-seed spread is
  not an independent-sample robustness result. Phase 4 drops that claim.
- Protocol and limitations: `docs/EVALUATION.md`; versioned evidence:
  `results/evaluation.md` and `results/evaluation.json`, with the same run at
  `results/phase4_evaluation.md` and `results/phase4_evaluation.json`.

## Phase 5 resolution (quality hardening and traceability)
- Added `tests/test_phase5.py` for known-series metric conventions, optimizer
  feasibility and `solver_status`, caps/sector limits, risk-free Sharpe
  consistency, audit tamper detection, suitability repair, and an end-to-end
  profile-to-hash trace.
- Fixed the optimizer's `expected_sharpe` so it uses the same point-in-time
  risk-free rate as the post-optimization annotation. The engine, counterfactual
  and backtester pass the rate through explicitly.
- The failed-solve fallback now uses a feasibility-aware start. When position
  caps and a sector cap are jointly feasible, the fallback records a
  `fallback:` status and still returns a feasible weight vector.
- Added a descriptive dev/validation/test table to the advisory report. The
  advisory strategies are not trained on those splits; the labels are applied
  only to the walk-forward return paths. The test window is reported even where
  it differs from validation.
- Versioned `results/evaluation.*` and `results/trace_example.*` so the README
  claims and the review trace point at committed evidence.
- The full suite is 96 tests. The suite and the generated reports are the
  reproducibility boundary; no live-trading claim is made.
- Worked trace: `results/trace_example.md` and `results/trace_example.json`,
  generated by `scripts/run_trace_example.py` on the deterministic synthetic
  provider. It is an audit demonstration, not market evidence.
- Audit scope remains local: the chain detects edited bodies and broken
  `previous_hash` links, but it is not a write-once ledger and has no external
  timestamp or notary.
