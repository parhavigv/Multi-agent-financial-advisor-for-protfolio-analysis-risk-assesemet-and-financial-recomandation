# Audit and phase resolutions

## Data
- The Phase 0 baseline was synthetic-only (seeded regime-switching factor model). Phase 1 added `RealMarket`; when the cached real files exist, subsequent evaluations use real ETF prices and FRED data (see the Phase 1 resolution below).
- Deterministic with seeds; regime macro path fixed across seeds (cross-section varies).

## Agents
- Rule-based agents by default (deterministic analyst). LLM layer optional via env vars (disabled unless keys provided).

## Benchmarks
- Phase 0 lacked explicit 1/N and 60/40 benchmarks. The Phase 3 ablation compares quant_only, agents_only, hybrid and hybrid_shuffled, but still does not report a standalone 1/N or 60/40 benchmark.

## Investor/profile
- Single reference investor in evaluation (`EVAL-001`); CLI uses demo profile. No large panel by default.

## Risk-free
- `risk_free_rate` defaults to 0.065 (6.5%) in config.

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
