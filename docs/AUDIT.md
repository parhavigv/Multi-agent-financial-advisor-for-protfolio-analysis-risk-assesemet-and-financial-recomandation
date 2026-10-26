# Audit (Phase 0)

## Data
- Synthetic-only by default (seeded regime-switching factor model). If `data/prices.csv` exists, `CsvMarket` can load it, but repo uses synthetic data.
- Deterministic with seeds; regime macro path fixed across seeds (cross-section varies).

## Agents
- Rule-based agents by default (deterministic analyst). LLM layer optional via env vars (disabled unless keys provided).

## Benchmarks
- Missing 1/N and 60/40 explicit benchmarks in current backtester configs (configs: quant_only, agents_only, hybrid). Not reported by default.

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
