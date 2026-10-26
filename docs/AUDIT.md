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
