# Walk-forward evaluation

Primary seed 20261023, window 2018-01-01 → 2026-06-30, rebalance every 21 trading days, 12.0 bps transaction cost. Market source: RealMarket. The macro regime path is fixed across seeds; the seed controls only cross-sectional asset draws.

**Configurations.** `quant_only` = constrained mean-variance optimizer with a neutral view set and no suitability gate. `agents_only` = direct agent-consensus allocation, bypassing every quantitative guardrail. `hybrid` = agent views enter the Black-Litterman posterior, the optimizer chooses weights, and the blocking suitability gate remediates the result. `hybrid_shuffled` = the same pipeline with the asset-to-tilt mapping permuted by a seeded RNG (placebo).


## Primary seed

| Metric | quant_only | agents_only | hybrid | hybrid_shuffled |
|---|---|---|---|---|
| Annual return | 11.64% | 13.31% | 9.87% | 9.86% |
| Volatility | 12.32% | 16.43% | 10.81% | 10.77% |
| Sharpe | 0.69 | 0.62 | 0.62 | 0.62 |
| Sortino | 0.97 | 0.87 | 0.88 | 0.88 |
| Max drawdown | -15.91% | -17.92% | -16.01% | -16.04% |
| CVaR95 (1-day) | 1.78% | 2.40% | 1.52% | 1.51% |
| Calmar | 0.73 | 0.74 | 0.62 | 0.61 |
| Turnover / rebalance | 6.8% | 20.2% | 6.6% | 6.4% |
| Annual cost | 9.9 bps | 29.5 bps | 9.6 bps | 9.3 bps |
| Suitability breach rate | 41% | 70% | 0% | 0% |
| Average equity weight | 74% | 85% | 61% | 60% |

## Robustness across 5 seeds

Mean ± standard deviation across seeds 20261023, 7, 101, 2718, 31337. Every seed shares the same macro regime path and differs only in cross-sectional draws.

| Metric | quant_only | agents_only | hybrid | hybrid_shuffled |
|---|---|---|---|---|
| Annual return | 11.64% ± 0.00% | 13.31% ± 0.00% | 9.87% ± 0.00% | 9.84% ± 0.02% |
| Volatility | 12.32% ± 0.00% | 16.43% ± 0.00% | 10.81% ± 0.00% | 10.77% ± 0.00% |
| Sharpe | 0.69 ± 0.00 | 0.62 ± 0.00 | 0.62 ± 0.00 | 0.62 ± 0.00 |
| Sortino | 0.97 ± 0.00 | 0.87 ± 0.00 | 0.88 ± 0.00 | 0.88 ± 0.00 |
| Max drawdown | -15.91% ± 0.00% | -17.92% ± 0.00% | -16.01% ± 0.00% | -16.07% ± 0.01% |
| CVaR95 (1-day) | 1.78% ± 0.00% | 2.40% ± 0.00% | 1.52% ± 0.00% | 1.51% ± 0.00% |
| Calmar | 0.73 ± 0.00 | 0.74 ± 0.00 | 0.62 ± 0.00 | 0.61 ± 0.00 |
| Turnover / rebalance | 6.8% ± 0.0% | 20.2% ± 0.0% | 6.6% ± 0.0% | 6.4% ± 0.0% |
| Annual cost | 9.9 bps ± 0.0 bps | 29.5 bps ± 0.0 bps | 9.6 bps ± 0.0 bps | 9.4 bps ± 0.0 bps |
| Suitability breach rate | 41% ± 0% | 70% ± 0% | 0% ± 0% | 0% ± 0% |
| Average equity weight | 74% ± 0% | 85% ± 0% | 61% ± 0% | 60% ± 0% |

## Reading the results

* **Suitability is the decisive axis.** The hybrid gate ends at 0.0% breaches versus 69.7% for unconstrained agent consensus.
* **Turnover.** Unconstrained agents trade 3.1× more per rebalance, which is why their cost drag is 29.5 bps/yr versus 9.6 bps/yr.
* **Risk-adjusted outcome.** Hybrid Sharpe 0.62 ± 0.00 versus 0.62 ± 0.00 for agents only; max drawdown -16.0% versus -17.9%.
* **Return is not the objective function.** Hybrid annual return is 9.87% versus 13.31% for unconstrained agents; volatility is 10.81% versus 16.43%, and CVaR95 is 1.52% versus 2.40%. The design optimises for suitability and risk control, not alpha.
* **Versus the numerical engine.** Hybrid annual return is 9.87% versus 11.64% for quant_only, with volatility 10.81% versus 12.32%, CVaR95 1.52% versus 1.78%, and suitability breaches 0.0% versus 40.9%. The agent layer is a risk-and-suitability layer here, not a return-enhancement layer.
* **Placebo check.** Permuting which asset receives each agent tilt leaves Sharpe at 0.62 ± 0.00 versus 0.62 ± 0.00 for the true mapping, and annual return at 9.84% versus 9.87%. The gap is negligible in this run, so the placebo does not support a claim that the asset-level mapping adds value. With 5 seeds this is descriptive, not a formal significance test.

