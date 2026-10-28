# Phase 4 evaluation: a claim-supporting comparison

Window 2018-01-01 to 2026-06-30; 252-day estimation window; rebalance every 21 trading days; 12.0 bps transaction cost. Market source: **RealMarket**.

The panel contains five investor profiles spanning binding suitability classes 1-5. Every strategy starts from cash and pays the same initial build cost. The fair baseline is `quant_only_gated`: it uses the same optimizer and the same blocking suitability gate as `hybrid`, but no agent views. `quant_only_no_gate` isolates the value of the gate.

Bootstrap intervals use a moving-block bootstrap with 21-day blocks and 1000 replicates. Paired tests resample the same block indices for hybrid and each baseline, preserving the shared trading calendar. The reported p-value is a two-sided percentile bootstrap p-value, not an exact finite-sample test; no multiple-testing correction is applied.

No seed-robustness claim is made. The real-data market path does not vary with the random seed, so repeated seeds are not independent trials and are not reported as a robustness panel.

## Investor panel

| Profile | Risk class | Horizon | Loss tolerance | Volatility limit |
|---|---:|---:|---:|---:|
| P1-CAPITAL | 1 | 3 yrs | 5% | 8.5% |
| P2-INCOME | 2 | 5 yrs | 10% | 11.0% |
| P3-BALANCED | 3 | 10 yrs | 15% | 14.5% |
| P4-GROWTH | 4 | 20 yrs | 20% | 18.5% |
| P5-AGGRESSIVE | 5 | 30 yrs | 30% | 24.0% |

## Aggregate results across the five profiles

| Config | Return | Vol | Sharpe | MaxDD | Turnover | Cost | Breach | Vol/limit | DD/limit | Stability | Agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `one_over_n` | 12.8% | 14.5% | 0.66 | -30.1% | 0.6% | 0.8 bps | 50% | 1.09x | 2.71x | 1.00 | n/a |
| `sixty_forty` | 10.0% | 11.6% | 0.58 | -23.8% | 0.6% | 0.8 bps | 47% | 0.86x | 2.14x | 1.00 | n/a |
| `min_variance` | 5.1% | 8.0% | 0.24 | -18.3% | 5.0% | 7.2 bps | 60% | 0.59x | 1.64x | 0.99 | n/a |
| `quant_only_no_gate` | 13.0% | 13.8% | 0.70 | -26.4% | 10.2% | 14.8 bps | 56% | 1.02x | 2.34x | 0.97 | n/a |
| `quant_only_gated` | 9.0% | 11.2% | 0.45 | -24.9% | 8.2% | 11.9 bps | 41% | 0.76x | 2.15x | 0.98 | n/a |
| `agents_only` | 7.4% | 13.3% | 0.33 | -27.9% | 76.1% | 110.5 bps | 74% | 0.91x | 2.26x | 0.26 | 0.78 |
| `hybrid` | 9.1% | 11.3% | 0.46 | -24.9% | 8.0% | 11.7 bps | 40% | 0.76x | 2.15x | 0.98 | 0.81 |

## Data split and test-window readout

The forecast models use dev through **2021-12-31**, validation **2022-01-01 to 2023-12-31**, and test from **2024-01-01**. The advisory strategies are not trained; these same fixed boundaries are applied descriptively to their walk-forward returns, and no advisory parameter is estimated from the test window.

| Config | Dev return | Validation return | Test return | Test Sharpe | Test MaxDD |
|---|---:|---:|---:|---:|---:|
| `one_over_n` | 16.9% | 1.8% | 17.4% | 1.34 | -10.8% |
| `sixty_forty` | 14.4% | 0.1% | 13.1% | 1.16 | -8.4% |
| `min_variance` | 7.7% | -3.3% | 9.2% | 0.90 | -5.6% |
| `quant_only_no_gate` | 14.0% | 4.9% | 18.6% | 1.28 | -10.5% |
| `quant_only_gated` | 9.9% | 1.6% | 14.2% | 1.00 | -8.9% |
| `agents_only` | 12.2% | -3.6% | 11.0% | 0.74 | -9.9% |
| `hybrid` | 10.1% | 1.6% | 14.3% | 1.00 | -8.9% |

The advisory panel is a walk-forward analysis, not a model-selection experiment. These split labels describe fixed calendar boundaries and are reported for transparency; they do not imply that a strategy was trained on dev or selected on validation.

## Per-profile results

| Profile | Class | Config | Return | Vol | Sharpe | MaxDD | Turnover | Cost | Breach | Vol/limit | DD/limit | Stability | Agreement |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1-CAPITAL | 1 | `one_over_n` | 12.8% | 14.5% | 0.66 | -30.1% | 0.6% | 0.8 bps | 100% | 1.71x | 6.01x | 1.00 | n/a |
| P1-CAPITAL | 1 | `sixty_forty` | 10.0% | 11.6% | 0.58 | -23.8% | 0.6% | 0.8 bps | 100% | 1.36x | 4.77x | 1.00 | n/a |
| P1-CAPITAL | 1 | `min_variance` | 5.1% | 8.0% | 0.24 | -18.3% | 5.0% | 7.2 bps | 100% | 0.94x | 3.65x | 0.99 | n/a |
| P1-CAPITAL | 1 | `quant_only_no_gate` | 12.2% | 13.0% | 0.69 | -25.4% | 9.3% | 13.6 bps | 100% | 1.53x | 5.07x | 0.98 | n/a |
| P1-CAPITAL | 1 | `quant_only_gated` | 4.1% | 7.6% | 0.12 | -22.8% | 0.6% | 0.8 bps | 100% | 0.90x | 4.56x | 1.00 | n/a |
| P1-CAPITAL | 1 | `agents_only` | 7.8% | 9.7% | 0.48 | -20.7% | 55.4% | 80.3 bps | 100% | 1.14x | 4.15x | 0.51 | 0.79 |
| P1-CAPITAL | 1 | `hybrid` | 4.1% | 7.6% | 0.12 | -22.8% | 0.6% | 0.8 bps | 100% | 0.90x | 4.56x | 1.00 | 0.82 |
| P2-INCOME | 2 | `one_over_n` | 12.8% | 14.5% | 0.66 | -30.1% | 0.6% | 0.8 bps | 100% | 1.32x | 3.01x | 1.00 | n/a |
| P2-INCOME | 2 | `sixty_forty` | 10.0% | 11.6% | 0.58 | -23.8% | 0.6% | 0.8 bps | 100% | 1.05x | 2.38x | 1.00 | n/a |
| P2-INCOME | 2 | `min_variance` | 5.1% | 8.0% | 0.24 | -18.3% | 5.0% | 7.2 bps | 100% | 0.72x | 1.83x | 0.99 | n/a |
| P2-INCOME | 2 | `quant_only_no_gate` | 12.3% | 13.3% | 0.69 | -25.8% | 10.0% | 14.5 bps | 97% | 1.21x | 2.58x | 0.97 | n/a |
| P2-INCOME | 2 | `quant_only_gated` | 5.3% | 8.2% | 0.26 | -22.7% | 8.6% | 12.5 bps | 89% | 0.75x | 2.27x | 0.97 | n/a |
| P2-INCOME | 2 | `agents_only` | 6.9% | 10.5% | 0.35 | -29.0% | 66.9% | 97.1 bps | 100% | 0.95x | 2.90x | 0.38 | 0.78 |
| P2-INCOME | 2 | `hybrid` | 5.6% | 8.2% | 0.29 | -22.7% | 7.5% | 10.9 bps | 88% | 0.75x | 2.27x | 0.97 | 0.79 |
| P3-BALANCED | 3 | `one_over_n` | 12.8% | 14.5% | 0.66 | -30.1% | 0.6% | 0.8 bps | 26% | 1.00x | 2.00x | 1.00 | n/a |
| P3-BALANCED | 3 | `sixty_forty` | 10.0% | 11.6% | 0.58 | -23.8% | 0.6% | 0.8 bps | 21% | 0.80x | 1.59x | 1.00 | n/a |
| P3-BALANCED | 3 | `min_variance` | 5.1% | 8.0% | 0.24 | -18.3% | 5.0% | 7.2 bps | 100% | 0.55x | 1.22x | 0.99 | n/a |
| P3-BALANCED | 3 | `quant_only_no_gate` | 13.0% | 13.9% | 0.70 | -26.4% | 10.4% | 15.2 bps | 72% | 0.96x | 1.76x | 0.97 | n/a |
| P3-BALANCED | 3 | `quant_only_gated` | 9.1% | 11.6% | 0.50 | -24.8% | 9.5% | 13.7 bps | 14% | 0.80x | 1.65x | 0.97 | n/a |
| P3-BALANCED | 3 | `agents_only` | 8.7% | 13.0% | 0.42 | -22.6% | 78.4% | 113.8 bps | 93% | 0.90x | 1.50x | 0.23 | 0.77 |
| P3-BALANCED | 3 | `hybrid` | 9.2% | 11.6% | 0.52 | -24.7% | 9.5% | 13.8 bps | 14% | 0.80x | 1.65x | 0.97 | 0.82 |
| P4-GROWTH | 4 | `one_over_n` | 12.8% | 14.5% | 0.66 | -30.1% | 0.6% | 0.8 bps | 14% | 0.79x | 1.50x | 1.00 | n/a |
| P4-GROWTH | 4 | `sixty_forty` | 10.0% | 11.6% | 0.58 | -23.8% | 0.6% | 0.8 bps | 12% | 0.63x | 1.19x | 1.00 | n/a |
| P4-GROWTH | 4 | `min_variance` | 5.1% | 8.0% | 0.24 | -18.3% | 5.0% | 7.2 bps | 0% | 0.43x | 0.91x | 0.99 | n/a |
| P4-GROWTH | 4 | `quant_only_no_gate` | 13.5% | 14.4% | 0.72 | -27.0% | 10.7% | 15.5 bps | 11% | 0.78x | 1.35x | 0.97 | n/a |
| P4-GROWTH | 4 | `quant_only_gated` | 12.7% | 14.2% | 0.67 | -27.0% | 11.3% | 16.4 bps | 0% | 0.77x | 1.35x | 0.97 | n/a |
| P4-GROWTH | 4 | `agents_only` | 7.1% | 16.0% | 0.24 | -29.9% | 89.8% | 130.3 bps | 47% | 0.87x | 1.50x | 0.09 | 0.77 |
| P4-GROWTH | 4 | `hybrid` | 12.6% | 14.2% | 0.67 | -27.1% | 11.5% | 16.7 bps | 0% | 0.77x | 1.35x | 0.97 | 0.81 |
| P5-AGGRESSIVE | 5 | `one_over_n` | 12.8% | 14.5% | 0.66 | -30.1% | 0.6% | 0.8 bps | 12% | 0.61x | 1.00x | 1.00 | n/a |
| P5-AGGRESSIVE | 5 | `sixty_forty` | 10.0% | 11.6% | 0.58 | -23.8% | 0.6% | 0.8 bps | 0% | 0.48x | 0.79x | 1.00 | n/a |
| P5-AGGRESSIVE | 5 | `min_variance` | 5.1% | 8.0% | 0.24 | -18.3% | 5.0% | 7.2 bps | 0% | 0.33x | 0.61x | 0.99 | n/a |
| P5-AGGRESSIVE | 5 | `quant_only_no_gate` | 13.8% | 14.6% | 0.73 | -27.4% | 10.7% | 15.6 bps | 1% | 0.61x | 0.91x | 0.97 | n/a |
| P5-AGGRESSIVE | 5 | `quant_only_gated` | 13.8% | 14.6% | 0.73 | -27.4% | 10.9% | 15.9 bps | 0% | 0.61x | 0.91x | 0.97 | n/a |
| P5-AGGRESSIVE | 5 | `agents_only` | 6.3% | 17.1% | 0.18 | -37.5% | 90.2% | 131.0 bps | 32% | 0.71x | 1.25x | 0.09 | 0.77 |
| P5-AGGRESSIVE | 5 | `hybrid` | 13.8% | 14.6% | 0.72 | -27.5% | 11.1% | 16.2 bps | 0% | 0.61x | 0.92x | 0.97 | 0.82 |

## Bootstrap intervals

| Profile | Config | Sharpe [95% CI] | MaxDD [95% CI] |
|---|---|---|---|
| P1-CAPITAL | `one_over_n` | 0.66 [-0.14, 1.54] | -30.1% [-46.3%, -11.1%] |
| P1-CAPITAL | `sixty_forty` | 0.58 [-0.13, 1.49] | -23.8% [-40.1%, -9.3%] |
| P1-CAPITAL | `min_variance` | 0.24 [-0.53, 1.21] | -18.3% [-35.2%, -8.4%] |
| P1-CAPITAL | `quant_only_no_gate` | 0.69 [-0.09, 1.59] | -25.4% [-41.8%, -10.2%] |
| P1-CAPITAL | `quant_only_gated` | 0.12 [-0.52, 0.80] | -22.8% [-26.4%, -8.4%] |
| P1-CAPITAL | `agents_only` | 0.48 [-0.26, 1.29] | -20.7% [-29.5%, -9.2%] |
| P1-CAPITAL | `hybrid` | 0.12 [-0.60, 0.82] | -22.8% [-28.3%, -8.9%] |
| P2-INCOME | `one_over_n` | 0.66 [-0.08, 1.62] | -30.1% [-47.2%, -10.8%] |
| P2-INCOME | `sixty_forty` | 0.58 [-0.24, 1.45] | -23.8% [-39.9%, -9.2%] |
| P2-INCOME | `min_variance` | 0.24 [-0.56, 1.13] | -18.3% [-35.8%, -8.4%] |
| P2-INCOME | `quant_only_no_gate` | 0.69 [-0.09, 1.63] | -25.8% [-42.9%, -10.9%] |
| P2-INCOME | `quant_only_gated` | 0.26 [-0.44, 0.93] | -22.7% [-27.2%, -8.6%] |
| P2-INCOME | `agents_only` | 0.35 [-0.47, 1.18] | -29.0% [-37.2%, -11.2%] |
| P2-INCOME | `hybrid` | 0.29 [-0.41, 1.04] | -22.7% [-25.8%, -8.5%] |
| P3-BALANCED | `one_over_n` | 0.66 [-0.13, 1.61] | -30.1% [-49.1%, -10.5%] |
| P3-BALANCED | `sixty_forty` | 0.58 [-0.20, 1.49] | -23.8% [-39.8%, -9.1%] |
| P3-BALANCED | `min_variance` | 0.24 [-0.52, 1.13] | -18.3% [-35.8%, -8.2%] |
| P3-BALANCED | `quant_only_no_gate` | 0.70 [-0.04, 1.63] | -26.4% [-42.5%, -11.3%] |
| P3-BALANCED | `quant_only_gated` | 0.50 [-0.29, 1.40] | -24.8% [-44.7%, -10.1%] |
| P3-BALANCED | `agents_only` | 0.42 [-0.33, 1.34] | -22.6% [-45.7%, -11.7%] |
| P3-BALANCED | `hybrid` | 0.52 [-0.28, 1.39] | -24.7% [-43.9%, -10.6%] |
| P4-GROWTH | `one_over_n` | 0.66 [-0.12, 1.63] | -30.1% [-43.8%, -11.1%] |
| P4-GROWTH | `sixty_forty` | 0.58 [-0.14, 1.47] | -23.8% [-40.6%, -9.0%] |
| P4-GROWTH | `min_variance` | 0.24 [-0.48, 1.16] | -18.3% [-36.0%, -8.2%] |
| P4-GROWTH | `quant_only_no_gate` | 0.72 [-0.07, 1.63] | -27.0% [-43.7%, -10.9%] |
| P4-GROWTH | `quant_only_gated` | 0.67 [-0.07, 1.58] | -27.0% [-44.9%, -11.1%] |
| P4-GROWTH | `agents_only` | 0.24 [-0.50, 1.15] | -29.9% [-57.5%, -15.5%] |
| P4-GROWTH | `hybrid` | 0.67 [-0.11, 1.63] | -27.1% [-45.1%, -11.7%] |
| P5-AGGRESSIVE | `one_over_n` | 0.66 [-0.08, 1.67] | -30.1% [-50.0%, -11.2%] |
| P5-AGGRESSIVE | `sixty_forty` | 0.58 [-0.20, 1.48] | -23.8% [-38.5%, -9.5%] |
| P5-AGGRESSIVE | `min_variance` | 0.24 [-0.61, 1.11] | -18.3% [-37.6%, -8.2%] |
| P5-AGGRESSIVE | `quant_only_no_gate` | 0.73 [-0.01, 1.70] | -27.4% [-45.1%, -11.4%] |
| P5-AGGRESSIVE | `quant_only_gated` | 0.73 [-0.03, 1.64] | -27.4% [-42.7%, -11.3%] |
| P5-AGGRESSIVE | `agents_only` | 0.18 [-0.52, 1.19] | -37.5% [-63.6%, -16.5%] |
| P5-AGGRESSIVE | `hybrid` | 0.72 [-0.02, 1.66] | -27.5% [-44.8%, -11.7%] |

## Paired tests: hybrid vs each baseline

Delta is hybrid minus baseline. For Sharpe, positive is better. For max drawdown, positive means the hybrid drawdown is less negative (better).

| Profile | Baseline | Delta Sharpe [95% CI] | p | Delta MaxDD [95% CI] | p |
|---|---|---:|---:|---:|---:|
| P1-CAPITAL | `one_over_n` | -0.54 [-1.44, 0.27] | 0.216 | 7.3% [-6.4%, 31.1%] | 0.256 |
| P1-CAPITAL | `sixty_forty` | -0.46 [-1.30, 0.28] | 0.200 | 1.0% [-7.6%, 24.2%] | 0.432 |
| P1-CAPITAL | `min_variance` | -0.12 [-0.64, 0.40] | 0.689 | -4.5% [-5.9%, 15.7%] | 0.611 |
| P1-CAPITAL | `quant_only_no_gate` | -0.57 [-1.43, 0.15] | 0.138 | 2.6% [-7.4%, 26.4%] | 0.338 |
| P1-CAPITAL | `quant_only_gated` | 0.00 [0.00, 0.00] | 1.000 | 0.0% [0.0%, 0.0%] | 1.000 |
| P1-CAPITAL | `agents_only` | -0.36 [-0.88, 0.16] | 0.214 | -2.1% [-7.2%, 10.9%] | 0.545 |
| P2-INCOME | `one_over_n` | -0.37 [-1.23, 0.40] | 0.346 | 7.4% [-4.5%, 29.8%] | 0.224 |
| P2-INCOME | `sixty_forty` | -0.30 [-1.08, 0.38] | 0.376 | 1.1% [-6.8%, 23.2%] | 0.390 |
| P2-INCOME | `min_variance` | 0.05 [-0.43, 0.51] | 0.881 | -4.5% [-5.1%, 15.8%] | 0.569 |
| P2-INCOME | `quant_only_no_gate` | -0.40 [-1.22, 0.31] | 0.232 | 3.1% [-5.9%, 26.6%] | 0.302 |
| P2-INCOME | `quant_only_gated` | 0.03 [-0.01, 0.09] | 0.282 | 0.0% [-0.1%, 1.2%] | 0.981 |
| P2-INCOME | `agents_only` | -0.06 [-0.63, 0.50] | 0.871 | 6.3% [-5.2%, 18.1%] | 0.258 |
| P3-BALANCED | `one_over_n` | -0.14 [-0.56, 0.19] | 0.404 | 5.4% [-4.7%, 9.8%] | 0.398 |
| P3-BALANCED | `sixty_forty` | -0.07 [-0.44, 0.30] | 0.707 | -0.9% [-7.8%, 4.1%] | 0.523 |
| P3-BALANCED | `min_variance` | 0.28 [-0.13, 0.76] | 0.172 | -6.4% [-11.7%, 3.1%] | 0.208 |
| P3-BALANCED | `quant_only_no_gate` | -0.19 [-0.41, -0.03] | 0.032 | 1.7% [-4.1%, 4.3%] | 0.603 |
| P3-BALANCED | `quant_only_gated` | 0.01 [-0.00, 0.03] | 0.090 | 0.1% [-0.1%, 0.6%] | 0.244 |
| P3-BALANCED | `agents_only` | 0.09 [-0.27, 0.54] | 0.597 | -2.1% [-6.6%, 14.3%] | 0.809 |
| P4-GROWTH | `one_over_n` | 0.00 [-0.31, 0.29] | 0.933 | 3.0% [-4.6%, 7.3%] | 0.525 |
| P4-GROWTH | `sixty_forty` | 0.08 [-0.22, 0.43] | 0.511 | -3.2% [-8.3%, 2.6%] | 0.250 |
| P4-GROWTH | `min_variance` | 0.43 [-0.01, 0.98] | 0.056 | -8.8% [-13.9%, 3.2%] | 0.190 |
| P4-GROWTH | `quant_only_no_gate` | -0.05 [-0.12, 0.00] | 0.064 | -0.0% [-1.5%, 1.0%] | 0.531 |
| P4-GROWTH | `quant_only_gated` | -0.00 [-0.01, 0.00] | 0.406 | -0.0% [-0.1%, 0.1%] | 0.575 |
| P4-GROWTH | `agents_only` | 0.42 [0.04, 0.81] | 0.026 | 2.9% [0.6%, 22.0%] | 0.030 |
| P5-AGGRESSIVE | `one_over_n` | 0.06 [-0.22, 0.32] | 0.595 | 2.6% [-4.1%, 7.2%] | 0.533 |
| P5-AGGRESSIVE | `sixty_forty` | 0.14 [-0.15, 0.46] | 0.286 | -3.6% [-8.2%, 2.3%] | 0.220 |
| P5-AGGRESSIVE | `min_variance` | 0.49 [0.02, 1.05] | 0.040 | -9.2% [-13.8%, 3.8%] | 0.184 |
| P5-AGGRESSIVE | `quant_only_no_gate` | -0.00 [-0.02, 0.01] | 0.535 | -0.0% [-0.2%, 0.1%] | 0.362 |
| P5-AGGRESSIVE | `quant_only_gated` | -0.00 [-0.01, 0.00] | 0.210 | -0.0% [-0.2%, 0.0%] | 0.340 |
| P5-AGGRESSIVE | `agents_only` | 0.54 [0.18, 0.88] | 0.004 | 10.0% [3.2%, 26.7%] | 0.006 |

## Regime windows

Mean cumulative return and mean max drawdown across the five profiles. The windows are fixed calendar periods, not selected from the results.

| Config | 2020 crash return | 2020 crash MaxDD | 2022 rates return | 2022 rates MaxDD |
|---|---:|---:|---:|---:|
| `one_over_n` | -12.0% | -30.1% | -7.6% | -16.9% |
| `sixty_forty` | -7.8% | -23.8% | -9.2% | -16.8% |
| `min_variance` | -5.2% | -18.1% | -11.6% | -17.9% |
| `quant_only_no_gate` | -12.2% | -26.4% | -0.8% | -15.1% |
| `quant_only_gated` | -6.2% | -20.1% | -5.9% | -16.1% |
| `agents_only` | -10.0% | -23.7% | -10.8% | -24.8% |
| `hybrid` | -6.2% | -20.1% | -5.8% | -16.0% |

## Reading the results

* **Gate effect.** Hybrid breach rate is 40% versus 56% for quant-only without the gate. The gate is doing real work; its value is not evidence of alpha.
* **Fair baseline.** Against `quant_only_gated`, hybrid annual return is 9.1% versus 9.0%; volatility is 11.3% versus 11.2%; Sharpe is 0.46 versus 0.45. This is the comparison that matters when asking whether agent views add value.
* **Strategic benchmark.** Hybrid annual return is 9.1% versus 10.0% for 60/40, with max drawdown -24.9% versus -23.8%. Beating 60/40 is not the same as beating the gated quantitative baseline.
* **Agents without guardrails.** `agents_only` is the failure mode the guardrails contain: 7.4% return, 76.1% turnover per rebalance, 110.5 bps annual cost, and a 74% breach rate. Holdings-aware portfolio views make it churn.
* **Paired bootstrap.** At the 5% level, 4 of 30 hybrid-vs-baseline Sharpe differences and 2 of 30 max-drawdown differences have a percentile p-value below 0.05. No multiple-testing correction is applied, so this is descriptive rather than a family-wise significance claim.
* **Regimes.** In the 2020 crash window hybrid returned -6.2% versus -7.8% for 60/40; in the 2022 rate shock hybrid returned -5.8% versus -9.2% for 60/40.
* **Test-window readout.** Hybrid annual return is 14.3% in the test window versus 1.6% in validation (higher), with Sharpe 1.00 versus -0.21 and test max drawdown -8.9%. This is a descriptive out-of-sample readout: the advisory weights are not fitted on the dev or validation labels, and the test window is not used to choose a strategy. Differences therefore reflect the macro path and sampling variation, not a tuned improvement.
* **What is not claimed.** The real-data path is identical across seeds, so the old five-seed standard deviations are not an independent-sample robustness result. This report replaces that claim with the five-profile panel and moving-block intervals.

