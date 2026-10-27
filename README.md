# Multi-Agent Financial Advisor for Portfolio Analysis, Risk Assessment & Financial Recommendations

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

A production-grade, multi-agent financial advisory system that combines quantitative portfolio optimization, multi-agent reasoning, explainable AI (XAI), and suitability-aware recommendations with a tamper-evident audit trail. Designed for portfolio analysis, risk assessment, and data-driven financial recommendations.

## 🎯 Key Features

- **Multi-Agent Architecture** - Specialized agents (Market, Fundamental, Sentiment, Risk, Portfolio, Recommendation) collaborate through an orchestrated workflow
- **Grounded Optional LLM Analysts** - Strict-schema LLM path with per-signal fact-sheet citations, numeric grounding and automatic deterministic fallback
- **Black-Litterman View Blending** - Coordinated agent signals become absolute views with confidence-scaled uncertainty and a hard cap on total influence
- **Suitability-Aware Recommendations** - Enforces investor profile constraints with automated repair mechanisms and hard blocking gates
- **Quantitative Portfolio Optimization** - Mean-variance optimization with constraints (position limits, sector caps, turnover penalties, transaction costs)
- **Comprehensive Risk Analytics** - VaR, CVaR, max drawdown, Sharpe/Sortino ratios, risk decomposition, and scenario analysis
- **Explainable AI** - Full traceability with agent rationale, evidence attribution, variance decomposition, and counterfactual analysis
- **Tamper-Evident Audit Trail** - Cryptographic hash chaining ensures complete auditability of every recommendation
- **Deterministic & Reproducible** - Seeded synthetic market data for testing and backtesting with full reproducibility
- **No-Lookahead Guarantee** - All data access respects point-in-time constraints to prevent information leakage
- **Multi-Profile Walk-Forward Evaluation** - Compare 1/N, 60/40, min-variance, quant-only, agents-only and hybrid across five suitability classes with block-bootstrap intervals and regime windows

## 🏗️ Architecture

![Multi-Agent Financial Advisory System Architecture](architecture.png)

*System architecture as designed for the INTELLIVEST-MAS framework.*

### Agent Responsibilities

| Agent | Role | Purpose |
|-------|------|---------|
| **Market Agent** | Technical Analysis | Analyzes trends, momentum, volatility regimes |
| **Fundamental Agent** | Valuation | Evaluates valuation metrics and fundamentals |
| **Sentiment Agent** | Market Sentiment | Gauges market sentiment and positioning |
| **Risk Agent** | Risk Assessment | Identifies tail risks and portfolio vulnerabilities |
| **Portfolio Agent** | Allocation Strategy | Proposes tactical allocation ideas |
| **Recommendation Agent** | Synthesis | Consolidates agent inputs into coherent advice |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/parhavigv/Multi-agent-financial-advisor-for-protfolio-analysis-risk-assesemet-and-financial-recomandation.git
cd Multi-agent-financial-advisor-for-protfolio-analysis-risk-assesemet-and-financial-recomandation

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Generate a Recommendation

```bash
# Use default demo profile
intellivest-advise

# Use custom profile
intellivest-advise --profile path/to/profile.json

# Output as JSON
intellivest-advise --json

# Enable the validated ML forecast overlay (opt-in)
intellivest-advise --ml
```

### Run Backtesting & Evaluation

```bash
# Run the Phase 4 multi-profile evaluation with bootstrap intervals
intellivest-evaluate

# Custom parameters
intellivest-evaluate --rebalance-every 21 --lookback-days 252 --bootstrap-samples 1000

# Fast run without bootstrap/paired resampling
intellivest-evaluate --no-bootstrap
```

### Real Market Data

By default the system runs on a deterministic synthetic market so it works
offline. To use real prices and a real risk-free rate:

```bash
# Downloads 14 US ETFs (SPY, XLK, ... VNQ) and FRED DGS3MO
python scripts/fetch_real_data.py

# then everything else is unchanged
intellivest-evaluate
```

Prices are cached to `data/prices.csv` and the rate series to
`data/risk_free.csv` (both gitignored). The risk-free rate is the FRED
3-month T-bill as of each decision date — never a hard-coded number. News is
not wired in for real data, so the sentiment agent abstains
(`confidence=0`) instead of inventing a signal. See
[docs/REAL_DATA.md](docs/REAL_DATA.md) for the ticker mapping and no-lookahead
guarantees.

### Forecasting Evaluation (Phase 2)

Return, volatility and regime models are evaluated with a purged, embargoed,
expanding walk-forward. The split is time-based: dev through 2021-12-31,
validation 2022-2023 (model selection only), and a single held-out test window
from 2024. Six return candidates (baselines plus Ridge/ElasticNet/LightGBM),
three volatility models (EWMA/HAR/GARCH), and a 2-3 state causal HMM are scored
with RMSE, MAE, rank IC (Newey-West t), hit rate, QLIKE and Diebold-Mariano
tests, plus a shuffled-label control.

```bash
# Dev + validation only (safe to run while developing)
python scripts/run_forecast_evaluation.py

# Final report, including the held-out test window
python scripts/run_forecast_evaluation.py --include-test --output results/forecast_report.md
```

The current evaluation retains `shrunk_mean` for returns and `ewma` for
volatility: the ML candidates did not clear the pre-registered stability and
margin rules. The full protocol, leakage controls, test results and limitations
are in [docs/FORECASTING.md](docs/FORECASTING.md); the generated evidence is
versioned at [results/forecast_report.md](results/forecast_report.md).

The same models are integrated into the pipeline behind
`AdvisoryEngine(use_ml=True)` / `intellivest-advise --ml`. The service makes a
single inner time split at the decision date, uses a candidate only if it beats
the baseline there, and otherwise falls back to `shrunk_mean` + `ewma`,
recording the rejection in `DecisionTrace.ml_forecast`. Market snapshots use the
causal HMM regime with a threshold fallback if the optional dependency is
missing.

### Optional LLM Analysts (Phase 3)

Deterministic agents are the default and the offline fallback. To enable the
optional LLM analyst path, set `INTELLIVEST_LLM_PROVIDER` (for example `mock`
for tests or an OpenAI-compatible provider), `INTELLIVEST_LLM_MODEL`,
`INTELLIVEST_LLM_BASE_URL` and `INTELLIVEST_LLM_API_KEY`.

The LLM never chooses weights, supplies performance metrics, or writes the
final evidence. It returns a strict JSON object; every signal must cite exact
dotted fact-sheet paths, and every number in a summary or rationale must be
represented by one of its cited fields. Invented Sharpe ratios, percentages,
dates, ratios and units are rejected. Evidence is rebuilt from the cited
fields, and any schema, citation, grounding or transport failure falls back to
the deterministic analyst with `degraded=True`. The full protocol is in
[docs/AGENTS.md](docs/AGENTS.md).

### Black-Litterman View Blending

Each coordinated agent tilt becomes an absolute view on one asset:
`Q_i = mu_i + tilt_i * view_shift`. The quantitative expected return is the
prior; confidence sets the diagonal view uncertainty
`Omega_i = P_i tau Sigma P_i' (1-c)/c`; a market view is added as an
equal-weight equity portfolio when the market signal is non-zero. The
optimizer then chooses weights from the posterior. A per-asset movement cap
and a global L1 influence cap prevent narrative conviction from dominating the
quantitative forecast. The advisory path uses mean-variance for every risk
class, because minimum-CVaR has no expected-return term and would silently
ignore the views; risk class still enters through risk aversion, position
limits, sector caps and the blocking suitability gate.

### Run Tests

```bash
pytest tests -v
```

## 📁 Project Structure

```
multi-agent-financial-advisor/
├── intellivest/
│   ├── agents/           # Multi-agent implementations
│   ├── api/              # FastAPI application
│   ├── data/             # Market data providers & universe
│   ├── evaluation/       # Backtesting framework
│   ├── explain/          # XAI & attribution modules
│   ├── forecasting/      # Phase-2 returns/vol/regime models & CV
│   ├── ml/               # Production bridge to the validated models
│   ├── orchestration/    # Agent coordination & audit trail
│   ├── personalization/  # Suitability engine & investor profiling
│   ├── quant/            # Quantitative models & optimization
│   ├── cli.py            # CLI entry points
│   ├── config.py         # Configuration management
│   └── contracts.py      # Pydantic data models
├── docs/                 # Audit, real-data and forecasting notes
├── scripts/              # Utility scripts
├── tests/                # Test suite
├── results/              # Run artifacts (gitignored, except the forecast report)
├── audits/               # Audit trail logs (gitignored)
└── pyproject.toml        # Project metadata & dependencies
```

## 🔒 Safety & Compliance Features

- **Suitability Enforcement** - Hard constraints based on risk tolerance, age, horizon, and loss tolerance
- **Automated Repair** - When constraints are violated, the system repairs portfolios iteratively with bounded iterations
- **Exclusion Lists** - Enforces ESG/compliance exclusions (fossil fuels, tobacco, etc.)
- **Evidence Dating** - All agent evidence must be timestamped and cannot reference future data
- **Audit Chain Integrity** - Cryptographic hashing creates an immutable audit trail
- **Grounding Requirements** - All recommendations are grounded in verifiable data and agent reports

## ⚙️ Configuration

Configure via environment variables or use defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `INTELLIVEST_SEED` | `20261023` | Random seed for reproducibility |
| `INTELLIVEST_DATA_START` | `2018-01-01` | Data window start |
| `INTELLIVEST_DATA_END` | `2026-06-30` | Data window end |
| `INTELLIVEST_MAX_WEIGHT` | `0.30` | Maximum single position weight |
| `INTELLIVEST_MAX_SECTOR_WEIGHT` | `0.45` | Maximum sector weight |
| `INTELLIVEST_TRANSACTION_COST_BPS` | `12.0` | Transaction costs in basis points |
| `INTELLIVEST_RISK_FREE_RATE` | `0.065` | Fallback rate only; real mode uses FRED DGS3MO |
| `INTELLIVEST_BL_TAU` | `0.05` | Black-Litterman prior uncertainty scale |
| `INTELLIVEST_BL_VIEW_SHIFT` | `0.03` | Maximum annualised return shift per view |
| `INTELLIVEST_BL_MAX_TOTAL_INFLUENCE` | `0.12` | Cap on the L1 sum of posterior movements |
| `INTELLIVEST_LLM_PROVIDER` | `deterministic` | LLM provider (`deterministic`, `mock` or a real endpoint) |
| `INTELLIVEST_LLM_MODEL` | `gpt-4o-mini` | Model name used when the LLM path is enabled |
| `INTELLIVEST_LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `INTELLIVEST_LLM_API_KEY` | unset | API key; the LLM path is disabled without it |

## 📊 Evaluation Results

Phase 4 evaluates seven strategies on the same cached real ETF prices, point-in-time FRED risk-free rate, rebalance dates and 12 bps transaction-cost assumption:

- **`one_over_n`** - equal weight across the allowed universe
- **`sixty_forty`** - 60% equal-weight equity sleeve / 40% equal-weight fixed-income sleeve
- **`min_variance`** - constrained long-only minimum variance
- **`quant_only_no_gate`** - constrained mean-variance with neutral views and no suitability gate
- **`quant_only_gated`** - the same optimizer and gate as the hybrid, with neutral views (the fair baseline)
- **`agents_only`** - direct agent consensus, bypassing the optimizer and gate
- **`hybrid`** - coordinated Black-Litterman views, optimizer weights, then the blocking suitability gate

`hybrid_shuffled` remains available as an opt-in placebo but is not part of the primary Phase 4 panel.

### Phase 4 protocol

- Five investor profiles spanning binding suitability classes 1-5.
- Window `2018-01-01` to `2026-06-30`, 252-day estimation window, 21-day rebalance interval.
- Fixed regime windows: the 2020 COVID crash (`2020-02-19` to `2020-04-30`) and the 2022 rate shock (`2022-01-01` to `2022-12-31`).
- Moving-block bootstrap with 21-day blocks and 1000 replicates for Sharpe and max-drawdown intervals.
- Paired bootstrap tests of hybrid versus every baseline, resampling the same block indices.
- Profile alignment: realised volatility versus policy limit, realised drawdown versus stated loss tolerance, weight stability between rebalances, and coordinator agreement for agent-driven arms.
- Holdings-aware agent decisions for `agents_only` and `hybrid`, recomputed at each rebalance with that arm's actual prior portfolio.
- No seed-robustness claim: the real-data path is identical across seeds, so repeated seeds are not independent trials. The five-profile panel and moving-block intervals replace that claim.

### Phase 4 results (real ETF data)

| Config | Return | Vol | Sharpe | MaxDD | Turnover | Cost | Breach |
|---|---:|---:|---:|---:|---:|---:|---:|
| `one_over_n` | 12.8% | 14.5% | 0.66 | -30.1% | 0.6% | 0.8 bps | 50% |
| `sixty_forty` | 10.0% | 11.6% | 0.58 | -23.8% | 0.6% | 0.8 bps | 47% |
| `min_variance` | 5.1% | 8.0% | 0.24 | -18.3% | 5.0% | 7.2 bps | 60% |
| `quant_only_no_gate` | 13.0% | 13.8% | 0.70 | -26.4% | 10.2% | 14.8 bps | 56% |
| `quant_only_gated` | 9.0% | 11.2% | 0.45 | -24.9% | 8.2% | 11.9 bps | 41% |
| `agents_only` | 7.4% | 13.3% | 0.33 | -27.9% | 76.1% | 110.5 bps | 74% |
| `hybrid` | 9.1% | 11.3% | 0.46 | -24.9% | 8.0% | 11.7 bps | 40% |

The gate reduces the breach rate from 56% without a gate to 40% with the hybrid gate, but it does not eliminate breaches: the conservative profile's 5% loss tolerance was already exceeded by the realised multi-year path. Against the fair `quant_only_gated` baseline, hybrid is essentially tied (9.1% versus 9.0% annual return; Sharpe 0.46 versus 0.45). The agent layer does not establish an alpha advantage, and `agents_only` without the optimizer or gate is the worst risk-adjusted arm: 7.4% return, 76.1% turnover per rebalance, and a 74% breach rate. Of 30 paired Sharpe comparisons, 4 have p < 0.05 before multiple-testing correction, and 2 of 30 max-drawdown comparisons do; the directions are mixed.

In the 2020 crash window hybrid returned -6.2% versus -7.8% for 60/40; in the 2022 rate shock hybrid returned -5.8% versus -9.2% for 60/40. These are risk-reduction observations in two fixed episodes, not a claim of general outperformance.

The versioned report is `results/phase4_evaluation.md` (JSON: `results/phase4_evaluation.json`). The full protocol, per-profile tables, confidence intervals, paired tests and limitations are in [docs/EVALUATION.md](docs/EVALUATION.md). The earlier `results/phase3_ablation.*` files are retained as historical placebo evidence; their five-seed spread is not used as a robustness claim.

Forecast-model evaluation remains separate in `results/forecast_report.md` (see [docs/FORECASTING.md](docs/FORECASTING.md)). Its selection is made on validation only; the test window is added by the final run and never changes the selected models.

## 🧪 Testing

The test suite ensures:
- No-lookahead in data snapshots (synthetic **and** real-shaped data)
- Deterministic synthetic data generation
- Point-in-time risk-free rate from the FRED series
- Sentiment abstention (`confidence=0`) when no real news exists
- Risk metric conventions (VaR/CVaR non-negative, drawdown non-positive)
- Suitability enforcement and repair
- Evidence dating constraints
- Audit chain integrity
- Exclusion list enforcement
- Forecast causality, purge/embargo and split-boundary exclusion
- Selection invariance when the test window is added (no test-driven tuning)
- Shuffled-label control (no cross-sectional signal after permutation)
- ML overlay end-to-end: trace fields, fallback on short history, JSON round-trip
- LLM strict schema, citation existence, numeric grounding and deterministic fallback
- Black-Litterman posterior, confidence-scaled Omega, per-asset and total influence caps
- Placebo permutation invariance and end-to-end Black-Litterman trace fields

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=intellivest --cov-report=html
```

## 🛠️ Technology Stack

- **Core**: Python 3.10+, NumPy, Pandas, SciPy
- **ML/Stats**: scikit-learn; optional `hmmlearn` (regimes), `arch` (GARCH), `lightgbm` (gradient boosting)
- **Validation**: Pydantic v2
- **API**: FastAPI, Uvicorn
- **Testing**: pytest
- **Data**: Synthetic regime-switching market by default; optional real ETF prices + FRED risk-free rate

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Parhavi G.V
- Akkihebbal Bhargava Surya
- Mekala Kirthana
- Shefali Chimmanda Ganapati

## 📞 Support

For questions or issues, please open an issue on [GitHub](https://github.com/parhavigv/Multi-agent-financial-advisor-for-protfolio-analysis-risk-assesemet-and-financial-recomandation/issues).
