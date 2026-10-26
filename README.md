# Multi-Agent Financial Advisor for Portfolio Analysis, Risk Assessment & Financial Recommendations

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

A production-grade, multi-agent financial advisory system that combines quantitative portfolio optimization, multi-agent reasoning, explainable AI (XAI), and suitability-aware recommendations with a tamper-evident audit trail. Designed for portfolio analysis, risk assessment, and data-driven financial recommendations.

## 🎯 Key Features

- **Multi-Agent Architecture** - Specialized agents (Market, Fundamental, Sentiment, Risk, Portfolio, Recommendation) collaborate through an orchestrated workflow
- **Suitability-Aware Recommendations** - Enforces investor profile constraints with automated repair mechanisms and hard blocking gates
- **Quantitative Portfolio Optimization** - Mean-variance optimization with constraints (position limits, sector caps, turnover penalties, transaction costs)
- **Comprehensive Risk Analytics** - VaR, CVaR, max drawdown, Sharpe/Sortino ratios, risk decomposition, and scenario analysis
- **Explainable AI** - Full traceability with agent rationale, evidence attribution, variance decomposition, and counterfactual analysis
- **Tamper-Evident Audit Trail** - Cryptographic hash chaining ensures complete auditability of every recommendation
- **Deterministic & Reproducible** - Seeded synthetic market data for testing and backtesting with full reproducibility
- **No-Lookahead Guarantee** - All data access respects point-in-time constraints to prevent information leakage
- **Walk-Forward Backtesting** - Compare quant-only, agents-only, and hybrid strategies with robustness across multiple seeds

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
```

### Run Backtesting & Evaluation

```bash
# Run walk-forward evaluation across configurations
intellivest-evaluate

# Custom parameters
intellivest-evaluate --rebalance-every 21 --seeds 20261023,7,101
```

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
│   ├── orchestration/    # Agent coordination & audit trail
│   ├── personalization/  # Suitability engine & investor profiling
│   ├── quant/            # Quantitative models & optimization
│   ├── cli.py            # CLI entry points
│   ├── config.py         # Configuration management
│   └── contracts.py      # Pydantic data models
├── scripts/              # Utility scripts
├── tests/                # Test suite
├── results/              # Evaluation outputs (gitignored)
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
| `INTELLIVEST_LLM_PROVIDER` | `deterministic` | LLM provider (deterministic/mock or real) |

## 📊 Evaluation Results

The system supports three configurations for comparison:
- **`quant_only`** - Pure constrained optimization, no agent inputs
- **`agents_only`** - Direct agent consensus, bypassing quantitative guardrails
- **`hybrid`** - Agent tilts bounded within optimizer, followed by suitability gate (recommended)

Results are saved to `results/evaluation.json` and `results/evaluation.md` with metrics including Sharpe ratio, Sortino ratio, max drawdown, CVaR, turnover, and suitability breach rates.

## 🧪 Testing

The test suite ensures:
- No-lookahead in data snapshots
- Deterministic synthetic data generation
- Risk metric conventions (VaR/CVaR non-negative, drawdown non-positive)
- Suitability enforcement and repair
- Evidence dating constraints
- Audit chain integrity
- Exclusion list enforcement

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=intellivest --cov-report=html
```

## 🛠️ Technology Stack

- **Core**: Python 3.10+, NumPy, Pandas, SciPy
- **ML/Stats**: scikit-learn
- **Validation**: Pydantic v2
- **API**: FastAPI, Uvicorn
- **Testing**: pytest
- **Data**: Synthetic market data with realistic regimes

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
