"""Quantify effect of Constraints.diversification on optimizer behavior."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from intellivest.config import load_settings
from intellivest.data.market import build_provider
from intellivest.orchestration.coordinator import Coordinator, CoordinatorDecision
from intellivest.orchestration.orchestrator import AdvisoryEngine
from intellivest.quant.engine import QuantEngine
from intellivest.quant.optimize import Constraints, mean_variance
from intellivest.data.universe import sector_of, excluded_tickers


def main() -> None:
    settings = load_settings()
    provider = build_provider(settings)
    snap = provider.snapshot(pd.Timestamp(settings.data_end).date())
    returns = snap.returns.dropna(how="all").fillna(0.0)
    q = QuantEngine(settings)
    feats = q.build(snap)
    neutral = CoordinatorDecision(tilts={}, market_score=0.0, portfolio_score=0.0, agreement=1.0, dispersion=0.0)
    policy_max_w = 0.30
    c1 = Constraints(max_weight=policy_max_w, max_sector_weight=0.45, turnover_penalty=0.0020, risk_aversion=2.0, diversification=1.0)
    c2 = Constraints(max_weight=policy_max_w, max_sector_weight=0.45, turnover_penalty=0.0020, risk_aversion=2.0, diversification=0.0)
    ex = excluded_tickers([])
    r1 = mean_variance(feats.expected_returns, feats.covariance, previous_weights={}, constraints=c1, sector_of=sector_of, exclude=ex)
    r2 = mean_variance(feats.expected_returns, feats.covariance, previous_weights={}, constraints=c2, sector_of=sector_of, exclude=ex)
    w1 = r1.weights
    w2 = r2.weights
    n = len(w1)
    hhi1 = sum(v*v for v in w1.values())
    hhi2 = sum(v*v for v in w2.values())
    out = {
        "n_assets": n,
        "diversification_1p0": {"hhi": hhi1, "effective_n": 1.0/hhi1 if hhi1>0 else 0, "max_weight": max(w1.values()), "weights": {k: float(v) for k,v in w1.items()}},
        "diversification_0p0": {"hhi": hhi2, "effective_n": 1.0/hhi2 if hhi2>0 else 0, "max_weight": max(w2.values()), "weights": {k: float(v) for k,v in w2.items()}},
        "note": "diversification>0 adds a concave term encouraging spread; effect depends on returns/covariance/constraints"
    }
    Path('results/baseline_synthetic').mkdir(parents=True, exist_ok=True)
    Path('results/baseline_synthetic/diversification_check.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
