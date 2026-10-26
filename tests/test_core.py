"""No-lookahead, audit-chain integrity, grounding and core invariants."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from intellivest.config import load_settings
from intellivest.contracts import InvestorProfile, Signal
from intellivest.data.market import SyntheticMarket
from intellivest.data.universe import TICKERS
from intellivest.orchestration.orchestrator import AdvisoryEngine
from intellivest.personalization.suitability import SuitabilityEngine
from intellivest.quant import risk


def test_snapshot_has_no_future_rows() -> None:
    settings = load_settings()
    provider = SyntheticMarket(settings)
    snap = provider.snapshot(date.fromisoformat("2022-06-30"))
    assert snap.as_of == date.fromisoformat("2022-06-30")
    assert snap.prices.index[-1].date() <= snap.as_of
    assert snap.returns.index[-1].date() <= snap.as_of


def test_synthetic_deterministic() -> None:
    settings = load_settings()
    p1 = SyntheticMarket(settings)
    p2 = SyntheticMarket(settings)
    assert p1.history().equals(p2.history())


def test_risk_conventions() -> None:
    series = pd.Series([0.01, -0.02, 0.015, -0.03, 0.005])
    assert risk.historical_var(series, 0.95) >= 0.0
    assert risk.historical_cvar(series, 0.95) >= 0.0
    dd = risk.max_drawdown(series)
    assert dd <= 0.0


def test_signal_neutral_band() -> None:
    from intellivest.agents.base import directional_signal

    s = directional_signal(name="x", subject="SPX", strength=0.05, rationale="mild")
    assert s.direction == "neutral"
    assert s.strength == 0.0


def test_advice_produces_trace_and_audit_passes() -> None:
    settings = load_settings()
    engine = AdvisoryEngine(settings, provider=SyntheticMarket(settings))
    profile = InvestorProfile(
        investor_id="TEST-001",
        as_of=date.fromisoformat(settings.data_end),
        age=34,
        horizon_years=12,
        risk_tolerance=4,
        loss_tolerance_pct=0.20,
        goal="growth",
        liquidity_need="low",
        current_holdings={"SPX": 0.35, "TECH": 0.30, "GOLD": 0.10, "GOVT": 0.25},
    )
    trace = engine.advise(profile)
    assert trace.trace_id
    assert trace.hash != trace.previous_hash or trace.previous_hash == "GENESIS"
    ok, msg = engine.verify_audit_chain()
    assert ok, msg


def test_suitability_blocks_aggressive_for_conservative() -> None:
    settings = load_settings()
    engine = AdvisoryEngine(settings, provider=SyntheticMarket(settings))
    profile = InvestorProfile(
        investor_id="TEST-002",
        as_of=date.fromisoformat(settings.data_end),
        age=34,
        horizon_years=12,
        risk_tolerance=1,
        loss_tolerance_pct=0.05,
        goal="capital_preservation",
        liquidity_need="low",
        current_holdings={"TECH": 1.0},
    )
    trace = engine.advise(profile)
    # The gate must act even if the agent/optimizer propose an aggressive mix.
    assert trace.suitability.risk_class == 1
    assert trace.suitability.repaired is True


def test_agent_evidence_must_be_dated_or_before() -> None:
    from intellivest.agents.base import AgentContext, AnalystAgent
    from intellivest.contracts import AgentReport, Evidence

    class Dummy(AnalystAgent):
        name = "dummy"
        role = "dummy"

        def analyze(self, ctx: AgentContext) -> AgentReport:
            future = Evidence(source="x", as_of=ctx.snapshot.as_of + timedelta(days=1), claim="bad", value=1.0)
            past = Evidence(source="y", as_of=ctx.snapshot.as_of, claim="ok", value=1.0)
            return self._report(ctx, summary="s", evidence=[future, past])

    settings = load_settings()
    provider = SyntheticMarket(settings)
    snap = provider.snapshot(date.fromisoformat(settings.data_end))
    engine = AdvisoryEngine(settings, provider=provider)
    # build features
    from intellivest.quant.engine import QuantEngine

    features = QuantEngine(settings).build(snap)
    profile = InvestorProfile(
        investor_id="TEST-003",
        as_of=date.fromisoformat(settings.data_end),
        age=40,
        horizon_years=10,
        risk_tolerance=3,
        loss_tolerance_pct=0.15,
        goal="balanced_growth",
        liquidity_need="low",
        current_holdings={},
    )
    ctx = __import__("intellivest.agents.base", fromlist=["AgentContext"]).AgentContext(
        profile=profile, snapshot=snap, features=features, current_weights={}, settings=settings
    )
    rep = Dummy().analyze(ctx)
    assert all(e.as_of <= ctx.snapshot.as_of for e in rep.evidence)
    assert len(rep.evidence) == 1


def test_exclusions_blocked() -> None:
    from intellivest.data.universe import excluded_tickers

    blocked = excluded_tickers(["fossil", "tobacco"])
    assert "ENRG" in blocked
