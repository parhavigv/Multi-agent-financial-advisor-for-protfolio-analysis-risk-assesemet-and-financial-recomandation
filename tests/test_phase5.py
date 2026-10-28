"""Phase 5: optimism checks, suitability repair and end-to-end traceability."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from intellivest.config import Settings, load_settings
from intellivest.contracts import InvestorProfile
from intellivest.data.market import SyntheticMarket
from intellivest.data.universe import TICKERS, asset_class_of
from intellivest.evaluation.backtest import equal_weight
from intellivest.evaluation.metrics import summarise
from intellivest.evaluation.profiles import phase4_profiles
from intellivest.orchestration.orchestrator import AdvisoryEngine
from intellivest.orchestration.trace import AuditChain, digest
from intellivest.personalization.suitability import SuitabilityEngine, policy_for
from intellivest.quant.engine import QuantEngine
from intellivest.quant.optimize import (
    Constraints,
    annotate,
    mean_variance,
    min_cvar,
    min_variance,
    risk_parity,
)

# ---------------------------------------------------------------------------
# known-series metric conventions
# ---------------------------------------------------------------------------


def test_metrics_on_a_known_series() -> None:
    """Every metric is checked against an independent hand calculation."""
    returns = pd.Series([0.10, -0.20, 0.05, -0.10])
    risk_free = 0.02
    summary = summarise(returns, risk_free=risk_free, periods=4)

    values = returns.to_numpy()
    annual_return = float(np.prod(1.0 + values) - 1.0)
    annual_volatility = float(np.std(values, ddof=1) * np.sqrt(4))
    hurdle = (1.0 + risk_free) ** (1.0 / 4.0) - 1.0
    downside = np.minimum(values - hurdle, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)) * np.sqrt(4))
    equity = np.cumprod(1.0 + values)
    max_drawdown = float((equity / np.maximum.accumulate(equity) - 1.0).min())

    assert summary.annual_return == pytest.approx(annual_return, abs=1e-6)
    assert summary.annual_volatility == pytest.approx(annual_volatility, abs=1e-6)
    assert summary.sharpe == pytest.approx(
        (annual_return - risk_free) / annual_volatility, abs=1e-4
    )
    assert summary.sortino == pytest.approx(
        (annual_return - risk_free) / downside_deviation, abs=1e-4
    )
    assert summary.max_drawdown == pytest.approx(max_drawdown, abs=1e-6)
    assert summary.cvar_95 == pytest.approx(0.20, abs=1e-9)
    assert summary.calmar == pytest.approx(annual_return / abs(max_drawdown), abs=1e-4)
    assert summary.hit_rate == pytest.approx(0.5)


def test_risk_free_rate_changes_sharpe_and_sortino() -> None:
    returns = pd.Series([0.01, -0.005, 0.008, 0.002] * 10)
    low = summarise(returns, risk_free=0.01, periods=4)
    high = summarise(returns, risk_free=0.05, periods=4)
    assert high.sharpe < low.sharpe
    assert high.sortino < low.sortino


# ---------------------------------------------------------------------------
# optimizer feasibility, status and the risk-free convention
# ---------------------------------------------------------------------------


@pytest.fixture()
def optimizer_inputs() -> tuple[pd.Series, pd.DataFrame, dict[str, str]]:
    tickers = ["A1", "A2", "B1", "B2", "C1", "C2"]
    sectors = {"A1": "A", "A2": "A", "B1": "B", "B2": "B", "C1": "C", "C2": "C"}
    expected = pd.Series(
        [0.20, 0.18, 0.05, 0.04, 0.03, 0.02],
        index=tickers,
    )
    covariance = pd.DataFrame(
        np.diag([0.012, 0.011, 0.010, 0.009, 0.008, 0.007]),
        index=tickers,
        columns=tickers,
    )
    return expected, covariance, sectors


def _sector_exposure(weights: dict[str, float], sectors: dict[str, str]) -> dict[str, float]:
    output: dict[str, float] = {}
    for ticker, weight in weights.items():
        output[sectors[ticker]] = output.get(sectors[ticker], 0.0) + weight
    return output


def test_mean_variance_respects_caps_sector_limits_and_exclusions(
    optimizer_inputs,
) -> None:
    expected, covariance, sectors = optimizer_inputs
    constraints = Constraints(
        max_weight=0.30,
        max_sector_weight=0.40,
        turnover_penalty=0.0,
        risk_aversion=0.5,
        diversification=0.1,
    )
    result = mean_variance(
        expected,
        covariance,
        constraints=constraints,
        sector_of=sectors.get,
    )

    assert result.solver_status == "optimal"
    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-8)
    assert max(result.weights.values()) <= constraints.max_weight + 1e-6
    assert all(
        exposure <= constraints.max_sector_weight + 1e-6
        for exposure in _sector_exposure(result.weights, sectors).values()
    )
    assert "sector_cap:A" in result.binding_constraints

    excluded = mean_variance(
        expected,
        covariance,
        constraints=constraints,
        sector_of=sectors.get,
        exclude={"A1"},
    )
    assert excluded.solver_status == "optimal"
    assert "A1" not in excluded.weights


def test_optimizer_failure_returns_a_constraint_aware_fallback(
    optimizer_inputs, monkeypatch
) -> None:
    expected, covariance, sectors = optimizer_inputs
    constraints = Constraints(
        max_weight=0.30,
        max_sector_weight=0.40,
        turnover_penalty=0.0,
        risk_aversion=0.5,
        diversification=0.1,
    )

    class FailedSolver:
        success = False
        message = "forced failure for the fallback test"
        x = np.zeros(len(expected))

    monkeypatch.setattr(
        "intellivest.quant.optimize.minimize",
        lambda *args, **kwargs: FailedSolver(),
    )
    result = mean_variance(
        expected,
        covariance,
        constraints=constraints,
        sector_of=sectors.get,
    )

    assert result.solver_status.startswith("fallback:")
    assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-8)
    assert max(result.weights.values()) <= constraints.max_weight + 1e-6
    assert all(
        exposure <= constraints.max_sector_weight + 1e-6
        for exposure in _sector_exposure(result.weights, sectors).values()
    )


def test_all_solvers_report_explicit_status_and_feasible_weights(optimizer_inputs) -> None:
    expected, covariance, sectors = optimizer_inputs
    constraints = Constraints(
        max_weight=0.30,
        max_sector_weight=0.40,
        turnover_penalty=0.0,
        risk_aversion=0.5,
        diversification=0.1,
    )
    returns = pd.DataFrame(
        np.random.default_rng(4).normal(0.0003, 0.01, size=(80, len(expected))),
        columns=expected.index,
    )

    results = [
        mean_variance(
            expected,
            covariance,
            constraints=constraints,
            sector_of=sectors.get,
        ),
        min_variance(covariance, constraints=constraints, sector_of=sectors.get),
        risk_parity(covariance, constraints=constraints),
        min_cvar(
            returns,
            covariance,
            constraints=constraints,
            sector_of=sectors.get,
        ),
    ]

    for result in results:
        assert result.solver_status
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-7)
        assert min(result.weights.values()) >= 0.0
        assert max(result.weights.values()) <= constraints.max_weight + 1e-6


def test_expected_sharpe_uses_the_supplied_risk_free_rate(optimizer_inputs) -> None:
    expected, covariance, sectors = optimizer_inputs
    constraints = Constraints(
        max_weight=0.30,
        max_sector_weight=0.40,
        turnover_penalty=0.0,
        risk_aversion=0.5,
        diversification=0.1,
    )
    risk_free = 0.035
    result = mean_variance(
        expected,
        covariance,
        constraints=constraints,
        sector_of=sectors.get,
        risk_free=risk_free,
    )
    expected_sharpe = (result.expected_return - risk_free) / result.expected_volatility
    assert result.expected_sharpe == pytest.approx(expected_sharpe, rel=1e-9)

    reannotated = annotate(
        result,
        expected,
        pd.DataFrame(
            np.zeros((20, len(expected))),
            columns=expected.index,
        ),
        risk_free=risk_free,
    )
    assert reannotated.expected_sharpe == pytest.approx(result.expected_sharpe, rel=1e-9)

    zero_rate = mean_variance(
        expected,
        covariance,
        constraints=constraints,
        sector_of=sectors.get,
        risk_free=0.0,
    )
    assert zero_rate.expected_sharpe > result.expected_sharpe


# ---------------------------------------------------------------------------
# audit tamper detection
# ---------------------------------------------------------------------------


def _audit_payload(trace_id: str) -> dict:
    return {
        "trace_id": trace_id,
        "investor_id": "AUDIT-001",
        "created_at": "2026-06-30T12:00:00+00:00",
        "recommendation": {"weights": {"SPX": 0.6, "GOVT": 0.4}},
        "optimizer": {"solver_status": "optimal"},
    }


def test_audit_chain_detects_tampered_recommendation(tmp_path) -> None:
    chain = AuditChain(tmp_path)
    chain.seal(_audit_payload("trace-tamper"))
    ok, message = chain.verify()
    assert ok, message

    trace_path = tmp_path / "traces" / "trace-tamper.json"
    snapshot = json.loads(trace_path.read_text(encoding="utf-8"))
    snapshot["recommendation"]["weights"]["SPX"] = 0.99
    trace_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    ok, message = chain.verify()
    assert not ok
    assert "content hash mismatch" in message


def test_audit_chain_detects_broken_previous_hash(tmp_path) -> None:
    chain = AuditChain(tmp_path)
    chain.seal(_audit_payload("trace-one"))
    chain.seal(_audit_payload("trace-two"))

    chain_path = tmp_path / "chain.jsonl"
    lines = chain_path.read_text(encoding="utf-8").splitlines()
    second = json.loads(lines[1])
    second["previous_hash"] = "0" * 64
    lines[1] = json.dumps(second, sort_keys=True, separators=(",", ":"))
    chain_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, message = chain.verify()
    assert not ok
    assert "previous_hash mismatch" in message


# ---------------------------------------------------------------------------
# suitability repair
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_context():
    settings = load_settings()
    provider = SyntheticMarket(settings)
    snapshot = provider.snapshot(date.fromisoformat(settings.data_end))
    features = QuantEngine(settings).build(snapshot, risk_free=provider.get_risk_free(snapshot.as_of))
    return settings, provider, snapshot, features


def test_repair_moves_an_aggressive_book_inside_policy(synthetic_context) -> None:
    settings, _, snapshot, features = synthetic_context
    profile = phase4_profiles(snapshot.as_of)[2]
    engine = SuitabilityEngine(settings)
    proposed = {"TECH": 1.0}

    before = engine.evaluate(proposed, profile, features, snapshot.returns)
    assert not before.passed
    assert {check.rule_id for check in before.checks if not check.passed and check.severity == "breach"}

    repaired, report = engine.repair(proposed, profile, features, snapshot.returns)

    assert report.repaired is True
    assert report.repair_iterations > 0
    assert report.passed is True
    assert sum(repaired.values()) == pytest.approx(1.0, abs=1e-8)
    policy = policy_for(profile)
    equity = sum(
        weight
        for ticker, weight in repaired.items()
        if ticker in TICKERS and asset_class_of(ticker) == "equity"
    )
    assert equity <= policy.max_equity + 1e-9
    assert max(repaired.values()) <= policy.max_single + 1e-9
    assert all(check.passed for check in report.checks if check.severity == "breach")


def test_repair_is_honest_when_the_policy_is_infeasible_on_realised_data(
    synthetic_context,
) -> None:
    settings, _, snapshot, features = synthetic_context
    profile = phase4_profiles(snapshot.as_of)[0]
    engine = SuitabilityEngine(settings)
    _, report = engine.repair(equal_weight(TICKERS), profile, features, snapshot.returns)

    assert report.repaired is True
    if not report.passed:
        assert report.violations


# ---------------------------------------------------------------------------
# one recommendation traced from profile to audit hash
# ---------------------------------------------------------------------------


def _trace_profile(as_of: date) -> InvestorProfile:
    return InvestorProfile(
        investor_id="TRACE-001",
        as_of=as_of,
        age=42,
        horizon_years=10,
        risk_tolerance=3,
        loss_tolerance_pct=0.15,
        goal="balanced_growth",
        liquidity_need="medium",
        current_holdings={"SPX": 0.30, "GOVT": 0.35, "GOLD": 0.10, "CRED": 0.25},
    )


def test_recommendation_trace_links_every_stage_to_an_audit_hash(tmp_path) -> None:
    base = load_settings()
    settings: Settings = replace(
        base,
        audits_dir=tmp_path / "audits",
        results_dir=tmp_path / "results",
        llm_provider="deterministic",
        llm_api_key=None,
    )
    provider = SyntheticMarket(settings)
    profile = _trace_profile(date.fromisoformat(settings.data_end))
    engine = AdvisoryEngine(settings, provider=provider)

    trace = engine.advise(profile)

    assert trace.profile_digest == digest(profile.model_dump(mode="json"))
    assert trace.agents
    assert any(report.signals for report in trace.agents)
    assert trace.black_litterman is not None
    assert trace.black_litterman.n_views > 0
    assert trace.optimizer.solver_status
    assert trace.optimizer.weights
    assert trace.suitability.checks
    assert trace.explanation.suitability_rationale
    assert trace.explanation.counterfactual is not None
    assert trace.risk_free_rate is not None
    assert trace.optimizer.expected_sharpe == pytest.approx(
        (trace.optimizer.expected_return - trace.risk_free_rate)
        / trace.optimizer.expected_volatility,
        rel=1e-9,
    )
    assert trace.recommendation.expected_sharpe == pytest.approx(
        trace.optimizer.expected_sharpe, rel=1e-9
    )
    assert len(trace.hash) == 64
    assert trace.previous_hash == "GENESIS"

    step_names = [step.name for step in trace.steps]
    for required in (
        "profile.normalise",
        "agents.analyse",
        "agents.coordinate",
        "portfolio.optimise",
        "suitability.validate",
        "explain.attribute",
        "recommendation.compose",
    ):
        assert required in step_names

    ok, message = engine.verify_audit_chain()
    assert ok, message
    snapshot_path = settings.audits_dir / "traces" / f"{trace.trace_id}.json"
    sealed = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert sealed["hash"] == trace.hash
    assert sealed["profile_digest"] == trace.profile_digest
    assert sealed["black_litterman"]["n_views"] == trace.black_litterman.n_views
    assert sealed["optimizer"]["weights"] == trace.optimizer.weights
    assert sealed["suitability"]["checks"] == [
        check.model_dump(mode="json") for check in trace.suitability.checks
    ]
    assert sealed["recommendation"]["weights"] == trace.recommendation.weights
