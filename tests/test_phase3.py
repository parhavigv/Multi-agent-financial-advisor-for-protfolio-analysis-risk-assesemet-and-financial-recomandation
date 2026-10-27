"""Phase 3: LLM grounding, Black-Litterman views and the placebo ablation."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from intellivest.agents import default_agents
from intellivest.agents.base import AgentContext
from intellivest.agents.llm import GroundingError, guard_numbers, numeric_tokens
from intellivest.agents.market import MarketAnalysisAgent
from intellivest.agents.schema import LLMValidationError, validate_llm_report
from intellivest.config import Settings, load_settings
from intellivest.contracts import AgentReport, InvestorProfile, Signal
from intellivest.data.market import SyntheticMarket
from intellivest.evaluation.backtest import CONFIGS, shuffle_decision
from intellivest.orchestration.coordinator import Coordinator, CoordinatorDecision
from intellivest.orchestration.orchestrator import AdvisoryEngine
from intellivest.quant.black_litterman import posterior_expected_returns
from intellivest.quant.engine import QuantEngine


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def settings() -> Settings:
    base = load_settings()
    return replace(base, llm_provider="deterministic", llm_api_key=None)


@pytest.fixture(scope="module")
def llm_settings(settings: Settings) -> Settings:
    return replace(settings, llm_provider="mock", llm_api_key="test-key")


@pytest.fixture(scope="module")
def snapshot(settings: Settings):
    return SyntheticMarket(settings).snapshot(date.fromisoformat(settings.data_end))


@pytest.fixture(scope="module")
def features(settings: Settings, snapshot):
    return QuantEngine(settings).build(snapshot)


@pytest.fixture(scope="module")
def profile(settings: Settings) -> InvestorProfile:
    return InvestorProfile(
        investor_id="PHASE3-001",
        as_of=date.fromisoformat(settings.data_end),
        age=34,
        horizon_years=12,
        risk_tolerance=4,
        loss_tolerance_pct=0.20,
        goal="growth",
        liquidity_need="low",
        current_holdings={"SPX": 0.35, "TECH": 0.30, "GOLD": 0.10, "GOVT": 0.25},
    )


def _context(settings: Settings, snapshot, features, profile) -> AgentContext:
    return AgentContext(
        profile=profile,
        snapshot=snapshot,
        features=features,
        current_weights={"SPX": 0.35, "TECH": 0.30, "GOLD": 0.10, "GOVT": 0.25},
        settings=settings,
    )


# ---------------------------------------------------------------------------
# numeric grounding
# ---------------------------------------------------------------------------


def test_guard_accepts_units_ratios_dates_and_labels() -> None:
    allowed = numeric_tokens(0.1234, 0.0012, 4.0, 12.0) | {"2026-06-30", "60/40"}
    guard_numbers(
        "Momentum 12.34%, cost 12 bps, risk class 4, horizon 12 years, "
        "as of 2026-06-30, ratio 60/40, 12-1 momentum.",
        allowed,
    )


def test_guard_checks_scientific_notation() -> None:
    allowed = numeric_tokens(0.001)
    guard_numbers("A 1e-3 effect is inside the interval.", allowed)
    with pytest.raises(GroundingError):
        guard_numbers("A 1e-4 effect is outside the interval.", allowed)


@pytest.mark.parametrize(
    "text",
    [
        "Sharpe 2.31",
        "Sharpe 9.99.",
        "Version 1.2.3",
        "Return 7.5%",
        "Cost 25 bps",
        "Date 2027-01-01",
        "Ratio 70/30",
        "Ratio 70/30.",
        "Sharpe 3",
        "Price $1,000",
    ],
)
def test_guard_rejects_invented_numbers(text: str, ) -> None:
    allowed = numeric_tokens(0.1234, 0.0012, 4.0, 12.0) | {"2026-06-30", "60/40"}
    with pytest.raises(GroundingError):
        guard_numbers(text, allowed)


def test_guard_normalises_month_dates() -> None:
    allowed = {"2026-06-30"}
    guard_numbers("As of June 30, 2026.", allowed)
    with pytest.raises(GroundingError):
        guard_numbers("As of July 1, 2026.", allowed)


def test_numeric_tokens_include_percent_bps_and_rounded_sharpe() -> None:
    tokens = numeric_tokens(0.000108, 0.6187, 0.134, 4.0, 12.0)
    assert "1.1" in tokens  # 1.08 bps, one decimal
    assert "1.08" in tokens
    guard_numbers(
        "Risk class 4, horizon 12 years, Sharpe 0.62, return 13.4%, cost 1.1 bps.",
        tokens,
    )


def test_tiny_nonzero_value_does_not_whitelist_zero() -> None:
    tokens = numeric_tokens(0.000001)
    assert "0" not in tokens
    with pytest.raises(GroundingError):
        guard_numbers("Sharpe 0", tokens)


def test_empty_robustness_panel_has_no_nan_aggregate() -> None:
    from intellivest.cli import _aggregate

    assert _aggregate([], ("quant_only", "hybrid")) == {}


# ---------------------------------------------------------------------------
# LLM schema and fallback
# ---------------------------------------------------------------------------


def _valid_payload(features, snapshot) -> dict:
    value = float(features.momentum["SPX"])
    direction = "bullish" if value >= 0 else "bearish"
    strength = 0.5 if value >= 0 else -0.5
    return {
        "summary": f"SPX 12-1 momentum is {value * 100:.1f}%.",
        "confidence": 0.72,
        "summary_citations": ["assets.SPX.momentum_12_1"],
        "signals": [
            {
                "name": "momentum",
                "subject": "SPX",
                "direction": direction,
                "strength": strength,
                "rationale": f"Momentum is {value * 100:.1f}%.",
                "confidence": 0.8,
                "citations": ["assets.SPX.momentum_12_1"],
            }
        ],
    }


def test_valid_llm_report_is_accepted_and_evidence_is_derived(
    monkeypatch, llm_settings, snapshot, features, profile
) -> None:
    payload = _valid_payload(features, snapshot)
    monkeypatch.setattr(
        "intellivest.agents.llm.complete_json", lambda *args, **kwargs: payload
    )
    ctx = _context(llm_settings, snapshot, features, profile)

    report = MarketAnalysisAgent().analyze(ctx)

    assert report.degraded is False
    assert report.summary == payload["summary"]
    assert report.metrics == {}
    assert report.signals[0].citations == ["assets.SPX.momentum_12_1"]
    assert report.evidence
    assert "assets.SPX.momentum_12_1" in report.evidence[0].claim


def _fake_summary(payload: dict) -> None:
    payload["summary"] = "Sharpe 9.99."


def _fake_rationale(payload: dict) -> None:
    payload["signals"][0]["rationale"] = "Upside 99.99%."


def _missing_field(payload: dict) -> None:
    payload["signals"][0]["citations"] = ["assets.SPX.not_a_field"]


def _missing_summary_field(payload: dict) -> None:
    payload["summary_citations"] = ["assets.SPX.not_a_field"]


def _missing_citations(payload: dict) -> None:
    del payload["signals"][0]["citations"]


def _extra_metrics(payload: dict) -> None:
    payload["metrics"] = {"sharpe": 9.99}


def _invalid_subject(payload: dict) -> None:
    payload["signals"][0]["subject"] = "DOGE"


def _no_signals(payload: dict) -> None:
    payload["signals"] = []


@pytest.mark.parametrize(
    "mutate",
    [
        _fake_summary,
        _fake_rationale,
        _missing_field,
        _missing_summary_field,
        _missing_citations,
        _extra_metrics,
        _invalid_subject,
        _no_signals,
    ],
    ids=[
        "fake-summary",
        "fake-rationale",
        "missing-field",
        "missing-summary-field",
        "missing-citations",
        "extra-metrics",
        "invalid-subject",
        "no-signals",
    ],
)
def test_invalid_llm_payload_falls_back_and_is_marked_degraded(
    monkeypatch, llm_settings, snapshot, features, profile, mutate
) -> None:
    payload = _valid_payload(features, snapshot)
    mutate(payload)
    monkeypatch.setattr(
        "intellivest.agents.llm.complete_json", lambda *args, **kwargs: payload
    )
    ctx = _context(llm_settings, snapshot, features, profile)

    report = MarketAnalysisAgent().analyze(ctx)

    assert report.degraded is True
    assert report.summary != payload["summary"]


def test_deterministic_mode_does_not_call_the_llm(
    monkeypatch, settings, snapshot, features, profile
) -> None:
    def _explode(*args, **kwargs):
        raise AssertionError("LLM must not be called in deterministic mode")

    monkeypatch.setattr("intellivest.agents.llm.complete_json", _explode)
    ctx = _context(settings, snapshot, features, profile)

    report = MarketAnalysisAgent().analyze(ctx)

    assert report.degraded is False
    assert all(not signal.citations for signal in report.signals)


def test_signal_citations_are_validated_and_deduplicated() -> None:
    with pytest.raises(ValueError):
        Signal(
            name="bad",
            subject="SPX",
            direction="bullish",
            strength=0.5,
            rationale="bad citation",
            citations=["not a path!"],
        )
    signal = Signal(
        name="ok",
        subject="SPX",
        direction="bullish",
        strength=0.5,
        rationale="ok",
        citations=["assets.SPX.momentum_12_1", "assets.SPX.momentum_12_1"],
    )
    assert signal.citations == ["assets.SPX.momentum_12_1"]


def test_schema_rejects_extra_metrics_directly() -> None:
    with pytest.raises(LLMValidationError):
        validate_llm_report(
            {
                "summary": "A grounded summary.",
                "confidence": 0.5,
                "summary_citations": [],
                "signals": [
                    {
                        "name": "s",
                        "subject": "SPX",
                        "direction": "bullish",
                        "strength": 0.2,
                        "rationale": "No numbers here.",
                        "confidence": 0.5,
                        "citations": ["assets.SPX.momentum_12_1"],
                    }
                ],
                "metrics": {"sharpe": 3.0},
            },
            agent="test",
            role="test",
            as_of=date(2026, 6, 30),
            fact_sheet={},
        )


# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------


def _prior_cov() -> tuple[pd.Series, pd.DataFrame]:
    prior = pd.Series({"A": 0.05, "B": 0.05})
    cov = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.09]], index=["A", "B"], columns=["A", "B"]
    )
    return prior, cov


def test_black_litterman_without_views_returns_the_prior() -> None:
    prior, cov = _prior_cov()
    result = posterior_expected_returns(prior, cov, {})
    assert result.n_views == 0
    assert result.total_influence == 0.0
    assert result.influence_capped is False
    assert np.allclose(result.posterior.to_numpy(), prior.to_numpy())


def test_black_litterman_view_moves_the_viewed_asset_only() -> None:
    prior, cov = _prior_cov()
    result = posterior_expected_returns(
        prior,
        cov,
        {"A": 0.8},
        confidence={"A": 0.9},
        max_total_influence=1.0,
    )
    assert result.n_views == 1
    assert result.posterior["A"] > prior["A"]
    assert result.posterior["B"] == pytest.approx(prior["B"])
    assert result.omega.iloc[0] > 0


def test_black_litterman_confidence_shrinks_omega_and_moves_more() -> None:
    prior, cov = _prior_cov()
    high = posterior_expected_returns(
        prior,
        cov,
        {"A": 0.5},
        confidence={"A": 0.95},
        max_total_influence=1.0,
    )
    low = posterior_expected_returns(
        prior,
        cov,
        {"A": 0.5},
        confidence={"A": 0.10},
        max_total_influence=1.0,
    )
    assert high.omega.iloc[0] < low.omega.iloc[0]
    assert high.posterior["A"] > low.posterior["A"] > prior["A"]


def test_black_litterman_caps_per_asset_and_total_influence() -> None:
    prior, cov = _prior_cov()
    result = posterior_expected_returns(
        prior,
        cov,
        {"A": 1.0, "B": -1.0},
        confidence={"A": 0.99, "B": 0.99},
        view_shift=0.03,
        max_total_influence=0.01,
    )
    assert result.influence_capped is True
    assert result.total_influence <= 0.01 + 1e-12
    for ticker in ("A", "B"):
        assert abs(result.posterior[ticker] - prior[ticker]) <= 0.03 + 1e-12


def test_black_litterman_market_view_is_an_equity_portfolio_view() -> None:
    tickers = ["SPX", "TECH", "GOVT"]
    prior = pd.Series({ticker: 0.05 for ticker in tickers})
    cov = pd.DataFrame(
        np.eye(len(tickers)) * 0.04, index=tickers, columns=tickers
    )
    result = posterior_expected_returns(
        prior,
        cov,
        {},
        market_score=1.0,
        market_confidence=0.9,
        max_total_influence=1.0,
    )
    assert result.n_views == 1
    assert result.posterior["SPX"] > prior["SPX"]
    assert result.posterior["TECH"] > prior["TECH"]
    assert result.posterior["GOVT"] == pytest.approx(prior["GOVT"])


def test_black_litterman_ignores_unknown_tickers() -> None:
    prior, cov = _prior_cov()
    result = posterior_expected_returns(prior, cov, {"ZZZ": 1.0})
    assert result.n_views == 0
    assert np.allclose(result.posterior.to_numpy(), prior.to_numpy())


# ---------------------------------------------------------------------------
# coordinator and placebo
# ---------------------------------------------------------------------------


def test_coordinator_records_view_confidence_and_dispersion() -> None:
    as_of = date(2026, 6, 30)
    aggressive = AgentReport(
        agent="market_analysis",
        role="market",
        as_of=as_of,
        summary="aggressive",
        confidence=0.8,
        signals=[
            Signal(
                name="asset",
                subject="SPX",
                direction="bullish",
                strength=0.6,
                rationale="strong",
                confidence=0.9,
            ),
            Signal(
                name="market",
                subject="MARKET",
                direction="bullish",
                strength=0.4,
                rationale="risk-on",
                confidence=0.7,
            ),
        ],
    )
    cautious = AgentReport(
        agent="risk_analysis",
        role="risk",
        as_of=as_of,
        summary="cautious",
        confidence=0.4,
        signals=[
            Signal(
                name="asset",
                subject="SPX",
                direction="bearish",
                strength=-0.4,
                rationale="crowded",
                confidence=0.2,
            )
        ],
    )
    decision = Coordinator(default_agents()).coordinate(
        [aggressive, cautious], ["SPX"]
    )
    assert 0.2 <= decision.view_confidence["SPX"] <= 0.9
    assert decision.view_dispersion["SPX"] > 0
    assert 0.0 <= decision.market_confidence <= 1.0


def test_shuffle_placebo_preserves_distribution_and_is_deterministic() -> None:
    decision = CoordinatorDecision(
        tilts={"SPX": 0.8, "TECH": -0.4, "GOLD": 0.2},
        market_score=0.1,
        portfolio_score=-0.2,
        agreement=0.5,
        dispersion=0.3,
        view_confidence={"SPX": 0.9, "TECH": 0.4, "GOLD": 0.6},
        view_dispersion={"SPX": 0.1, "TECH": 0.2, "GOLD": 0.3},
    )
    shuffled = shuffle_decision(decision, ["SPX", "TECH", "GOLD"], seed=3)

    assert shuffled.tilts != decision.tilts
    assert sorted(shuffled.tilts.values()) == sorted(decision.tilts.values())
    assert sorted(
        zip(shuffled.tilts.values(), shuffled.view_confidence.values())
    ) == sorted(zip(decision.tilts.values(), decision.view_confidence.values()))
    assert shuffle_decision(decision, ["SPX", "TECH", "GOLD"], seed=3) == shuffled
    assert "hybrid" in CONFIGS
    assert "hybrid_shuffled" not in CONFIGS


# ---------------------------------------------------------------------------
# end-to-end integration
# ---------------------------------------------------------------------------


def test_engine_uses_black_litterman_and_records_the_trace(
    settings: Settings, snapshot, features, profile
) -> None:
    provider = SyntheticMarket(settings)
    engine = AdvisoryEngine(settings, provider=provider, agents=default_agents())
    trace = engine.advise(profile)

    assert trace.black_litterman is not None
    assert trace.black_litterman.n_views >= 1
    assert trace.black_litterman.total_influence <= settings.bl_max_total_influence + 1e-9
    assert abs(sum(trace.optimizer.weights.values()) - 1.0) < 1e-6
    assert trace.black_litterman.posterior_expected_returns
    assert (
        trace.black_litterman.prior_expected_returns
        != trace.black_litterman.posterior_expected_returns
    )
    assert engine.verify_audit_chain()[0] is True


def test_optimiser_receives_the_black_litterman_posterior(
    settings: Settings, snapshot, features, profile
) -> None:
    engine = AdvisoryEngine(settings, provider=SyntheticMarket(settings))
    neutral = CoordinatorDecision(
        tilts={},
        market_score=0.0,
        portfolio_score=0.0,
        agreement=1.0,
        dispersion=0.0,
    )
    viewed = replace(
        neutral,
        tilts={"SPX": 1.0},
        view_confidence={"SPX": 0.99},
        market_confidence=0.5,
    )

    base, base_mu, _ = engine.optimise(
        profile, features, snapshot.returns, neutral, {}
    )
    shifted, shifted_mu, bl = engine.optimise(
        profile, features, snapshot.returns, viewed, {}
    )

    assert shifted_mu["SPX"] > base_mu["SPX"]
    assert shifted.expected_return != base.expected_return
    assert bl.n_views == 1
