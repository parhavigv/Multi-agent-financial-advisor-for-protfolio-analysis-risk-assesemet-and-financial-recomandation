"""Phase 4: profiles, fair baselines, bootstrap intervals and regime reporting."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from intellivest.config import Settings, load_settings
from intellivest.data.market import SyntheticMarket
from intellivest.data.universe import TICKERS, asset_class_of
from intellivest.evaluation.backtest import (
    BASELINE_CONFIGS,
    CONFIGS,
    PLACEBO_CONFIG,
    Backtester,
    equal_weight,
    regime_statistics,
    sixty_forty_weights,
    weight_stability,
)
from intellivest.evaluation.bootstrap import (
    bootstrap_ci,
    moving_block_indices,
    paired_bootstrap_test,
)
from intellivest.evaluation.phase4 import run_phase4
from intellivest.evaluation.profiles import phase4_profiles
from intellivest.personalization.suitability import policy_for


@pytest.fixture(scope="module")
def settings() -> Settings:
    base = load_settings()
    return replace(
        base,
        data_start="2024-01-01",
        data_end="2025-06-30",
        lookback_days=60,
        llm_provider="deterministic",
        llm_api_key=None,
    )


@pytest.fixture(scope="module")
def provider(settings: Settings) -> SyntheticMarket:
    return SyntheticMarket(settings)


@pytest.fixture(scope="module")
def profile(settings: Settings):
    return phase4_profiles(date.fromisoformat(settings.data_end))[2]


# ---------------------------------------------------------------------------
# protocol fixed before looking at results
# ---------------------------------------------------------------------------


def test_phase4_profiles_span_every_suitability_class() -> None:
    profiles = phase4_profiles(date(2026, 6, 30))
    assert [profile.suitability_class for profile in profiles] == [1, 2, 3, 4, 5]
    assert len({profile.investor_id for profile in profiles}) == 5
    assert all(not profile.current_holdings for profile in profiles)
    assert profiles[0].loss_tolerance_pct < profiles[-1].loss_tolerance_pct


def test_phase4_configs_are_the_seven_primary_arms() -> None:
    assert CONFIGS == (
        "one_over_n",
        "sixty_forty",
        "min_variance",
        "quant_only_no_gate",
        "quant_only_gated",
        "agents_only",
        "hybrid",
    )
    assert BASELINE_CONFIGS == CONFIGS[:-1]
    assert PLACEBO_CONFIG == "hybrid_shuffled"
    assert PLACEBO_CONFIG not in CONFIGS


# ---------------------------------------------------------------------------
# pure strategy and diagnostic helpers
# ---------------------------------------------------------------------------


def test_equal_weight_and_sixty_forty_sleeves() -> None:
    equal = equal_weight(TICKERS)
    assert sum(equal.values()) == pytest.approx(1.0)
    assert len(equal) == len(TICKERS)

    balanced = sixty_forty_weights(TICKERS)
    assert sum(balanced.values()) == pytest.approx(1.0)
    equity = sum(
        weight for ticker, weight in balanced.items() if asset_class_of(ticker) == "equity"
    )
    fixed = sum(
        weight
        for ticker, weight in balanced.items()
        if asset_class_of(ticker) == "fixed_income"
    )
    assert equity == pytest.approx(0.60)
    assert fixed == pytest.approx(0.40)


def test_weight_stability_is_a_cosine_similarity() -> None:
    assert weight_stability([{"SPX": 1.0}, {"SPX": 1.0}]) == pytest.approx(1.0)
    assert weight_stability([{"SPX": 1.0}, {"TECH": 1.0}]) == pytest.approx(0.0)
    assert weight_stability([]) == 1.0


def test_regime_statistics_use_the_fixed_window() -> None:
    series = pd.Series(
        [0.10, -0.20, 0.05],
        index=pd.to_datetime(["2020-02-19", "2020-03-16", "2020-04-30"]),
    )
    stats = regime_statistics(series, date(2020, 2, 19), date(2020, 4, 30))
    assert stats["observations"] == 3
    assert stats["cumulative_return"] == pytest.approx(1.10 * 0.80 * 1.05 - 1.0)
    assert stats["max_drawdown"] == pytest.approx(-0.20)


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


def test_moving_block_indices_are_bounded_and_reproducible() -> None:
    first = moving_block_indices(100, 10, np.random.default_rng(7))
    second = moving_block_indices(100, 10, np.random.default_rng(7))
    assert first.shape == (100,)
    assert np.array_equal(first, second)
    assert first.min() >= 0
    assert first.max() < 100
    assert moving_block_indices(5, 10, np.random.default_rng(0)).tolist() == [0, 1, 2, 3, 4]


def test_bootstrap_ci_returns_ordered_percentile_interval() -> None:
    values = np.random.default_rng(3).normal(0.001, 0.01, 400)
    interval = bootstrap_ci(
        values,
        lambda sample: float(np.mean(sample)),
        block_length=10,
        n_boot=300,
        seed=11,
    )
    assert interval.point == pytest.approx(float(np.mean(values)))
    assert interval.low <= interval.high
    assert interval.n_boot == 300
    assert interval.block_length == 10


def test_paired_bootstrap_detects_a_large_paired_difference() -> None:
    rng = np.random.default_rng(42)
    better = rng.normal(0.002, 0.01, 300)
    worse = rng.normal(-0.002, 0.01, 300)
    result = paired_bootstrap_test(
        better,
        worse,
        lambda sample: float(np.mean(sample)),
        metric_name="mean",
        block_length=21,
        n_boot=300,
        seed=7,
    )
    assert result.difference > 0
    assert result.low > 0
    assert result.p_value < 0.05

    identical = paired_bootstrap_test(
        better,
        better,
        lambda sample: float(np.mean(sample)),
        metric_name="mean",
        n_boot=100,
        seed=7,
    )
    assert identical.difference == 0.0
    assert identical.p_value == 1.0


# ---------------------------------------------------------------------------
# integration
# ---------------------------------------------------------------------------


def test_backtester_records_alignment_and_bootstrap(
    settings: Settings, provider: SyntheticMarket, profile
) -> None:
    backtester = Backtester(
        settings,
        provider=provider,
        rebalance_every=21,
        lookback_days=settings.lookback_days,
    )
    results = backtester.run(
        profile,
        ("one_over_n", "hybrid"),
        bootstrap=True,
        n_boot=25,
        block_length=5,
    )
    by_config = {result.config: result for result in results}
    passive = by_config["one_over_n"]
    hybrid = by_config["hybrid"]

    assert not passive.daily_returns.empty
    assert passive.profile_alignment is not None
    assert passive.profile_alignment.volatility_limit == pytest.approx(
        policy_for(profile).max_volatility
    )
    assert 0.0 <= passive.profile_alignment.recommendation_stability <= 1.0 + 1e-9
    assert passive.profile_alignment.inter_agent_agreement is None
    assert hybrid.profile_alignment.inter_agent_agreement is not None
    assert set(passive.bootstrap) == {"sharpe", "max_drawdown"}
    assert passive.bootstrap["sharpe"].low <= passive.bootstrap["sharpe"].high
    assert hybrid.weights_history
    assert passive.turnovers


def test_phase4_runner_writes_report_without_seed_robustness_claim(
    settings: Settings, provider: SyntheticMarket, profile, tmp_path, monkeypatch
) -> None:
    import intellivest.evaluation.phase4 as phase4_module

    monkeypatch.setattr(phase4_module, "phase4_profiles", lambda as_of: [profile])
    payload, markdown = run_phase4(
        settings,
        provider=provider,
        output=tmp_path,
        lookback_days=settings.lookback_days,
        bootstrap=False,
    )

    assert (tmp_path / "phase4_evaluation.json").exists()
    assert (tmp_path / "phase4_evaluation.md").exists()
    assert (tmp_path / "evaluation.json").exists()
    assert (tmp_path / "equity_curves.csv").exists()
    assert payload["generated_with"]["configs"] == list(CONFIGS)
    assert payload["generated_with"]["seed_robustness_claim"] is False
    assert payload["generated_with"]["bootstrap"]["n_boot"] == 0
    assert payload["generated_with"]["data_split"]["test_start"] == "2024-01-01"
    assert payload["split_aggregate"]["test"]["hybrid"]["observations"] > 0
    assert payload["profiles"][0]["paired_tests"]
    assert payload["profiles"][0]["results"][0]["profile_alignment"]
    assert payload["profiles"][0]["results"][0]["splits"]
    assert "No seed-robustness claim is made" in markdown
    assert "Robustness across" not in markdown
    assert "## Bootstrap intervals" not in markdown
    assert "## Data split and test-window readout" in markdown
    assert "Paired tests: hybrid vs each baseline" in markdown
    assert "Regime windows" in markdown
