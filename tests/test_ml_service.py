"""Tests for the production ML forecast bridge and HMM regime wiring."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from intellivest.config import load_settings
from intellivest.contracts import InvestorProfile
from intellivest.data.market import SyntheticMarket, infer_market_regime
from intellivest.ml.service import MLForecastService


def _snapshot(rows: int | None = None):
    settings = load_settings()
    provider = SyntheticMarket(settings)
    snapshot = provider.snapshot(date.fromisoformat(settings.data_end))
    if rows is None:
        return settings, snapshot
    returns = snapshot.returns.iloc[-rows:].copy()
    prices = snapshot.prices.loc[: returns.index[-1]].iloc[-(rows + 1) :].copy()
    return settings, replace(snapshot, prices=prices, returns=returns)


def test_service_falls_back_when_history_is_too_short() -> None:
    settings, snapshot = _snapshot(rows=80)
    result = MLForecastService(settings.seed).forecast(snapshot, settings)

    assert result.fallbacks, "short history must be reported as a fallback"
    assert all("simple" in name for name in result.model_identifiers)
    assert list(result.expected_returns.index) == list(snapshot.returns.columns)
    assert np.isfinite(result.expected_returns.to_numpy()).all()
    assert (result.volatility_forecast > 0).all()
    assert np.isfinite(result.volatility_forecast.to_numpy()).all()


def test_service_selects_a_model_with_inner_validation() -> None:
    settings, snapshot = _snapshot()
    result = MLForecastService(settings.seed).forecast(snapshot, settings)

    assert not result.fallbacks
    assert result.model_identifiers
    assert any(name.startswith("returns:") for name in result.model_identifiers)
    assert any(name.startswith("volatility:") for name in result.model_identifiers)
    # Both a baseline and a candidate must be scored on the inner window.
    assert "returns:shrunk_mean" in result.validation_metrics
    assert "volatility:ewma" in result.validation_metrics
    # Forecasts must be non-degenerate and within the documented bounds.
    assert result.expected_returns.std() > 0
    assert result.expected_returns.abs().max() <= 0.35 + 1e-9
    assert (result.volatility_forecast > 0).all()
    assert result.horizon == 21


def test_infer_market_regime_labels_and_short_history_fallback() -> None:
    settings, snapshot = _snapshot()
    label, confidence = infer_market_regime(snapshot.benchmark_returns)
    assert label in {"calm", "elevated", "stress"}
    assert 0.0 <= confidence <= 1.0

    short = snapshot.benchmark_returns.iloc[-40:]
    label, confidence = infer_market_regime(short)
    assert label == "elevated"
    assert confidence == 0.4


def test_ml_overlay_is_recorded_in_the_trace() -> None:
    """``use_ml=True`` must populate the trace, not just return weights."""
    from intellivest.orchestration.orchestrator import AdvisoryEngine

    settings = load_settings()
    profile = InvestorProfile(
        investor_id="ML-TRACE",
        as_of=date.fromisoformat(settings.data_end),
        age=34,
        horizon_years=12,
        risk_tolerance=3,
        loss_tolerance_pct=0.15,
        goal="balanced_growth",
        liquidity_need="low",
    )
    trace = AdvisoryEngine(settings=settings, use_ml=True).advise(profile)
    section = trace.ml_forecast
    assert section is not None
    assert section.model_identifiers
    assert section.validation_metrics
    assert section.regime_prediction in {"calm", "elevated", "stress"}
    assert section.forecast_units == "annualized_return"
    assert len(section.forecast_values) == len(trace.recommendation.weights)
    # JSON round-trip: the trace must remain serialisable with the new fields.
    payload = trace.model_dump(mode="json")
    assert payload["ml_forecast"]["model_identifiers"] == section.model_identifiers
