"""Typed contracts shared across the pipeline.

Every agent, the optimizer, the suitability engine and the API exchange these
models. Keeping the boundary typed is what prevents the free-form-text
fragility described in the survey: an agent cannot hand the optimizer a
sentence, and the optimizer cannot hand the narrator a number it invented.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_CITATION_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*$")

Direction = Literal["bullish", "bearish", "neutral"]
Severity = Literal["info", "warning", "breach"]
Goal = Literal[
    "capital_preservation",
    "income",
    "balanced_growth",
    "growth",
    "aggressive_growth",
]
Liquidity = Literal["low", "medium", "high"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ---------------------------------------------------------------------------
# Investor side
# ---------------------------------------------------------------------------


class InvestorProfile(StrictModel):
    """Investor inputs. Suitability is computed from these, never assumed."""

    investor_id: str = Field(min_length=1, max_length=64)
    as_of: date
    age: int = Field(ge=18, le=100)
    horizon_years: float = Field(gt=0, le=50)
    # Self-reported tolerance on a 1 (very low) .. 5 (very high) scale.
    risk_tolerance: int = Field(ge=1, le=5)
    # Maximum drawdown the investor states they can tolerate, e.g. 0.15 = 15%.
    loss_tolerance_pct: float = Field(gt=0, le=1)
    goal: Goal = "balanced_growth"
    liquidity_need: Liquidity = "low"
    # ticker -> weight. May be empty for a fresh investor.
    current_holdings: dict[str, float] = Field(default_factory=dict)
    # Sectors / asset classes the investor refuses to hold (e.g. "tobacco").
    exclusions: list[str] = Field(default_factory=list)
    notes: str | None = Field(default=None, max_length=500)

    @field_validator("current_holdings")
    @classmethod
    def _weights_are_fractions(cls, value: dict[str, float]) -> dict[str, float]:
        for ticker, weight in value.items():
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"holding weight for {ticker} must be in [0, 1]")
        total = sum(value.values())
        if value and total > 1.0 + 1e-6:
            raise ValueError("current holdings sum to more than 100%")
        return {k: float(v) for k, v in value.items()}

    @field_validator("exclusions")
    @classmethod
    def _normalise_exclusions(cls, value: list[str]) -> list[str]:
        return sorted({item.strip().lower() for item in value if item.strip()})

    @property
    def risk_capacity(self) -> int:
        """Objective capacity to absorb risk, independent of stated tolerance.

        Young investors with long horizons can recover from drawdowns; older
        investors with short horizons cannot. Capacity is capped by tolerance
        so a nervous investor is never pushed past their stated comfort.
        """
        horizon_score = 1 + min(4, int(self.horizon_years // 3))
        age_score = 5 if self.age < 35 else 4 if self.age < 50 else 3 if self.age < 62 else 2
        capacity = min(horizon_score, age_score)
        return max(1, min(5, capacity))

    @property
    def suitability_class(self) -> int:
        """Binding risk class: the stricter of capacity and stated tolerance."""
        loss_score = 1 if self.loss_tolerance_pct < 0.05 else (
            2 if self.loss_tolerance_pct < 0.10 else (
                3 if self.loss_tolerance_pct < 0.20 else (
                    4 if self.loss_tolerance_pct < 0.30 else 5)))
        return max(1, min(self.risk_tolerance, self.risk_capacity, loss_score))

    @property
    def is_short_horizon(self) -> bool:
        return self.horizon_years < 3 or self.liquidity_need == "high"


# ---------------------------------------------------------------------------
# Agent layer
# ---------------------------------------------------------------------------


class Evidence(StrictModel):
    """A dated, auditable fact supporting a signal."""

    source: str = Field(min_length=1, max_length=120)
    as_of: date
    claim: str = Field(min_length=1, max_length=400)
    value: float | None = None


class Signal(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    subject: str = Field(min_length=1, max_length=32)  # ticker or "PORTFOLIO" / "MARKET"
    direction: Direction
    # Signed conviction in [-1, 1]; magnitude is used by the coordinator.
    strength: float = Field(ge=-1.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=400)
    confidence: float = Field(ge=0.0, le=1.0, default=0.6)
    # Dotted fact-sheet paths used by an LLM signal. Deterministic signals have
    # no citations because their inputs are the local numeric objects.
    citations: list[str] = Field(default_factory=list, max_length=16)

    @field_validator("citations")
    @classmethod
    def _citation_paths(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for path in value:
            if not _CITATION_PATH.fullmatch(path):
                raise ValueError(f"invalid citation path: {path!r}")
            if path not in cleaned:
                cleaned.append(path)
        return cleaned

    @model_validator(mode="after")
    def _direction_matches_strength(self) -> "Signal":
        if self.direction == "bullish" and self.strength < 0:
            raise ValueError("bullish signal cannot have negative strength")
        if self.direction == "bearish" and self.strength > 0:
            raise ValueError("bearish signal cannot have positive strength")
        if self.direction == "neutral" and abs(self.strength) > 0.15:
            raise ValueError("neutral signal must have |strength| <= 0.15")
        return self


class AgentReport(StrictModel):
    agent: str
    role: str
    as_of: date
    summary: str = Field(min_length=1, max_length=1200)
    confidence: float = Field(ge=0.0, le=1.0)
    signals: list[Signal] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    # True when an LLM was requested but the deterministic analyst was used.
    degraded: bool = False

    @field_validator("evidence")
    @classmethod
    def _evidence_is_dated(cls, value: list[Evidence]) -> list[Evidence]:
        return value


# ---------------------------------------------------------------------------
# Portfolio / risk
# ---------------------------------------------------------------------------


class RiskMetrics(StrictModel):
    annual_return: float
    annual_volatility: float
    sharpe: float
    sortino: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    beta: float
    effective_n: float
    herfindahl: float
    tracking_error: float | None = None


class OptimizationResult(StrictModel):
    method: str
    weights: dict[str, float]
    expected_return: float
    expected_volatility: float
    expected_sharpe: float
    expected_cvar_95: float
    turnover: float
    objective: float
    solver_status: str
    binding_constraints: list[str] = Field(default_factory=list)

    @field_validator("weights")
    @classmethod
    def _weights_normalised(cls, value: dict[str, float]) -> dict[str, float]:
        total = sum(value.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"weights must sum to 1.0, got {total:.6f}")
        return {k: float(v) for k, v in value.items() if v > 1e-9}


class SuitabilityCheck(StrictModel):
    rule_id: str
    description: str
    passed: bool
    observed: float | str
    limit: float | str | None = None
    severity: Severity = "info"
    remediation: str | None = None


class SuitabilityReport(StrictModel):
    passed: bool
    risk_class: int
    checks: list[SuitabilityCheck]
    violations: list[str] = Field(default_factory=list)
    repaired: bool = False
    repair_iterations: int = 0

    @model_validator(mode="after")
    def _violations_match_checks(self) -> "SuitabilityReport":
        failed = [c.rule_id for c in self.checks if not c.passed and c.severity == "breach"]
        if self.passed and failed:
            raise ValueError("report cannot pass with failing mandatory checks")
        if not self.passed and not failed:
            raise ValueError("a failing report must contain at least one breach")
        return self


class Recommendation(StrictModel):
    investor_id: str
    as_of: date
    headline: str
    rationale: str
    weights: dict[str, float]
    allocation_changes: dict[str, float]
    expected_return: float
    expected_volatility: float
    expected_sharpe: float
    cvar_95: float
    max_drawdown_estimate: float
    turnover: float
    estimated_cost_bps: float
    suitability: SuitabilityReport
    confidence: float = Field(ge=0.0, le=1.0)
    agent_agreement: float = Field(ge=0.0, le=1.0)
    degraded: bool = False
    disclaimers: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Explainability / trace
# ---------------------------------------------------------------------------


class TraceStep(StrictModel):
    name: str
    started_at: datetime
    duration_ms: float
    status: Literal["ok", "degraded", "failed"]
    inputs_digest: str
    output_digest: str
    notes: str | None = None


class Explanation(StrictModel):
    """Decision-path explanation, not a post-hoc afterthought."""

    portfolio_variance_contributions: dict[str, float]
    return_contributions: dict[str, float]
    agent_contributions: dict[str, float]
    binding_constraints: list[str]
    suitability_rationale: list[str]
    counterfactual: dict[str, Any] | None = None


class BlackLittermanSection(StrictModel):
    """Audit record of how agent views entered the expected-return estimate."""

    tau: float
    view_shift: float
    max_total_influence: float
    prior_expected_returns: dict[str, float]
    posterior_expected_returns: dict[str, float]
    views: dict[str, float]
    view_confidence: dict[str, float]
    view_dispersion: dict[str, float] = Field(default_factory=dict)
    view_returns: dict[str, float] = Field(default_factory=dict)
    omega: dict[str, float] = Field(default_factory=dict)
    total_influence: float
    influence_capped: bool
    n_views: int


class MLForecastSection(StrictModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, protected_namespaces=())
    forecast_timestamp: datetime | None = None
    data_cutoff: date | None = None
    model_identifiers: list[str] = Field(default_factory=list)
    model_versions: dict[str, str] = Field(default_factory=dict)
    model_weights: dict[str, float] = Field(default_factory=dict)
    validation_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    forecast_values: dict[str, float] = Field(default_factory=dict)
    forecast_units: str = "annualized_return"
    top_factors: list[dict[str, Any]] = Field(default_factory=list)
    attribution_method: str = "none"
    regime_prediction: str | None = None
    regime_confidence: float | None = None
    fallbacks: list[str] = Field(default_factory=list)
    rejected_models: list[dict[str, str]] = Field(default_factory=list)
    disagreement: float | None = None
    uncertainty: float | None = None


class DecisionTrace(StrictModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, protected_namespaces=())
    trace_id: str
    created_at: datetime
    as_of: date
    investor_id: str
    profile_digest: str
    config_digest: str
    steps: list[TraceStep]
    agents: list[AgentReport]
    optimizer: OptimizationResult
    suitability: SuitabilityReport
    explanation: Explanation
    recommendation: Recommendation
    previous_hash: str
    hash: str
    ml_forecast: MLForecastSection | None = None
    black_litterman: BlackLittermanSection | None = None
    risk_free_rate: float | None = None
    risk_free_source: str | None = None


class AdviceResponse(StrictModel):
    trace_id: str
    recommendation: Recommendation
    agents: list[AgentReport]
    explanation: Explanation
    steps: list[TraceStep]
