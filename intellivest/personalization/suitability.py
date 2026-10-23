"""Suitability policy and remediation.

The survey's central personalization finding is that an allocation can look
excellent financially and still be unsuitable for the investor stating it.
This module makes suitability an explicit, blocking stage with named rules,
observed values, limits and remediation — not a paragraph bolted onto the end
of a report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from intellivest.config import Settings
from intellivest.contracts import (
    InvestorProfile,
    SuitabilityCheck,
    SuitabilityReport,
)
from intellivest.data.universe import (
    asset_class_of,
    excluded_tickers,
    sector_of,
    sector_weights,
)
from intellivest.quant import risk
from intellivest.quant.engine import QuantFeatures


@dataclass(frozen=True)
class RiskPolicy:
    risk_class: int
    max_equity: float
    max_volatility: float
    max_cvar: float
    min_defensive: float
    max_single: float
    min_effective_n: float


POLICIES: dict[int, RiskPolicy] = {
    1: RiskPolicy(1, 0.30, 0.085, 0.018, 0.55, 0.15, 4.0),
    2: RiskPolicy(2, 0.50, 0.110, 0.024, 0.35, 0.20, 4.0),
    3: RiskPolicy(3, 0.72, 0.145, 0.031, 0.15, 0.25, 4.0),
    4: RiskPolicy(4, 0.90, 0.185, 0.040, 0.00, 0.30, 3.0),
    5: RiskPolicy(5, 1.00, 0.240, 0.052, 0.00, 0.35, 3.0),
}

DEFENSIVE_CLASSES = {"fixed_income", "commodity"}
CLASS1_CORE = {"GOVT": 0.50, "CRED": 0.30, "GOLD": 0.20}
CLASS2_CORE = {"GOVT": 0.42, "CRED": 0.28, "GOLD": 0.16}
CLASS3_CORE = {"GOVT": 0.35, "CRED": 0.25, "GOLD": 0.12}
CLASS4_CORE = {"GOVT": 0.30, "CRED": 0.20, "GOLD": 0.10}
CLASS5_CORE = {"GOVT": 0.25, "CRED": 0.15, "GOLD": 0.10}

CORE_BY_CLASS = {1: CLASS1_CORE, 2: CLASS2_CORE, 3: CLASS3_CORE, 4: CLASS4_CORE, 5: CLASS5_CORE}

# Remediation anchors sit *below* the policy ceiling on purpose: a repair
# target should restore headroom, not merely touch the limit again.
ANCHOR_EQUITY = {1: 0.15, 2: 0.30, 3: 0.50, 4: 0.65, 5: 0.78}

# Last-resort anchors. When even the class reference allocation cannot satisfy a
# binding rule (for example the investor's loss tolerance is tighter than the
# realised drawdown of a 65/35 book in the current window), the repair ladder
# escalates to a capital-preservation mix with no equity. Without this step a
# genuinely infeasible profile would return an unrepaired portfolio, whereas the
# gate should always move toward the safest admissible allocation it can build.
CAPITAL_PRESERVATION: dict[int, dict[str, float]] = {
    1: {"GOVT": 0.55, "CRED": 0.30, "GOLD": 0.15},
    2: {"GOVT": 0.50, "CRED": 0.30, "GOLD": 0.20},
    3: {"GOVT": 0.45, "CRED": 0.30, "GOLD": 0.25},
    4: {"GOVT": 0.40, "CRED": 0.35, "GOLD": 0.25},
    5: {"GOVT": 0.35, "CRED": 0.35, "GOLD": 0.30},
}


def policy_for(profile: InvestorProfile) -> RiskPolicy:
    return POLICIES[profile.suitability_class]


def reference_allocation(
    profile: InvestorProfile, tickers: list[str]
) -> dict[str, float]:
    """A conservative class-appropriate anchor used for remediation."""
    policy = policy_for(profile)
    blocked = excluded_tickers(profile.exclusions)
    allowed = [t for t in tickers if t not in blocked]
    equity = [t for t in allowed if asset_class_of(t) == "equity"]
    defensive = [t for t in allowed if asset_class_of(t) in DEFENSIVE_CLASSES]

    equity_budget = min(policy.max_equity, ANCHOR_EQUITY[policy.risk_class])
    defensive_budget = 1.0 - equity_budget

    target: dict[str, float] = {}
    core = CORE_BY_CLASS[policy.risk_class]
    core_available = {t: w for t, w in core.items() if t in defensive}
    if core_available:
        total = sum(core_available.values())
        for ticker, weight in core_available.items():
            target[ticker] = defensive_budget * weight / total
    elif defensive:
        for ticker in defensive:
            target[ticker] = defensive_budget / len(defensive)
    else:
        equity_budget = 1.0

    if equity:
        for ticker in equity:
            target[ticker] = target.get(ticker, 0.0) + equity_budget / len(equity)
    elif defensive:
        scale = 1.0 / sum(target.values())
        target = {t: w * scale for t, w in target.items()}
    return {t: w for t, w in target.items() if w > 1e-9}


def capital_preservation_allocation(
    profile: InvestorProfile, tickers: list[str]
) -> dict[str, float]:
    """No-equity anchor used as the final rung of the repair ladder."""
    blocked = excluded_tickers(profile.exclusions)
    defensive = [
        t for t in tickers
        if t not in blocked and asset_class_of(t) in DEFENSIVE_CLASSES
    ]
    core = CAPITAL_PRESERVATION[policy_for(profile).risk_class]
    available = {t: w for t, w in core.items() if t in defensive}
    if not available:
        if not defensive:
            return reference_allocation(profile, tickers)
        return {t: 1.0 / len(defensive) for t in defensive}
    total = sum(available.values())
    return {t: w / total for t, w in available.items()}


class SuitabilityEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # -- evaluation -------------------------------------------------------
    def evaluate(
        self,
        weights: Mapping[str, float],
        profile: InvestorProfile,
        features: QuantFeatures,
        returns: pd.DataFrame,
        *,
        turnover: float = 0.0,
    ) -> SuitabilityReport:
        policy = policy_for(profile)
        weights = {t: float(w) for t, w in weights.items() if w > 1e-9}
        blocked = excluded_tickers(profile.exclusions)

        portfolio = risk.portfolio_returns(weights, returns)
        vector = pd.Series(weights).reindex(features.covariance.columns).fillna(0.0).to_numpy()
        variance = float(vector @ features.covariance.to_numpy() @ vector)
        volatility = float(np.sqrt(max(variance, 0.0)))
        cvar = risk.historical_cvar(portfolio, 0.95)
        drawdown = abs(risk.max_drawdown(portfolio))
        equity_weight = sum(w for t, w in weights.items() if asset_class_of(t) == "equity")
        defensive_weight = sum(
            w for t, w in weights.items() if asset_class_of(t) in DEFENSIVE_CLASSES
        )
        single_max = max(weights.values()) if weights else 0.0
        sectors = sector_weights(weights)
        worst_sector, worst_sector_weight = ("none", 0.0) if not sectors else max(
            sectors.items(), key=lambda item: item[1]
        )
        effective = risk.effective_n(weights)
        held_blocked = sorted(set(weights) & blocked)

        checks: list[SuitabilityCheck] = [
            SuitabilityCheck(
                rule_id="R-EQUITY",
                description="Equity exposure within the investor's risk class",
                passed=equity_weight <= policy.max_equity + 1e-9,
                observed=round(equity_weight, 4),
                limit=policy.max_equity,
                severity="breach",
                remediation=f"Reduce equity to <= {policy.max_equity:.0%}",
            ),
            SuitabilityCheck(
                rule_id="R-VOLATILITY",
                description="Expected annualised volatility within policy",
                passed=volatility <= policy.max_volatility + 1e-9,
                observed=round(volatility, 4),
                limit=policy.max_volatility,
                severity="breach",
                remediation=f"Lower-volatility mix targeting <= {policy.max_volatility:.1%}",
            ),
            SuitabilityCheck(
                rule_id="R-CVAR",
                description="One-day 95% CVaR within policy",
                passed=cvar <= policy.max_cvar + 1e-9,
                observed=round(cvar, 4),
                limit=policy.max_cvar,
                severity="breach",
                remediation=f"Reduce tail risk to <= {policy.max_cvar:.1%} daily",
            ),
            SuitabilityCheck(
                rule_id="R-DRAWDOWN",
                description="Historical max drawdown within stated loss tolerance",
                passed=drawdown <= profile.loss_tolerance_pct + 1e-9,
                observed=round(drawdown, 4),
                limit=profile.loss_tolerance_pct,
                severity="breach",
                remediation="Shift toward defensive assets to respect loss tolerance",
            ),
            SuitabilityCheck(
                rule_id="R-SINGLE",
                description="Single-position cap",
                passed=single_max <= min(policy.max_single, self.settings.max_weight) + 1e-9,
                observed=round(single_max, 4),
                limit=round(min(policy.max_single, self.settings.max_weight), 4),
                severity="breach",
                remediation="Diversify the largest position",
            ),
            SuitabilityCheck(
                rule_id="R-SECTOR",
                description="Sector concentration cap",
                passed=worst_sector_weight <= self.settings.max_sector_weight + 1e-9,
                observed=round(worst_sector_weight, 4),
                limit=self.settings.max_sector_weight,
                severity="warning",
                remediation="Reduce the most concentrated sector",
            ),
            SuitabilityCheck(
                rule_id="R-DIVERSIFICATION",
                description="Effective number of holdings",
                passed=effective >= policy.min_effective_n - 1e-9,
                observed=round(effective, 3),
                limit=policy.min_effective_n,
                severity="warning",
                remediation="Add uncorrelated positions",
            ),
            SuitabilityCheck(
                rule_id="R-EXCLUSION",
                description="No holdings that violate stated exclusions",
                passed=not held_blocked,
                observed=",".join(held_blocked) if held_blocked else "none",
                limit="none",
                severity="breach",
                remediation="Remove excluded holdings",
            ),
            SuitabilityCheck(
                rule_id="R-DEFENSIVE",
                description="Defensive allocation floor for the risk class",
                passed=defensive_weight >= policy.min_defensive - 1e-9,
                observed=round(defensive_weight, 4),
                limit=policy.min_defensive,
                severity="warning",
                remediation=f"Hold at least {policy.min_defensive:.0%} in defensive assets",
            ),
            SuitabilityCheck(
                rule_id="R-LIQUIDITY",
                description="Defensive sleeve appropriate to a short horizon",
                passed=(not profile.is_short_horizon) or defensive_weight >= 0.20 - 1e-9,
                observed=round(defensive_weight, 4),
                limit=0.20 if profile.is_short_horizon else None,
                severity="breach",
                remediation="Increase liquid defensive assets for a short horizon",
            ),
            SuitabilityCheck(
                rule_id="R-TURNOVER",
                description="Turnover budget",
                passed=turnover <= 0.60 + 1e-9,
                observed=round(turnover, 4),
                limit=0.60,
                severity="warning",
                remediation="Stage the rebalance across periods",
            ),
        ]

        breaches = [c for c in checks if not c.passed and c.severity == "breach"]
        return SuitabilityReport(
            passed=not breaches,
            risk_class=policy.risk_class,
            checks=checks,
            violations=[f"{c.rule_id}: {c.remediation}" for c in breaches],
        )

    # -- remediation ------------------------------------------------------
    def repair(
        self,
        weights: Mapping[str, float],
        profile: InvestorProfile,
        features: QuantFeatures,
        returns: pd.DataFrame,
        *,
        turnover: float = 0.0,
    ) -> tuple[dict[str, float], SuitabilityReport]:
        original = {t: float(w) for t, w in weights.items() if w > 1e-9}
        report = self.evaluate(original, profile, features, returns, turnover=turnover)
        if report.passed:
            return original, report

        reference = reference_allocation(profile, features.tickers)
        fallback = capital_preservation_allocation(profile, features.tickers)
        steps = self.settings.max_repair_iterations
        # Rung 1: blend toward the class-appropriate anchor. Rung 2: if that is
        # still infeasible, continue toward a no-equity capital-preservation mix.
        candidates: list[tuple[dict[str, float], float]] = [
            (reference, i / float(steps)) for i in range(1, steps + 1)
        ] + [(fallback, t) for t in (0.5, 0.75, 1.0)]
        for iteration, (anchor, blend) in enumerate(candidates, start=1):
            mixed = _blend(original, anchor, blend)
            candidate_report = self.evaluate(mixed, profile, features, returns, turnover=turnover)
            if candidate_report.passed:
                return mixed, candidate_report.model_copy(
                    update={"repaired": True, "repair_iterations": iteration}
                )
        # The defensive anchor itself still breaches: return it and report the
        # residual violations instead of silently passing.
        mixed = _blend(original, fallback, 1.0)
        final_report = self.evaluate(mixed, profile, features, returns, turnover=turnover)
        return mixed, final_report.model_copy(
            update={"repaired": True, "repair_iterations": len(candidates)}
        )


def _blend(
    weights: Mapping[str, float], reference: Mapping[str, float], t: float
) -> dict[str, float]:
    keys = set(weights) | set(reference)
    mixed = {
        key: (1.0 - t) * float(weights.get(key, 0.0)) + t * float(reference.get(key, 0.0))
        for key in keys
    }
    total = sum(mixed.values())
    mixed = {k: v / total for k, v in mixed.items() if v / total > 1e-9}
    return mixed


def suitability_summary(report: SuitabilityReport) -> list[str]:
    """Human-readable, decision-path suitable rationale lines."""
    lines: list[str] = []
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        limit = f"{check.limit}" if check.limit is not None else "n/a"
        lines.append(f"[{status}] {check.rule_id}: observed {check.observed} (limit {limit})")
    return lines


__all__ = [
    "POLICIES",
    "RiskPolicy",
    "SuitabilityEngine",
    "capital_preservation_allocation",
    "policy_for",
    "reference_allocation",
    "suitability_summary",
]
