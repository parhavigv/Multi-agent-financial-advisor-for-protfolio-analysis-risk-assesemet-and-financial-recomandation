"""Investor personalization: risk class derivation and suitability rules."""

from intellivest.personalization.suitability import (
    POLICIES,
    RiskPolicy,
    SuitabilityEngine,
    policy_for,
    reference_allocation,
    suitability_summary,
)

__all__ = [
    "POLICIES",
    "RiskPolicy",
    "SuitabilityEngine",
    "policy_for",
    "reference_allocation",
    "suitability_summary",
]
