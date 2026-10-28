"""Investor profiles used by the Phase 4 evaluation panel.

The panel deliberately spans the full suitability scale: risk classes 1-5,
short and long horizons, and different stated loss tolerances. Each profile
has no legacy holdings so every strategy starts from the same cash position;
the first rebalance therefore pays the same initial build cost in every arm.
"""

from __future__ import annotations

from datetime import date

from intellivest.contracts import InvestorProfile


def phase4_profiles(as_of: date) -> list[InvestorProfile]:
    """Return one profile for each binding suitability class (1-5)."""
    profiles = [
        InvestorProfile(
            investor_id="P1-CAPITAL",
            as_of=as_of,
            age=68,
            horizon_years=3.0,
            risk_tolerance=1,
            loss_tolerance_pct=0.05,
            goal="capital_preservation",
            liquidity_need="high",
            current_holdings={},
        ),
        InvestorProfile(
            investor_id="P2-INCOME",
            as_of=as_of,
            age=58,
            horizon_years=5.0,
            risk_tolerance=2,
            loss_tolerance_pct=0.10,
            goal="income",
            liquidity_need="medium",
            current_holdings={},
        ),
        InvestorProfile(
            investor_id="P3-BALANCED",
            as_of=as_of,
            age=45,
            horizon_years=10.0,
            risk_tolerance=3,
            loss_tolerance_pct=0.15,
            goal="balanced_growth",
            liquidity_need="low",
            current_holdings={},
        ),
        InvestorProfile(
            investor_id="P4-GROWTH",
            as_of=as_of,
            age=34,
            horizon_years=20.0,
            risk_tolerance=4,
            loss_tolerance_pct=0.20,
            goal="growth",
            liquidity_need="low",
            current_holdings={},
        ),
        InvestorProfile(
            investor_id="P5-AGGRESSIVE",
            as_of=as_of,
            age=30,
            horizon_years=30.0,
            risk_tolerance=5,
            loss_tolerance_pct=0.30,
            goal="aggressive_growth",
            liquidity_need="low",
            current_holdings={},
        ),
    ]
    assert [profile.suitability_class for profile in profiles] == [1, 2, 3, 4, 5]
    return profiles


__all__ = ["phase4_profiles"]
