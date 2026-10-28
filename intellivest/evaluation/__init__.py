"""Evaluation harness: Phase 4 multi-profile, multi-regime comparison."""

from intellivest.evaluation.backtest import (
    AGENT_CONFIGS,
    BASELINE_CONFIGS,
    CONFIGS,
    PLACEBO_CONFIG,
    REGIME_WINDOWS,
    BacktestResult,
    Backtester,
    ProfileAlignment,
    default_profile,
    regime_statistics,
)
from intellivest.evaluation.bootstrap import (
    BootstrapCI,
    PairedBootstrap,
    bootstrap_ci,
    moving_block_indices,
    paired_bootstrap_test,
)
from intellivest.evaluation.metrics import PerformanceSummary, summarise
from intellivest.evaluation.phase4 import run_phase4
from intellivest.evaluation.profiles import phase4_profiles

__all__ = [
    "AGENT_CONFIGS",
    "BASELINE_CONFIGS",
    "CONFIGS",
    "PLACEBO_CONFIG",
    "REGIME_WINDOWS",
    "Backtester",
    "BacktestResult",
    "BootstrapCI",
    "PairedBootstrap",
    "PerformanceSummary",
    "ProfileAlignment",
    "bootstrap_ci",
    "default_profile",
    "moving_block_indices",
    "paired_bootstrap_test",
    "phase4_profiles",
    "regime_statistics",
    "run_phase4",
    "summarise",
]
