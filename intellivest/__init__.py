"""INTELLIVEST-MAS — hybrid multi-agent financial advisory system.

Pipeline (see ``docs/architecture.md``):

    Investor Profiling -> Data Layer -> Specialized Agents
        -> Quantitative Engine -> Portfolio Optimization
        -> Suitability Validation -> Recommendation -> Decision Trace

Design principle: LLM/agents handle reasoning, coordination, interpretation
and communication. Numerical forecasting, risk measurement, optimization and
constraint validation are performed by deterministic, testable code.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
