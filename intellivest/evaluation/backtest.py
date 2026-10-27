"""Walk-forward comparison of four advisory configurations.

A — **quant_only**: constrained mean-variance optimization, no agent views and
    no suitability remediation. Isolates the numerical engine.
B — **agents_only**: allocation taken directly from agent consensus (top
    positive convictions, equal-weighted), bypassing the optimizer, position
    caps, exclusions and suitability. This is the "let the agents decide"
    failure mode the survey warns about.
C — **hybrid**: coordinated agent views enter the Black-Litterman posterior,
    the optimizer chooses weights, then a blocking suitability gate remediates
    the result. The proposed design.
D — **hybrid_shuffled**: the same pipeline as C, but the asset-to-tilt mapping
    is permuted with a seeded RNG. This placebo keeps the distribution of
    agent views and the market view while destroying asset-level information.
    If C does not beat D, the apparent value of the agents is not attributable
    to their asset mapping.

All configurations share the same seeded market, rebalance dates, lookback and
transaction-cost assumption. Only the decision logic differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from intellivest.agents import AgentContext, default_agents
from intellivest.agents.base import AnalystAgent
from intellivest.config import Settings
from intellivest.contracts import InvestorProfile, OptimizationResult
from intellivest.data.market import MarketDataProvider, MarketSnapshot, build_provider
from intellivest.data.universe import asset_class_of, TICKERS
from intellivest.evaluation.metrics import PerformanceSummary, summarise
from intellivest.orchestration.coordinator import Coordinator, CoordinatorDecision
from intellivest.orchestration.orchestrator import AdvisoryEngine
from intellivest.personalization.suitability import SuitabilityEngine, policy_for
from intellivest.quant.engine import QuantEngine, QuantFeatures
from intellivest.quant.risk import portfolio_returns

CONFIGS: tuple[str, ...] = ("quant_only", "agents_only", "hybrid", "hybrid_shuffled")


@dataclass
class BacktestResult:
    config: str
    summary: PerformanceSummary
    equity_curve: pd.Series
    weights_history: list[dict[str, float]] = field(default_factory=list)
    period_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config,
            "metrics": self.summary.to_dict(),
            "final_equity": round(float(self.equity_curve.iloc[-1]), 6),
            "periods": len(self.weights_history),
        }


class Backtester:
    def __init__(
        self,
        settings: Settings | None = None,
        provider: MarketDataProvider | None = None,
        agents: list[AnalystAgent] | None = None,
        rebalance_every: int = 21,
    ) -> None:
        from intellivest.config import load_settings

        self.settings = settings or load_settings()
        self.provider = provider or build_provider(self.settings)
        self.quant = QuantEngine(self.settings)
        self.agents = agents or default_agents()
        self.coordinator = Coordinator(self.agents)
        self.engine = AdvisoryEngine(self.settings, provider=self.provider, agents=self.agents)
        self.suitability = SuitabilityEngine(self.settings)
        self.rebalance_every = rebalance_every

    # -- data preparation -------------------------------------------------
    def _periods(self, profile: InvestorProfile) -> list[dict[str, object]]:
        history = self.provider.history()
        lookback = self.settings.lookback_days
        if len(history) < lookback + 30:
            raise ValueError("history shorter than lookback; extend the data window")
        start = lookback + 5
        indices = list(range(start, len(history) - 1, self.rebalance_every))
        periods: list[dict[str, object]] = []
        for index in indices:
            as_of = history.index[index].date()
            snapshot = self.provider.snapshot(as_of)
            features = self.quant.build(snapshot, risk_free=self._risk_free_at(as_of))
            ctx = AgentContext(
                profile=profile,
                snapshot=snapshot,
                features=features,
                current_weights={},
                settings=self.settings,
            )
            reports = [agent.analyze(ctx) for agent in self.agents]
            decision = self.coordinator.coordinate(reports, features.tickers)
            periods.append(
                {
                    "index": index,
                    "as_of": as_of,
                    "snapshot": snapshot,
                    "features": features,
                    "ctx": ctx,
                    "decision": decision,
                    "next_index": min(index + self.rebalance_every, len(history) - 1),
                }
            )
        return periods

    # -- configuration logic ---------------------------------------------
    def _weights_for(
        self,
        config: str,
        profile: InvestorProfile,
        period: Mapping[str, object],
        previous: Mapping[str, float],
    ) -> tuple[dict[str, float], OptimizationResult | None, bool]:
        ctx: AgentContext = period["ctx"]  # type: ignore[assignment]
        decision: CoordinatorDecision = period["decision"]  # type: ignore[assignment]
        features: QuantFeatures = period["features"]  # type: ignore[assignment]
        ctx = AgentContext(
            profile=profile,
            snapshot=ctx.snapshot,
            features=features,
            current_weights=dict(previous),
            settings=self.settings,
        )

        if config in {"hybrid", "hybrid_shuffled"}:
            effective = decision
            if config == "hybrid_shuffled":
                effective = shuffle_decision(
                    decision,
                    features.tickers,
                    seed=self.settings.seed + int(period["index"]),
                )
            optimization, suitability, _ = self.engine.construct(profile, ctx, effective)
            return optimization.weights, optimization, not suitability.passed

        if config == "quant_only":
            neutral = CoordinatorDecision(
                tilts={},
                market_score=0.0,
                portfolio_score=0.0,
                agreement=1.0,
                dispersion=0.0,
            )
            optimization, _, _ = self.engine.optimise(
                profile, features, ctx.snapshot.returns, neutral, dict(previous)
            )
            report = self.suitability.evaluate(
                optimization.weights,
                profile,
                features,
                ctx.snapshot.returns,
                turnover=optimization.turnover,
            )
            return optimization.weights, optimization, not report.passed

        # agents_only: direct consensus allocation, no quantitative guardrails.
        candidates = sorted(
            ((t, s) for t, s in decision.tilts.items() if s > 0.05),
            key=lambda pair: pair[1],
            reverse=True,
        )[:5]
        if not candidates:
            ranked = features.expected_returns.sort_values(ascending=False).head(5)
            candidates = [(t, float(score)) for t, score in ranked.items()]
        total = sum(score for _, score in candidates)
        weights = {ticker: score / total for ticker, score in candidates if total > 0}
        report = self.suitability.evaluate(
            weights, profile, features, ctx.snapshot.returns, turnover=0.0
        )
        return weights, None, not report.passed

    # -- execution --------------------------------------------------------
    def _risk_free_at(self, as_of: date) -> float:
        """Point-in-time risk-free rate for a single rebalance date."""
        getter = getattr(self.provider, "get_risk_free", None)
        if not callable(getter):
            return float(self.settings.risk_free_rate)
        try:
            return float(getter(as_of))
        except Exception:
            return float(self.settings.risk_free_rate)

    def _risk_free_for(self, returns: pd.Series) -> float:
        """Average point-in-time risk-free rate over the backtest sample.

        Uses the FRED DGS3MO series (via the provider) so the rate matches the
        period actually traded instead of a hard-coded constant. Falls back to
        the configured value when no real series is available.
        """
        if returns.empty:
            return float(self.settings.risk_free_rate)
        getter = getattr(self.provider, "get_risk_free", None)
        if not callable(getter):
            return float(self.settings.risk_free_rate)
        rates = []
        for stamp in (returns.index[0], returns.index[len(returns) // 2], returns.index[-1]):
            try:
                rates.append(float(getter(pd.Timestamp(stamp).date())))
            except Exception:
                continue
        if not rates:
            return float(self.settings.risk_free_rate)
        return float(sum(rates) / len(rates))

    def run(self, profile: InvestorProfile, configs: Iterable[str] = CONFIGS) -> list[BacktestResult]:
        history = self.provider.history()
        daily_returns = history.pct_change().fillna(0.0)
        periods = self._periods(profile)
        results: list[BacktestResult] = []

        for config in configs:
            previous: dict[str, float] = {}
            pieces: list[pd.Series] = []
            turnovers: list[float] = []
            equity_weights: list[float] = []
            breaches: list[bool] = []
            weights_history: list[dict[str, float]] = []
            notes: list[str] = []

            for period in periods:
                index = int(period["index"])  # type: ignore[arg-type]
                next_index = int(period["next_index"])  # type: ignore[arg-type]
                weights, _, breached = self._weights_for(config, profile, period, previous)
                weights_history.append(dict(weights))
                breaches.append(breached)
                equity_weights.append(
                    float(sum(w for t, w in weights.items() if asset_class_of(t) == "equity"))
                )
                turnover = _turnover(weights, previous)
                turnovers.append(turnover)
                previous = weights

                window = daily_returns.iloc[index + 1 : next_index + 1]
                series = portfolio_returns(weights, window)
                if not series.empty:
                    cost = turnover * self.settings.transaction_cost_bps / 10_000.0
                    series = series.copy()
                    series.iloc[0] = (1.0 + series.iloc[0]) * (1.0 - cost) - 1.0
                pieces.append(series)
                if breached:
                    notes.append(f"{period['as_of']}: suitability breach recorded")

            combined = pd.concat(pieces) if pieces else pd.Series(dtype=float)
            rf = self._risk_free_for(combined)
            summary = summarise(
                combined,
                risk_free=rf,
                periods=self.settings.trading_days,
                turnovers=turnovers,
                equity_weights=equity_weights,
                breaches=breaches,
                cost_bps=self.settings.transaction_cost_bps,
            )
            equity_curve = (1.0 + combined.fillna(0.0)).cumprod()
            results.append(
                BacktestResult(
                    config=config,
                    summary=summary,
                    equity_curve=equity_curve,
                    weights_history=weights_history,
                    period_notes=notes,
                )
            )
        return results


def shuffle_decision(
    decision: CoordinatorDecision,
    tickers: Iterable[str],
    *,
    seed: int,
) -> CoordinatorDecision:
    """Permute asset-level views while preserving their distribution.

    The market and portfolio scores are left unchanged: this is a placebo for
    the *asset mapping*, not for the existence of a macro view. Confidence and
    dispersion travel with their tilt so the Black-Litterman uncertainty is
    shuffled consistently rather than accidentally reset.
    """
    order = [ticker for ticker in tickers if ticker in decision.tilts]
    if len(order) < 2:
        return decision
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(len(order))
    if np.array_equal(permutation, np.arange(len(order))):
        permutation = np.roll(permutation, 1)
    shuffled_tilts = {
        order[index]: decision.tilts[order[int(source)]]
        for index, source in enumerate(permutation)
    }
    shuffled_confidence = {
        order[index]: decision.view_confidence.get(
            order[int(source)], decision.view_confidence.get(order[index], 0.5)
        )
        for index, source in enumerate(permutation)
    }
    shuffled_dispersion = {
        order[index]: decision.view_dispersion.get(
            order[int(source)], decision.view_dispersion.get(order[index], 0.0)
        )
        for index, source in enumerate(permutation)
    }
    return replace(
        decision,
        tilts=shuffled_tilts,
        view_confidence=shuffled_confidence,
        view_dispersion=shuffled_dispersion,
    )


def _turnover(weights: Mapping[str, float], previous: Mapping[str, float]) -> float:
    keys = set(weights) | set(previous)
    return float(sum(abs(weights.get(k, 0.0) - previous.get(k, 0.0)) for k in keys) / 2.0)


def default_profile(as_of: date) -> InvestorProfile:
    """The reference investor used by the reported evaluation."""
    return InvestorProfile(
        investor_id="EVAL-001",
        as_of=as_of,
        age=34,
        horizon_years=12.0,
        risk_tolerance=4,
        loss_tolerance_pct=0.20,
        goal="growth",
        liquidity_need="low",
        current_holdings={"SPX": 0.35, "TECH": 0.30, "GOLD": 0.10, "GOVT": 0.25},
    )


__all__ = [
    "Backtester",
    "BacktestResult",
    "CONFIGS",
    "default_profile",
    "shuffle_decision",
]
