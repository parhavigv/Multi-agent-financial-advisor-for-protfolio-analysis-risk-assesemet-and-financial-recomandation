"""Walk-forward comparison of Phase 4 advisory strategies.

The evaluation answers a narrower question than a return-maximisation study:
does the proposed hybrid add anything once the comparison includes a *fair*
suitability-gated quantitative baseline, realistic transaction costs, a
five-profile investor panel, and uncertainty intervals that respect serial
dependence?

Strategies
----------
``one_over_n``
    Equal weight across every allowed asset. No optimizer, no agent views.
``sixty_forty``
    60% in the equity sleeve (equal weight), 40% in fixed income (equal
    weight). The classic strategic benchmark.
``min_variance``
    Long-only constrained minimum variance with global position and sector
    caps. No suitability gate.
``quant_only_no_gate``
    Constrained mean-variance on the quantitative prior with a neutral view
    set. No suitability gate. This is the "optimizer alone" baseline.
``quant_only_gated``
    The same quantitative optimizer followed by the blocking suitability gate.
    This is the fair baseline for the hybrid: identical numerical engine and
    identical gate, but no agent views.
``agents_only``
    Direct agent-consensus allocation, bypassing the optimizer and the gate.
    The "let the agents decide" failure mode.
``hybrid``
    Coordinated agent views enter a Black-Litterman posterior, the constrained
    optimizer chooses weights, and the blocking suitability gate remediates the
    result.

``hybrid_shuffled`` is retained as an opt-in placebo/backward-compatible
configuration but is not part of the Phase 4 primary panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from intellivest.agents import AgentContext, default_agents
from intellivest.agents.base import AnalystAgent
from intellivest.config import Settings
from intellivest.contracts import InvestorProfile, OptimizationResult
from intellivest.data.market import MarketDataProvider
from intellivest.data.universe import asset_class_of, sector_of
from intellivest.evaluation.bootstrap import BootstrapCI, bootstrap_ci
from intellivest.evaluation.metrics import PerformanceSummary, summarise
from intellivest.orchestration.coordinator import Coordinator, CoordinatorDecision
from intellivest.orchestration.orchestrator import AdvisoryEngine
from intellivest.personalization.suitability import SuitabilityEngine, policy_for
from intellivest.quant import risk
from intellivest.quant.engine import QuantEngine, QuantFeatures
from intellivest.quant.optimize import Constraints, annotate, min_variance
from intellivest.quant.risk import portfolio_returns

BASELINE_CONFIGS: tuple[str, ...] = (
    "one_over_n",
    "sixty_forty",
    "min_variance",
    "quant_only_no_gate",
    "quant_only_gated",
    "agents_only",
)
CONFIGS: tuple[str, ...] = BASELINE_CONFIGS + ("hybrid",)
PLACEBO_CONFIG = "hybrid_shuffled"
AGENT_CONFIGS: frozenset[str] = frozenset({"agents_only", "hybrid", PLACEBO_CONFIG})

# Regime windows are fixed before looking at strategy results. They are
# descriptive sub-periods, not tuned buckets.
REGIME_WINDOWS: dict[str, dict[str, object]] = {
    "2020_crash": {
        "label": "COVID crash",
        "start": date(2020, 2, 19),
        "end": date(2020, 4, 30),
    },
    "2022_rates": {
        "label": "Rate shock",
        "start": date(2022, 1, 1),
        "end": date(2022, 12, 31),
    },
}


@dataclass(frozen=True)
class ProfileAlignment:
    """How a realised path lines up with the investor's stated constraints."""

    profile_id: str
    risk_class: int
    realised_volatility: float
    volatility_limit: float
    volatility_ratio: float
    volatility_ok: bool
    realised_drawdown: float
    drawdown_limit: float
    drawdown_ratio: float
    drawdown_ok: bool
    recommendation_stability: float
    inter_agent_agreement: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "risk_class": self.risk_class,
            "realised_volatility": round(self.realised_volatility, 6),
            "volatility_limit": round(self.volatility_limit, 6),
            "volatility_ratio": round(self.volatility_ratio, 4),
            "volatility_ok": self.volatility_ok,
            "realised_drawdown": round(self.realised_drawdown, 6),
            "drawdown_limit": round(self.drawdown_limit, 6),
            "drawdown_ratio": round(self.drawdown_ratio, 4),
            "drawdown_ok": self.drawdown_ok,
            "recommendation_stability": round(self.recommendation_stability, 4),
            "inter_agent_agreement": (
                round(self.inter_agent_agreement, 4)
                if self.inter_agent_agreement is not None
                else None
            ),
        }


@dataclass
class BacktestResult:
    config: str
    summary: PerformanceSummary
    equity_curve: pd.Series
    daily_returns: pd.Series = field(
        repr=False, default_factory=lambda: pd.Series(dtype=float)
    )
    risk_free: float = 0.0
    weights_history: list[dict[str, float]] = field(default_factory=list)
    turnovers: list[float] = field(default_factory=list)
    agreements: list[float] = field(default_factory=list)
    profile_alignment: ProfileAlignment | None = None
    bootstrap: dict[str, BootstrapCI] = field(default_factory=dict)
    period_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "config": self.config,
            "metrics": self.summary.to_dict(),
            "final_equity": round(float(self.equity_curve.iloc[-1]), 6)
            if not self.equity_curve.empty
            else 0.0,
            "periods": len(self.weights_history),
            "risk_free": round(self.risk_free, 6),
            "profile_alignment": self.profile_alignment.to_dict()
            if self.profile_alignment is not None
            else None,
            "bootstrap": {
                name: interval.to_dict() for name, interval in self.bootstrap.items()
            },
        }
        return payload


class Backtester:
    def __init__(
        self,
        settings: Settings | None = None,
        provider: MarketDataProvider | None = None,
        agents: list[AnalystAgent] | None = None,
        rebalance_every: int = 21,
        lookback_days: int | None = None,
    ) -> None:
        from intellivest.config import load_settings

        self.settings = settings or load_settings()
        self.provider = provider
        if self.provider is None:
            from intellivest.data.market import build_provider

            self.provider = build_provider(self.settings)
        self.quant = QuantEngine(self.settings)
        self.agents = agents or default_agents()
        self.coordinator = Coordinator(self.agents)
        self.engine = AdvisoryEngine(
            self.settings, provider=self.provider, agents=self.agents
        )
        self.suitability = SuitabilityEngine(self.settings)
        self.rebalance_every = rebalance_every
        self.lookback_days = int(lookback_days or self.settings.lookback_days)
        self._base_periods_cache: list[dict[str, object]] | None = None

    # -- data preparation -------------------------------------------------
    def _base_periods(self) -> list[dict[str, object]]:
        """Snapshots and features shared by every profile.

        The HMM regime fit and the feature build are profile-independent. In a
        five-profile panel this cache cuts that work by roughly five times while
        leaving the profile-specific agent decisions intact.
        """
        if self._base_periods_cache is not None:
            return self._base_periods_cache
        history = self.provider.history()
        if len(history) < self.lookback_days + 30:
            raise ValueError(
                "history shorter than lookback; extend the data window or reduce lookback_days"
            )
        start = self.lookback_days + 5
        indices = list(range(start, len(history) - 1, self.rebalance_every))
        base: list[dict[str, object]] = []
        for index in indices:
            as_of = history.index[index].date()
            snapshot = self.provider.snapshot(as_of, lookback_days=self.lookback_days)
            features = self.quant.build(snapshot, risk_free=self._risk_free_at(as_of))
            base.append(
                {
                    "index": index,
                    "as_of": as_of,
                    "snapshot": snapshot,
                    "features": features,
                    "next_index": min(index + self.rebalance_every, len(history) - 1),
                }
            )
        self._base_periods_cache = base
        return base

    def _agent_decision(
        self,
        profile: InvestorProfile,
        period: Mapping[str, object],
        previous: Mapping[str, float],
    ) -> CoordinatorDecision:
        """Run the panel with the strategy's actual current holdings.

        Agent views for the portfolio and risk analysts depend on the weights
        the strategy is carrying into the rebalance. Computing one decision per
        profile with empty holdings and reusing it for every strategy would give
        those agents the wrong portfolio state and would not be a fair test of
        the hybrid.
        """
        features: QuantFeatures = period["features"]  # type: ignore[assignment]
        snapshot = period["snapshot"]
        profile_at = profile.model_copy(update={"as_of": period["as_of"]})  # type: ignore[arg-type]
        ctx = AgentContext(
            profile=profile_at,
            snapshot=snapshot,  # type: ignore[arg-type]
            features=features,
            current_weights=dict(previous),
            settings=self.settings,
        )
        reports = [agent.analyze(ctx) for agent in self.agents]
        return self.coordinator.coordinate(reports, features.tickers)

    # -- strategy logic ---------------------------------------------------
    def _weights_for(
        self,
        config: str,
        profile: InvestorProfile,
        period: Mapping[str, object],
        previous: Mapping[str, float],
        decision: CoordinatorDecision,
    ) -> tuple[dict[str, float], OptimizationResult | None, bool]:
        features: QuantFeatures = period["features"]  # type: ignore[assignment]
        snapshot = period["snapshot"]
        # The profile's as_of is the evaluation end date; each decision must be
        # evaluated with the point-in-time risk-free rate at the rebalance date.
        profile_at = profile.model_copy(update={"as_of": period["as_of"]})  # type: ignore[arg-type]
        excluded = _excluded_set(profile)
        allowed = [ticker for ticker in features.tickers if ticker not in excluded]
        ctx_at = AgentContext(
            profile=profile_at,
            snapshot=snapshot,  # type: ignore[arg-type]
            features=features,
            current_weights=dict(previous),
            settings=self.settings,
        )

        if config == "one_over_n":
            weights = equal_weight(allowed)
            return self._unoptimized(weights, profile_at, features, ctx_at, previous)

        if config == "sixty_forty":
            weights = sixty_forty_weights(allowed)
            return self._unoptimized(weights, profile_at, features, ctx_at, previous)

        if config == "min_variance":
            constraints = Constraints(
                max_weight=self.settings.max_weight,
                max_sector_weight=self.settings.max_sector_weight,
                turnover_penalty=self.settings.turnover_penalty,
                diversification=0.6,
            )
            cov = features.covariance.loc[allowed, allowed]
            rf = self._risk_free_at(snapshot.as_of)  # type: ignore[union-attr]
            optimization = min_variance(
                cov,
                previous_weights=dict(previous),
                constraints=constraints,
                sector_of=sector_of,
                exclude=excluded,
                risk_free=rf,
            )
            optimization = annotate(
                optimization,
                features.expected_returns,
                snapshot.returns,  # type: ignore[union-attr]
                rf,
            )
            return self._unoptimized(
                optimization.weights, profile_at, features, ctx_at, previous, optimization
            )

        neutral = CoordinatorDecision(
            tilts={},
            market_score=0.0,
            portfolio_score=0.0,
            agreement=1.0,
            dispersion=0.0,
        )
        if config in {"quant_only_no_gate", "quant_only_gated"}:
            optimization, _, _ = self.engine.optimise(
                profile_at,
                features,
                snapshot.returns,  # type: ignore[union-attr]
                neutral,
                dict(previous),
            )
            if config == "quant_only_gated":
                optimization, suitability = self.engine.apply_suitability(
                    profile_at,
                    features,
                    snapshot.returns,  # type: ignore[union-attr]
                    optimization,
                    features.expected_returns,
                    dict(previous),
                )
                return optimization.weights, optimization, not suitability.passed
            return self._unoptimized(
                optimization.weights, profile_at, features, ctx_at, previous, optimization
            )

        if config in {"hybrid", PLACEBO_CONFIG}:
            effective = decision
            if config == PLACEBO_CONFIG:
                effective = shuffle_decision(
                    decision,
                    features.tickers,
                    seed=self.settings.seed + int(period["index"]),  # type: ignore[arg-type]
                )
            optimization, suitability, _ = self.engine.construct(profile_at, ctx_at, effective)
            return optimization.weights, optimization, not suitability.passed

        if config == "agents_only":
            weights = consensus_weights(decision, features, allowed)
            return self._unoptimized(weights, profile_at, features, ctx_at, previous)

        raise ValueError(f"unknown configuration: {config}")

    def _unoptimized(
        self,
        weights: Mapping[str, float],
        profile: InvestorProfile,
        features: QuantFeatures,
        ctx: AgentContext,
        previous: Mapping[str, float],
        optimization: OptimizationResult | None = None,
    ) -> tuple[dict[str, float], OptimizationResult | None, bool]:
        clean = {ticker: float(weight) for ticker, weight in weights.items() if weight > 1e-9}
        total = sum(clean.values())
        if total <= 0:
            clean = equal_weight(features.tickers)
        else:
            clean = {ticker: weight / total for ticker, weight in clean.items()}
        report = self.suitability.evaluate(
            clean,
            profile,
            features,
            ctx.snapshot.returns,
            turnover=_turnover(clean, previous),
        )
        return clean, optimization, not report.passed

    # -- execution --------------------------------------------------------
    def _risk_free_at(self, as_of: date) -> float:
        getter = getattr(self.provider, "get_risk_free", None)
        if not callable(getter):
            return float(self.settings.risk_free_rate)
        try:
            return float(getter(as_of))
        except Exception:
            return float(self.settings.risk_free_rate)

    def _risk_free_for(self, returns: pd.Series) -> float:
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

    def run(
        self,
        profile: InvestorProfile,
        configs: Iterable[str] = CONFIGS,
        *,
        bootstrap: bool = False,
        n_boot: int = 1000,
        block_length: int = 21,
    ) -> list[BacktestResult]:
        history = self.provider.history()
        daily_returns = history.pct_change().fillna(0.0)
        periods = self._base_periods()
        neutral = CoordinatorDecision(
            tilts={},
            market_score=0.0,
            portfolio_score=0.0,
            agreement=1.0,
            dispersion=0.0,
        )
        results: list[BacktestResult] = []

        for config_index, config in enumerate(configs):
            previous: dict[str, float] = {}
            pieces: list[pd.Series] = []
            turnovers: list[float] = []
            equity_weights: list[float] = []
            breaches: list[bool] = []
            weights_history: list[dict[str, float]] = []
            agreements: list[float] = []
            notes: list[str] = []

            for period in periods:
                index = int(period["index"])  # type: ignore[arg-type]
                next_index = int(period["next_index"])  # type: ignore[arg-type]
                decision = (
                    self._agent_decision(profile, period, previous)
                    if config in AGENT_CONFIGS
                    else neutral
                )
                weights, _, breached = self._weights_for(
                    config, profile, period, previous, decision
                )
                weights_history.append(dict(weights))
                breaches.append(breached)
                if config in AGENT_CONFIGS:
                    agreements.append(float(decision.agreement))
                equity_weights.append(
                    float(
                        sum(
                            weight
                            for ticker, weight in weights.items()
                            if asset_class_of(ticker) == "equity"
                        )
                    )
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
            policy = policy_for(profile)
            realised_drawdown = abs(summary.max_drawdown)
            alignment = ProfileAlignment(
                profile_id=profile.investor_id,
                risk_class=profile.suitability_class,
                realised_volatility=summary.annual_volatility,
                volatility_limit=policy.max_volatility,
                volatility_ratio=(
                    summary.annual_volatility / policy.max_volatility
                    if policy.max_volatility > 1e-12
                    else 0.0
                ),
                volatility_ok=summary.annual_volatility <= policy.max_volatility + 1e-9,
                realised_drawdown=realised_drawdown,
                drawdown_limit=profile.loss_tolerance_pct,
                drawdown_ratio=(
                    realised_drawdown / profile.loss_tolerance_pct
                    if profile.loss_tolerance_pct > 1e-12
                    else 0.0
                ),
                drawdown_ok=realised_drawdown <= profile.loss_tolerance_pct + 1e-9,
                recommendation_stability=weight_stability(weights_history),
                inter_agent_agreement=(
                    float(np.mean(agreements)) if config in AGENT_CONFIGS and agreements else None
                ),
            )

            intervals: dict[str, BootstrapCI] = {}
            if bootstrap:
                seed = self.settings.seed + 1000 * profile.suitability_class + config_index
                intervals["sharpe"] = bootstrap_ci(
                    combined.to_numpy(dtype=float),
                    lambda values, rf=rf: risk.sharpe_ratio(
                        pd.Series(values), rf, self.settings.trading_days
                    ),
                    block_length=block_length,
                    n_boot=n_boot,
                    seed=seed,
                )
                intervals["max_drawdown"] = bootstrap_ci(
                    combined.to_numpy(dtype=float),
                    lambda values: risk.max_drawdown(pd.Series(values)),
                    block_length=block_length,
                    n_boot=n_boot,
                    seed=seed + 1,
                )

            results.append(
                BacktestResult(
                    config=config,
                    summary=summary,
                    equity_curve=equity_curve,
                    daily_returns=combined,
                    risk_free=rf,
                    weights_history=weights_history,
                    turnovers=turnovers,
                    agreements=agreements,
                    profile_alignment=alignment,
                    bootstrap=intervals,
                    period_notes=notes,
                )
            )
        return results


def equal_weight(tickers: Iterable[str]) -> dict[str, float]:
    universe = [ticker for ticker in tickers if ticker]
    if not universe:
        return {}
    weight = 1.0 / len(universe)
    return {ticker: weight for ticker in universe}


def sixty_forty_weights(tickers: Iterable[str]) -> dict[str, float]:
    """60% equal-weight equity sleeve, 40% equal-weight fixed-income sleeve."""
    universe = [ticker for ticker in tickers if ticker]
    equity = [ticker for ticker in universe if asset_class_of(ticker) == "equity"]
    fixed = [ticker for ticker in universe if asset_class_of(ticker) == "fixed_income"]
    if not equity or not fixed:
        return equal_weight(universe)
    weights = {ticker: 0.60 / len(equity) for ticker in equity}
    weights.update({ticker: 0.40 / len(fixed) for ticker in fixed})
    total = sum(weights.values())
    return {ticker: weight / total for ticker, weight in weights.items()}


def consensus_weights(
    decision: CoordinatorDecision,
    features: QuantFeatures,
    tickers: Iterable[str],
) -> dict[str, float]:
    """Direct agent-consensus allocation without optimizer or gate."""
    allowed = set(tickers)
    candidates = sorted(
        (
            (ticker, score)
            for ticker, score in decision.tilts.items()
            if ticker in allowed and score > 0.05
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )[:5]
    if not candidates:
        ranked = (
            features.expected_returns.reindex(list(allowed)).dropna().sort_values(ascending=False)
        )
        candidates = [(ticker, 1.0) for ticker in ranked.head(5).index]
    if not candidates:
        return equal_weight(tickers)
    total = sum(score for _, score in candidates)
    if total <= 0:
        return equal_weight([ticker for ticker, _ in candidates])
    return {ticker: score / total for ticker, score in candidates}


def weight_stability(history: list[dict[str, float]]) -> float:
    """Mean cosine similarity between consecutive rebalance weight vectors."""
    if len(history) < 2:
        return 1.0
    values: list[float] = []
    for left, right in zip(history, history[1:]):
        keys = set(left) | set(right)
        a = np.array([left.get(key, 0.0) for key in keys], dtype=float)
        b = np.array([right.get(key, 0.0) for key in keys], dtype=float)
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        values.append(float(a @ b / denominator) if denominator > 1e-12 else 1.0)
    return float(np.mean(values)) if values else 1.0


def regime_statistics(
    daily_returns: pd.Series,
    start: date,
    end: date,
) -> dict[str, float | int]:
    """Cumulative return and max drawdown inside a fixed calendar window."""
    if daily_returns.empty:
        return {"observations": 0, "cumulative_return": 0.0, "max_drawdown": 0.0}
    window = daily_returns.loc[
        (daily_returns.index >= pd.Timestamp(start))
        & (daily_returns.index <= pd.Timestamp(end))
    ]
    if window.empty:
        return {"observations": 0, "cumulative_return": 0.0, "max_drawdown": 0.0}
    return {
        "observations": int(len(window)),
        "cumulative_return": float((1.0 + window).prod() - 1.0),
        "max_drawdown": float(risk.max_drawdown(window)),
    }


def shuffle_decision(
    decision: CoordinatorDecision,
    tickers: Iterable[str],
    *,
    seed: int,
) -> CoordinatorDecision:
    """Permute asset-level views while preserving their distribution."""
    from dataclasses import replace

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
    return float(
        sum(abs(weights.get(key, 0.0) - previous.get(key, 0.0)) for key in keys) / 2.0
    )


def _excluded_set(profile: InvestorProfile) -> set[str]:
    from intellivest.data.universe import excluded_tickers

    return excluded_tickers(profile.exclusions)


def default_profile(as_of: date) -> InvestorProfile:
    """Backward-compatible reference investor."""
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
    "AGENT_CONFIGS",
    "BASELINE_CONFIGS",
    "Backtester",
    "BacktestResult",
    "CONFIGS",
    "PLACEBO_CONFIG",
    "ProfileAlignment",
    "REGIME_WINDOWS",
    "consensus_weights",
    "default_profile",
    "equal_weight",
    "regime_statistics",
    "shuffle_decision",
    "sixty_forty_weights",
    "weight_stability",
]
