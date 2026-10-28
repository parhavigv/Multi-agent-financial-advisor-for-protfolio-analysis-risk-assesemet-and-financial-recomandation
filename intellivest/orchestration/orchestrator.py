"""End-to-end advisory orchestration.

    profile -> data -> quant features -> specialist agents -> coordinator
            -> optimizer -> suitability (with repair) -> recommendation
            -> explanation -> hash-chained decision trace

The orchestrator is the only place that knows the ordering. Each stage is
swappable, and every stage records a digest into the trace.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Mapping

import pandas as pd

from intellivest.agents import AgentContext, AnalystAgent, RecommendationAgent, default_agents
from intellivest.config import Settings, load_settings
from intellivest.contracts import (
    AgentReport,
    BlackLittermanSection,
    DecisionTrace,
    Explanation,
    InvestorProfile,
    MLForecastSection,
    OptimizationResult,
    RiskMetrics,
    SuitabilityReport,
)
from intellivest.data.market import MarketDataProvider, build_provider
from intellivest.data.universe import asset_class_of, excluded_tickers, sector_of
from intellivest.explain.attribution import (
    return_contributions,
    variance_shapley,
)
from intellivest.orchestration.coordinator import Coordinator, CoordinatorDecision
from intellivest.orchestration.trace import AuditChain, TraceRecorder, digest
from intellivest.personalization.suitability import (
    SuitabilityEngine,
    policy_for,
    suitability_summary,
)
from intellivest.quant.black_litterman import BlackLittermanResult, posterior_expected_returns
from intellivest.quant.engine import QuantEngine, QuantFeatures
from intellivest.quant.optimize import Constraints, annotate, mean_variance, portfolio_volatility


class AdvisoryEngine:
    def __init__(
        self,
        settings: Settings | None = None,
        provider: MarketDataProvider | None = None,
        agents: list[AnalystAgent] | None = None,
        use_ml: bool = False,
    ) -> None:
        self.settings = settings or load_settings()
        self.provider = provider or build_provider(self.settings)
        self.quant = QuantEngine(self.settings, use_ml=use_ml)
        self.agents = agents or default_agents()
        self.coordinator = Coordinator(self.agents)
        self.suitability = SuitabilityEngine(self.settings)
        self.recommender = RecommendationAgent(self.settings)
        self.audit = AuditChain(self.settings.audits_dir)
        self.use_ml = use_ml

    # -- public -----------------------------------------------------------
    def advise(self, profile: InvestorProfile) -> DecisionTrace:
        recorder = TraceRecorder()
        trace_id = uuid.uuid4().hex[:16]
        created_at = datetime.now(timezone.utc)
        try:
            rf = self._risk_free(profile)
        except Exception:
            rf = self.settings.risk_free_rate
        self._last_rf = rf
        config_digest = digest(
            {
                "seed": self.settings.seed,
                "max_weight": self.settings.max_weight,
                "max_sector_weight": self.settings.max_sector_weight,
                "risk_free": rf,
                "turnover_penalty": self.settings.turnover_penalty,
                "transaction_cost_bps": self.settings.transaction_cost_bps,
                "bl_tau": self.settings.bl_tau,
                "bl_view_shift": self.settings.bl_view_shift,
                "bl_max_total_influence": self.settings.bl_max_total_influence,
                "llm_provider": self.settings.llm_provider,
                "llm_model": self.settings.llm_model if self.settings.llm_enabled else "none",
                "covariance": self.quant.covariance_method,
            }
        )
        profile_digest = digest(profile.model_dump(mode="json"))
        current_weights = _normalise_holdings(profile.current_holdings)

        with recorder.step("profile.normalise") as step:
            step["inputs"] = profile.model_dump(mode="json")
            step["outputs"] = {
                "risk_class": profile.suitability_class,
                "risk_capacity": profile.risk_capacity,
                "horizon_years": profile.horizon_years,
                "exclusions": profile.exclusions,
                "current_weights": current_weights,
            }
            step["notes"] = (
                f"Binding risk class {profile.suitability_class} = min(tolerance "
                f"{profile.risk_tolerance}, capacity {profile.risk_capacity}, loss tolerance)."
            )

        with recorder.step("data.snapshot") as step:
            snapshot = self.provider.snapshot(profile.as_of)
            step["inputs"] = {"as_of": profile.as_of.isoformat(), "lookback_days": self.settings.lookback_days}
            step["outputs"] = {
                "rows": len(snapshot.prices),
                "first": str(snapshot.prices.index[0].date()),
                "last": str(snapshot.prices.index[-1].date()),
                "regime": snapshot.regime,
                "news_items": len(snapshot.news),
            }
            step["notes"] = "Prices truncated at as_of; no future row can enter the pipeline."

        with recorder.step("quant.features") as step:
            features = self.quant.build(snapshot, risk_free=rf)
            step["inputs"] = {"returns_rows": len(snapshot.returns)}
            step["outputs"] = {
                "covariance_method": features.covariance_method,
                "expected_returns": features.expected_returns.round(4).to_dict(),
                "volatility": features.volatility_forecast.round(4).to_dict(),
                "regime": features.regime,
                "risk_free": rf,
                "risk_free_source": getattr(self.provider, "risk_free_source", "settings"),
            }

        with recorder.step("agents.analyse") as step:
            ctx = AgentContext(
                profile=profile,
                snapshot=snapshot,
                features=features,
                current_weights=current_weights,
                settings=self.settings,
            )
            reports: list[AgentReport] = [agent.analyze(ctx) for agent in self.agents]
            step["inputs"] = {"agents": [agent.name for agent in self.agents]}
            step["outputs"] = {
                "reports": [
                    {"agent": r.agent, "signals": len(r.signals), "confidence": r.confidence,
                     "degraded": r.degraded}
                    for r in reports
                ]
            }
            degraded_agents = any(report.degraded for report in reports)
            if degraded_agents:
                step["status"] = "degraded"
                step["notes"] = "One or more agents fell back to the deterministic analyst."

        with recorder.step("agents.coordinate") as step:
            decision = self.coordinator.coordinate(reports, features.tickers)
            step["inputs"] = {"tickers": features.tickers}
            step["outputs"] = {
                "tilts": decision.tilts,
                "agreement": decision.agreement,
                "dispersion": decision.dispersion,
                "view_confidence": decision.view_confidence,
                "view_dispersion": decision.view_dispersion,
                "market_confidence": decision.market_confidence,
                "portfolio_confidence": decision.portfolio_confidence,
                "conflicts": decision.conflicts,
                "contribution": decision.per_agent_contribution,
            }
            if decision.conflicts:
                step["status"] = "degraded"
                step["notes"] = "; ".join(decision.conflicts)

        with recorder.step("portfolio.optimise") as step:
            optimization, adjusted_expected, bl_result = self.optimise(
                profile, features, snapshot.returns, decision, current_weights
            )
            step["inputs"] = {
                "method": optimization.method,
                "exclusions": sorted(excluded_tickers(profile.exclusions)),
                "previous_weights": current_weights,
                "bl_views": bl_result.n_views,
            }
            step["outputs"] = optimization.model_dump(mode="json")
            step["notes"] = (
                f"Solver status: {optimization.solver_status}; "
                f"Black-Litterman views {bl_result.n_views}, total influence "
                f"{bl_result.total_influence:.4f} (cap {self.settings.bl_max_total_influence:.4f})"
            )

        with recorder.step("suitability.validate") as step:
            optimization, suitability = self.apply_suitability(
                profile,
                features,
                snapshot.returns,
                optimization,
                adjusted_expected,
                current_weights,
            )
            step["inputs"] = {"weights": optimization.weights}
            step["outputs"] = suitability.model_dump(mode="json")
            if not suitability.passed:
                step["status"] = "failed"
                step["notes"] = "Residual suitability breaches require human review."

        with recorder.step("explain.attribute") as step:
            metrics = features.metrics_for(
                optimization.weights, snapshot.returns, self.settings
            )
            contributions = variance_shapley(optimization.weights, features.covariance)
            counterfactual = self._counterfactual(
                profile, features, snapshot.returns, decision, current_weights, optimization
            )
            explanation = Explanation(
                portfolio_variance_contributions=contributions,
                return_contributions=return_contributions(
                    optimization.weights, adjusted_expected
                ),
                agent_contributions=decision.per_agent_contribution,
                binding_constraints=optimization.binding_constraints,
                suitability_rationale=suitability_summary(suitability),
                counterfactual=counterfactual,
            )
            step["inputs"] = {"weights": optimization.weights}
            step["outputs"] = explanation.model_dump(mode="json")

        with recorder.step("recommendation.compose") as step:
            recommendation = self.recommender.build(
                ctx,
                optimization,
                suitability,
                metrics,
                agreement=decision.agreement,
                degraded=degraded_agents or not suitability.passed,
                max_drawdown_estimate=abs(metrics.max_drawdown),
            )
            step["inputs"] = {"agreement": decision.agreement, "suitability_passed": suitability.passed}
            step["outputs"] = recommendation.model_dump(mode="json")

        ml_section = None
        try:
            ml_res = getattr(self.quant, "last_ml_forecast", None)
            if ml_res is not None:
                er = getattr(ml_res, "expected_returns", pd.Series())
                try:
                    fv = {k: float(v) for k, v in er.to_dict().items()} if hasattr(er, "to_dict") else {}
                except Exception:
                    fv = {}
                attr = getattr(ml_res, "attribution", []) or []
                ml_section = MLForecastSection(
                    forecast_timestamp=created_at,
                    data_cutoff=profile.as_of,
                    model_identifiers=list(getattr(ml_res, "model_identifiers", []) or []),
                    model_versions={},
                    model_weights=getattr(ml_res, "weights", {}) or {},
                    validation_metrics=getattr(ml_res, "validation_metrics", {}) or {},
                    forecast_values=fv,
                    forecast_units=str(getattr(ml_res, "forecast_units", "annualized_return")),
                    top_factors=attr,
                    attribution_method="ridge_coefficients" if attr else "none",
                    regime_prediction=getattr(ml_res, "regime", None),
                    regime_confidence=float(getattr(ml_res, "regime_confidence")) if getattr(ml_res, "regime_confidence", None) is not None else None,
                    fallbacks=list(getattr(ml_res, "fallbacks", []) or []),
                    rejected_models=getattr(ml_res, "rejected", []) or [],
                    disagreement=float(getattr(ml_res, "disagreement")) if getattr(ml_res, "disagreement", None) is not None else None,
                    uncertainty=float(getattr(ml_res, "uncertainty")) if getattr(ml_res, "uncertainty", None) is not None else None,
                )
        except Exception:
            ml_section = None

        bl_section = BlackLittermanSection(
            tau=float(self.settings.bl_tau),
            view_shift=float(self.settings.bl_view_shift),
            max_total_influence=float(self.settings.bl_max_total_influence),
            prior_expected_returns={
                str(k): float(v) for k, v in bl_result.prior.to_dict().items()
            },
            posterior_expected_returns={
                str(k): float(v) for k, v in bl_result.posterior.to_dict().items()
            },
            views={str(k): float(v) for k, v in bl_result.views.to_dict().items()},
            view_confidence={
                str(k): float(v) for k, v in bl_result.confidence.to_dict().items()
            },
            view_dispersion={
                str(k): float(v) for k, v in decision.view_dispersion.items()
            },
            view_returns={
                str(k): float(v) for k, v in bl_result.view_returns.to_dict().items()
            },
            omega={str(k): float(v) for k, v in bl_result.omega.to_dict().items()},
            total_influence=float(bl_result.total_influence),
            influence_capped=bool(bl_result.influence_capped),
            n_views=int(bl_result.n_views),
        )

        body = {
            "trace_id": trace_id,
            "created_at": created_at.isoformat(),
            "as_of": profile.as_of.isoformat(),
            "investor_id": profile.investor_id,
            "profile_digest": profile_digest,
            "config_digest": config_digest,
            "steps": [step.model_dump(mode="json") for step in recorder.steps],
            "agents": [report.model_dump(mode="json") for report in reports],
            "optimizer": optimization.model_dump(mode="json"),
            "suitability": suitability.model_dump(mode="json"),
            "explanation": explanation.model_dump(mode="json"),
            "recommendation": recommendation.model_dump(mode="json"),
            "black_litterman": bl_section.model_dump(mode="json"),
        }
        body["risk_free_rate"] = float(rf)
        body["risk_free_source"] = str(
            getattr(self.provider, "risk_free_source", "settings")
        )
        if ml_section is not None:
            body["ml_forecast"] = ml_section.model_dump(mode="json")
        previous_hash, final_hash = self.audit.seal(body)
        return DecisionTrace(**body, previous_hash=previous_hash, hash=final_hash)

    # -- internals --------------------------------------------------------
    def _black_litterman(
        self, features: QuantFeatures, decision: CoordinatorDecision
    ) -> BlackLittermanResult:
        """Blend coordinated agent views with the quantitative prior.

        A single implementation is used by both the advisory path and the
        counterfactual, so the two cannot drift apart.
        """
        return posterior_expected_returns(
            features.expected_returns,
            features.covariance,
            decision.tilts,
            confidence=decision.view_confidence,
            market_score=decision.market_score,
            market_confidence=decision.market_confidence,
            portfolio_score=decision.portfolio_score,
            tau=self.settings.bl_tau,
            view_shift=self.settings.bl_view_shift,
            max_total_influence=self.settings.bl_max_total_influence,
        )

    def optimise(
        self,
        profile: InvestorProfile,
        features: QuantFeatures,
        returns: pd.DataFrame,
        decision: CoordinatorDecision,
        current_weights: Mapping[str, float],
    ) -> tuple[OptimizationResult, pd.Series, BlackLittermanResult]:
        """Optimize using the BL posterior for every risk class.

        The advisory path uses mean-variance for all classes because minimum
        CVaR has no expected-return term and would silently ignore every agent
        view. Risk class still enters through risk aversion, position/sector
        caps and the blocking suitability gate. Minimum-CVaR remains available
        in ``quant.optimize`` as a tail-risk alternative for callers that want
        it.
        """
        policy = policy_for(profile)
        risk_class = profile.suitability_class
        constraints = Constraints(
            max_weight=min(self.settings.max_weight, policy.max_single),
            max_sector_weight=self.settings.max_sector_weight,
            turnover_penalty=self.settings.turnover_penalty,
            risk_aversion={1: 6.0, 2: 4.5, 3: 3.0, 4: 2.0, 5: 1.4}[risk_class],
            diversification=0.6,
        )
        exclude = excluded_tickers(profile.exclusions)

        bl_result = self._black_litterman(features, decision)
        adjusted_expected = bl_result.posterior
        rf = self._risk_free(profile)
        result = mean_variance(
            adjusted_expected,
            features.covariance,
            previous_weights=dict(current_weights),
            constraints=constraints,
            sector_of=sector_of,
            exclude=exclude,
            risk_free=rf,
        )
        result = annotate(result, adjusted_expected, returns, rf)
        return result, adjusted_expected, bl_result

    def apply_suitability(
        self,
        profile: InvestorProfile,
        features: QuantFeatures,
        returns: pd.DataFrame,
        optimization: OptimizationResult,
        adjusted_expected: pd.Series,
        current_weights: Mapping[str, float],
    ) -> tuple[OptimizationResult, "SuitabilityReport"]:
        """Validate and, if necessary, remediate the optimized weights."""
        repaired_weights, suitability = self.suitability.repair(
            optimization.weights,
            profile,
            features,
            returns,
            turnover=optimization.turnover,
        )
        if _weights_differ(repaired_weights, optimization.weights):
            optimization = optimization.model_copy(
                update={
                    "weights": repaired_weights,
                    "expected_volatility": portfolio_volatility(
                        repaired_weights, features.covariance
                    ),
                    "turnover": _turnover(repaired_weights, current_weights),
                    "method": f"{optimization.method}+suitability_repair",
                }
            )
            try:
                rf = self._risk_free(profile)
            except Exception:
                rf = self.settings.risk_free_rate
            optimization = annotate(
                optimization, adjusted_expected, returns, rf
            )
        return optimization, suitability

    def construct(
        self,
        profile: InvestorProfile,
        ctx: AgentContext,
        decision: CoordinatorDecision,
    ) -> tuple[OptimizationResult, "SuitabilityReport", "RiskMetrics"]:
        """Optimize, validate and measure in one call (used by the backtester)."""
        optimization, adjusted, _ = self.optimise(
            profile, ctx.features, ctx.snapshot.returns, decision, ctx.current_weights
        )
        optimization, suitability = self.apply_suitability(
            profile,
            ctx.features,
            ctx.snapshot.returns,
            optimization,
            adjusted,
            ctx.current_weights,
        )
        metrics = ctx.features.metrics_for(
            optimization.weights, ctx.snapshot.returns, self.settings
        )
        return optimization, suitability, metrics

    def _counterfactual(
        self,
        profile: InvestorProfile,
        features: QuantFeatures,
        returns: pd.DataFrame,
        decision: CoordinatorDecision,
        current_weights: Mapping[str, float],
        current: OptimizationResult,
    ) -> dict[str, object]:
        """One-step "what if the investor were more risk tolerant?" comparison."""
        if profile.suitability_class >= 5:
            return {
                "question": "Already at the highest risk class; counterfactual omitted.",
                "alternative_risk_class": profile.suitability_class,
            }
        relaxed = profile.model_copy(
            update={
                "risk_tolerance": min(5, profile.risk_tolerance + 1),
                "loss_tolerance_pct": min(1.0, profile.loss_tolerance_pct * 1.5),
            }
        )
        policy = policy_for(relaxed)
        constraints = Constraints(
            max_weight=min(self.settings.max_weight, policy.max_single),
            max_sector_weight=self.settings.max_sector_weight,
            turnover_penalty=self.settings.turnover_penalty,
            risk_aversion=max(1.0, {1: 6.0, 2: 4.5, 3: 3.0, 4: 2.0, 5: 1.4}[relaxed.suitability_class] - 1.0),
        )
        adjusted = self._black_litterman(features, decision).posterior
        alternative = mean_variance(
            adjusted,
            features.covariance,
            previous_weights=dict(current_weights),
            constraints=constraints,
            sector_of=sector_of,
            exclude=excluded_tickers(profile.exclusions),
            risk_free=self._risk_free(profile),
        )
        current_equity = sum(
            w for t, w in current.weights.items() if asset_class_of(t) == "equity"
        )
        alternative_equity = sum(
            w for t, w in alternative.weights.items() if asset_class_of(t) == "equity"
        )
        return {
            "question": "What changes if the investor accepts one risk class higher?",
            "alternative_risk_class": relaxed.suitability_class,
            "current_equity_weight": round(current_equity, 4),
            "alternative_equity_weight": round(alternative_equity, 4),
            "equity_shift": round(alternative_equity - current_equity, 4),
            "alternative_expected_volatility": round(alternative.expected_volatility, 4),
            "alternative_max_single_weight": round(max(alternative.weights.values()), 4),
        }

    # -- utilities --------------------------------------------------------
    def _risk_free(self, profile: InvestorProfile) -> float:
        """Point-in-time risk-free rate for ``profile.as_of``.

        Real data providers return the latest FRED DGS3MO observation on or
        before the decision date; the synthetic provider returns the configured
        constant. The rate is never taken from a later date.
        """
        try:
            getter = getattr(self.provider, "get_risk_free", None)
            if callable(getter):
                return float(getter(profile.as_of))
        except Exception:
            pass
        return float(self.settings.risk_free_rate)

    def verify_audit_chain(self) -> tuple[bool, str]:
        return self.audit.verify()


def _normalise_holdings(holdings: Mapping[str, float]) -> dict[str, float]:
    clean = {ticker: float(weight) for ticker, weight in holdings.items() if weight > 1e-9}
    total = sum(clean.values())
    if total <= 0:
        return {}
    return {ticker: weight / total for ticker, weight in clean.items()}


def _weights_differ(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    keys = set(left) | set(right)
    return any(abs(left.get(k, 0.0) - right.get(k, 0.0)) > 1e-6 for k in keys)


def _turnover(weights: Mapping[str, float], previous: Mapping[str, float]) -> float:
    keys = set(weights) | set(previous)
    return float(sum(abs(weights.get(k, 0.0) - previous.get(k, 0.0)) for k in keys) / 2.0)


__all__ = ["AdvisoryEngine"]
