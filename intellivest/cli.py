"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intellivest.config import load_settings
from intellivest.contracts import InvestorProfile
from intellivest.orchestration.orchestrator import AdvisoryEngine

EXAMPLE_PROFILE = {
    "investor_id": "DEMO-001",
    "as_of": "2026-06-30",
    "age": 34,
    "horizon_years": 12,
    "risk_tolerance": 4,
    "loss_tolerance_pct": 0.20,
    "goal": "growth",
    "liquidity_need": "low",
    "current_holdings": {"SPX": 0.35, "TECH": 0.30, "GOLD": 0.10, "GOVT": 0.25},
}


def _use_utf8_stdout() -> None:
    """Keep report glyphs (→, ±, ×) printable on Windows code pages."""
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - non-standard streams
        pass


def _print_trace(trace) -> None:
    recommendation = trace.recommendation
    print("=" * 78)
    print(f"INTELLIVEST-MAS — advice for {recommendation.investor_id} as of {recommendation.as_of}")
    print("=" * 78)
    print(f"Trace ID        : {trace.trace_id}")
    print(f"Headline        : {recommendation.headline}")
    print(f"Risk class      : {recommendation.suitability.risk_class}")
    print(f"Expected return : {recommendation.expected_return:.2%}")
    print(f"Volatility      : {recommendation.expected_volatility:.2%}")
    print(f"Sharpe          : {recommendation.expected_sharpe:.2f}")
    print(f"CVaR95 (1-day)  : {recommendation.cvar_95:.2%}")
    print(f"Max drawdown    : {recommendation.max_drawdown_estimate:.2%}")
    print(f"Turnover        : {recommendation.turnover:.1%}")
    print(f"Cost            : {recommendation.estimated_cost_bps:.1f} bps")
    print(f"Agent agreement : {recommendation.agent_agreement:.0%}")
    print(f"Suitability     : {'PASS' if recommendation.suitability.passed else 'FAIL'}"
          f" (repaired={recommendation.suitability.repaired})")
    print("-" * 78)
    print("Allocation")
    for ticker, weight in sorted(recommendation.weights.items(), key=lambda kv: -kv[1]):
        change = recommendation.allocation_changes.get(ticker, 0.0)
        print(f"  {ticker:6s} {weight:6.2%}   change {change:+7.2%}")
    print("-" * 78)
    print("Rationale")
    print(recommendation.rationale)
    print("-" * 78)
    print("Risk contribution (variance shares)")
    for ticker, share in sorted(
        trace.explanation.portfolio_variance_contributions.items(), key=lambda kv: -kv[1]
    )[:6]:
        print(f"  {ticker:6s} {share:6.2%}")
    print("-" * 78)
    print("Suitability checks")
    for line in trace.explanation.suitability_rationale:
        print(f"  {line}")
    if trace.ml_forecast is not None:
        section = trace.ml_forecast
        print("-" * 78)
        print("ML forecast")
        print(f"  models    : {', '.join(section.model_identifiers)}")
        if section.regime_prediction:
            confidence = section.regime_confidence
            suffix = f" ({confidence:.0%} confidence)" if confidence is not None else ""
            print(f"  regime    : {section.regime_prediction}{suffix}")
        for fallback in section.fallbacks:
            print(f"  fallback  : {fallback}")
        for rejected in section.rejected_models:
            print(
                f"  rejected  : {rejected.get('model', '?')} - "
                f"{rejected.get('reason', '')}"
            )
    if trace.explanation.counterfactual:
        print("-" * 78)
        print("Counterfactual")
        for key, value in trace.explanation.counterfactual.items():
            print(f"  {key}: {value}")
    print("-" * 78)
    print(f"Trace hash: {trace.hash}")
    print(f"Prev hash : {trace.previous_hash}")
    print("=" * 78)


def advise_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a suitability-checked recommendation.")
    parser.add_argument("--profile", type=Path, help="Path to an investor-profile JSON file.")
    parser.add_argument("--as-of", type=str, help="Override the as_of date (YYYY-MM-DD).")
    parser.add_argument("--json", action="store_true", help="Emit the full trace as JSON.")
    parser.add_argument(
        "--ml",
        action="store_true",
        help="Enable the validated ML forecast overlay (Ridge/log-HAR with fallback).",
    )
    args = parser.parse_args(argv)
    _use_utf8_stdout()

    payload = (
        json.loads(args.profile.read_text(encoding="utf-8")) if args.profile else dict(EXAMPLE_PROFILE)
    )
    if args.as_of:
        payload["as_of"] = args.as_of
    profile = InvestorProfile(**payload)

    trace = AdvisoryEngine(load_settings(), use_ml=args.ml).advise(profile)
    if args.json:
        print(json.dumps(trace.model_dump(mode="json"), indent=2))
    else:
        _print_trace(trace)
    return 0 if trace.suitability.passed else 2


def evaluate_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Phase 4 multi-profile, multi-regime evaluation."
    )
    parser.add_argument(
        "--rebalance-every",
        type=int,
        default=21,
        help="Trading days between rebalances (default: 21).",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=252,
        help="Estimation window in trading days (default: 252).",
    )
    parser.add_argument("--output", type=Path, default=None, help="Directory for JSON/Markdown results.")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Moving-block bootstrap replicates (default: 1000).",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=21,
        help="Moving-block length in trading days (default: 21).",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip bootstrap intervals and paired resampling for a fast run.",
    )
    args = parser.parse_args(argv)
    _use_utf8_stdout()

    from intellivest.evaluation.phase4 import run_phase4

    settings = load_settings()
    _, markdown = run_phase4(
        settings,
        output=args.output,
        rebalance_every=args.rebalance_every,
        lookback_days=args.lookback_days,
        bootstrap=not args.no_bootstrap,
        n_boot=args.bootstrap_samples,
        block_length=args.block_length,
    )
    print(markdown)
    return 0


DEFAULT_SEEDS: tuple[int, ...] = (20261023, 7, 101, 2718, 31337)

_ROBUSTNESS_KEYS: tuple[str, ...] = (
    "annual_return",
    "annual_volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "cvar_95",
    "calmar",
    "turnover_per_rebalance",
    "annual_cost_bps",
    "suitability_breach_rate",
    "average_equity_weight",
)


def _aggregate(robustness: list[dict[str, object]], configs) -> dict[str, object]:
    """Mean and standard deviation of each metric across seeds."""
    import numpy as np

    if not robustness:
        return {}
    aggregate: dict[str, object] = {}
    for config in configs:
        rows = []
        for entry in robustness:
            for result in entry["results"]:  # type: ignore[index]
                if result["config"] == config:
                    rows.append(result["metrics"])
        aggregate[config] = {
            key: {
                "mean": round(float(np.mean([row[key] for row in rows])), 6),
                "std": round(float(np.std([row[key] for row in rows], ddof=1)), 6)
                if len(rows) > 1
                else 0.0,
            }
            for key in _ROBUSTNESS_KEYS
        }
    return aggregate


def _markdown(results, payload) -> str:
    generated = payload["generated_with"]
    header = (
        "# Walk-forward evaluation\n\n"
        f"Primary seed {generated['seed']}, window {generated['data_start']} → "
        f"{generated['data_end']}, rebalance every {generated['rebalance_every']} trading days, "
        f"{generated['transaction_cost_bps']} bps transaction cost. Market source: "
        f"{generated.get('market_source', 'unknown')}. The macro regime path is fixed across "
        "seeds; the seed controls only cross-sectional asset draws.\n\n"
        "**Configurations.** `quant_only` = constrained mean-variance optimizer with a neutral "
        "view set and no suitability gate. `agents_only` = direct agent-consensus allocation, "
        "bypassing every quantitative guardrail. `hybrid` = agent views enter the Black-Litterman "
        "posterior, the optimizer chooses weights, and the blocking suitability gate remediates "
        "the result. `hybrid_shuffled` = the same pipeline with the asset-to-tilt mapping permuted "
        "by a seeded RNG (placebo).\n\n"
    )
    columns = [
        ("Annual return", "annual_return", "{:.2%}"),
        ("Volatility", "annual_volatility", "{:.2%}"),
        ("Sharpe", "sharpe", "{:.2f}"),
        ("Sortino", "sortino", "{:.2f}"),
        ("Max drawdown", "max_drawdown", "{:.2%}"),
        ("CVaR95 (1-day)", "cvar_95", "{:.2%}"),
        ("Calmar", "calmar", "{:.2f}"),
        ("Turnover / rebalance", "turnover_per_rebalance", "{:.1%}"),
        ("Annual cost", "annual_cost_bps", "{:.1f} bps"),
        ("Suitability breach rate", "suitability_breach_rate", "{:.0%}"),
        ("Average equity weight", "average_equity_weight", "{:.0%}"),
    ]
    lines = [header, "## Primary seed\n", "| Metric | " + " | ".join(r.config for r in results) + " |"]
    lines.append("|---|" + "---|" * len(results))
    for label, key, fmt in columns:
        row = " | ".join(fmt.format(r.summary.to_dict()[key]) for r in results)
        lines.append(f"| {label} | {row} |")

    aggregate = payload.get("robustness", {}).get("aggregate") or {}
    robustness = payload.get("robustness", {})
    if aggregate:
        seeds = robustness.get("seeds", [])
        lines.append("")
        lines.append(f"## Robustness across {len(seeds)} seeds\n")
        lines.append(
            "Mean ± standard deviation across seeds " + ", ".join(str(s) for s in seeds) + ". "
            "Every seed shares the same macro regime path and differs only in cross-sectional draws."
        )
        lines.append("")
        lines.append("| Metric | " + " | ".join(aggregate) + " |")
        lines.append("|---|" + "---|" * len(aggregate))
        for label, key, fmt in columns:
            cells = []
            for config in aggregate:
                entry = aggregate[config][key]
                cells.append(f"{fmt.format(entry['mean'])} ± {fmt.format(entry['std'])}")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append(_reading(payload))
    return "\n".join(lines) + "\n"


def _reading(payload) -> str:
    """A short, automatically derived interpretation of the tables."""
    aggregate = payload.get("robustness", {}).get("aggregate") or {}
    if not aggregate:
        return "## Reading the results\n\nRobustness panel skipped.\n"
    lines = ["## Reading the results\n"]
    if "hybrid" in aggregate and "agents_only" in aggregate:
        h = aggregate["hybrid"]
        a = aggregate["agents_only"]
        lines.append(
            f"* **Suitability is the decisive axis.** The hybrid gate ends at "
            f"{h['suitability_breach_rate']['mean']:.1%} breaches versus "
            f"{a['suitability_breach_rate']['mean']:.1%} for unconstrained agent consensus."
        )
        if a["turnover_per_rebalance"]["mean"] > 0:
            ratio = a["turnover_per_rebalance"]["mean"] / max(
                h["turnover_per_rebalance"]["mean"], 1e-9
            )
            lines.append(
                f"* **Turnover.** Unconstrained agents trade {ratio:.1f}× more per rebalance, "
                f"which is why their cost drag is "
                f"{a['annual_cost_bps']['mean']:.1f} bps/yr versus "
                f"{h['annual_cost_bps']['mean']:.1f} bps/yr."
            )
        lines.append(
            f"* **Risk-adjusted outcome.** Hybrid Sharpe "
            f"{h['sharpe']['mean']:.2f} ± {h['sharpe']['std']:.2f} versus "
            f"{a['sharpe']['mean']:.2f} ± {a['sharpe']['std']:.2f} for agents only; "
            f"max drawdown {h['max_drawdown']['mean']:.1%} versus "
            f"{a['max_drawdown']['mean']:.1%}."
        )
        lines.append(
            "* **Return is not the objective function.** "
            f"Hybrid annual return is {h['annual_return']['mean']:.2%} versus "
            f"{a['annual_return']['mean']:.2%} for unconstrained agents; "
            f"volatility is {h['annual_volatility']['mean']:.2%} versus "
            f"{a['annual_volatility']['mean']:.2%}, and CVaR95 is "
            f"{h['cvar_95']['mean']:.2%} versus {a['cvar_95']['mean']:.2%}. "
            "The design optimises for suitability and risk control, not alpha."
        )
    if "hybrid" in aggregate and "quant_only" in aggregate:
        h = aggregate["hybrid"]
        q = aggregate["quant_only"]
        lines.append(
            f"* **Versus the numerical engine.** Hybrid annual return is "
            f"{h['annual_return']['mean']:.2%} versus {q['annual_return']['mean']:.2%} "
            f"for quant_only, with volatility {h['annual_volatility']['mean']:.2%} "
            f"versus {q['annual_volatility']['mean']:.2%}, CVaR95 "
            f"{h['cvar_95']['mean']:.2%} versus {q['cvar_95']['mean']:.2%}, and "
            f"suitability breaches {h['suitability_breach_rate']['mean']:.1%} versus "
            f"{q['suitability_breach_rate']['mean']:.1%}. The agent layer is a "
            "risk-and-suitability layer here, not a return-enhancement layer."
        )
    if "hybrid" in aggregate and "hybrid_shuffled" in aggregate:
        h = aggregate["hybrid"]
        s = aggregate["hybrid_shuffled"]
        gap_sharpe = h["sharpe"]["mean"] - s["sharpe"]["mean"]
        gap_return = h["annual_return"]["mean"] - s["annual_return"]["mean"]
        if abs(gap_sharpe) < 0.05 and abs(gap_return) < 0.005:
            conclusion = (
                "The gap is negligible in this run, so the placebo does not "
                "support a claim that the asset-level mapping adds value."
            )
        elif gap_sharpe > 0 and gap_return > 0:
            conclusion = (
                "The true mapping is ahead on both metrics, which is consistent "
                "with asset-level agent information adding value."
            )
        else:
            conclusion = (
                "The placebo is not worse, so this run does not support a claim "
                "that the asset-level mapping adds value."
            )
        lines.append(
            f"* **Placebo check.** Permuting which asset receives each agent tilt leaves "
            f"Sharpe at {s['sharpe']['mean']:.2f} ± {s['sharpe']['std']:.2f} versus "
            f"{h['sharpe']['mean']:.2f} ± {h['sharpe']['std']:.2f} for the true mapping, "
            f"and annual return at {s['annual_return']['mean']:.2%} versus "
            f"{h['annual_return']['mean']:.2%}. {conclusion} With "
            f"{len(payload.get('robustness', {}).get('seeds', []))} seeds this is "
            "descriptive, not a formal significance test."
        )
    return "\n".join(lines) + "\n"



if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(advise_main())
