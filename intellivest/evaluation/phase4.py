"""Phase 4 evaluation runner and report.

The runner is deliberately explicit about the protocol:

* five investor profiles spanning risk classes 1-5;
* seven strategies, including a suitability-gated quantitative baseline that is
  identical to the hybrid except for the absence of agent views;
* real cached ETF prices where available, with a one-year estimation window so
  the sample includes both the 2020 crash and the 2022 rate shock;
* transaction costs and turnover;
* moving-block bootstrap intervals and paired hybrid-vs-baseline tests;
* profile-alignment diagnostics;
* no seed-robustness claim when the market path does not vary with the seed.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd

from intellivest.config import Settings
from intellivest.data.market import MarketDataProvider, build_provider
from intellivest.evaluation.backtest import (
    BASELINE_CONFIGS,
    CONFIGS,
    REGIME_WINDOWS,
    BacktestResult,
    Backtester,
    regime_statistics,
)
from intellivest.evaluation.bootstrap import paired_bootstrap_test
from intellivest.evaluation.profiles import phase4_profiles
from intellivest.evaluation.splits import DEFAULT_SPLIT
from intellivest.quant import risk

PHASE4_LOOKBACK_DAYS = 252
PHASE4_REBALANCE_EVERY = 21
PHASE4_BOOTSTRAP_SAMPLES = 1000
PHASE4_BLOCK_LENGTH = 21

_AGGREGATE_METRICS: tuple[str, ...] = (
    "annual_return",
    "annual_volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "cvar_95",
    "turnover_per_rebalance",
    "annual_cost_bps",
    "suitability_breach_rate",
    "average_equity_weight",
)
_ALIGNMENT_METRICS: tuple[str, ...] = (
    "realised_volatility",
    "volatility_ratio",
    "realised_drawdown",
    "drawdown_ratio",
    "recommendation_stability",
    "inter_agent_agreement",
)


def run_phase4(
    settings: Settings,
    *,
    provider: MarketDataProvider | None = None,
    output: Path | None = None,
    rebalance_every: int = PHASE4_REBALANCE_EVERY,
    lookback_days: int = PHASE4_LOOKBACK_DAYS,
    bootstrap: bool = True,
    n_boot: int = PHASE4_BOOTSTRAP_SAMPLES,
    block_length: int = PHASE4_BLOCK_LENGTH,
) -> tuple[dict[str, Any], str]:
    """Run the panel, write JSON/Markdown, and return the payload and report."""
    provider = provider or build_provider(settings)
    backtester = Backtester(
        settings,
        provider=provider,
        rebalance_every=rebalance_every,
        lookback_days=lookback_days,
    )
    as_of = _last_available_date(settings)
    profiles = phase4_profiles(as_of)

    profile_payloads: list[dict[str, Any]] = []
    curves: dict[str, pd.Series] = {}
    effective_boot = n_boot if bootstrap else 0
    for profile in profiles:
        results = backtester.run(
            profile,
            CONFIGS,
            bootstrap=bootstrap,
            n_boot=effective_boot,
            block_length=block_length,
        )
        by_config = {result.config: result for result in results}
        for result in results:
            curves[f"{profile.investor_id}:{result.config}"] = result.equity_curve
        paired = _paired_tests(by_config, settings, block_length, effective_boot)
        regimes = {
            key: {
                config: regime_statistics(
                    by_config[config].daily_returns,
                    start=pd.Timestamp(window["start"]).date(),
                    end=pd.Timestamp(window["end"]).date(),
                )
                for config in CONFIGS
            }
            for key, window in REGIME_WINDOWS.items()
        }
        profile_payloads.append(
            {
                "profile": profile.model_dump(mode="json"),
                "results": [
                    {
                        **result.to_dict(),
                        "splits": _split_statistics(
                            result.daily_returns,
                            result.risk_free,
                            settings.trading_days,
                        ),
                    }
                    for result in results
                ],
                "paired_tests": paired,
                "regimes": regimes,
            }
        )

    aggregate = _aggregate(profile_payloads)
    split_aggregate = _split_aggregate(profile_payloads)
    regime_aggregate = _regime_aggregate(profile_payloads)
    payload: dict[str, Any] = {
        "generated_with": {
            "data_start": settings.data_start,
            "data_end": settings.data_end,
            "lookback_days": lookback_days,
            "rebalance_every": rebalance_every,
            "transaction_cost_bps": settings.transaction_cost_bps,
            "market_source": type(provider).__name__,
            "configs": list(CONFIGS),
            "profiles": [profile.investor_id for profile in profiles],
            "data_split": {
                "dev_end": DEFAULT_SPLIT.dev_end.isoformat(),
                "validation_start": DEFAULT_SPLIT.val_start.isoformat(),
                "validation_end": DEFAULT_SPLIT.val_end.isoformat(),
                "test_start": DEFAULT_SPLIT.test_start.isoformat(),
                "note": (
                    "The advisory strategies are not trained; these same fixed "
                    "boundaries are applied descriptively to their walk-forward "
                    "returns, and no advisory parameter is estimated from the test "
                    "window."
                ),
            },
            "bootstrap": {
                "method": "moving_block",
                "block_length": block_length,
                "n_boot": effective_boot,
                "confidence": 0.95,
                "paired": bootstrap,
            },
            "seed_robustness_claim": False,
            "note": (
                "The real-data market path does not vary with the random seed, so "
                "repeated seeds are not independent trials and are not reported as "
                "a robustness panel."
            ),
        },
        "profiles": profile_payloads,
        "aggregate": aggregate,
        "split_aggregate": split_aggregate,
        "regime_windows": {
            key: {
                "label": window["label"],
                "start": window["start"].isoformat(),
                "end": window["end"].isoformat(),
            }
            for key, window in REGIME_WINDOWS.items()
        },
        "regime_aggregate": regime_aggregate,
    }

    markdown = render_markdown(payload)
    destination = output or settings.results_dir
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "phase4_evaluation.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (destination / "phase4_evaluation.md").write_text(markdown, encoding="utf-8")
    # Keep the historical filenames populated for compatibility.
    (destination / "evaluation.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (destination / "evaluation.md").write_text(markdown, encoding="utf-8")
    if curves:
        frame = pd.concat(curves, axis=1)
        frame.index.name = "date"
        frame.to_csv(destination / "equity_curves.csv", float_format="%.6f")
    return payload, markdown


def _last_available_date(settings: Settings) -> Any:
    from datetime import date

    return date.fromisoformat(settings.data_end)


def _paired_tests(
    results: dict[str, BacktestResult],
    settings: Settings,
    block_length: int,
    n_boot: int,
) -> dict[str, Any]:
    hybrid = results["hybrid"]
    output: dict[str, Any] = {}
    for index, baseline in enumerate(BASELINE_CONFIGS):
        left = results[baseline]
        joined = pd.concat(
            [hybrid.daily_returns, left.daily_returns], axis=1, join="inner"
        ).dropna()
        if joined.empty:
            continue
        a = joined.iloc[:, 0].to_numpy(dtype=float)
        b = joined.iloc[:, 1].to_numpy(dtype=float)
        rf = hybrid.risk_free
        sharpe = paired_bootstrap_test(
            a,
            b,
            lambda values, rf=rf: risk.sharpe_ratio(
                pd.Series(values), rf, settings.trading_days
            ),
            metric_name="sharpe",
            block_length=block_length,
            n_boot=n_boot,
            seed=settings.seed + 1000 * index + 17,
        )
        drawdown = paired_bootstrap_test(
            a,
            b,
            lambda values: risk.max_drawdown(pd.Series(values)),
            metric_name="max_drawdown",
            block_length=block_length,
            n_boot=n_boot,
            seed=settings.seed + 1000 * index + 29,
        )
        output[baseline] = {
            "sharpe": sharpe.to_dict(),
            "max_drawdown": drawdown.to_dict(),
        }
    return output


def _aggregate(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for config in CONFIGS:
        rows = [
            next(result for result in entry["results"] if result["config"] == config)
            for entry in profiles
        ]
        metrics = {}
        for key in _AGGREGATE_METRICS:
            values = [float(row["metrics"][key]) for row in rows]
            metrics[key] = {
                "mean": round(float(mean(values)), 6),
                "min": round(float(min(values)), 6),
                "max": round(float(max(values)), 6),
            }
        alignment = {}
        for key in _ALIGNMENT_METRICS:
            values = [
                float(row["profile_alignment"][key])
                for row in rows
                if row.get("profile_alignment")
                and row["profile_alignment"].get(key) is not None
            ]
            alignment[key] = (
                {
                    "mean": round(float(mean(values)), 6),
                    "min": round(float(min(values)), 6),
                    "max": round(float(max(values)), 6),
                }
                if values
                else {"mean": None, "min": None, "max": None}
            )
        output[config] = {"metrics": metrics, "alignment": alignment}
    return output


def _split_statistics(
    daily_returns: pd.Series,
    risk_free: float,
    periods: int,
) -> dict[str, Any]:
    """Descriptive performance for the fixed dev/validation/test boundaries."""
    output: dict[str, Any] = {}
    boundaries = {
        "dev": (None, DEFAULT_SPLIT.dev_end),
        "validation": (DEFAULT_SPLIT.val_start, DEFAULT_SPLIT.val_end),
        "test": (DEFAULT_SPLIT.test_start, None),
    }
    for name, (start, end) in boundaries.items():
        subset = daily_returns.dropna()
        if start is not None:
            subset = subset[subset.index >= pd.Timestamp(start)]
        if end is not None:
            subset = subset[subset.index <= pd.Timestamp(end)]
        if subset.empty:
            output[name] = {
                "observations": 0,
                "annual_return": 0.0,
                "annual_volatility": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
            }
            continue
        output[name] = {
            "observations": int(subset.size),
            "annual_return": round(risk.annualized_return(subset, periods), 6),
            "annual_volatility": round(risk.annualized_volatility(subset, periods), 6),
            "sharpe": round(risk.sharpe_ratio(subset, risk_free, periods), 4),
            "max_drawdown": round(risk.max_drawdown(subset), 6),
        }
    return output


def _split_aggregate(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in ("dev", "validation", "test"):
        output[split] = {}
        for config in CONFIGS:
            rows = [
                next(result for result in entry["results"] if result["config"] == config)[
                    "splits"
                ][split]
                for entry in profiles
            ]
            metrics: dict[str, Any] = {}
            for key in ("annual_return", "annual_volatility", "sharpe", "max_drawdown"):
                values = [float(row[key]) for row in rows]
                metrics[key] = {
                    "mean": round(float(mean(values)), 6),
                    "min": round(float(min(values)), 6),
                    "max": round(float(max(values)), 6),
                }
            output[split][config] = {
                "metrics": metrics,
                "observations": rows[0]["observations"] if rows else 0,
            }
    return output


def _regime_aggregate(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for window in REGIME_WINDOWS:
        output[window] = {}
        for config in CONFIGS:
            rows = [
                entry["regimes"][window][config]
                for entry in profiles
                if entry["regimes"].get(window)
            ]
            output[window][config] = {
                "mean_cumulative_return": round(
                    float(mean(row["cumulative_return"] for row in rows)), 6
                )
                if rows
                else 0.0,
                "mean_max_drawdown": round(
                    float(mean(row["max_drawdown"] for row in rows)), 6
                )
                if rows
                else 0.0,
                "profiles": len(rows),
            }
    return output


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def render_markdown(payload: dict[str, Any]) -> str:
    generated = payload["generated_with"]
    bootstrap = generated["bootstrap"]
    if bootstrap["n_boot"] > 0:
        bootstrap_note = (
            "Bootstrap intervals use a moving-block bootstrap with "
            f"{bootstrap['block_length']}-day blocks and "
            f"{bootstrap['n_boot']} replicates. Paired tests resample the same "
            "block indices for hybrid and each baseline, preserving the shared "
            "trading calendar. The reported p-value is a two-sided percentile "
            "bootstrap p-value, not an exact finite-sample test; no "
            "multiple-testing correction is applied."
        )
    else:
        bootstrap_note = (
            "Bootstrap intervals and paired resampling were disabled for this "
            "run; the point estimates and regime tables are still reported."
        )
    lines: list[str] = [
        "# Phase 4 evaluation: a claim-supporting comparison",
        "",
        f"Window {generated['data_start']} to {generated['data_end']}; "
        f"{generated['lookback_days']}-day estimation window; rebalance every "
        f"{generated['rebalance_every']} trading days; "
        f"{generated['transaction_cost_bps']:.1f} bps transaction cost. "
        f"Market source: **{generated['market_source']}**.",
        "",
        "The panel contains five investor profiles spanning binding suitability "
        "classes 1-5. Every strategy starts from cash and pays the same initial "
        "build cost. The fair baseline is `quant_only_gated`: it uses the same "
        "optimizer and the same blocking suitability gate as `hybrid`, but no "
        "agent views. `quant_only_no_gate` isolates the value of the gate.",
        "",
        bootstrap_note,
        "",
        "No seed-robustness claim is made. " + generated["note"],
        "",
        "## Investor panel",
        "",
        "| Profile | Risk class | Horizon | Loss tolerance | Volatility limit |",
        "|---|---:|---:|---:|---:|",
    ]
    for entry in payload["profiles"]:
        profile = entry["profile"]
        alignment = entry["results"][0]["profile_alignment"] if entry["results"] else None
        vol_limit = alignment["volatility_limit"] if alignment else None
        lines.append(
            f"| {profile['investor_id']} | {profile['risk_tolerance']} | "
            f"{profile['horizon_years']:.0f} yrs | "
            f"{profile['loss_tolerance_pct']:.0%} | "
            f"{_fmt_pct(vol_limit)} |"
        )

    lines.extend(["", "## Aggregate results across the five profiles", ""])
    lines.append(
        "| Config | Return | Vol | Sharpe | MaxDD | Turnover | Cost | Breach | "
        "Vol/limit | DD/limit | Stability | Agreement |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for config in CONFIGS:
        aggregate = payload["aggregate"].get(config, {})
        metrics = aggregate.get("metrics", {})
        alignment = aggregate.get("alignment", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{config}`",
                    _fmt_pct(_mean(metrics, "annual_return")),
                    _fmt_pct(_mean(metrics, "annual_volatility")),
                    _fmt_num(_mean(metrics, "sharpe")),
                    _fmt_pct(_mean(metrics, "max_drawdown")),
                    _fmt_pct(_mean(metrics, "turnover_per_rebalance")),
                    _fmt_num(_mean(metrics, "annual_cost_bps"), 1) + " bps",
                    _fmt_pct(_mean(metrics, "suitability_breach_rate"), 0),
                    _fmt_num(_mean(alignment, "volatility_ratio"), 2) + "x",
                    _fmt_num(_mean(alignment, "drawdown_ratio"), 2) + "x",
                    _fmt_num(_mean(alignment, "recommendation_stability"), 2),
                    _fmt_num(_mean(alignment, "inter_agent_agreement"), 2),
                ]
            )
            + " |"
        )

    split = generated.get("data_split")
    if split:
        lines.extend(["", "## Data split and test-window readout", ""])
        lines.append(
            f"The forecast models use dev through **{split['dev_end']}**, "
            f"validation **{split['validation_start']} to {split['validation_end']}**, "
            f"and test from **{split['test_start']}**. {split['note']}"
        )
        lines.append("")
        lines.append(
            "| Config | Dev return | Validation return | Test return | Test Sharpe | Test MaxDD |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for config in CONFIGS:
            values = [f"`{config}`"]
            for period in ("dev", "validation"):
                entry = payload.get("split_aggregate", {}).get(period, {}).get(config, {})
                values.append(_fmt_pct(_mean(entry.get("metrics", {}), "annual_return")))
            test = payload.get("split_aggregate", {}).get("test", {}).get(config, {})
            test_metrics = test.get("metrics", {})
            values.extend(
                [
                    _fmt_pct(_mean(test_metrics, "annual_return")),
                    _fmt_num(_mean(test_metrics, "sharpe")),
                    _fmt_pct(_mean(test_metrics, "max_drawdown")),
                ]
            )
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
        lines.append(
            "The advisory panel is a walk-forward analysis, not a model-selection "
            "experiment. These split labels describe fixed calendar boundaries and "
            "are reported for transparency; they do not imply that a strategy was "
            "trained on dev or selected on validation."
        )

    lines.extend(["", "## Per-profile results", ""])
    lines.append(
        "| Profile | Class | Config | Return | Vol | Sharpe | MaxDD | Turnover | "
        "Cost | Breach | Vol/limit | DD/limit | Stability | Agreement |"
    )
    lines.append(
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for entry in payload["profiles"]:
        profile = entry["profile"]
        for result in entry["results"]:
            metrics = result["metrics"]
            alignment = result["profile_alignment"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        profile["investor_id"],
                        str(profile["risk_tolerance"]),
                        f"`{result['config']}`",
                        _fmt_pct(metrics["annual_return"]),
                        _fmt_pct(metrics["annual_volatility"]),
                        _fmt_num(metrics["sharpe"]),
                        _fmt_pct(metrics["max_drawdown"]),
                        _fmt_pct(metrics["turnover_per_rebalance"]),
                        _fmt_num(metrics["annual_cost_bps"], 1) + " bps",
                        _fmt_pct(metrics["suitability_breach_rate"], 0),
                        _fmt_num(alignment["volatility_ratio"], 2) + "x",
                        _fmt_num(alignment["drawdown_ratio"], 2) + "x",
                        _fmt_num(alignment["recommendation_stability"], 2),
                        _fmt_num(alignment["inter_agent_agreement"], 2),
                    ]
                )
                + " |"
            )

    if payload["generated_with"]["bootstrap"]["n_boot"] > 0:
        lines.extend(["", "## Bootstrap intervals", ""])
        lines.append("| Profile | Config | Sharpe [95% CI] | MaxDD [95% CI] |")
        lines.append("|---|---|---|---|")
        for entry in payload["profiles"]:
            profile_id = entry["profile"]["investor_id"]
            for result in entry["results"]:
                interval = result.get("bootstrap", {})
                if not interval:
                    continue
                lines.append(
                    f"| {profile_id} | `{result['config']}` | "
                    f"{_fmt_ci(interval.get('sharpe'), percent=False)} | "
                    f"{_fmt_ci(interval.get('max_drawdown'), percent=True)} |"
                )

    lines.extend(["", "## Paired tests: hybrid vs each baseline", ""])
    lines.append(
        "Delta is hybrid minus baseline. For Sharpe, positive is better. For max "
        "drawdown, positive means the hybrid drawdown is less negative (better)."
    )
    lines.append("")
    lines.append("| Profile | Baseline | Delta Sharpe [95% CI] | p | Delta MaxDD [95% CI] | p |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for entry in payload["profiles"]:
        profile_id = entry["profile"]["investor_id"]
        for baseline in BASELINE_CONFIGS:
            test = entry["paired_tests"].get(baseline)
            if not test:
                continue
            sharpe = test["sharpe"]
            drawdown = test["max_drawdown"]
            lines.append(
                f"| {profile_id} | `{baseline}` | "
                f"{_fmt_delta(sharpe, percent=False)} | "
                f"{sharpe['p_value']:.3f} | "
                f"{_fmt_delta(drawdown, percent=True)} | "
                f"{drawdown['p_value']:.3f} |"
            )

    lines.extend(["", "## Regime windows", ""])
    lines.append(
        "Mean cumulative return and mean max drawdown across the five profiles. "
        "The windows are fixed calendar periods, not selected from the results."
    )
    lines.append("")
    lines.append(
        "| Config | 2020 crash return | 2020 crash MaxDD | 2022 rates return | 2022 rates MaxDD |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for config in CONFIGS:
        values = []
        for window in ("2020_crash", "2022_rates"):
            row = payload["regime_aggregate"][window][config]
            values.extend(
                [
                    _fmt_pct(row["mean_cumulative_return"]),
                    _fmt_pct(row["mean_max_drawdown"]),
                ]
            )
        lines.append(f"| `{config}` | " + " | ".join(values) + " |")

    lines.extend(["", _reading(payload)])
    return "\n".join(lines) + "\n"


def _reading(payload: dict[str, Any]) -> str:
    lines = ["## Reading the results", ""]
    aggregate = payload["aggregate"]
    hybrid = aggregate["hybrid"]
    gated = aggregate["quant_only_gated"]
    ungated = aggregate["quant_only_no_gate"]
    baseline = aggregate["sixty_forty"]

    lines.append(
        "* **Gate effect.** Hybrid breach rate is "
        f"{_fmt_pct(_mean(hybrid['metrics'], 'suitability_breach_rate'), 0)} versus "
        f"{_fmt_pct(_mean(ungated['metrics'], 'suitability_breach_rate'), 0)} for "
        "quant-only without the gate. The gate is doing real work; its value is "
        "not evidence of alpha."
    )
    lines.append(
        "* **Fair baseline.** Against `quant_only_gated`, hybrid annual return is "
        f"{_fmt_pct(_mean(hybrid['metrics'], 'annual_return'))} versus "
        f"{_fmt_pct(_mean(gated['metrics'], 'annual_return'))}; volatility is "
        f"{_fmt_pct(_mean(hybrid['metrics'], 'annual_volatility'))} versus "
        f"{_fmt_pct(_mean(gated['metrics'], 'annual_volatility'))}; Sharpe is "
        f"{_fmt_num(_mean(hybrid['metrics'], 'sharpe'))} versus "
        f"{_fmt_num(_mean(gated['metrics'], 'sharpe'))}. This is the comparison "
        "that matters when asking whether agent views add value."
    )
    lines.append(
        "* **Strategic benchmark.** Hybrid annual return is "
        f"{_fmt_pct(_mean(hybrid['metrics'], 'annual_return'))} versus "
        f"{_fmt_pct(_mean(baseline['metrics'], 'annual_return'))} for 60/40, with "
        f"max drawdown {_fmt_pct(_mean(hybrid['metrics'], 'max_drawdown'))} versus "
        f"{_fmt_pct(_mean(baseline['metrics'], 'max_drawdown'))}. Beating 60/40 is "
        "not the same as beating the gated quantitative baseline."
    )
    agents = aggregate["agents_only"]
    lines.append(
        "* **Agents without guardrails.** `agents_only` is the failure mode the "
        f"guardrails contain: {_fmt_pct(_mean(agents['metrics'], 'annual_return'))} "
        "return, "
        f"{_fmt_pct(_mean(agents['metrics'], 'turnover_per_rebalance'))} turnover "
        "per rebalance, "
        f"{_fmt_num(_mean(agents['metrics'], 'annual_cost_bps'), 1)} bps annual "
        "cost, and a "
        f"{_fmt_pct(_mean(agents['metrics'], 'suitability_breach_rate'), 0)} "
        "breach rate. Holdings-aware portfolio views make it churn."
    )

    significant_sharpe = 0
    significant_drawdown = 0
    total_tests = 0
    for entry in payload["profiles"]:
        for baseline_name in BASELINE_CONFIGS:
            test = entry["paired_tests"].get(baseline_name)
            if not test:
                continue
            total_tests += 1
            significant_sharpe += int(test["sharpe"]["p_value"] < 0.05)
            significant_drawdown += int(test["max_drawdown"]["p_value"] < 0.05)
    lines.append(
        "* **Paired bootstrap.** At the 5% level, "
        f"{significant_sharpe} of {total_tests} hybrid-vs-baseline Sharpe "
        f"differences and {significant_drawdown} of {total_tests} max-drawdown "
        "differences have a percentile p-value below 0.05. No multiple-testing "
        "correction is applied, so this is descriptive rather than a family-wise "
        "significance claim."
    )

    crash = payload["regime_aggregate"]["2020_crash"]
    rates = payload["regime_aggregate"]["2022_rates"]
    lines.append(
        "* **Regimes.** In the 2020 crash window hybrid returned "
        f"{_fmt_pct(crash['hybrid']['mean_cumulative_return'])} versus "
        f"{_fmt_pct(crash['sixty_forty']['mean_cumulative_return'])} for 60/40; "
        "in the 2022 rate shock hybrid returned "
        f"{_fmt_pct(rates['hybrid']['mean_cumulative_return'])} versus "
        f"{_fmt_pct(rates['sixty_forty']['mean_cumulative_return'])} for 60/40."
    )
    split_aggregate = payload.get("split_aggregate", {})
    if split_aggregate:
        validation = split_aggregate["validation"]["hybrid"]["metrics"]
        test = split_aggregate["test"]["hybrid"]["metrics"]
        validation_return = _mean(validation, "annual_return")
        test_return = _mean(test, "annual_return")
        validation_sharpe = _mean(validation, "sharpe")
        test_sharpe = _mean(test, "sharpe")
        direction = "lower" if test_return < validation_return else "higher"
        lines.append(
            "* **Test-window readout.** Hybrid annual return is "
            f"{_fmt_pct(test_return)} in the test window versus "
            f"{_fmt_pct(validation_return)} in validation ({direction}), with "
            f"Sharpe {_fmt_num(test_sharpe)} versus {_fmt_num(validation_sharpe)} "
            "and test max drawdown "
            f"{_fmt_pct(_mean(test, 'max_drawdown'))}. This is a descriptive "
            "out-of-sample readout: the advisory weights are not fitted on the "
            "dev or validation labels, and the test window is not used to choose "
            "a strategy. Differences therefore reflect the macro path and "
            "sampling variation, not a tuned improvement."
        )
    lines.append(
        "* **What is not claimed.** The real-data path is identical across seeds, "
        "so the old five-seed standard deviations are not an independent-sample "
        "robustness result. This report replaces that claim with the five-profile "
        "panel and moving-block intervals."
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------


def _fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}%}"


def _fmt_num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _fmt_ci(interval: dict[str, Any] | None, *, percent: bool) -> str:
    if not interval:
        return "n/a"
    formatter = _fmt_pct if percent else lambda value: _fmt_num(value)
    return (
        f"{formatter(interval['point'])} "
        f"[{formatter(interval['low'])}, {formatter(interval['high'])}]"
    )


def _fmt_delta(interval: dict[str, Any], *, percent: bool) -> str:
    formatter = _fmt_pct if percent else lambda value: _fmt_num(value)
    return (
        f"{formatter(interval['difference'])} "
        f"[{formatter(interval['low'])}, {formatter(interval['high'])}]"
    )


def _mean(metrics: dict[str, Any], key: str) -> Any:
    entry = metrics.get(key)
    return entry.get("mean") if isinstance(entry, dict) else None


__all__ = [
    "PHASE4_BLOCK_LENGTH",
    "PHASE4_BOOTSTRAP_SAMPLES",
    "PHASE4_LOOKBACK_DAYS",
    "PHASE4_REBALANCE_EVERY",
    "render_markdown",
    "run_phase4",
]
