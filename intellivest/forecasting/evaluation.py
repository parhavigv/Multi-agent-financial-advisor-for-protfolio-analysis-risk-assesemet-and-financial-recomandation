"""End-to-end forecast evaluation with an honest time-based protocol.

Protocol
--------
* Features are causal and lagged one day (see :mod:`intellivest.forecasting.data`).
* Out-of-fold predictions come from a purged, embargoed expanding walk-forward
  (see :mod:`intellivest.forecasting.cv`).
* Model **selection** uses the validation window (2022-2023) only. The test
  window (2024 onward) is evaluated exactly once, by the final report run, and
  never feeds back into selection or hyper-parameters.
* A shuffled-label control must produce a rank IC indistinguishable from zero;
  if it does not, the evaluation pipeline itself is suspect.
* Train-vs-validation gaps are flagged with an explicit rule so an overfit
  candidate cannot slip through on a single lucky validation window.

Split membership is assigned by the feature date, but the last ``horizon``
feature dates before each boundary are excluded from the *scored* set: their
target window reaches into the next split. This keeps validation metrics from
ever consuming a test-period return, in development and final runs alike.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from intellivest.evaluation.splits import TimeSplit

from . import metrics as M
from .base import PanelData
from .cv import out_of_fold_predictions
from .data import build_panel
from .regime import HMMRegime
from .return_models import (
    BASELINES,
    CANDIDATES,
    ElasticNetForecast,
    LightGBMForecast,
    MomentumForecast,
    OneOverNForecast,
    RidgeForecast,
    ShrunkMeanForecast,
)
from .vol_models import EWMAVolatility, GARCH11Volatility, HARVolatility

SPLITS = ("dev", "validation", "test")
VOL_BASELINE = "ewma"

# Gap rule: a candidate is flagged if its validation rank IC is more than 0.10
# below its in-sample rank IC, or if validation error is more than 50% above
# the in-sample error. For volatility models the QLIKE analogue is used.
IC_GAP_LIMIT = 0.10
ERROR_RATIO_LIMIT = 1.50
QLIKE_GAP_LIMIT = 0.30
# A candidate whose out-of-fold rank IC moves this much between the dev and
# validation windows is not stable enough to promote on two years of validation.
DEV_VAL_SWING_LIMIT = 0.15

# Minimum validation advantage before a candidate replaces the baseline.
# QLIKE can be negative, so the volatility margin is absolute, not relative.
MIN_IC_EDGE = 0.01
MIN_QLIKE_MARGIN = 0.01


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------


@dataclass
class ModelReport:
    name: str
    role: str  # "baseline", "candidate", "control"
    kind: str  # "return" or "volatility"
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    train_metrics: dict[str, float] = field(default_factory=dict)
    dm: dict[str, float] = field(default_factory=dict)
    dm_by_split: dict[str, dict[str, float]] = field(default_factory=dict)
    flagged: bool = False
    flag_reasons: list[str] = field(default_factory=list)
    coefficients: dict[str, float] = field(default_factory=dict)
    fit_seconds: float = 0.0
    seed_ics: list[float] = field(default_factory=list)

    @property
    def val(self) -> dict[str, float]:
        return self.metrics.get("validation", {})

    @property
    def test(self) -> dict[str, float]:
        return self.metrics.get("test", {})


@dataclass
class RegimeReport:
    n_states: int
    bic: float
    fitted_through: str
    labels: pd.Series = field(repr=False, default_factory=lambda: pd.Series(dtype=object))
    frequencies: dict[str, dict[str, float]] = field(default_factory=dict)
    conditional: dict[str, dict[str, dict[str, dict[str, float]]]] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    data: dict[str, object]
    returns_board: list[ModelReport]
    vol_board: list[ModelReport]
    control: ModelReport | None
    regime: RegimeReport | None
    selected_return: str
    selected_vol: str
    selection_note: str
    include_test: bool
    elapsed_seconds: float

    def to_dict(self) -> dict[str, object]:
        def report(r: ModelReport) -> dict[str, object]:
            return {
                "name": r.name,
                "role": r.role,
                "kind": r.kind,
                "metrics": r.metrics,
                "train_metrics": r.train_metrics,
                "dm_vs_baseline": r.dm,
                "dm_by_split": r.dm_by_split,
                "flagged": r.flagged,
                "flag_reasons": r.flag_reasons,
                "coefficients": r.coefficients,
                "fit_seconds": r.fit_seconds,
                "seed_ics": r.seed_ics,
            }

        regime = None
        if self.regime is not None:
            regime = {
                "n_states": self.regime.n_states,
                "bic": self.regime.bic,
                "fitted_through": self.regime.fitted_through,
                "frequencies": self.regime.frequencies,
                "conditional": self.regime.conditional,
                "labels": {str(k.date()): v for k, v in self.regime.labels.items()},
            }
        return _json_safe(
            {
                "data": self.data,
                "returns": [report(r) for r in self.returns_board],
                "volatility": [report(r) for r in self.vol_board],
                "shuffled_control": report(self.control) if self.control else None,
                "regime": regime,
                "selected_return": self.selected_return,
                "selected_vol": self.selected_vol,
                "selection_note": self.selection_note,
                "include_test": self.include_test,
                "elapsed_seconds": self.elapsed_seconds,
            }
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if not math.isfinite(f) else f
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


def _split_dates(dates: pd.DatetimeIndex, horizon: int) -> dict[str, pd.DatetimeIndex]:
    """Date masks for dev/validation/test, excluding boundary-crossing targets."""
    ts = TimeSplit()
    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    if len(dates) == 0:
        return {"dev": dates, "validation": dates, "test": dates}

    def _cut(end: pd.Timestamp) -> pd.Timestamp:
        pos = int(np.searchsorted(dates.values, np.datetime64(end), side="right")) - 1
        pos = max(0, pos - max(int(horizon), 0))
        return dates[pos]

    dev_cut = _cut(pd.Timestamp(ts.dev_end))
    val_cut = _cut(pd.Timestamp(ts.val_end))
    val_start = pd.Timestamp(ts.val_start)
    test_start = pd.Timestamp(ts.test_start)
    return {
        "dev": dates[dates <= dev_cut],
        "validation": dates[(dates >= val_start) & (dates <= val_cut)],
        "test": dates[dates >= test_start],
    }


def _make_return_model(name: str, horizon: int):
    factories: dict[str, Callable[[], object]] = {
        "shrunk_mean": lambda: ShrunkMeanForecast(),
        "momentum_12_1": lambda: MomentumForecast(),
        "one_over_n": lambda: OneOverNForecast(),
        "ridge": lambda: RidgeForecast(horizon=horizon),
        "elastic_net": lambda: ElasticNetForecast(horizon=horizon),
        "lightgbm": lambda: LightGBMForecast(horizon=horizon),
    }
    return factories[name]


def _make_vol_model(name: str, horizon: int):
    factories: dict[str, Callable[[], object]] = {
        "ewma": lambda: EWMAVolatility(),
        "har": lambda: HARVolatility(horizon=horizon),
        "garch11": lambda: GARCH11Volatility(horizon=horizon),
    }
    return factories[name]


def _metrics_by_split(
    predictions: pd.Series,
    truth: pd.Series,
    dates: pd.DatetimeIndex,
    *,
    kind: str,
    horizon: int,
) -> dict[str, dict[str, float]]:
    splits = _split_dates(dates, horizon)
    out: dict[str, dict[str, float]] = {}
    for split in SPLITS:
        split_dates = splits[split]
        if len(split_dates) == 0:
            continue
        mask = predictions.index.get_level_values("date").isin(split_dates)
        p = predictions.loc[mask]
        t = truth.loc[mask]
        if p.empty:
            continue
        out[split] = (
            M.return_metrics(t, p, horizon=horizon)
            if kind == "return"
            else M.volatility_metrics(t, p)
        )
    return out


def _loss_series(predictions: pd.Series, truth: pd.Series, kind: str) -> pd.Series:
    common = predictions.index.intersection(truth.index)
    p = predictions.loc[common].astype(float)
    t = truth.loc[common].astype(float)
    if kind == "return":
        return (t - p) ** 2
    true_var = np.maximum(t.to_numpy() ** 2, 1e-18)
    pred_var = np.maximum(p.to_numpy() ** 2, 1e-12)
    return pd.Series(np.log(pred_var) + true_var / pred_var, index=common)


def _top_coefficients(coefficients: dict[str, float], k: int = 6) -> dict[str, float]:
    items = [(name, float(v)) for name, v in coefficients.items() if np.isfinite(v)]
    items.sort(key=lambda item: abs(item[1]), reverse=True)
    return {name: value for name, value in items[:k]}


def _flag_return(report: ModelReport) -> list[str]:
    reasons: list[str] = []
    train_ic = report.train_metrics.get("rank_ic", float("nan"))
    val_ic = report.val.get("rank_ic", float("nan"))
    dev_ic = report.metrics.get("dev", {}).get("rank_ic", float("nan"))
    train_rmse = report.train_metrics.get("rmse", float("nan"))
    val_rmse = report.val.get("rmse", float("nan"))
    if np.isfinite(train_ic) and np.isfinite(val_ic):
        gap = train_ic - val_ic
        if gap > IC_GAP_LIMIT:
            reasons.append(f"rank IC falls {gap:.3f} from train to validation")
    if report.role == "candidate" and np.isfinite(dev_ic) and np.isfinite(val_ic):
        swing = abs(dev_ic - val_ic)
        if swing > DEV_VAL_SWING_LIMIT:
            reasons.append(
                f"dev-to-validation rank IC swings {swing:.3f} "
                f"(dev {dev_ic:.3f}, validation {val_ic:.3f})"
            )
    if np.isfinite(train_rmse) and train_rmse > 1e-12 and np.isfinite(val_rmse):
        ratio = val_rmse / train_rmse
        if ratio > ERROR_RATIO_LIMIT:
            reasons.append(f"validation RMSE is {ratio:.2f}x the train RMSE")
    if report.val.get("constant", 0.0) > 0 and report.role == "candidate":
        reasons.append("degenerate constant forecast")
    return reasons


def _flag_vol(report: ModelReport) -> list[str]:
    reasons: list[str] = []
    train_q = report.train_metrics.get("qlike", float("nan"))
    val_q = report.val.get("qlike", float("nan"))
    if np.isfinite(train_q) and np.isfinite(val_q) and (val_q - train_q) > QLIKE_GAP_LIMIT:
        reasons.append(f"QLIKE rises {val_q - train_q:.3f} from train to validation")
    train_mae = report.train_metrics.get("mae", float("nan"))
    val_mae = report.val.get("mae", float("nan"))
    if np.isfinite(train_mae) and train_mae > 1e-12 and np.isfinite(val_mae):
        ratio = val_mae / train_mae
        if ratio > ERROR_RATIO_LIMIT:
            reasons.append(f"validation vol MAE is {ratio:.2f}x the train MAE")
    return reasons


def _train_metrics(
    fold_info: list[dict[str, object]],
    kind: str,
    horizon: int,
    *,
    max_date: pd.Timestamp | None = None,
) -> dict[str, float]:
    """In-sample metrics on the final training window.

    ``max_date`` caps the diagnostic at folds whose training window ends on or
    before that date. Selection must be invariant to whether the test window is
    included, so the caller passes the validation end here even on a final run:
    otherwise the last fold would be trained on (and its in-sample metrics
    computed from) test-period data, silently changing the stability flags.
    """
    last = _last_fold_info(fold_info, max_date=max_date)
    if last is None:
        return {}
    pred = last.get("train_pred")
    true = last.get("train_true")
    if not isinstance(pred, pd.Series) or not isinstance(true, pd.Series) or pred.empty:
        return {}
    return (
        M.return_metrics(true, pred, horizon=horizon)
        if kind == "return"
        else M.volatility_metrics(true, pred)
    )


def _last_fold_info(
    fold_info: list[dict[str, object]], *, max_date: pd.Timestamp | None = None
) -> dict[str, object] | None:
    """Last fold whose *prediction block* ends on or before ``max_date``.

    Filtering on the cutoff is not sufficient: a fold can start before the
    validation boundary yet predict into the test window. Selecting on the
    prediction end reproduces exactly the folds that a development-only run
    would have seen.
    """
    if not fold_info:
        return None
    eligible = fold_info
    if max_date is not None:
        cutoff = pd.Timestamp(max_date)

        def _end(entry: dict[str, object]) -> pd.Timestamp:
            value = entry.get("predict_end", entry.get("cutoff"))
            return pd.Timestamp(value)  # type: ignore[arg-type]

        eligible = [entry for entry in fold_info if _end(entry) <= cutoff]
    return eligible[-1] if eligible else None


def _run_board(
    panel: PanelData,
    *,
    names: list[str],
    kind: str,
    role_for: Callable[[str], str],
    horizon: int,
    min_train: int,
    embargo: int,
    step: int,
    max_date: pd.Timestamp | None,
    seed: int,
    train_metrics_max_date: pd.Timestamp | None = None,
    dates: pd.DatetimeIndex | None = None,
) -> tuple[list[ModelReport], dict[str, pd.Series], pd.Series | None]:
    reports: list[ModelReport] = []
    predictions_map: dict[str, pd.Series] = {}
    truth: pd.Series | None = None
    dates = (
        pd.DatetimeIndex(sorted(panel.X.index.get_level_values("date").unique()))
        if dates is None
        else pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    )
    for name in names:
        started = time.perf_counter()
        factory = (
            _make_return_model(name, horizon) if kind == "return" else _make_vol_model(name, horizon)
        )
        predictions, targets, info = out_of_fold_predictions(
            panel,
            factory,
            min_train=min_train,
            horizon=horizon,
            embargo=embargo,
            step=step,
            max_date=max_date,
            seed=seed,
        )
        predictions_map[name] = predictions
        truth = targets
        reports.append(
            ModelReport(
                name=name,
                role=role_for(name),
                kind=kind,
                metrics=_metrics_by_split(predictions, targets, dates, kind=kind, horizon=horizon),
                train_metrics=_train_metrics(
                    info, kind, horizon, max_date=train_metrics_max_date
                ),
                coefficients=_top_coefficients(
                    (_last_fold_info(info, max_date=train_metrics_max_date) or {}).get(
                        "coefficients", {}
                    )
                ),
                fit_seconds=time.perf_counter() - started,
            )
        )
    return reports, predictions_map, truth


def _best_baseline(reports: list[ModelReport], kind: str) -> ModelReport:
    baselines = [r for r in reports if r.role == "baseline"]
    if not baselines:
        raise ValueError("no baseline models")

    def score(report: ModelReport) -> tuple[float, float]:
        val = report.val
        if kind == "return":
            ic = val.get("rank_ic", float("nan"))
            err = val.get("rmse", float("inf"))
            return (
                ic if np.isfinite(ic) else -np.inf,
                -err if np.isfinite(err) else -np.inf,
            )
        q = val.get("qlike", float("nan"))
        return (-q if np.isfinite(q) else -np.inf, 0.0)

    return max(baselines, key=score)


def _attach_dm(
    reports: list[ModelReport],
    predictions_map: dict[str, pd.Series],
    truth: pd.Series,
    baseline: ModelReport,
    kind: str,
    horizon: int,
    dates: pd.DatetimeIndex | None = None,
) -> None:
    """Diebold-Mariano against the selected baseline, computed per split.

    A single statistic over dev+validation+test would silently mix periods and
    change when the test window is added, so each split gets its own test. The
    split boundaries are taken from the full reporting calendar (not from the
    prediction index), otherwise skipping the boundary-crossing fold in a
    development run would move the validation cutoff.
    """
    common = (
        pd.DatetimeIndex(sorted(truth.index.get_level_values("date").unique()))
        if dates is None
        else pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    )
    splits = _split_dates(common, horizon)
    split_data: dict[str, tuple[pd.Series, dict[str, pd.Series]]] = {}
    for split, split_dates in splits.items():
        if len(split_dates) == 0:
            continue
        mask = truth.index.get_level_values("date").isin(split_dates)
        split_data[split] = (
            truth.loc[mask],
            {name: series.loc[mask] for name, series in predictions_map.items()},
        )
    for report in reports:
        report.dm_by_split = {}
        for split, (split_truth, split_predictions) in split_data.items():
            base_loss = _loss_series(
                split_predictions[baseline.name], split_truth, kind
            )
            if report.name == baseline.name:
                report.dm_by_split[split] = {
                    "dm_stat": 0.0,
                    "dm_pvalue": float("nan"),
                    "mean_diff": 0.0,
                    "n_dates": float(
                        base_loss.index.get_level_values("date").nunique()
                    ),
                    "baseline": baseline.name,  # type: ignore[dict-item]
                }
                continue
            model_loss = _loss_series(
                split_predictions[report.name], split_truth, kind
            )
            stats = M.diebold_mariano_nw(
                model_loss, base_loss, lags=max(horizon - 1, 0)
            )
            stats["baseline"] = baseline.name  # type: ignore[assignment]
            report.dm_by_split[split] = stats
        # Backwards-compatible field: the validation comparison.
        report.dm = report.dm_by_split.get("validation", {})


# ---------------------------------------------------------------------------
# regime
# ---------------------------------------------------------------------------


def _regime_report(
    returns: pd.DataFrame,
    predictions_map: dict[str, pd.Series],
    truth: pd.Series,
    models: list[str],
    *,
    horizon: int,
    max_states: int,
    include_test: bool,
    dates: pd.DatetimeIndex | None = None,
) -> RegimeReport | None:
    benchmark = returns["SPX"] if "SPX" in returns.columns else returns.mean(axis=1)
    dev_end = pd.Timestamp(TimeSplit().dev_end)
    train = benchmark.loc[:dev_end].dropna()
    if len(train) < 120:
        return None
    detector = HMMRegime(max_states=max_states, random_state=12345).fit(train)
    labels = detector.labels(benchmark.dropna())
    if labels.empty:
        return None
    if not include_test:
        # Development reports must not display test-period regime statistics.
        labels = labels.loc[labels.index <= pd.Timestamp(TimeSplit().val_end)]
        if labels.empty:
            return None

    label_dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(labels.index).unique()))
    splits = _split_dates(label_dates, horizon)
    frequencies: dict[str, dict[str, float]] = {}
    for split, split_dates in splits.items():
        if len(split_dates) == 0:
            continue
        subset = labels.loc[labels.index.isin(split_dates)]
        if subset.empty:
            continue
        frequencies[split] = {
            str(k): float(v) for k, v in subset.value_counts(normalize=True).items()
        }

    date_splits = _split_dates(
        pd.DatetimeIndex(sorted(predictions_map[models[0]].index.get_level_values("date").unique()))
        if dates is None and models and not predictions_map.get(models[0], pd.Series(dtype=float)).empty
        else (label_dates if dates is None else dates),
        horizon,
    )

    conditional: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for model_name in models:
        preds = predictions_map.get(model_name)
        if preds is None or preds.empty:
            continue
        conditional[model_name] = {}
        for split in SPLITS:
            if split == "test" and not include_test:
                continue
            split_dates = date_splits[split]
            if len(split_dates) == 0:
                continue
            mask = preds.index.get_level_values("date").isin(split_dates)
            p = preds.loc[mask]
            t = truth.loc[mask]
            if p.empty:
                continue
            frame = pd.DataFrame({"p": p, "t": t})
            frame["regime"] = frame.index.get_level_values("date").map(labels)
            per_regime: dict[str, dict[str, float]] = {}
            for regime, block in frame.dropna(subset=["regime"]).groupby("regime"):
                per_regime[str(regime)] = M.return_metrics(
                    block["t"], block["p"], horizon=horizon
                )
            conditional[model_name][split] = per_regime
    return RegimeReport(
        n_states=int(detector.n_states),
        bic=float(detector.bic_),
        fitted_through=str(dev_end.date()),
        labels=labels,
        frequencies=frequencies,
        conditional=conditional,
    )


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def evaluate(
    returns: pd.DataFrame,
    prices: pd.DataFrame | None,
    *,
    horizon: int = 21,
    min_train: int = 378,
    embargo: int = 21,
    step: int = 21,
    include_test: bool = False,
    seed: int = 12345,
    max_states: int = 3,
) -> EvaluationResult:
    """Run the full forecast evaluation.

    Folds are always generated from the full date calendar and a development
    run (``include_test=False``) skips every fold whose prediction block reaches
    past the validation boundary *before* fitting. That makes the dev and
    validation fold schedule - and therefore the stability flags and the model
    selection - byte-for-byte identical whether or not the test window is
    included. The test rows may exist in the causal feature matrix, but no test
    value enters a fitted model, a prediction, a metric, a flag or the
    selection; scored validation dates are those whose targets end on or before
    ``val_end``, and every eligible fold's training targets end before it.
    """
    started = time.perf_counter()
    if returns.empty:
        raise ValueError("returns frame is empty")
    returns = returns.dropna(how="all").fillna(0.0)

    return_panel = build_panel(returns, prices, kind="return", horizon=horizon).to_panel()
    vol_panel = build_panel(returns, prices, kind="volatility", horizon=horizon).to_panel()
    selection_end = pd.Timestamp(TimeSplit().val_end)
    max_date = None if include_test else selection_end

    all_dates = pd.DatetimeIndex(sorted(return_panel.X.index.get_level_values("date").unique()))
    reported_dates = (
        all_dates if include_test else all_dates[all_dates <= selection_end]
    )
    splits = _split_dates(reported_dates, horizon)

    returns_board, return_predictions, return_truth = _run_board(
        return_panel,
        names=[cls.name for cls in (*BASELINES, *CANDIDATES)],
        kind="return",
        role_for=lambda n: "baseline" if n in {c.name for c in BASELINES} else "candidate",
        horizon=horizon,
        min_train=min_train,
        embargo=embargo,
        step=step,
        max_date=max_date,
        seed=seed,
        train_metrics_max_date=selection_end,
        dates=reported_dates,
    )
    for report in returns_board:
        report.flag_reasons = _flag_return(report)
        report.flagged = bool(report.flag_reasons)
    best_baseline = _best_baseline(returns_board, "return")
    if return_truth is not None:
        _attach_dm(
            returns_board,
            return_predictions,
            return_truth,
            best_baseline,
            "return",
            horizon,
            dates=reported_dates,
        )

    # Shuffled-label control: the same Ridge pipeline with labels permuted across
    # tickers within each date. One seed has a high-variance random coefficient
    # direction, so average several seeds; an honest pipeline has no consistent
    # cross-sectional signal after the permutation.
    control_started = time.perf_counter()
    control_seeds = (seed, 1, 2)
    per_seed_reports: list[ModelReport] = []
    for control_seed in control_seeds:
        control_preds, control_truth, control_info = out_of_fold_predictions(
            return_panel,
            _make_return_model("ridge", horizon),
            min_train=min_train,
            horizon=horizon,
            embargo=embargo,
            step=step,
            max_date=max_date,
            shuffle_train_labels=True,
            seed=control_seed,
        )
        per_seed_reports.append(
            ModelReport(
                name="ridge_shuffled_control",
                role="control",
                kind="return",
                metrics=_metrics_by_split(
                    control_preds,
                    control_truth,
                    reported_dates,
                    kind="return",
                    horizon=horizon,
                ),
                train_metrics=_train_metrics(
                    control_info, "return", horizon, max_date=selection_end
                ),
            )
        )

    def _mean_metrics(reports: list[ModelReport]) -> dict[str, dict[str, float]]:
        averaged: dict[str, dict[str, float]] = {}
        for split in SPLITS:
            blocks = [r.metrics[split] for r in reports if split in r.metrics]
            if not blocks:
                continue
            keys = {k for block in blocks for k in block}
            averaged[split] = {}
            for key in keys:
                values = [block.get(key, float("nan")) for block in blocks]
                finite = [v for v in values if np.isfinite(v)]
                averaged[split][key] = float(np.mean(finite)) if finite else float("nan")
        return averaged

    def _mean_train(reports: list[ModelReport]) -> dict[str, float]:
        keys = {k for report in reports for k in report.train_metrics}
        out: dict[str, float] = {}
        for key in keys:
            values = [
                report.train_metrics.get(key, float("nan")) for report in reports
            ]
            finite = [v for v in values if np.isfinite(v)]
            out[key] = float(np.mean(finite)) if finite else float("nan")
        return out

    control = ModelReport(
        name="ridge_shuffled_control",
        role="control",
        kind="return",
        metrics=_mean_metrics(per_seed_reports),
        train_metrics=_mean_train(per_seed_reports),
        fit_seconds=time.perf_counter() - control_started,
        seed_ics=[
            float(r.metrics.get("validation", {}).get("rank_ic", float("nan")))
            for r in per_seed_reports
        ],
    )

    vol_board, vol_predictions, vol_truth = _run_board(
        vol_panel,
        names=[EWMAVolatility.name, HARVolatility.name, GARCH11Volatility.name],
        kind="volatility",
        role_for=lambda n: "baseline" if n == VOL_BASELINE else "candidate",
        horizon=horizon,
        min_train=min_train,
        embargo=embargo,
        step=step,
        max_date=max_date,
        seed=seed,
        train_metrics_max_date=selection_end,
        dates=reported_dates,
    )
    for report in vol_board:
        report.flag_reasons = _flag_vol(report)
        report.flagged = bool(report.flag_reasons)
    best_vol_baseline = _best_baseline(vol_board, "volatility")
    if vol_truth is not None:
        _attach_dm(
            vol_board,
            vol_predictions,
            vol_truth,
            best_vol_baseline,
            "volatility",
            horizon,
            dates=reported_dates,
        )

    # Selection on validation only; flagged candidates are not eligible.
    def val_ic(report: ModelReport) -> float:
        value = report.val.get("rank_ic", float("nan"))
        return value if np.isfinite(value) else -np.inf

    eligible = [
        r for r in returns_board if r.role == "candidate" and not r.flagged and val_ic(r) > 0.0
    ]
    selected_return = best_baseline.name
    if eligible:
        best_candidate = max(eligible, key=val_ic)
        if val_ic(best_candidate) > val_ic(best_baseline) + MIN_IC_EDGE:
            selected_return = best_candidate.name

    vol_candidates = [
        r
        for r in vol_board
        if r.role == "candidate"
        and not r.flagged
        and np.isfinite(r.val.get("qlike", float("nan")))
    ]
    selected_vol = best_vol_baseline.name
    if vol_candidates:
        best_vol_candidate = min(vol_candidates, key=lambda r: r.val["qlike"])
        base_qlike = best_vol_baseline.val.get("qlike", float("nan"))
        if np.isfinite(base_qlike) and best_vol_candidate.val["qlike"] < (
            base_qlike - MIN_QLIKE_MARGIN
        ):
            selected_vol = best_vol_candidate.name

    note = (
        "Best model per task is chosen on the validation window only. A candidate "
        "must beat the best baseline by at least 0.01 rank IC (returns) or 0.01 "
        "QLIKE (volatility), must have positive validation rank IC for returns, and "
        "must not be flagged for a large train/validation gap; otherwise the "
        "baseline is retained."
    )

    regime = None
    if return_truth is not None:
        focus = [selected_return]
        if best_baseline.name not in focus:
            focus.append(best_baseline.name)
        regime = _regime_report(
            returns,
            return_predictions,
            return_truth,
            focus,
            horizon=horizon,
            max_states=max_states,
            include_test=include_test,
            dates=reported_dates,
        )

    panel_dates = return_panel.X.index.get_level_values("date")
    reported_mask = panel_dates.isin(reported_dates)
    data = {
        "tickers": list(returns.columns),
        "n_tickers": int(len(returns.columns)),
        "start": str(reported_dates.min().date()) if len(reported_dates) else None,
        "end": str(reported_dates.max().date()) if len(reported_dates) else None,
        "n_dates": int(len(reported_dates)),
        "n_observations": int(reported_mask.sum()),
        "horizon_days": int(horizon),
        "min_train_days": int(min_train),
        "embargo_days": int(embargo),
        "step_days": int(step),
        "features": list(return_panel.X.columns),
        "split_counts": {name: int(len(dates)) for name, dates in splits.items() if len(dates)},
        "split_boundaries": {
            "dev_end": str(TimeSplit().dev_end),
            "validation": [str(TimeSplit().val_start), str(TimeSplit().val_end)],
            "test_start": str(TimeSplit().test_start),
        },
        "test_included": bool(include_test),
    }

    return EvaluationResult(
        data=data,
        returns_board=returns_board,
        vol_board=vol_board,
        control=control,
        regime=regime,
        selected_return=selected_return,
        selected_vol=selected_vol,
        selection_note=note,
        include_test=include_test,
        elapsed_seconds=time.perf_counter() - started,
    )


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _split_names(include_test: bool) -> list[str]:
    return ["dev", "validation", "test"] if include_test else ["dev", "validation"]


def _model_table(board: list[ModelReport], include_test: bool, kind: str) -> list[str]:
    splits = _split_names(include_test)
    header = ["model", "role"]
    header += [f"{split} RMSE" for split in splits]
    header += [f"{split} MAE" for split in splits]
    if kind == "return":
        header.append("train in-sample IC")
        header += [f"{split} IC" for split in splits]
        header += ["validation IC t", "hit rate (val)"]
        header += [f"DM ({split})" for split in splits]
        header.append("flag")
    else:
        header.append("train in-sample QLIKE")
        header += [f"{split} QLIKE" for split in splits]
        header += [f"DM ({split})" for split in splits]
        header.append("flag")

    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for report in board:
        row = [f"`{report.name}`", report.role]
        row += [_fmt(report.metrics.get(split, {}).get("rmse")) for split in splits]
        row += [_fmt(report.metrics.get(split, {}).get("mae")) for split in splits]
        if kind == "return":
            row.append(_fmt(report.train_metrics.get("rank_ic")))
            row += [_fmt(report.metrics.get(split, {}).get("rank_ic")) for split in splits]
            row.append(_fmt(report.val.get("rank_ic_t"), 2))
            row.append(_fmt(report.val.get("hit_rate"), 3))
        else:
            row.append(_fmt(report.train_metrics.get("qlike")))
            row += [_fmt(report.metrics.get(split, {}).get("qlike")) for split in splits]
        row += [
            _fmt(report.dm_by_split.get(split, {}).get("dm_stat"), 2)
            for split in splits
        ]
        row.append("**FLAGGED**" if report.flagged else "ok")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_markdown(result: EvaluationResult) -> str:
    data = result.data
    lines: list[str] = []
    add = lines.append

    add("# Forecast model evaluation")
    add("")
    add(
        "Generated by `scripts/run_forecast_evaluation.py`. Every number below comes "
        "from that script; nothing is hand-entered. The test window is included only "
        "when the run passes `--include-test`. Model selection never uses test "
        "results: stability diagnostics are restricted to folds that end at the "
        "validation boundary, and a regression test asserts that adding the test "
        "window leaves the dev/validation metrics, flags and selected models "
        "unchanged."
    )
    add("")
    add("## Protocol")
    add("")
    add(f"- Universe: {data['n_tickers']} tickers  -  {', '.join(data['tickers'])}")
    add(
        f"- Sample: {data['start']} to {data['end']} "
        f"({data['n_dates']} feature dates, {data['n_observations']} asset-date rows)"
    )
    add(f"- Forecast horizon: {data['horizon_days']} trading days")
    add(
        f"- Cross-validation: purged expanding walk-forward, min train "
        f"{data['min_train_days']} days, purge {data['horizon_days']} days, "
        f"embargo {data['embargo_days']} days, step {data['step_days']} days"
    )
    add(
        f"- Dev: through {data['split_boundaries']['dev_end']}; validation: "
        f"{data['split_boundaries']['validation'][0]} to "
        f"{data['split_boundaries']['validation'][1]}; test: from "
        f"{data['split_boundaries']['test_start']}"
    )
    counts = ", ".join(f"{k} {v}" for k, v in data["split_counts"].items())
    add(f"- Scored feature dates per split: {counts}")
    add(f"- Test included in this run: **{'yes' if result.include_test else 'no'}**")
    add("")
    add(
        "All features are lagged by one trading day and targets are forward "
        "returns/vol, so a training row whose target window overlaps a prediction "
        "block is purged. The last `horizon` feature dates before each split boundary "
        "are excluded from the scored set because their targets reach into the next "
        "split. The shuffled control re-runs the Ridge pipeline with training labels "
        "permuted across tickers within each date, averaged over three seeds so that "
        "one random coefficient direction cannot masquerade as skill."
    )
    add("")

    add("## Selection")
    add("")
    add(f"- Selected return model: **`{result.selected_return}`**")
    add(f"- Selected volatility model: **`{result.selected_vol}`**")
    add(f"- Rule: {result.selection_note}")
    add("")

    add("## Return forecasts")
    add("")
    lines.extend(_model_table(result.returns_board, result.include_test, "return"))
    add("")
    add(
        "Rank IC is the mean Spearman rank correlation across tickers per date; its "
        "t-statistic uses a Newey-West variance with `horizon - 1` lags to account for "
        "overlapping targets. `one_over_n` is constant across assets, so its rank IC is "
        "0 by construction and its t-statistic is undefined."
    )
    add("")

    if result.control is not None:
        c = result.control
        add("### Shuffled-label control")
        add("")
        add("| split | rank IC | rank IC t | RMSE | hit rate |")
        add("|---|---|---|---|---|")
        for split in _split_names(result.include_test):
            m = c.metrics.get(split, {})
            if m:
                add(
                    f"| {split} | {_fmt(m.get('rank_ic'))} | {_fmt(m.get('rank_ic_t'), 2)} | "
                    f"{_fmt(m.get('rmse'))} | {_fmt(m.get('hit_rate'), 3)} |"
                )
        add("")
        if c.seed_ics:
            seed_text = ", ".join(_fmt(ic) for ic in c.seed_ics)
            add(f"Per-seed validation rank IC: {seed_text}.")
            add("")
        control_val_ic = c.metrics.get("validation", {}).get("rank_ic", float("nan"))
        control_val_t = c.metrics.get("validation", {}).get("rank_ic_t", float("nan"))
        passed = (
            np.isfinite(control_val_ic)
            and abs(control_val_ic) < 0.05
            and (not np.isfinite(control_val_t) or abs(control_val_t) < 2.0)
        )
        add(
            f"Seed-averaged validation rank IC of the shuffled control: "
            f"**{_fmt(control_val_ic)}** (t = {_fmt(control_val_t, 2)}), which "
            f"{'passes' if passed else 'FAILS'} the |IC| < 0.05 and |t| < 2 sanity "
            "threshold. A failure would mean the evaluation leaks label information: "
            "after permuting labels across tickers within each date, a model with no "
            "real signal should have no cross-sectional predictive power."
        )
        add("")

    add("## Volatility forecasts")
    add("")
    lines.extend(_model_table(result.vol_board, result.include_test, "volatility"))
    add("")
    add(
        "QLIKE is computed on variance, `log(sigma^2_pred) + sigma^2_true / sigma^2_pred` (lower is "
        "better). `ewma` is the designated baseline; GARCH parameters are estimated "
        "with `arch` on each training fold and filtered forward with the standard "
        "GARCH(1,1) recursion."
    )
    add("")

    if result.include_test:
        selected_return = next(
            (report for report in result.returns_board if report.name == result.selected_return),
            None,
        )
        selected_vol = next(
            (report for report in result.vol_board if report.name == result.selected_vol),
            None,
        )
        add("## Final test readout")
        add("")
        add(
            "The selected models were fixed using the validation split. The test "
            "split is reported below for transparency and is not used to re-select "
            "or tune a model."
        )
        add("")
        add("| Task | Model | Metric | Validation | Test |")
        add("|---|---|---|---:|---:|")
        for task, report, metrics in (
            ("return", selected_return, ("rank_ic", "rmse", "mae")),
            ("volatility", selected_vol, ("qlike", "rmse", "mae")),
        ):
            if report is None:
                continue
            for metric in metrics:
                validation_value = report.val.get(metric)
                test_value = report.test.get(metric)
                if validation_value is None or test_value is None:
                    continue
                add(
                    f"| {task} | `{report.name}` | {metric} | "
                    f"{_fmt(validation_value, 4)} | {_fmt(test_value, 4)} |"
                )
        add("")
        if selected_return is not None:
            add(
                f"Return rank IC moved from {_fmt(selected_return.val.get('rank_ic'), 4)} "
                f"in validation to {_fmt(selected_return.test.get('rank_ic'), 4)} in "
                f"test, and RMSE moved from {_fmt(selected_return.val.get('rmse'), 4)} "
                f"to {_fmt(selected_return.test.get('rmse'), 4)}."
            )
            add("")
        if selected_vol is not None:
            val_qlike = selected_vol.val.get("qlike")
            test_qlike = selected_vol.test.get("qlike")
            val_rmse = selected_vol.val.get("rmse")
            test_rmse = selected_vol.test.get("rmse")
            val_mae = selected_vol.val.get("mae")
            test_mae = selected_vol.test.get("mae")
            qlike_word = (
                "improved"
                if test_qlike is not None
                and val_qlike is not None
                and test_qlike < val_qlike
                else "worsened"
            )
            error_word = (
                "improved"
                if test_rmse is not None
                and val_rmse is not None
                and test_rmse < val_rmse
                and test_mae is not None
                and val_mae is not None
                and test_mae < val_mae
                else "worsened or was mixed"
            )
            add(
                f"Volatility QLIKE {qlike_word} "
                f"({_fmt(val_qlike, 4)} to {_fmt(test_qlike, 4)}), while absolute "
                f"error {error_word} (RMSE {_fmt(val_rmse, 4)} to "
                f"{_fmt(test_rmse, 4)}; MAE {_fmt(val_mae, 4)} to "
                f"{_fmt(test_mae, 4)}). This distinction matters: QLIKE is the "
                "selection criterion, but it does not mean the forecast is closer "
                "in absolute volatility terms."
            )
            add("")
            best_test_error = min(
                (
                    report
                    for report in result.vol_board
                    if report.name != selected_vol.name
                    and report.test.get("rmse") is not None
                ),
                key=lambda report: report.test["rmse"],
                default=None,
            )
            if best_test_error is not None and best_test_error.test.get("rmse") < (
                selected_vol.test.get("rmse") or float("inf")
            ):
                add(
                    f"`{best_test_error.name}` has a lower test RMSE than the selected "
                    f"`{selected_vol.name}` ({_fmt(best_test_error.test.get('rmse'), 4)} "
                    f"versus {_fmt(selected_vol.test.get('rmse'), 4)}), but the "
                    "selection rule is fixed on validation QLIKE. The test window is "
                    "not used to promote it after the fact."
                )
                add("")
        add(
            "The test result is therefore mixed rather than uniformly better: the "
            "forecast ranking/QLIKE objective does not guarantee lower absolute "
            "errors, and the advisory evaluation should not be read as if it did."
        )
        add("")

    add("## Train-vs-validation gaps")
    add("")
    flagged = [r for r in (*result.returns_board, *result.vol_board) if r.flagged]
    if not flagged:
        add("No model exceeded the gap limits.")
    else:
        for report in flagged:
            add(f"- **`{report.name}`** ({report.kind}): " + "; ".join(report.flag_reasons))
    add("")
    add(
        f"Limits: return rank IC may fall by at most {IC_GAP_LIMIT:.2f}; error "
        f"(RMSE/MAE) may rise by at most {ERROR_RATIO_LIMIT:.0%}; volatility QLIKE may "
        f"rise by at most {QLIKE_GAP_LIMIT:.2f}. Candidate return models are also "
        f"flagged when their out-of-fold rank IC moves more than "
        f"{DEV_VAL_SWING_LIMIT:.2f} between the dev and validation windows."
    )
    add("")

    if result.regime is not None:
        regime = result.regime
        add("## Regime-conditional results")
        add("")
        add(
            f"Gaussian HMM with {regime.n_states} states selected by BIC "
            f"({regime.bic:.1f}), fitted on benchmark returns through "
            f"{regime.fitted_through}. Labels use filtered probabilities "
            "`P(state_t | x_1..x_t)`, never Viterbi or smoothed paths."
        )
        add("")
        add("| split | calm | elevated | stress |")
        add("|---|---|---|---|")
        for split, freq in regime.frequencies.items():
            add(
                "| "
                + split
                + " | "
                + " | ".join(
                    _fmt(freq.get(label), 3) for label in ("calm", "elevated", "stress")
                )
                + " |"
            )
        add("")
        for model_name, by_split in regime.conditional.items():
            add(f"**`{model_name}`**  -  rank IC (RMSE) by regime")
            add("")
            add("| split | regime | rank IC | rank IC t | RMSE | n dates |")
            add("|---|---|---|---|---|---|")
            for split, per_regime in by_split.items():
                for regime_name, values in per_regime.items():
                    add(
                        f"| {split} | {regime_name} | {_fmt(values.get('rank_ic'))} | "
                        f"{_fmt(values.get('rank_ic_t'), 2)} | {_fmt(values.get('rmse'))} | "
                        f"{_fmt(values.get('n_dates'), 0)} |"
                    )
            add("")

    add("## Reproduction")
    add("")
    add("```bash")
    add("python scripts/fetch_real_data.py                    # refresh ETF prices + FRED DGS3MO")
    add("python scripts/run_forecast_evaluation.py            # dev + validation only")
    add("python scripts/run_forecast_evaluation.py --include-test  # final report")
    add("```")
    add("")
    add(
        "The final run adds the test window without re-selecting anything. Results "
        "are reported as-is, and the selected models are the same ones chosen on "
        "dev/validation alone."
    )
    add("")
    return "\n".join(lines)


def write_report(result: EvaluationResult, path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown(result), encoding="utf-8")
    json_path = target.with_suffix(".json")
    json_path.write_text(
        json.dumps(result.to_dict(), indent=2, allow_nan=False), encoding="utf-8"
    )


__all__ = [
    "EvaluationResult",
    "ModelReport",
    "RegimeReport",
    "evaluate",
    "render_markdown",
    "write_report",
]
