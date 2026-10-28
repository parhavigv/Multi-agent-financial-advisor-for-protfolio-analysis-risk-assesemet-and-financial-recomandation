"""Purged, embargoed expanding walk-forward cross-validation.

The forecast horizon is ``h`` days. A training row at date ``t`` has a target
that covers ``(t, t+h]``. If a prediction block starts at date ``c``, keeping
training rows with ``t > c - h`` would leak the future into the fit, so those
rows are **purged**. An additional embargo of ``embargo`` days is then removed
to absorb residual serial correlation (e.g. overlapping targets, stale closes).

Only the past is ever used: each fold trains on ``[start, cutoff - h - embargo)``
and predicts the next ``step`` dates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import PanelData


@dataclass(frozen=True)
class Fold:
    train_dates: pd.DatetimeIndex
    predict_dates: pd.DatetimeIndex
    cutoff: pd.Timestamp

    @property
    def train_mask(self) -> np.ndarray:
        return np.ones(len(self.train_dates), dtype=bool)


def purged_expanding_folds(
    dates: pd.DatetimeIndex,
    *,
    min_train: int = 504,
    horizon: int = 21,
    embargo: int = 21,
    step: int = 21,
) -> list[Fold]:
    """Build expanding folds over ``dates``.

    ``min_train`` counts *dates*, the purge/embargo count dates as well (the
    panel is daily, so this equals trading days). The first prediction block
    starts once ``min_train + horizon + embargo`` dates are available.
    """
    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    n = len(dates)
    gap = max(int(horizon), 1) + max(int(embargo), 0)
    folds: list[Fold] = []
    start = max(int(min_train) + gap, gap)
    for cut in range(start, n, max(int(step), 1)):
        cutoff = dates[cut]
        train_end = cut - gap  # exclusive
        if train_end < 1:
            continue
        block_end = min(cut + max(int(step), 1), n)
        folds.append(
            Fold(
                train_dates=dates[:train_end],
                predict_dates=dates[cut:block_end],
                cutoff=cutoff,
            )
        )
    return folds


def subset(panel: PanelData, dates: pd.DatetimeIndex) -> PanelData:
    """Rows of ``panel`` whose date is in ``dates``."""
    keep = pd.DatetimeIndex(dates)
    mask = panel.X.index.get_level_values("date").isin(keep)
    return PanelData(X=panel.X.loc[mask], y=panel.y.loc[mask], tickers=panel.tickers)


def out_of_fold_predictions(
    panel: PanelData,
    model_factory,
    *,
    min_train: int = 504,
    horizon: int = 21,
    embargo: int = 21,
    step: int = 21,
    max_date: pd.Timestamp | None = None,
    shuffle_train_labels: bool = False,
    seed: int = 12345,
) -> tuple[pd.Series, pd.Series, list[dict[str, object]]]:
    """Fit ``model_factory()`` on each fold and collect out-of-fold predictions.

    Returns ``(predictions, targets, fold_info)`` where predictions/targets are
    indexed by ``(date, ticker)`` and each date appears exactly once. In-sample
    training predictions are also recorded in ``fold_info`` so the caller can
    measure a genuine train-vs-validation gap without re-fitting.
    """
    folds = purged_expanding_folds(
        panel.dates, min_train=min_train, horizon=horizon, embargo=embargo, step=step
    )
    preds: list[pd.Series] = []
    truth: list[pd.Series] = []
    info: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)

    for fold in folds:
        # Never run a fold whose prediction block reaches beyond ``max_date``:
        # development runs must not see the held-out test period at all.
        if max_date is not None and fold.predict_dates.max() > pd.Timestamp(max_date):
            continue
        train = subset(panel, fold.train_dates)
        test = subset(panel, fold.predict_dates)
        if train.X.empty or test.X.empty:
            continue

        y_train = train.y.copy()
        if shuffle_train_labels:
            # Permute labels *across tickers within each date*. This preserves
            # the per-date return distribution (so the control is not trivially
            # easy) but destroys any feature-to-asset alignment: a model with
            # genuine skill must collapse to zero rank IC on the held-out data.
            shuffled = np.empty(len(y_train), dtype=float)
            values = y_train.to_numpy(dtype=float)
            dates = y_train.index.get_level_values("date")
            for date in np.unique(dates):
                idx = np.flatnonzero(dates == date)
                shuffled[idx] = rng.permutation(values[idx])
            y_train = pd.Series(shuffled, index=y_train.index, name=y_train.name)
            train = PanelData(X=train.X, y=y_train, tickers=train.tickers)

        model = model_factory()
        model.fit(train)
        out = model.predict(test)
        out = out.reindex(test.X.index)

        preds.append(out.dropna())
        truth.append(test.y.reindex(out.dropna().index))

        # In-sample predictions on the fold's own training window.
        in_sample = model.predict(train).reindex(train.X.index)
        info.append(
            {
                "cutoff": fold.cutoff,
                "predict_end": fold.predict_dates.max(),
                "train_rows": int(len(train.X)),
                "predict_rows": int(len(test.X)),
                "train_pred": in_sample,
                "train_true": train.y,
                "coefficients": model.coefficients(),
            }
        )

    if preds:
        predictions = pd.concat(preds).sort_index()
        targets = pd.concat(truth).sort_index()
    else:  # pragma: no cover - only when the sample is too short
        predictions = pd.Series(dtype=float)
        targets = pd.Series(dtype=float)
    predictions.index = predictions.index.set_names(["date", "ticker"])
    targets.index = targets.index.set_names(["date", "ticker"])
    return predictions.rename("prediction"), targets.rename("target"), info


__all__ = ["Fold", "purged_expanding_folds", "subset", "out_of_fold_predictions"]
