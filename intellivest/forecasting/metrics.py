"""Forecast evaluation metrics.

The interesting quantities for a cross-sectional forecast are not just RMSE
(pooled over tickers) but the *rank* information the allocator actually
consumes.  With an overlapping ``h``-day target, both the rank-IC t-statistic
and the Diebold-Mariano test need a Newey-West correction; the plain i.i.d.
formulas overstate significance by roughly ``sqrt(h)``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from intellivest.ml.base import mae, rmse

__all__ = [
    "diebold_mariano_nw",
    "hit_rate",
    "mean_tstat_nw",
    "per_date_rank_ic",
    "qlike_variance",
    "return_metrics",
    "volatility_metrics",
]


def _aligned(
    y_true: pd.Series, y_pred: pd.Series
) -> tuple[pd.Series, pd.Series]:
    common = y_true.index.intersection(y_pred.index)
    a = y_true.loc[common].astype(float)
    b = y_pred.loc[common].astype(float)
    mask = np.isfinite(a.to_numpy()) & np.isfinite(b.to_numpy())
    return a.loc[mask], b.loc[mask]


def per_date_rank_ic(
    y_true: pd.Series, y_pred: pd.Series, min_names: int = 5
) -> pd.Series:
    """Spearman rank IC per date across the cross-section of tickers."""
    a, b = _aligned(y_true, y_pred)
    if a.empty:
        return pd.Series(dtype=float)
    frame = pd.DataFrame({"y": a, "p": b})
    values: dict[pd.Timestamp, float] = {}
    for date, block in frame.groupby(level="date"):
        if len(block) < min_names or block["y"].nunique() < 2 or block["p"].nunique() < 2:
            continue
        values[pd.Timestamp(date)] = float(
            block["y"].rank().corr(block["p"].rank())
        )
    return pd.Series(values, dtype=float).sort_index()


def _newey_west_variance(x: np.ndarray, lags: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    demeaned = x - x.mean()
    gamma0 = float(np.dot(demeaned, demeaned) / n)
    var = gamma0
    max_lag = int(min(max(lags, 0), n - 1))
    for lag in range(1, max_lag + 1):
        gamma = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1.0)
        var += 2.0 * weight * gamma
    # Guard against a numerically negative HAC estimate.
    return max(var / n, 0.0)


def mean_tstat_nw(values: pd.Series | np.ndarray, lags: int = 0) -> tuple[float, float, int]:
    """Mean of a serially correlated series with a Newey-West t-statistic."""
    x = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(x))
    var = _newey_west_variance(x, lags=lags)
    if not np.isfinite(var) or var <= 0:
        return mean, float("nan"), len(x)
    return mean, mean / math.sqrt(var), len(x)


def hit_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Share of observations whose predicted and realised signs agree."""
    a, b = _aligned(y_true, y_pred)
    if a.empty:
        return float("nan")
    return float(np.mean(np.sign(a.to_numpy()) == np.sign(b.to_numpy())))


def qlike_variance(true_vol: pd.Series, pred_vol: pd.Series) -> float:
    """QLIKE on variance: ``log(sigma2) + sigma2_true / sigma2_pred``."""
    a, b = _aligned(true_vol, pred_vol)
    if a.empty:
        return float("nan")
    true_var = np.maximum(a.to_numpy() ** 2, 1e-18)
    pred_var = np.maximum(b.to_numpy() ** 2, 1e-12)
    return float(np.mean(np.log(pred_var) + true_var / pred_var))


def _per_date_loss(
    loss: pd.Series,
) -> pd.Series:
    return loss.groupby(level="date").mean().sort_index()


def diebold_mariano_nw(
    loss_model: pd.Series,
    loss_baseline: pd.Series,
    *,
    lags: int = 20,
) -> dict[str, float]:
    """Diebold-Mariano test on per-date loss differentials.

    Negative statistic => the model has *lower* loss than the baseline.
    """
    common = loss_model.index.intersection(loss_baseline.index)
    if len(common) == 0:
        return {"dm_stat": float("nan"), "dm_pvalue": float("nan"), "mean_diff": float("nan"), "n_dates": 0.0}
    a = _per_date_loss(loss_model.loc[common])
    b = _per_date_loss(loss_baseline.loc[common])
    diff = (a - b).dropna()
    mean, stat, n = mean_tstat_nw(diff, lags=lags)
    # Identical forecasts have a zero loss differential with zero variance: the
    # correct conclusion is "no difference", not an undefined statistic.
    if np.isfinite(mean) and abs(mean) < 1e-15:
        stat = 0.0
    try:
        from scipy.stats import norm

        pvalue = float(2.0 * (1.0 - norm.cdf(abs(stat)))) if np.isfinite(stat) else float("nan")
    except Exception:  # pragma: no cover
        pvalue = float("nan")
    return {
        "dm_stat": float(stat),
        "dm_pvalue": float(pvalue),
        "mean_diff": float(mean),
        "n_dates": float(n),
    }


def return_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    horizon: int = 21,
) -> dict[str, float]:
    """Pooled error metrics plus cross-sectional rank information."""
    a, b = _aligned(y_true, y_pred)
    if a.empty:
        return {
            "n_obs": 0.0,
            "n_dates": 0.0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "rank_ic": float("nan"),
            "rank_ic_t": float("nan"),
            "rank_ic_positive_share": float("nan"),
            "hit_rate": float("nan"),
            "dispersion": float("nan"),
        }
    ic = per_date_rank_ic(a, b)
    mean_ic, tstat, n_dates = mean_tstat_nw(ic, lags=max(int(horizon) - 1, 0))
    pred = b.to_numpy()
    dispersion = float(np.std(pred, ddof=0))
    # A forecast can vary across dates (because each fold has its own intercept)
    # yet be constant *within* every date, which is what matters for rank IC.
    per_date_unique = b.groupby(level="date").nunique()
    constant = bool(len(per_date_unique) and int(per_date_unique.max()) <= 1)
    constant = constant or dispersion < 1e-12
    # A constant forecast has no cross-sectional ranking. Report IC as exactly
    # 0.0 (rather than dropping the model) and mark the t-statistic undefined.
    if constant:
        mean_ic, tstat = 0.0, float("nan")
    return {
        "n_obs": float(len(a)),
        "n_dates": float(n_dates),
        "rmse": float(rmse(a, b)),
        "mae": float(mae(a, b)),
        "rank_ic": float(mean_ic),
        "rank_ic_t": float(tstat),
        "rank_ic_positive_share": float(np.mean(ic.to_numpy() > 0)) if len(ic) else float("nan"),
        "hit_rate": float(hit_rate(a, b)),
        # Cross-sectional dispersion of the forecast: a degenerate (constant)
        # forecaster has zero dispersion and an undefined rank IC.
        "dispersion": dispersion,
        "constant": float(1.0 if constant else 0.0),
    }


def volatility_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    a, b = _aligned(y_true, y_pred)
    if a.empty:
        return {
            "n_obs": 0.0,
            "n_dates": 0.0,
            "qlike": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
        }
    return {
        "n_obs": float(len(a)),
        "n_dates": float(a.index.get_level_values("date").nunique()),
        "qlike": qlike_variance(a, b),
        "mae": float(mae(a, b)),
        "rmse": float(rmse(a, b)),
    }
