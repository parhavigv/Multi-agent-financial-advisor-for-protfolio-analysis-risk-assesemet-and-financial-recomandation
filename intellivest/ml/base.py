"""Base utilities for ML evaluation - leakage-safe, deterministic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("nan")
    diff = y_true[mask] - y_pred[mask]
    return float(np.sqrt(np.mean(diff**2)))


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def qlike(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """QLIKE for volatility: log(sigma^2) + y^2/sigma^2 (lower is better)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return float("nan")
    sigma2 = np.maximum(y_pred[mask] ** 2, 1e-18)
    y2 = y_true[mask] ** 2
    return float(np.mean(np.log(sigma2) + y2 / sigma2))


def ic(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Pearson correlation (IC)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return float("nan")
    try:
        return float(np.corrcoef(y_true[mask], y_pred[mask])[0, 1])
    except Exception:
        return float("nan")


def rank_ic(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return float("nan")
    try:
        import scipy.stats

        return float(scipy.stats.spearmanr(y_true[mask], y_pred[mask]).correlation)
    except Exception:
        # fallback
        rt = pd.Series(y_true[mask]).rank(pct=True)
        rp = pd.Series(y_pred[mask]).rank(pct=True)
        return float(rt.corr(rp))


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Simplified DM test (asymptotic normal approximation)."""
    e1 = np.asarray(e1, dtype=float).ravel()
    e2 = np.asarray(e2, dtype=float).ravel()
    mask = np.isfinite(e1) & np.isfinite(e2)
    e1, e2 = e1[mask], e2[mask]
    d = e1**2 - e2**2
    if len(d) < 5:
        return float("nan"), float("nan")
    dbar = float(np.mean(d))
    # Newey-West style rough variance
    T = len(d)
    var = float(np.var(d, ddof=1)) / T
    if var <= 0 or not np.isfinite(var):
        return float("nan"), float("nan")
    dm_stat = dbar / math.sqrt(var)
    try:
        from scipy.stats import norm

        pval = 2 * (1 - norm.cdf(abs(dm_stat)))
    except Exception:
        pval = float("nan")
    return dm_stat, pval


def block_bootstrap_diff(e1: np.ndarray, e2: np.ndarray, n_boot: int = 200, block_size: int = 5) -> tuple[float, float]:
    e1 = np.asarray(e1, dtype=float).ravel()
    e2 = np.asarray(e2, dtype=float).ravel()
    mask = np.isfinite(e1) & np.isfinite(e2)
    e1, e2 = e1[mask], e2[mask]
    if len(e1) < block_size * 2:
        return float("nan"), float("nan")
    rng = np.random.RandomState(12345)
    d0 = np.mean(e1**2 - e2**2)
    T = len(e1)
    blocks = int(math.ceil(T / block_size))
    stats = []
    for _ in range(n_boot):
        idx = []
        for _ in range(blocks):
            start = rng.randint(0, max(1, T - block_size))
            idx.extend(range(start, min(start + block_size, T)))
        idx = np.array(idx[:T])
        stats.append(np.mean(e1[idx] ** 2 - e2[idx] ** 2))
    stats = np.array(stats)
    pval = float(np.mean(np.abs(stats) >= abs(d0)))
    return float(d0), pval


def chronological_split(n: int, train_frac: float = 0.6, val_frac: float = 0.2):
    """Chronological split indices."""
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return (0, train_end), (train_end, val_end), (val_end, n)
