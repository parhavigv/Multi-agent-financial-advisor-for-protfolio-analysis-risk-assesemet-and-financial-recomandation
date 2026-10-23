"""Covariance estimation.

Mean-variance optimization is famously sensitive to estimation error [5], [8].
The default is Ledoit–Wolf shrinkage, with a diagonal fallback for very short
samples so the optimizer never receives a non-PSD matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def _annualize(cov: np.ndarray, periods: int) -> np.ndarray:
    return cov * periods


def sample_cov(returns: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    clean = returns.dropna(how="all").fillna(0.0)
    cov = np.cov(clean.to_numpy(dtype=float), rowvar=False, ddof=1)
    return pd.DataFrame(_annualize(np.atleast_2d(cov), periods), index=clean.columns, columns=clean.columns)


def ledoit_wolf_cov(returns: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    clean = returns.dropna(how="all").fillna(0.0)
    if clean.shape[0] < 5:
        return diagonal_cov(returns, periods)
    estimator = LedoitWolf(assume_centered=False)
    estimator.fit(clean.to_numpy(dtype=float))
    cov = _annualize(estimator.covariance_, periods)
    return pd.DataFrame(cov, index=clean.columns, columns=clean.columns)


def ewma_cov(returns: pd.DataFrame, halflife: int = 63, periods: int = 252) -> pd.DataFrame:
    clean = returns.dropna(how="all").fillna(0.0)
    cov = clean.ewm(halflife=halflife, adjust=False).cov().iloc[-len(clean.columns):]
    matrix = cov.to_numpy(dtype=float)
    # ``ewm().cov()`` returns the last cross-covariance block in column order.
    n = len(clean.columns)
    matrix = matrix.reshape(n, n) if matrix.size == n * n else matrix[-n:, -n:]
    return pd.DataFrame(_annualize(matrix, periods), index=clean.columns, columns=clean.columns)


def diagonal_cov(returns: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    clean = returns.dropna(how="all").fillna(0.0)
    variances = clean.var(ddof=1).to_numpy(dtype=float) * periods
    variances = np.where(np.isfinite(variances) & (variances > 1e-10), variances, 1e-6)
    return pd.DataFrame(np.diag(variances), index=clean.columns, columns=clean.columns)


def shrink_to_diagonal(cov: pd.DataFrame, intensity: float = 0.15) -> pd.DataFrame:
    diag = pd.DataFrame(
        np.diag(np.diag(cov.to_numpy(dtype=float))), index=cov.index, columns=cov.columns
    )
    blended = (1.0 - intensity) * cov + intensity * diag
    return make_psd(blended)


def make_psd(cov: pd.DataFrame, floor: float = 1e-8) -> pd.DataFrame:
    matrix = cov.to_numpy(dtype=float)
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, floor, None)
    repaired = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return pd.DataFrame(repaired, index=cov.index, columns=cov.columns)


def estimate_covariance(
    returns: pd.DataFrame, *, method: str = "ledoit_wolf", periods: int = 252
) -> pd.DataFrame:
    if returns.shape[1] == 0:
        raise ValueError("returns frame has no columns")
    if method == "sample":
        cov = sample_cov(returns, periods)
    elif method == "ewma":
        cov = ewma_cov(returns, periods=periods)
    elif method == "diagonal":
        cov = diagonal_cov(returns, periods)
    else:
        cov = ledoit_wolf_cov(returns, periods)
    cov = cov.fillna(0.0)
    return make_psd(cov)


__all__ = [
    "diagonal_cov",
    "estimate_covariance",
    "ewma_cov",
    "ledoit_wolf_cov",
    "make_psd",
    "sample_cov",
    "shrink_to_diagonal",
]
