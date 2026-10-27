"""Moving-block bootstrap for serially dependent return series.

Daily portfolio returns are not iid. The moving-block bootstrap resamples
contiguous blocks (default: one trading month) so serial correlation and
volatility clustering survive the resampling. Confidence intervals are
percentile intervals. The paired test resamples the *same* block indices for
both strategies, preserving the cross-strategy dependence that a paired
comparison requires.

The p-value is a two-sided percentile bootstrap p-value. It is not a
finite-sample exact test; it is reported as an uncertainty diagnostic alongside
the interval, not as proof of a trading edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    low: float
    high: float
    n_boot: int
    block_length: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "point": round(self.point, 6),
            "low": round(self.low, 6),
            "high": round(self.high, 6),
            "n_boot": self.n_boot,
            "block_length": self.block_length,
        }


@dataclass(frozen=True)
class PairedBootstrap:
    metric: str
    difference: float
    low: float
    high: float
    p_value: float
    n_boot: int
    block_length: int

    def to_dict(self) -> dict[str, float | str | int]:
        return {
            "metric": self.metric,
            "difference": round(self.difference, 6),
            "low": round(self.low, 6),
            "high": round(self.high, 6),
            "p_value": round(self.p_value, 6),
            "n_boot": self.n_boot,
            "block_length": self.block_length,
        }


def moving_block_indices(
    n: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n`` indices by concatenating sampled contiguous blocks.

    Start positions are sampled uniformly from ``0 .. n-block_length``, so the
    resampled path contains only observations that existed in the original
    sample. The final block is truncated to the required length.
    """
    if n <= 0:
        return np.empty(0, dtype=int)
    length = max(1, int(block_length))
    if length >= n or length == 1:
        return np.arange(n, dtype=int)
    n_blocks = int(np.ceil(n / length))
    starts = rng.integers(0, n - length + 1, size=n_blocks)
    pieces = [np.arange(start, start + length, dtype=int) for start in starts]
    return np.concatenate(pieces)[:n]


def bootstrap_ci(
    values: np.ndarray | list[float],
    metric: Callable[[np.ndarray], float],
    *,
    block_length: int = 21,
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> BootstrapCI:
    """Percentile moving-block bootstrap CI for one return path."""
    array = np.asarray(values, dtype=float).ravel()
    array = array[np.isfinite(array)]
    point = float(metric(array)) if array.size else 0.0
    if array.size < 2 or n_boot <= 0:
        return BootstrapCI(point, point, point, max(0, n_boot), block_length)

    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_boot), dtype=float)
    for index in range(int(n_boot)):
        sample = array[moving_block_indices(array.size, block_length, rng)]
        draws[index] = float(metric(sample))
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return BootstrapCI(point, point, point, int(n_boot), block_length)
    alpha = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return BootstrapCI(point, float(low), float(high), int(n_boot), block_length)


def paired_bootstrap_test(
    left: np.ndarray | list[float],
    right: np.ndarray | list[float],
    metric: Callable[[np.ndarray], float],
    *,
    metric_name: str,
    block_length: int = 21,
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> PairedBootstrap:
    """Paired moving-block bootstrap for ``metric(left) - metric(right)``.

    The same block indices are applied to both paths on every draw, so the
    comparison respects the fact that both strategies traded the same days.
    """
    a = np.asarray(left, dtype=float).ravel()
    b = np.asarray(right, dtype=float).ravel()
    size = min(a.size, b.size)
    a, b = a[:size], b[:size]
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    difference = float(metric(a) - metric(b)) if a.size else 0.0
    if a.size < 2 or n_boot <= 0:
        return PairedBootstrap(
            metric_name, difference, difference, difference, 1.0, max(0, n_boot), block_length
        )

    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_boot), dtype=float)
    for index in range(int(n_boot)):
        indices = moving_block_indices(a.size, block_length, rng)
        draws[index] = float(metric(a[indices]) - metric(b[indices]))
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return PairedBootstrap(
            metric_name, difference, difference, difference, 1.0, int(n_boot), block_length
        )
    alpha = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    # Two-sided percentile mass opposite the observed difference. The +1
    # correction prevents a zero p-value from a finite bootstrap sample.
    below = float(np.sum(draws <= 0.0))
    above = float(np.sum(draws >= 0.0))
    p_value = min(
        1.0,
        2.0
        * min(
            (below + 1.0) / (draws.size + 1.0),
            (above + 1.0) / (draws.size + 1.0),
        ),
    )
    return PairedBootstrap(
        metric_name,
        difference,
        float(low),
        float(high),
        float(p_value),
        int(n_boot),
        block_length,
    )


__all__ = [
    "BootstrapCI",
    "PairedBootstrap",
    "bootstrap_ci",
    "moving_block_indices",
    "paired_bootstrap_test",
]
