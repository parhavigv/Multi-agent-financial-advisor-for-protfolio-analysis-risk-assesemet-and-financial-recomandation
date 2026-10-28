"""Walk-forward evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import returns as rmod
from . import volatility as vmod
from .base import diebold_mariano


@dataclass
class WalkForwardResult:
    metrics: dict[str, float]
    predictions: pd.DataFrame


def expanding_window_splits(n: int, min_train: int = 252, step: int = 63):
    splits = []
    i = min_train
    while i + 21 <= n:
        splits.append((0, i, i, min(i + 21, n)))
        i += step
    return splits
