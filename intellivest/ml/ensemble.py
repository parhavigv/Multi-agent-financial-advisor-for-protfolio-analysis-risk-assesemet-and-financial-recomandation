"""Ensemble methods with shrinkage."""

from __future__ import annotations

import numpy as np
import pandas as pd


def inverse_error_weights(errors: dict[str, float], eps: float = 1e-12) -> dict[str, float]:
    w = {}
    for k, e in errors.items():
        if not np.isfinite(e) or e <= 0:
            w[k] = 0.0
        else:
            w[k] = 1.0 / (e * e + eps)
    s = sum(w.values())
    if s <= 0:
        return {k: 1.0 / len(errors) for k in errors}
    return {k: v / s for k, v in w.items()}


def shrink_toward_mean(weights: dict[str, float], shrinkage: float = 0.2) -> dict[str, float]:
    if not weights:
        return weights
    m = np.mean(list(weights.values()))
    return {k: (1 - shrinkage) * v + shrinkage * m for k, v in weights.items()}
