"""HMM/regime detection using only observable data."""

from __future__ import annotations

import numpy as np
import pandas as pd


class RegimeDetector:
    def __init__(self, n_states: int = 2, random_state: int = 12345):
        self.n_states = n_states
        self.random_state = random_state
        self.model = None
        self.fitted = False

    def fit_predict(self, returns: pd.Series) -> pd.Series:
        """Fit on observable returns only - no hidden ground truth."""
        r = returns.dropna().fillna(0.0)
        if len(r) < 60:
            # fallback to rule-based
            vol = r.rolling(21).std(ddof=0).fillna(r.std(ddof=0))
            thresh = vol.median()
            regime = (vol > thresh).astype(int)
            return pd.Series(regime.values, index=r.index, name="regime")
        try:
            from hmmlearn.hmm import GaussianHMM

            X = np.column_stack([r.values, (r**2).values])
            # select by BIC simple
            best_bic = np.inf
            best_model = None
            for k in [2, 3]:
                try:
                    m = GaussianHMM(n_components=k, n_iter=100, random_state=self.random_state)
                    m.fit(X)
                    logL = m.score(X)
                    p = k * (2 * X.shape[1] + X.shape[1] * (X.shape[1] + 1) / 2) + k * (k - 1)
                    bic = -2 * logL + p * np.log(len(X))
                    if bic < best_bic:
                        best_bic = bic
                        best_model = m
                        self.n_states = k
                except Exception:
                    continue
            if best_model is None:
                raise ValueError("hmm failed")
            states = best_model.predict(X)
            self.model = best_model
            self.fitted = True
            return pd.Series(states, index=r.index, name="regime")
        except Exception:
            vol = r.rolling(21).std(ddof=0).fillna(r.std(ddof=0))
            thresh = vol.median()
            regime = (vol > thresh).astype(int)
            return pd.Series(regime.values, index=r.index, name="regime")

    def predict_next(self, returns: pd.Series) -> int:
        """Forward prediction using fitted model on observed history only."""
        if not self.fitted or self.model is None:
            pred = self.fit_predict(returns)
            return int(pred.iloc[-1]) if len(pred) else 0
        r = returns.dropna().fillna(0.0)
        try:
            X = np.column_stack([r.values, (r**2).values])
            states = self.model.predict(X)
            return int(states[-1])
        except Exception:
            pred = self.fit_predict(returns)
            return int(pred.iloc[-1]) if len(pred) else 0
