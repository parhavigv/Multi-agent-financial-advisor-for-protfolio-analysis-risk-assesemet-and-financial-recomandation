"""Hidden-Markov market regimes with causal (filtered) inference.

The threshold rule in :func:`intellivest.quant.risk.detect_regime` is replaced
by a Gaussian HMM.  Two details matter for honesty:

1. The number of states (2 or 3) is chosen by BIC on a **training window only**.
2. Labels come from the *filtered* distribution ``P(state_t | x_1..x_t)``, not
   Viterbi decoding or smoothed probabilities.  Smoothed paths use future
   observations and would leak information into the backtest.

State names are assigned by volatility so they mean the same thing across fits:
the lowest-volatility state is ``calm``, the highest is ``stress`` and a middle
state (when selected) is ``elevated``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REGIME_LABELS = ("calm", "elevated", "stress")


def _design(returns: pd.Series) -> np.ndarray:
    """Two-dimensional emission: signed return and squared return."""
    r = pd.Series(returns).astype(float)
    return np.column_stack([r.to_numpy(), (r.to_numpy() ** 2)])


def _log_gaussian(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Log N(x; mean, diag(var)) for a single observation x."""
    var = np.maximum(var, 1e-12)
    return -0.5 * (
        len(x) * np.log(2.0 * np.pi)
        + np.sum(np.log(var))
        + np.sum((x - mean) ** 2 / var)
    )


def _emission_matrix(X: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    """(T, K) matrix of Gaussian log-emission densities."""
    T = X.shape[0]
    K = means.shape[0]
    out = np.empty((T, K), dtype=float)
    for k in range(K):
        var = np.diag(covars[k]) if covars[k].ndim == 2 else covars[k]
        for t in range(T):
            out[t, k] = _log_gaussian(X[t], means[k], var)
    return out


@dataclass
class HMMRegime:
    """Fitted 2/3-state Gaussian HMM with causal filtering."""

    n_states: int = 2
    random_state: int = 12345
    max_states: int = 3
    model: object | None = None
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None
    means_: np.ndarray | None = None
    covars_: np.ndarray | None = None
    transmat_: np.ndarray | None = None
    startprob_: np.ndarray | None = None
    state_order_: list[int] = field(default_factory=list)
    bic_: float = float("nan")
    trained_on_: pd.Timestamp | None = None
    fallback_: bool = False

    # ------------------------------------------------------------------ fit
    def fit(self, returns: pd.Series, *, max_states: int | None = None) -> "HMMRegime":
        r = pd.Series(returns).dropna().astype(float)
        if len(r) > 0:
            self.trained_on_ = pd.Timestamp(r.index[-1])
        max_states = int(max_states or self.max_states)
        if len(r) < 120:
            self.fallback_ = True
            self.n_states = 2
            return self

        X = _design(r)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-12] = 1.0
        Xs = (X - self.mean_) / self.std_

        try:
            from hmmlearn.hmm import GaussianHMM
        except Exception:  # pragma: no cover - dependency guard
            self.fallback_ = True
            return self

        best = None
        for k in range(2, max_states + 1):
            try:
                model = GaussianHMM(
                    n_components=k,
                    covariance_type="diag",
                    n_iter=300,
                    tol=1e-4,
                    random_state=self.random_state,
                )
                model.fit(Xs)
                log_likelihood = float(model.score(Xs))
                n_params = (
                    k * Xs.shape[1]              # means
                    + k * Xs.shape[1]             # diagonal variances
                    + k * (k - 1)                 # transition rows
                    + (k - 1)                     # initial probabilities
                )
                bic = -2.0 * log_likelihood + n_params * np.log(len(Xs))
                if best is None or bic < best[0]:
                    best = (bic, model)
            except Exception:
                continue

        if best is None:
            self.fallback_ = True
            return self

        self.bic_, self.model = best
        self.n_states = int(self.model.n_components)
        self.means_ = np.asarray(self.model.means_, dtype=float)
        self.covars_ = np.asarray(self.model.covars_, dtype=float)
        self.transmat_ = np.asarray(self.model.transmat_, dtype=float)
        self.startprob_ = np.asarray(self.model.startprob_, dtype=float)
        # hmmlearn stores diagonal covariances as full matrices; the return
        # dimension is the first diagonal entry. Order states by its variance.
        covars = np.asarray(self.covars_, dtype=float)
        if covars.ndim == 3:
            return_var = covars[:, 0, 0]
        else:  # pragma: no cover - future hmmlearn layout
            return_var = covars[:, 0]
        vol = np.sqrt(np.maximum(return_var, 1e-18))
        self.state_order_ = [int(i) for i in np.argsort(vol)]
        self.fallback_ = False
        return self

    # -------------------------------------------------------------- filtering
    def filtered_probabilities(self, returns: pd.Series) -> pd.DataFrame:
        """Causal ``P(state_t | x_1..x_t)`` for each date in ``returns``."""
        r = pd.Series(returns).astype(float)
        if r.empty:
            return pd.DataFrame(columns=REGIME_LABELS[: self.n_states])
        if self.fallback_ or self.model is None:
            return self._fallback_probabilities(r)

        X = _design(r)
        Xs = (X - self.mean_) / self.std_
        log_emit = _emission_matrix(Xs, self.means_, self.covars_)
        # Normalise emissions per date before the forward recursion.
        log_emit -= log_emit.max(axis=1, keepdims=True)

        K = self.n_states
        probs = np.zeros((len(r), K), dtype=float)
        alpha = self.startprob_ * np.exp(log_emit[0])
        alpha /= max(alpha.sum(), 1e-300)
        probs[0] = alpha
        for t in range(1, len(r)):
            predicted = alpha @ self.transmat_
            posterior = predicted * np.exp(log_emit[t])
            total = posterior.sum()
            alpha = posterior / total if total > 1e-300 else predicted
            probs[t] = alpha
        return pd.DataFrame(probs, index=r.index, columns=[f"p_{i}" for i in range(K)])

    def _fallback_probabilities(self, returns: pd.Series) -> pd.DataFrame:
        """Deterministic volatility-threshold fallback, one-hot encoded."""
        r = pd.Series(returns).astype(float)
        vol = r.rolling(21, min_periods=10).std(ddof=0)
        vol = vol.fillna(float(r.std(ddof=0)) if len(r) > 1 else 0.0)
        median = float(vol.median()) if len(vol) else 0.0
        high = (vol > median).to_numpy()
        probs = np.zeros((len(r), 2), dtype=float)
        probs[:, 0] = ~high
        probs[:, 1] = high
        return pd.DataFrame(probs, index=r.index, columns=["p_0", "p_1"])

    def labels(self, returns: pd.Series) -> pd.Series:
        """Filtered regime label per date, in ``calm/elevated/stress`` terms."""
        probs = self.filtered_probabilities(returns)
        if probs.empty:
            return pd.Series(dtype=object)
        order = self.state_order_ or list(range(probs.shape[1]))
        # Map fitted state index -> semantic label position.
        mapping = {
            int(state): REGIME_LABELS[min(pos, len(REGIME_LABELS) - 1)]
            for pos, state in enumerate(order)
        }
        if probs.shape[1] == 2:
            mapping = {int(order[0]): "calm", int(order[1]): "stress"}
        columns = [f"p_{i}" for i in range(probs.shape[1])]
        chosen = probs[columns].to_numpy().argmax(axis=1)
        return pd.Series(
            [mapping.get(int(i), "elevated") for i in chosen],
            index=probs.index,
            name="regime",
        )

    def latest(self, returns: pd.Series) -> tuple[str, float]:
        """Most recent filtered label and its probability."""
        probs = self.filtered_probabilities(returns)
        if probs.empty:
            return "elevated", 0.4
        label = self.labels(returns).iloc[-1]
        confidence = float(probs.iloc[-1].max())
        return str(label), confidence


def infer_regime(
    returns: pd.Series,
    *,
    max_states: int = 3,
    random_state: int = 12345,
    fit_window: int = 756,
    trading_days: int = 252,
) -> tuple[str, float]:
    """Convenience wrapper: fit on the supplied history and return the latest label.

    Callers must pass an already-truncated series (``as_of``), which is how the
    market providers use it. Only the trailing ``fit_window`` observations are
    used, which keeps production snapshots fast and avoids letting a decade of
    stale regimes dominate the fit.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 120:
        from intellivest.quant.risk import detect_regime

        return detect_regime(r, trading_days)
    window = r.iloc[-max(120, int(fit_window)) :]
    regime = HMMRegime(random_state=random_state, max_states=max_states).fit(window)
    if regime.fallback_:
        from intellivest.quant.risk import detect_regime

        return detect_regime(r, trading_days)
    return regime.latest(window)


__all__ = ["HMMRegime", "REGIME_LABELS", "infer_regime"]
