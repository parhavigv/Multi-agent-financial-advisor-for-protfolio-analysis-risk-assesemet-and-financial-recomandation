"""Constrained portfolio optimization.

Implements the quantitative core the survey calls for: mean-variance,
minimum-variance, risk parity and minimum-CVaR, all under the same practical
constraint set (long-only, position caps, sector caps, exclusions, turnover
regularisation).

Solvers are deterministic. If SLSQP fails to converge, the caller receives an
explicit ``solver_status`` and a feasible inverse-volatility fallback rather
than an unchecked weight vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from intellivest.contracts import OptimizationResult
from intellivest.quant.risk import historical_cvar, portfolio_returns


@dataclass(frozen=True)
class Constraints:
    max_weight: float = 0.30
    max_sector_weight: float = 0.45
    min_weight: float = 0.0
    turnover_penalty: float = 0.002
    risk_aversion: float = 2.5
    # Ridge penalty toward a neutral diversified prior. Estimation error makes
    # the unregularised mean-variance solution corner-heavy; this is the
    # weight-space analogue of the covariance shrinkage used above.
    diversification: float = 1.0

    def caps_for(self, tickers: Sequence[str]) -> np.ndarray:
        return np.full(len(tickers), self.max_weight, dtype=float)


def _sector_matrix(tickers: Sequence[str], sector_of) -> dict[str, np.ndarray]:
    groups: dict[str, np.ndarray] = {}
    for i, ticker in enumerate(tickers):
        sector = sector_of(ticker)
        if sector not in groups:
            groups[sector] = np.zeros(len(tickers))
        groups[sector][i] = 1.0
    return groups


def project_to_caps(weights: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Project weights onto {sum=1, 0 <= w <= cap} by iterative clipping."""
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    if w.sum() <= 0:
        w = np.full_like(w, 1.0 / len(w))
    w = w / w.sum()
    caps = np.asarray(caps, dtype=float)
    for _ in range(200):
        over = w > caps + 1e-12
        if not over.any():
            break
        excess = float((w[over] - caps[over]).sum())
        w[over] = caps[over]
        under = (~over) & (w < caps - 1e-12)
        if not under.any() or excess <= 0:
            break
        capacity = float((caps[under] - w[under]).sum())
        if capacity <= 1e-12:
            break
        w[under] += excess * (caps[under] - w[under]) / capacity
    total = w.sum()
    return w / total if total > 0 else np.full_like(w, 1.0 / len(w))


def inverse_volatility(cov: np.ndarray) -> np.ndarray:
    vols = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    inv = 1.0 / vols
    return inv / inv.sum()


def _group_penalty(w: np.ndarray, groups: dict[str, np.ndarray], limit: float) -> float:
    penalty = 0.0
    for vector in groups.values():
        exposure = float(vector @ w)
        if exposure > limit:
            penalty += (exposure - limit) ** 2
    return penalty


def _sector_ok(
    weights: np.ndarray,
    groups: dict[str, np.ndarray],
    limit: float,
    tol: float = 1e-8,
) -> bool:
    return all(float(vector @ weights) <= limit + tol for vector in groups.values())


def _feasible_start(
    target: np.ndarray,
    caps: np.ndarray,
    groups: dict[str, np.ndarray],
    limit: float,
) -> np.ndarray:
    """Project a target onto sum-to-one, position caps and sector caps.

    The iterative cap projection is exact for the position-cap problem, but it
    does not know about sector limits. When the projected target breaches a
    sector cap, solve a small feasibility LP so the fallback path remains
    feasible whenever the constraint set itself is feasible.
    """
    candidate = project_to_caps(target, caps)
    if not groups or _sector_ok(candidate, groups, limit):
        return candidate
    if sum(caps) < 1.0 - 1e-9:
        return candidate
    matrix = np.vstack(list(groups.values()))
    result = linprog(
        -np.asarray(target, dtype=float),
        A_ub=matrix,
        b_ub=np.full(len(groups), float(limit)),
        A_eq=np.ones((1, len(candidate))),
        b_eq=[1.0],
        bounds=[(0.0, float(cap)) for cap in caps],
        method="highs",
    )
    if not result.success:
        return candidate
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, caps)
    total = float(weights.sum())
    if total <= 0:
        return candidate
    weights = weights / total
    return weights if _sector_ok(weights, groups, limit, tol=1e-7) else candidate


def _solve(
    objective,
    n: int,
    caps: np.ndarray,
    *,
    x0: np.ndarray,
    extra_constraints: list[dict] | None = None,
    fallback: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    constraints: list[dict] = [
        {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0), "jac": lambda w: np.ones(n)}
    ]
    if extra_constraints:
        constraints.extend(extra_constraints)
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(0.0, float(c)) for c in caps],
        constraints=constraints,
        options={"maxiter": 400, "ftol": 1e-10},
    )
    if result.success:
        weights = np.clip(result.x, 0.0, caps)
        total = weights.sum()
        return (weights / total if total > 0 else inverse_volatility(np.eye(n))), "optimal"
    return (fallback if fallback is not None else x0), f"fallback:{result.message[:80]}"


def _result(
    method: str,
    tickers: Sequence[str],
    weights: np.ndarray,
    mu: pd.Series,
    cov: pd.DataFrame,
    previous: Mapping[str, float] | None,
    status: str,
    binding: list[str] | None = None,
    *,
    risk_free: float = 0.0,
) -> OptimizationResult:
    w = pd.Series(weights, index=list(tickers))
    expected_return = float(w @ mu.reindex(tickers).fillna(0.0))
    variance = float(w.to_numpy() @ cov.loc[tickers, tickers].to_numpy() @ w.to_numpy())
    volatility = float(np.sqrt(max(variance, 0.0)))
    previous = previous or {}
    turnover = float(
        sum(abs(w.get(t, 0.0) - previous.get(t, 0.0)) for t in tickers) / 2.0
    )
    weights_dict = {t: float(v) for t, v in w.items() if v > 1e-6}
    total = sum(weights_dict.values())
    weights_dict = {t: v / total for t, v in weights_dict.items()}
    return OptimizationResult(
        method=method,
        weights=weights_dict,
        expected_return=expected_return,
        expected_volatility=volatility,
        expected_sharpe=(
            (expected_return - float(risk_free)) / volatility
            if volatility > 1e-9
            else 0.0
        ),
        expected_cvar_95=0.0,
        turnover=turnover,
        objective=expected_return - 0.5 * variance,
        solver_status=status,
        binding_constraints=binding or [],
    )


def _binding_constraints(
    weights: np.ndarray,
    caps: np.ndarray,
    groups: dict[str, np.ndarray],
    limit: float,
    tickers: Sequence[str],
    tol: float = 1e-4,
) -> list[str]:
    result = [f"weight_cap:{tickers[i]}" for i in np.where(weights >= caps - tol)[0] if caps[i] > 0]
    for name, vector in groups.items():
        if float(vector @ weights) >= limit - tol:
            result.append(f"sector_cap:{name}")
    return result


def mean_variance(
    expected: pd.Series,
    cov: pd.DataFrame,
    *,
    previous_weights: Mapping[str, float] | None = None,
    constraints: Constraints | None = None,
    sector_of=None,
    exclude: set[str] | None = None,
    risk_free: float = 0.0,
) -> OptimizationResult:
    constraints = constraints or Constraints()
    tickers = [t for t in cov.columns if t in expected.index]
    exclude = exclude or set()
    caps = np.array([0.0 if t in exclude else constraints.max_weight for t in tickers])
    mu = expected.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
    sigma = cov.loc[tickers, tickers].to_numpy(dtype=float)
    groups = _sector_matrix(tickers, sector_of) if sector_of else {}
    prev = np.array([(previous_weights or {}).get(t, 0.0) for t in tickers])
    prior = np.ones(len(tickers), dtype=float) / len(tickers)

    def objective(w: np.ndarray) -> float:
        variance = float(w @ sigma @ w)
        turnover = float(((w - prev) ** 2).sum())
        concentration = float(((w - prior) ** 2).sum())
        return (
            -float(w @ mu)
            + 0.5 * constraints.risk_aversion * variance
            + constraints.turnover_penalty * turnover
            + constraints.diversification * concentration
            + 1000.0 * _group_penalty(w, groups, constraints.max_sector_weight)
        )

    start = _feasible_start(
        inverse_volatility(sigma), caps, groups, constraints.max_sector_weight
    )
    group_constraints = [
        {
            "type": "ineq",
            "fun": (lambda w, vec=vec: float(constraints.max_sector_weight - vec @ w)),
        }
        for vec in groups.values()
    ]
    weights, status = _solve(
        objective,
        len(tickers),
        caps,
        x0=start,
        extra_constraints=group_constraints,
        fallback=start,
    )
    weights = project_to_caps(weights, caps)
    return _result(
        "mean_variance",
        tickers,
        weights,
        expected,
        cov,
        previous_weights,
        status,
        _binding_constraints(weights, caps, groups, constraints.max_sector_weight, tickers),
        risk_free=risk_free,
    )


def min_variance(
    cov: pd.DataFrame,
    *,
    previous_weights: Mapping[str, float] | None = None,
    constraints: Constraints | None = None,
    sector_of=None,
    exclude: set[str] | None = None,
    risk_free: float = 0.0,
) -> OptimizationResult:
    constraints = constraints or Constraints()
    tickers = list(cov.columns)
    exclude = exclude or set()
    caps = np.array([0.0 if t in exclude else constraints.max_weight for t in tickers])
    sigma = cov.to_numpy(dtype=float)
    groups = _sector_matrix(tickers, sector_of) if sector_of else {}
    zero_mu = pd.Series(0.0, index=tickers)

    def objective(w: np.ndarray) -> float:
        return float(w @ sigma @ w) + 1000.0 * _group_penalty(w, groups, constraints.max_sector_weight)

    start = _feasible_start(
        inverse_volatility(sigma), caps, groups, constraints.max_sector_weight
    )
    group_constraints = [
        {"type": "ineq", "fun": (lambda w, vec=vec: float(constraints.max_sector_weight - vec @ w))}
        for vec in groups.values()
    ]
    weights, status = _solve(
        objective,
        len(tickers),
        caps,
        x0=start,
        extra_constraints=group_constraints,
        fallback=start,
    )
    weights = project_to_caps(weights, caps)
    return _result(
        "min_variance",
        tickers,
        weights,
        zero_mu,
        cov,
        previous_weights,
        status,
        _binding_constraints(weights, caps, groups, constraints.max_sector_weight, tickers),
        risk_free=risk_free,
    )


def risk_parity(
    cov: pd.DataFrame,
    *,
    previous_weights: Mapping[str, float] | None = None,
    constraints: Constraints | None = None,
    exclude: set[str] | None = None,
    risk_free: float = 0.0,
) -> OptimizationResult:
    constraints = constraints or Constraints()
    tickers = list(cov.columns)
    exclude = exclude or set()
    caps = np.array([0.0 if t in exclude else constraints.max_weight for t in tickers])
    sigma = cov.to_numpy(dtype=float)
    n = len(tickers)
    zero_mu = pd.Series(0.0, index=tickers)

    def objective(w: np.ndarray) -> float:
        variance = float(w @ sigma @ w)
        if variance <= 0:
            return 1e6
        return 0.5 * variance - (1.0 / n) * float(np.sum(np.log(np.clip(w, 1e-9, None))))

    start = project_to_caps(inverse_volatility(sigma), caps)
    weights, status = _solve(objective, n, caps, x0=start, fallback=start)
    weights = project_to_caps(weights, caps)
    return _result(
        "risk_parity",
        tickers,
        weights,
        zero_mu,
        cov,
        previous_weights,
        status,
        risk_free=risk_free,
    )


def min_cvar(
    returns: pd.DataFrame,
    cov: pd.DataFrame,
    *,
    previous_weights: Mapping[str, float] | None = None,
    constraints: Constraints | None = None,
    sector_of=None,
    exclude: set[str] | None = None,
    alpha: float = 0.95,
    risk_free: float = 0.0,
) -> OptimizationResult:
    """Rockafellar–Uryasev LP formulation of minimum CVaR."""
    constraints = constraints or Constraints()
    tickers = [t for t in cov.columns if t in returns.columns]
    exclude = exclude or set()
    caps = np.array([0.0 if t in exclude else constraints.max_weight for t in tickers])
    matrix = returns[tickers].dropna(how="all").fillna(0.0).to_numpy(dtype=float)
    n_obs, n_assets = matrix.shape
    if n_obs < 30:
        return min_variance(
            cov,
            previous_weights=previous_weights,
            constraints=constraints,
            sector_of=sector_of,
            exclude=exclude,
            risk_free=risk_free,
        )
    groups = _sector_matrix(tickers, sector_of) if sector_of else {}
    zero_mu = pd.Series(0.0, index=tickers)

    # Variables: w (n), zeta (1), u (n_obs)
    c = np.concatenate([np.zeros(n_assets), [1.0], np.full(n_obs, 1.0 / (n_obs * (1 - alpha)))])
    a_ub = np.zeros((n_obs, n_assets + 1 + n_obs))
    a_ub[:, :n_assets] = -matrix
    a_ub[:, n_assets] = -1.0
    a_ub[:, n_assets + 1:] = -np.eye(n_obs)
    b_ub = np.zeros(n_obs)

    if groups:
        group_rows = np.zeros((len(groups), n_assets + 1 + n_obs))
        for i, vec in enumerate(groups.values()):
            group_rows[i, :n_assets] = vec
        a_ub = np.vstack([a_ub, group_rows])
        b_ub = np.concatenate([b_ub, np.full(len(groups), constraints.max_sector_weight)])

    a_eq = np.zeros((1, n_assets + 1 + n_obs))
    a_eq[0, :n_assets] = 1.0
    bounds = [(0.0, float(cap)) for cap in caps] + [(None, None)] + [(0.0, None)] * n_obs

    result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[1.0], bounds=bounds, method="highs")
    if result.success:
        weights = project_to_caps(result.x[:n_assets], caps)
        status = "optimal"
    else:
        weights = _feasible_start(
            inverse_volatility(cov.loc[tickers, tickers].to_numpy()),
            caps,
            groups,
            constraints.max_sector_weight,
        )
        status = f"fallback:{result.message[:80]}"
    return _result(
        "min_cvar",
        tickers,
        weights,
        zero_mu,
        cov,
        previous_weights,
        status,
        _binding_constraints(weights, caps, groups, constraints.max_sector_weight, tickers),
        risk_free=risk_free,
    )


def with_cvar(result: OptimizationResult, returns: pd.DataFrame) -> OptimizationResult:
    """Attach the realised CVaR of the resulting weights (estimation step)."""
    cvar = historical_cvar(portfolio_returns(result.weights, returns), 0.95)
    return result.model_copy(update={"expected_cvar_95": cvar})


def annotate(
    result: OptimizationResult,
    expected: pd.Series,
    returns: pd.DataFrame,
    risk_free: float = 0.0,
) -> OptimizationResult:
    """Recompute return/risk attributes for any weight vector.

    Used after suitability repair, when the final weights are no longer the
    optimizer's own output.
    """
    weights = pd.Series(result.weights, dtype=float)
    mu = float((weights * expected.reindex(weights.index).fillna(0.0)).sum())
    series = portfolio_returns(result.weights, returns)
    cvar = historical_cvar(series, 0.95)
    sharpe = (mu - risk_free) / result.expected_volatility if result.expected_volatility > 1e-9 else 0.0
    return result.model_copy(
        update={"expected_return": mu, "expected_sharpe": sharpe, "expected_cvar_95": cvar}
    )


def portfolio_volatility(weights: Mapping[str, float], cov: pd.DataFrame) -> float:
    tickers = [t for t in weights if t in cov.columns]
    if not tickers:
        return 0.0
    w = np.array([weights[t] for t in tickers])
    sigma = cov.loc[tickers, tickers].to_numpy()
    return float(np.sqrt(max(w @ sigma @ w, 0.0)))


def portfolio_beta(weights: Mapping[str, float], betas: Mapping[str, float]) -> float:
    return float(sum(weights.get(t, 0.0) * betas.get(t, 0.0) for t in weights))


__all__ = [
    "Constraints",
    "annotate",
    "inverse_volatility",
    "mean_variance",
    "min_cvar",
    "min_variance",
    "portfolio_beta",
    "portfolio_volatility",
    "project_to_caps",
    "risk_parity",
    "with_cvar",
]
