"""Budget allocation: λ-search (MCKP LP relaxation) and greedy V/cost baseline."""

from __future__ import annotations

import numpy as np

from .config import ARM_COSTS


def net_value_matrix(tau: np.ndarray, annual_value: np.ndarray) -> np.ndarray:
    """tau: (n, 3) retention uplift for arms 1..3; returns V for arms 1..3."""
    av = annual_value.reshape(-1, 1)
    costs = np.array([ARM_COSTS[1], ARM_COSTS[2], ARM_COSTS[3]], dtype=float)
    return tau * av - costs


def allocate_lambda(
    value_scores: np.ndarray,
    candidate_arms: np.ndarray | None = None,
    costs: np.ndarray | None = None,
    *,
    budget: float,
    tol: float = 1.0,
    max_iter: int = 60,
) -> tuple[np.ndarray, float]:
    """Assign each user argmax_a (V_a - λ cost_a), including a=0 with V=0.

    Parameters
    ----------
    value_scores : (n, 3) net values for arms 1,2,3 (already V = τ*AV - cost)
    candidate_arms : unused legacy; arms inferred as 1..3
    costs : unused if value_scores already net of cost; kept for API compat
    """
    del candidate_arms, costs  # values already net of offer cost
    v = np.asarray(value_scores, dtype=float)
    n = v.shape[0]
    arm_costs = np.array([ARM_COSTS[1], ARM_COSTS[2], ARM_COSTS[3]], dtype=float)

    # Negative net value => never assign (sleeping dogs / non-responders).
    v = np.where(v > 0, v, -np.inf)

    def assign(lam: float) -> tuple[np.ndarray, float]:
        # score_a = V_a - λ * cost_a; control score = 0
        scores = v - lam * arm_costs.reshape(1, -1)
        best_idx = np.argmax(scores, axis=1)
        best_score = scores[np.arange(n), best_idx]
        arms = np.where(best_score > 0, best_idx + 1, 0).astype(int)
        spend = float(sum(ARM_COSTS[int(a)] for a in arms))
        return arms, spend

    # λ = 0 => take all positive V; large λ => take nothing.
    lo, hi = 0.0, 1e3
    arms_hi, spend_hi = assign(hi)
    if spend_hi > budget:
        # Even expensive λ still overspends — rare; fall back to greedy.
        return allocate_greedy(v, budget=budget)

    arms_best, spend_best = assign(0.0)
    if spend_best <= budget:
        return arms_best, 0.0

    best_arms = arms_hi
    best_lam = hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        arms, spend = assign(mid)
        if spend > budget:
            lo = mid
        else:
            hi = mid
            best_arms, best_lam = arms, mid
            if budget - spend <= tol:
                break
    return best_arms, best_lam


def allocate_greedy(value_scores: np.ndarray, *, budget: float) -> tuple[np.ndarray, float]:
    """Secondary baseline: rank users by max positive V/cost, fill until budget."""
    v = np.asarray(value_scores, dtype=float)
    n = v.shape[0]
    arm_costs = np.array([ARM_COSTS[1], ARM_COSTS[2], ARM_COSTS[3]], dtype=float)

    best_idx = np.argmax(v, axis=1)
    best_v = v[np.arange(n), best_idx]
    best_cost = arm_costs[best_idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        density = np.where(best_v > 0, best_v / best_cost, -np.inf)

    order = np.argsort(-density, kind="mergesort")
    arms = np.zeros(n, dtype=int)
    spend = 0.0
    for i in order:
        if not np.isfinite(density[i]) or density[i] <= 0:
            break
        c = float(best_cost[i])
        if spend + c <= budget:
            arms[i] = int(best_idx[i] + 1)
            spend += c
    return arms, spend


def allocate_naive_churn(
    churn_score: np.ndarray,
    *,
    budget: float,
    prefer_arms: tuple[int, ...] = (3, 2, 1),
) -> np.ndarray:
    """Give high-churn users the biggest affordable offers until budget exhausted."""
    n = len(churn_score)
    order = np.argsort(-churn_score, kind="mergesort")
    arms = np.zeros(n, dtype=int)
    spend = 0.0
    for i in order:
        for a in prefer_arms:
            c = ARM_COSTS[a]
            if spend + c <= budget:
                arms[i] = a
                spend += c
                break
    return arms
