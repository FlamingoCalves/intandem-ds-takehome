"""v1 naive churn, v2 segment shrunken net-ATE, v3 cross-fitted T-learner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .allocation import allocate_greedy, allocate_lambda, allocate_naive_churn, net_value_matrix
from .config import ARM_COSTS, BUDGET, RANDOM_SEED
from .data import FEATURE_COLS, make_preprocessor


@dataclass
class PolicyBundle:
    name: str
    arms: np.ndarray
    uplift_score: np.ndarray
    tau: np.ndarray  # (n, 3) retention uplift arms 1..3
    values: np.ndarray  # (n, 3) net values
    meta: dict


def ranking_uplift_score(tau: np.ndarray, annual_value: np.ndarray) -> np.ndarray:
    """Scalar score for holdout_scores.csv / Qini ranking.

    Uses best-arm expected incremental revenue ``max_a τ_a * AV`` (no cost
    subtraction). Allocation still optimizes net value ``τ*AV - cost`` under the
    budget; ranking quality is a separate deliverable.
    """
    av = np.asarray(annual_value, dtype=float).reshape(-1, 1)
    return np.max(np.asarray(tau, dtype=float) * av, axis=1)


def _churn_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("pre", make_preprocessor()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=5,
                    learning_rate=0.06,
                    max_iter=150,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def fit_v1_naive(
    train: pd.DataFrame,
    target: pd.DataFrame,
    *,
    budget: float = BUDGET,
) -> PolicyBundle:
    """Control-only churn model → big offers to high-risk users."""
    ctrl = train[train["offer_arm"] == 0]
    pipe = _churn_pipeline()
    pipe.fit(ctrl[FEATURE_COLS], ctrl["churned"])
    churn_score = pipe.predict_proba(target[FEATURE_COLS])[:, 1]
    arms = allocate_naive_churn(churn_score, budget=budget)

    # For Qini / scores: high churn risk ranks first (the trap ranking).
    # Fake tau: not causal — zeros except using -churn as "uplift" nonsense for SD count.
    tau = np.tile(churn_score.reshape(-1, 1), (1, 3))
    values = net_value_matrix(tau, target["annual_value"].to_numpy())
    return PolicyBundle(
        name="v1_naive_churn",
        arms=arms,
        uplift_score=churn_score,
        tau=tau,
        values=values,
        meta={"model": "control_only_histgbdt_churn"},
    )


def _assign_segments(df: pd.DataFrame, tenure_bins: np.ndarray, activity_med: float) -> pd.Series:
    tenure_q = pd.cut(
        df["tenure_months"],
        bins=tenure_bins,
        labels=["T1", "T2", "T3", "T4"],
        include_lowest=True,
    )
    activity = np.where(df["active_days_30d"] >= activity_med, "HiAct", "LoAct")
    return pd.Series([f"{t}_{a}" for t, a in zip(tenure_q.astype(str), activity)], index=df.index)


def fit_segment_table(train: pd.DataFrame, shrink: float = 0.35) -> tuple[dict, dict]:
    """Shrunken segment×arm retention ATE and net value per dollar of mean AV."""
    tenure_bins = np.quantile(train["tenure_months"], [0, 0.25, 0.5, 0.75, 1.0])
    # Ensure unique bin edges
    tenure_bins = np.unique(tenure_bins)
    if len(tenure_bins) < 3:
        tenure_bins = np.array(
            [
                train["tenure_months"].min() - 1e-6,
                train["tenure_months"].median(),
                train["tenure_months"].max() + 1e-6,
            ]
        )
    activity_med = float(train["active_days_30d"].median())
    seg = _assign_segments(train, tenure_bins, activity_med)
    d = train.copy()
    d["segment"] = seg.values

    global_ate = {}
    ctrl_rate = d.loc[d["offer_arm"] == 0, "churned"].mean()
    for arm in (1, 2, 3):
        global_ate[arm] = float(ctrl_rate - d.loc[d["offer_arm"] == arm, "churned"].mean())

    table = {}
    for segment, g in d.groupby("segment"):
        rates = g.groupby("offer_arm")["churned"].mean()
        av_mean = float(g["annual_value"].mean())
        if 0 not in rates.index:
            continue
        c = float(rates.loc[0])
        entry = {}
        for arm in (1, 2, 3):
            if arm not in rates.index:
                raw = global_ate[arm]
            else:
                raw = float(c - rates.loc[arm])
            n_arm = int((g["offer_arm"] == arm).sum())
            # Shrink toward global ATE; more shrink when cell is small.
            w = n_arm / (n_arm + 200.0)
            shrunk = w * raw + (1 - w) * global_ate[arm]
            # Extra global blend for stability.
            shrunk = (1 - shrink) * shrunk + shrink * global_ate[arm]
            net = shrunk * av_mean - ARM_COSTS[arm]
            entry[arm] = {"tau": shrunk, "net": net, "n": n_arm}
        table[str(segment)] = entry

    meta = {"tenure_bins": tenure_bins, "activity_med": activity_med, "global_ate": global_ate}
    return table, meta


def fit_v2_segments(
    train: pd.DataFrame,
    target: pd.DataFrame,
    *,
    budget: float = BUDGET,
) -> PolicyBundle:
    table, meta = fit_segment_table(train)
    seg = _assign_segments(target, meta["tenure_bins"], meta["activity_med"])

    n = len(target)
    tau = np.zeros((n, 3))
    values = np.zeros((n, 3))
    for i, s in enumerate(seg.astype(str)):
        cell = table.get(s)
        if cell is None:
            # fall back to global
            for j, arm in enumerate((1, 2, 3)):
                t = meta["global_ate"][arm]
                tau[i, j] = t
                values[i, j] = t * target.iloc[i]["annual_value"] - ARM_COSTS[arm]
        else:
            for j, arm in enumerate((1, 2, 3)):
                tau[i, j] = cell[arm]["tau"]
                # Use user-level AV for allocation economics.
                values[i, j] = cell[arm]["tau"] * target.iloc[i]["annual_value"] - ARM_COSTS[arm]

    arms, lam = allocate_lambda(values, budget=budget)
    uplift_score = ranking_uplift_score(tau, target["annual_value"].to_numpy())
    return PolicyBundle(
        name="v2_segment_shrunk_ate",
        arms=arms,
        uplift_score=uplift_score,
        tau=tau,
        values=values,
        meta={**meta, "lambda": lam, "segment_table_size": len(table)},
    )


def _crossfit_tlearner_tau(train: pd.DataFrame, n_splits: int = 4) -> np.ndarray:
    """Out-of-fold retention uplift τ_a = p0 - pa for arms 1..3 on train rows."""
    n = len(train)
    oof_p = {a: np.zeros(n) for a in range(4)}
    y = train["churned"].to_numpy()
    # Stratify on arm*2+churn for balance.
    strat = train["offer_arm"].to_numpy() * 2 + y
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    for fold, (tr_idx, va_idx) in enumerate(kfold.split(train, strat)):
        tr = train.iloc[tr_idx]
        va = train.iloc[va_idx]
        models = {}
        for arm in range(4):
            sub = tr[tr["offer_arm"] == arm]
            pipe = _churn_pipeline()
            pipe.fit(sub[FEATURE_COLS], sub["churned"])
            models[arm] = pipe
        for arm in range(4):
            oof_p[arm][va_idx] = models[arm].predict_proba(va[FEATURE_COLS])[:, 1]

    tau = np.column_stack([oof_p[0] - oof_p[a] for a in (1, 2, 3)])
    return tau


def _shrink_tau(tau: np.ndarray, global_ate: np.ndarray, shrink: float = 0.4) -> np.ndarray:
    # Clip extremes then blend toward global ATE.
    lo = np.quantile(tau, 0.01, axis=0)
    hi = np.quantile(tau, 0.99, axis=0)
    clipped = np.clip(tau, lo, hi)
    return (1 - shrink) * clipped + shrink * global_ate.reshape(1, -1)


def fit_v3_tlearner(
    train: pd.DataFrame,
    target: pd.DataFrame,
    *,
    budget: float = BUDGET,
    fit_full_for_target: bool = True,
) -> PolicyBundle:
    """Cross-fitted T-learner with shrinkage + λ allocation."""
    ctrl_rate = train.loc[train["offer_arm"] == 0, "churned"].mean()
    global_ate = np.array(
        [
            ctrl_rate - train.loc[train["offer_arm"] == a, "churned"].mean()
            for a in (1, 2, 3)
        ]
    )

    # OOF on train for diagnostics; for target scoring, refit on full train.
    models = {}
    for arm in range(4):
        sub = train[train["offer_arm"] == arm]
        pipe = _churn_pipeline()
        pipe.fit(sub[FEATURE_COLS], sub["churned"])
        models[arm] = pipe

    p = {a: models[a].predict_proba(target[FEATURE_COLS])[:, 1] for a in range(4)}
    tau = np.column_stack([p[0] - p[a] for a in (1, 2, 3)])
    tau = _shrink_tau(tau, global_ate, shrink=0.45)
    values = net_value_matrix(tau, target["annual_value"].to_numpy())
    arms, lam = allocate_lambda(values, budget=budget)
    uplift_score = ranking_uplift_score(tau, target["annual_value"].to_numpy())

    # Also compute OOF train tau for optional validation reporting.
    oof_tau = _crossfit_tlearner_tau(train) if fit_full_for_target else None

    return PolicyBundle(
        name="v3_tlearner_lambda",
        arms=arms,
        uplift_score=uplift_score,
        tau=tau,
        values=values,
        meta={
            "lambda": lam,
            "global_ate": global_ate.tolist(),
            "oof_tau_mean": oof_tau.mean(axis=0).tolist() if oof_tau is not None else None,
            "uplift_score_def": "max_a tau_a * annual_value",
        },
    )


def compare_greedy(values: np.ndarray, *, budget: float = BUDGET) -> np.ndarray:
    arms, _ = allocate_greedy(values, budget=budget)
    return arms
