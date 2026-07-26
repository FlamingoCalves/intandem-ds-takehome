"""Off-policy evaluation, Qini, and policy summary metrics.

Holdout DR/IPW nuisances must be fit on train (or cross-fit within train) only —
never refit on holdout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from .config import ARM_COSTS, PROPENSITY, RANDOM_SEED
from .data import FEATURE_COLS, make_preprocessor


@dataclass
class PolicyEval:
    name: str
    budget: float
    spend: float
    n_treated: int
    arm_mix: dict
    ipw_value: float
    ipw_incremental: float
    dr_value: float
    dr_incremental: float
    qini: float
    sleeping_dogs_contacted: int
    fraction_untreated: float
    qini_value: float = 0.0
    ipw_incremental_total: float = 0.0
    dr_incremental_total: float = 0.0


def assign_cost(arms: np.ndarray) -> np.ndarray:
    return np.vectorize(ARM_COSTS.get)(arms).astype(float)


def ipw_policy_value(
    df: pd.DataFrame,
    policy_arms: np.ndarray,
    *,
    propensity: dict | None = None,
) -> float:
    """IPW estimate of E[R*AV - cost(π)] under known randomization probs."""
    e = propensity or PROPENSITY
    a = df["offer_arm"].to_numpy()
    pi = np.asarray(policy_arms)
    retained = 1.0 - df["churned"].to_numpy()
    av = df["annual_value"].to_numpy()
    cost = assign_cost(pi)
    match = (a == pi).astype(float)
    w = np.array([e[int(x)] for x in a])
    reward = retained * av - cost
    return float(np.mean(match / w * reward))


def ipw_control_value(df: pd.DataFrame, *, propensity: dict | None = None) -> float:
    e = propensity or PROPENSITY
    a = df["offer_arm"].to_numpy()
    retained = 1.0 - df["churned"].to_numpy()
    av = df["annual_value"].to_numpy()
    match = (a == 0).astype(float)
    return float(np.mean(match / e[0] * (retained * av)))


def fit_outcome_models_on_train(
    train: pd.DataFrame,
) -> dict[int, Pipeline]:
    """Fit P(churn | X, A=a) on train only — used as DR nuisances on holdout."""
    models: dict[int, Pipeline] = {}
    for arm in range(4):
        sub = train[train["offer_arm"] == arm]
        pipe = Pipeline(
            steps=[
                ("pre", make_preprocessor()),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_depth=4,
                        learning_rate=0.05,
                        max_iter=120,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
        pipe.fit(sub[FEATURE_COLS], sub["churned"])
        models[arm] = pipe
    return models


def dr_policy_value(
    df: pd.DataFrame,
    policy_arms: np.ndarray,
    outcome_models: dict[int, Pipeline],
    *,
    propensity: dict | None = None,
) -> float:
    """Doubly robust policy value using train-fit outcome models + IPW residual."""
    e = propensity or PROPENSITY
    a = df["offer_arm"].to_numpy()
    pi = np.asarray(policy_arms)
    retained = 1.0 - df["churned"].to_numpy()
    av = df["annual_value"].to_numpy()
    cost = assign_cost(pi)
    x = df[FEATURE_COLS]

    # Predicted retention under policy arm and under observed arm.
    p_churn_pi = np.zeros(len(df))
    p_churn_a = np.zeros(len(df))
    for arm in range(4):
        pred = outcome_models[arm].predict_proba(x)[:, 1]
        p_churn_pi = np.where(pi == arm, pred, p_churn_pi)
        p_churn_a = np.where(a == arm, pred, p_churn_a)

    mu_pi = (1.0 - p_churn_pi) * av - cost
    observed_reward = retained * av - cost
    # When A != π, the residual term is 0; when A == π, correct the DM prediction.
    match = (a == pi).astype(float)
    w = np.array([e[int(x_)] for x_ in a])
    residual = match / w * (observed_reward - ((1.0 - p_churn_a) * av - cost))
    # Direct method uses mu_pi for everyone; residual only on matched arms.
    # Standard DR for policy value under known propensities:
    #   μ̂_π + 1[A=π]/e_A * (R_obs - μ̂_A) but reward includes policy cost.
    return float(np.mean(mu_pi + residual))


def binary_uplift_qini(
    y_churn: np.ndarray,
    treatment: np.ndarray,
    score: np.ndarray,
) -> float:
    """Normalized Qini (perfect≈1, random≈0) for binary treatment.

    Outcome is retention. Follows the usual uplift-curve normalization used in
    take-home rubrics (TASK.md: perfect=1, random=0).
    """
    y = 1.0 - np.asarray(y_churn, dtype=float)  # retention
    t = np.asarray(treatment, dtype=int)
    s = np.asarray(score, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    y, t = y[order], t[order]

    n = len(y)
    n_t = int(t.sum())
    n_c = n - n_t
    if n_t == 0 or n_c == 0:
        return 0.0

    # Number of treated/control in top-k as we walk the ranking.
    cum_t = np.cumsum(t)
    cum_c = np.cumsum(1 - t)
    cum_yt = np.cumsum(y * t)
    cum_yc = np.cumsum(y * (1 - t))

    # Classic Qini curve: n * (Y_t / N_t - Y_c / N_c) accumulated in ranking order,
    # expressed as incremental retained count vs random.
    qini = cum_yt - cum_yc * n_t / n_c

    xs = np.arange(1, n + 1)
    # Random baseline: straight line to overall treatment effect endpoint.
    random_line = qini[-1] * xs / n
    area = float(np.trapezoid(qini - random_line, xs / n))

    # Perfect ranking: all positive-effect units first. Approximate ideal area as
    # triangle with the same endpoint (standard normalization → random=0, perfect=1
    # when endpoint > 0). When endpoint ≤ 0 (harmful ranking), scale by |endpoint|.
    perfect_area = abs(qini[-1]) / 2.0
    if perfect_area < 1e-12:
        return 0.0
    return float(area / perfect_area)


def value_weighted_qini(
    df: pd.DataFrame,
    score: np.ndarray,
) -> float:
    """Qini using value-weighted retention (retained * annual_value) as outcome."""
    y = (1.0 - df["churned"].to_numpy()) * df["annual_value"].to_numpy()
    t = (df["offer_arm"].to_numpy() > 0).astype(int)
    s = np.asarray(score, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    y, t = y[order], t[order]
    n = len(y)
    n_t = int(t.sum())
    n_c = n - n_t
    if n_t == 0 or n_c == 0:
        return 0.0
    cum_yt = np.cumsum(y * t)
    cum_yc = np.cumsum(y * (1 - t))
    qini = cum_yt - cum_yc * n_t / n_c
    xs = np.arange(1, n + 1)
    random_line = qini[-1] * xs / n
    area = float(np.trapezoid(qini - random_line, xs / n))
    perfect_area = abs(qini[-1]) / 2.0
    if perfect_area < 1e-12:
        return 0.0
    return float(area / perfect_area)


def estimate_value_capture(
    incremental_value: float,
    *,
    naive_incremental: float | None = None,
) -> dict:
    """Without oracle perfect foresight, report incremental $ and vs-naive ratio."""
    out = {"incremental_value": incremental_value}
    if naive_incremental is not None and abs(naive_incremental) > 1e-9:
        out["lift_vs_naive"] = incremental_value - naive_incremental
    return out


def summarize_policy(
    name: str,
    df: pd.DataFrame,
    policy_arms: np.ndarray,
    uplift_score: np.ndarray,
    outcome_models: dict[int, Pipeline],
    *,
    budget: float,
    pred_tau: np.ndarray | None = None,
) -> PolicyEval:
    arms = np.asarray(policy_arms)
    spend = float(assign_cost(arms).sum())
    mix = {int(a): int((arms == a).sum()) for a in range(4)}
    ipw_v = ipw_policy_value(df, arms)
    ipw_0 = ipw_control_value(df)
    dr_v = dr_policy_value(df, arms, outcome_models)
    dr_0 = dr_policy_value(df, np.zeros(len(df), dtype=int), outcome_models)

    # Sleeping dogs contacted: offered a paid arm where predicted τ < 0 (if available).
    sd = 0
    if pred_tau is not None:
        # pred_tau shape (n, 3) for arms 1..3
        for i, a in enumerate(arms):
            if a > 0 and pred_tau[i, a - 1] < 0:
                sd += 1

    # Qini diagnostics: any paid offer vs control, ranked by uplift_score.
    qini = binary_uplift_qini(
        df["churned"].to_numpy(),
        (df["offer_arm"].to_numpy() > 0).astype(int),
        uplift_score,
    )
    qini_value = value_weighted_qini(df, uplift_score)

    return PolicyEval(
        name=name,
        budget=budget,
        spend=spend,
        n_treated=int((arms > 0).sum()),
        arm_mix=mix,
        ipw_value=ipw_v,
        ipw_incremental=ipw_v - ipw_0,
        dr_value=dr_v,
        dr_incremental=dr_v - dr_0,
        qini=qini,
        sleeping_dogs_contacted=sd,
        fraction_untreated=float((arms == 0).mean()),
        qini_value=qini_value,
        ipw_incremental_total=(ipw_v - ipw_0) * len(df),
        dr_incremental_total=(dr_v - dr_0) * len(df),
    )


def bootstrap_incremental(
    df: pd.DataFrame,
    policy_arms: np.ndarray,
    outcome_models: dict[int, Pipeline],
    *,
    n_boot: int = 200,
    seed: int = RANDOM_SEED,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(df)
    ipw_incs = []
    dr_incs = []
    arms = np.asarray(policy_arms)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub = df.iloc[idx].reset_index(drop=True)
        sub_arms = arms[idx]
        ipw_v = ipw_policy_value(sub, sub_arms)
        ipw_0 = ipw_control_value(sub)
        dr_v = dr_policy_value(sub, sub_arms, outcome_models)
        dr_0 = dr_policy_value(sub, np.zeros(len(sub), dtype=int), outcome_models)
        ipw_incs.append(ipw_v - ipw_0)
        dr_incs.append(dr_v - dr_0)
    ipw_incs = np.asarray(ipw_incs)
    dr_incs = np.asarray(dr_incs)
    return {
        "ipw_incremental_mean": float(ipw_incs.mean()),
        "ipw_incremental_ci95": (
            float(np.quantile(ipw_incs, 0.025)),
            float(np.quantile(ipw_incs, 0.975)),
        ),
        "dr_incremental_mean": float(dr_incs.mean()),
        "dr_incremental_ci95": (
            float(np.quantile(dr_incs, 0.025)),
            float(np.quantile(dr_incs, 0.975)),
        ),
    }


def policy_value_curve(
    df: pd.DataFrame,
    value_scores: np.ndarray,
    best_arms: np.ndarray,
    costs: np.ndarray,
    outcome_models: dict[int, Pipeline],
    budgets: list[float],
) -> pd.DataFrame:
    """Re-allocate under several budgets via greedy fill of ranked positive-value offers."""
    from .allocation import allocate_lambda

    rows = []
    for b in budgets:
        arms, _ = allocate_lambda(value_scores, best_arms, costs, budget=b)
        ev = summarize_policy(
            f"budget_{int(b)}",
            df,
            arms,
            uplift_score=np.max(value_scores, axis=1),
            outcome_models=outcome_models,
            budget=b,
        )
        rows.append(
            {
                "budget": b,
                "spend": ev.spend,
                "ipw_incremental": ev.ipw_incremental,
                "dr_incremental": ev.dr_incremental,
                "n_treated": ev.n_treated,
            }
        )
    return pd.DataFrame(rows)
