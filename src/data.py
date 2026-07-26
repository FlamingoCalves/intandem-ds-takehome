"""Data loading, schema checks, and feature matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from .config import FEATURE_COLS, LEAKY_FEATURE

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def load_splits(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = Path(data_dir) if data_dir is not None else DATA_DIR
    train = pd.read_csv(d / "train.csv")
    holdout = pd.read_csv(d / "holdout.csv")
    scoring = pd.read_csv(d / "scoring.csv")
    return train, holdout, scoring


def assert_data_contract(
    train: pd.DataFrame, holdout: pd.DataFrame, scoring: pd.DataFrame
) -> dict:
    """Validate schemas and basic economics; return summary stats for the notebook."""
    for name, df, need_outcomes in [
        ("train", train, True),
        ("holdout", holdout, True),
        ("scoring", scoring, False),
    ]:
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{name} missing features: {missing}")
        if LEAKY_FEATURE not in df.columns:
            raise ValueError(f"{name} missing leaky column {LEAKY_FEATURE}")
        if need_outcomes:
            for c in ("offer_arm", "churned"):
                if c not in df.columns:
                    raise ValueError(f"{name} missing {c}")

    av_ok = np.allclose(train["annual_value"], 12 * train["mrr"], rtol=1e-4, atol=0.05)
    if not av_ok:
        raise ValueError("annual_value is not approximately 12 * mrr on train")

    arm_share = train["offer_arm"].value_counts(normalize=True).sort_index().to_dict()
    leak_by_churn = (
        train.groupby("churned")[LEAKY_FEATURE].mean().rename({0: "retained", 1: "churned"})
    )
    scoring_leak = {
        "mean": float(scoring[LEAKY_FEATURE].mean()),
        "min": int(scoring[LEAKY_FEATURE].min()),
        "max": int(scoring[LEAKY_FEATURE].max()),
        "n_unique": int(scoring[LEAKY_FEATURE].nunique()),
    }

    return {
        "shapes": {
            "train": train.shape,
            "holdout": holdout.shape,
            "scoring": scoring.shape,
        },
        "arm_share_train": arm_share,
        "churn_rate_train": float(train["churned"].mean()),
        "annual_value_eq_12mrr": True,
        "leak_mean_by_outcome": leak_by_churn.to_dict(),
        "scoring_leak_looks_realistic": scoring_leak,
    }


def retention_ate_by_arm(df: pd.DataFrame) -> pd.DataFrame:
    """Population retention ATE vs control (arm 0)."""
    rates = df.groupby("offer_arm")["churned"].mean()
    control = rates.loc[0]
    rows = []
    for arm in sorted(rates.index):
        churn = float(rates.loc[arm])
        ret_uplift = float(control - churn)  # positive => offer reduces churn
        rows.append(
            {
                "offer_arm": int(arm),
                "churn_rate": churn,
                "retention_ate_vs_control": ret_uplift,
                "n": int((df["offer_arm"] == arm).sum()),
            }
        )
    return pd.DataFrame(rows)


def tenure_quartile_ates(df: pd.DataFrame) -> pd.DataFrame:
    """Retention ATE by tenure quartile × arm (sleeping-dog diagnostic)."""
    d = df.copy()
    d.loc[:, "tenure_q"] = pd.qcut(
        d["tenure_months"], 4, labels=["Q1", "Q2", "Q3", "Q4"]
    )
    rows = []
    for q, g in d.groupby("tenure_q", observed=True):
        rates = g.groupby("offer_arm")["churned"].mean()
        if 0 not in rates.index:
            continue
        control = float(rates.loc[0])
        for arm in [1, 2, 3]:
            if arm not in rates.index:
                continue
            rows.append(
                {
                    "tenure_q": str(q),
                    "offer_arm": arm,
                    "retention_ate": float(control - float(rates.loc[arm])),
                    "n": int((g["offer_arm"] == arm).sum()),
                }
            )
    return pd.DataFrame(rows)


def make_preprocessor() -> ColumnTransformer:
    numeric = [c for c in FEATURE_COLS if c != "acquisition_channel"]
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["acquisition_channel"],
            ),
        ]
    )


def design_matrix(
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
    *,
    fit: bool = False,
) -> np.ndarray:
    X = df[FEATURE_COLS]
    if fit:
        return preprocessor.fit_transform(X)
    return preprocessor.transform(X)
