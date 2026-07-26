"""Shared constants for the In Tandem retention allocation take-home."""

from __future__ import annotations

RANDOM_SEED = 2026
BUDGET = 40_000
HALF_BUDGET = 20_000

ARM_COSTS = {0: 0, 1: 1, 2: 5, 3: 15}
ARM_NAMES = {
    0: "none",
    1: "nudge",
    2: "discount",
    3: "concierge",
}

# Equal-probability randomized experiment (approx. 1/4 each).
N_ARMS = 4
PROPENSITY = {a: 1.0 / N_ARMS for a in range(N_ARMS)}

LEAKY_FEATURE = "offer_window_logins"

FEATURE_COLS = [
    "tenure_months",
    "active_days_30d",
    "sessions_30d",
    "family_members",
    "features_used",
    "support_tickets_90d",
    "price_tier",
    "months_since_last_active",
    "autopay",
    "prior_offers",
    "acquisition_channel",
    "mrr",
]

# TASK.md reference band for the naive trap (plumbing bug-check).
NAIVE_CAPTURE_REF = -0.02
NAIVE_QINI_REF = -0.03
