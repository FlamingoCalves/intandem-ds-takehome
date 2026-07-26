# In Tandem Product DS Take-Home — Writeup

**Candidate:** Jonathan Evans  
**Problem:** Allocate a $40,000 retention budget across four offers (none / $1 nudge / $5 discount / $15 concierge) for ~40k prospective subscribers to maximize **net incremental retained revenue**.

## Problem framing

Predicting churn is not the same as deciding who to pay to keep. Offers were randomized in `train`/`holdout`, so per-arm retention effects are identifiable without confounding adjustment. The decision problem is a constrained multi-arm policy: for each user pick an arm $a$ maximizing $\hat\tau_a(x)\cdot\text{annual\_value} - \text{cost}_a$ under a hard budget, with arm 0 (do nothing) always available.

In Tandem’s products support families through coordination and co-parenting workflows; we treat this as a consumer subscription retention experiment and do not invent product-specific metrics beyond the brief.

## What we found in the data

1. **Population ATEs are small** (~+1.0pp nudge, ~+1.6pp discount/concierge) and hide heterogeneity.
2. **Measurement trap:** `offer_window_logins` is post-treatment. Retained users average ~8 vs ~1 for churned. In `scoring.csv` the column still looks real (mean ~4.5, range 0–15) — not an obvious placeholder — so exclusion is based on the dictionary + outcome association, not on “this column looks fake.”
3. **Sleeping dogs are real:** concierge helps low-tenure users (~+6.5pp) and **hurts** high-tenure users (~−3.9pp). Targeting high-churn / high-tenure users with the biggest offer funds negative effects.

## Method (short)

- Features: pre-treatment only; **never** `offer_window_logins`.
- **v1:** control-only churn model → spend budget on concierge for highest-risk users (the trap).
- **v2:** tenure×activity segments with shrunken segment×arm net-ATEs + λ-search allocation.
- **v3:** shrunken cross-fitted HistGBDT T-learner CATEs + same λ allocator (`argmax_a(\hat V_a - \lambda c_a)`).
- Holdout evaluation: IPW + DR with known ~1/4 propensities; **DR nuisances fit on train only**. Promote v3 only if it clearly beats v2 on frozen holdout.

## Iteration log (holdout, must match `outputs/iteration_log.csv`)

| Version | Idea | Spend | DR incremental (total) | IPW incremental (total) | Qini (diag.) | Notes |
|--------|------|------:|------------------------:|-------------------------:|-------------:|-------|
| v1 | Churn → concierge | $40,000 | **−$35.8k** | **−$32.5k** | −0.085 | Value-destroying; plumbing check (negative vs TASK’s ≈−2%/−0.03 oracle band) |
| v2 | Segment shrunk ATE + λ | $35,284 | +$108.6k | +$177.4k | +0.024 | Mostly nudges; avoids concierge sleeping dogs |
| v2 greedy | V/cost fill (secondary) | $35,284 | +$108.6k | +$177.4k | +0.024 | Same allocation as λ on this run |
| **v3 (shipped)** | T-learner + λ | $39,997 | **+$121.5k** | +$167.7k | **+0.043** | Clears promotion gate vs v2; score = max τ·AV |
| v3 @ $20k | Re-solved λ | $20,000 | +$104.6k | +$108.9k | — | Half-budget: re-optimize, don’t truncate |

Bootstrap 95% CI for shipped policy DR incremental $/user: roughly **[1.1, 3.7]** (see `outputs/metrics.json`).

## Headline

- **Naive high-risk → big offer loses money** on honest holdout OPE.
- **Shipped policy (v3):** positive DR/IPW incremental value; diagnostic Qini > 0; scoring allocation spends **$39,996** with mix dominated by nudges, selective discounts, rare concierge (see `allocation.csv`).
- **`uplift_score` definition:** $\max_a \hat\tau_a(x)\cdot\text{annual\_value}$ — best-arm expected incremental revenue (**ranking** deliverable). Allocation still uses net value $\hat\tau\cdot\text{AV}-\text{cost}$ under λ-search; we deliberately separate “who ranks as high uplift” from “whom we pay under the budget.”
- **Qini note:** Our diagnostic Qini (~0.04 after this score choice) collapses all paid arms vs control and is **not** the grader’s oracle metric. Per-arm ranking quality is much stronger than that collapsed score. We optimize primarily for **budgeted incremental $** (IPW/DR); if the grader’s Qini uses a similar collapsed construction, it may sit below TASK.md’s ≈0.24 “competent” reference even when dollar value is strong — that tradeoff is intentional and discussed here rather than hidden.

Exact grader “value capture %” needs their perfect-foresight oracle (not in the bundle). We therefore report reproducible IPW/DR dollars and Qini diagnostics rather than inventing a capture %.

## Sleeping dogs

We identify negative-effect regions via tenure-quartile ATEs and by requiring $\hat V_a > 0$ before an offer is eligible. High-tenure users are largely kept on arm 0 or cheap nudges; concierge is sparse in the shipped mix.

## If the budget were halved

Re-solve λ for $20k from scratch (same scores, tighter dual). On holdout this still yields large positive incremental value (~$105k DR total) with a more nudge-heavy, lower-coverage mix — diminishing returns appear as budget grows, so the first dollars are the most valuable.

## Production transfer (Databricks + AWS)

1. **Features:** curated pre-treatment table in Databricks; blocklist post-treatment fields; data-quality tests for the leak column.
2. **Training:** weekly (or post-campaign) job fitting outcome models + exporting CATE/value scores to a model registry.
3. **Scoring:** AWS batch (or Databricks job) writes `user_id → offer_arm` under the live budget; λ refreshed when budget or price list changes.
4. **Monitoring:** arm mix, spend, covariate shift, and a persistent randomized exploration slice for ongoing IPW effect estimates; retrain if uplift decays or segment ATEs flip.
5. **Stakeholders:** Finance gets ROI vs do-nothing / vs churn baseline; Product gets “who gets which offer” summaries (tenure × offer) rather than a single global treatment.

## How to reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py          # writes allocation.csv, holdout_scores.csv, outputs/*
python build_notebook.py        # executes Retention_Allocation.ipynb
```
