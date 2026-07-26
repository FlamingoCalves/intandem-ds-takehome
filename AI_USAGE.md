# AI Usage Disclosure

**Tools:** Cursor (agent-assisted coding), plus a prior multi-agent methodology critique (`ds-critique`: methodologist, statistician, ML engineer, devil’s advocate) and an independent second-model review of the plan.

## How AI was used

AI helped draft the methodology, scaffold the Python package (`src/`), pipeline runner, notebook builder, and writeup structure. It also proposed evaluation details (IPW/DR, λ allocation, segment shrinkage) that were then implemented and checked against the provided data.

## What was independently verified

- Row counts, arm balance, and population ATEs recomputed from `data/*.csv`.
- Leakage: `offer_window_logins` means by outcome; scoring distribution is realistic-looking (not all zeros).
- Concierge tenure-quartile sign-flip (~+6.5pp Q1 vs ~−3.9pp Q4) recomputed before relying on it in the narrative.
- Every headline number in `WRITEUP.md` is taken from a fresh `python run_pipeline.py` run (`outputs/metrics.json` / `outputs/iteration_log.csv`).
- Deliverable checks: `allocation.csv` row count = scoring size, unique `user_id`, total spend ≤ $40,000; `holdout_scores.csv` covers every holdout row.

## Where judgment overrode AI defaults

- Shipped the graded core (notebook with visible iteration + short writeup + CSVs) **and** an In Tandem–branded stakeholder brief in `interactive_brief/` with interactive Plotly charts; notebook HTML export also embeds Plotly.
- Required a **v1 negative plumbing check** before trusting later OPE numbers.
- Required an explicit **promotion gate** for v3 vs v2 (relative + absolute holdout DR lift); would have shipped v2 if the gate failed. That means holdout was used for model selection, not only as a one-shot final exam.
- Kept sklearn HistGBDT T-learner rather than pulling in econml/causalml for the shippable path. Final scoring models are refit on full train; cross-fitting was a development diagnostic, not the production scorer.
- Treated grader “perfect foresight capture %” as **not estimable** from the public bundle; reported IPW/DR dollars instead of inventing an oracle.
- Separated **allocation** (net value under λ-budget) from **`uplift_score`** ranking (`max_a τ·AV`) after a cheap score pass improved diagnostic Qini without changing the dollar policy.

## What AI did *not* do

- Fabricate holdout metrics. All reported figures are from executed code on the provided CSVs.
- Modify or regenerate the competition data files.
