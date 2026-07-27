# AI Usage Disclosure

I used AI assistants (Cursor / Claude) to help draft code, structure the analysis, and iterate on methodology. I independently verified every reported number by re-running the pipeline from a clean environment, checked the leakage and sleeping-dogs findings against the raw data myself, and made the final calls on model promotion and packaging.

## How AI was used

AI helped draft methodology options, scaffold the Python package (`src/`), pipeline runner, notebook builder, writeup, and stakeholder brief. It also proposed evaluation details (IPW/DR, λ allocation, segment shrinkage) that I then implemented and checked against the provided data.

## What I verified myself

- Row counts, arm balance, and population ATEs recomputed from `data/*.csv`.
- Leakage: `offer_window_logins` means by outcome; scoring distribution is realistic-looking (not all zeros).
- Concierge tenure-quartile sign-flip (~+6.5pp Q1 vs ~−3.9pp Q4) recomputed before relying on it in the narrative.
- Every headline number in `WRITEUP.md` is taken from a fresh `python run_pipeline.py` run (`outputs/metrics.json` / `outputs/iteration_log.csv`).
- Deliverable checks: `allocation.csv` row count = scoring size, unique `user_id`, total spend ≤ $40,000; `holdout_scores.csv` covers every holdout row.

## Where I overrode AI suggestions

- Shipped the graded core (notebook with visible iteration + short writeup + CSVs) and added an In Tandem–branded stakeholder brief in `interactive_brief/` as a supplement; notebook HTML export also embeds Plotly.
- Required a **v1 negative plumbing check** before trusting later OPE numbers.
- Required an explicit **promotion gate** for v3 vs v2 (relative + absolute holdout DR lift); would have shipped v2 if the gate failed. That means holdout was used for model selection, not only as a one-shot final exam.
- Kept a sklearn HistGBDT T-learner rather than pulling in econml/causalml for the shippable path. Final scoring models are refit on full train; cross-fitting was a development diagnostic, not the production scorer.
- Treated grader “perfect foresight capture %” as **not estimable** from the public bundle; reported IPW/DR dollars instead of inventing an oracle.
- Separated **allocation** (net value under λ-budget) from **`uplift_score`** ranking (`max_a τ·AV`) after a cheap score pass improved diagnostic Qini without changing the dollar policy.

## What AI did *not* do

- Fabricate holdout metrics. All reported figures are from executed code on the provided CSVs.
- Modify or regenerate the competition data files.
