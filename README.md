# In Tandem — Product Data Scientist take-home

**Allocate a retention budget across offers.** Decide which offer (none / $1 nudge / $5 discount / $15 concierge) to give each of ~40,000 prospective subscribers within a **$40,000** budget to maximize **net incremental retained revenue**.

Official brief: [`TASK.md`](TASK.md) · Data docs: [`data/data_dictionary.md`](data/data_dictionary.md)

## Start here (what is graded)

1. **[`Retention_Allocation.ipynb`](Retention_Allocation.ipynb)** — analysis with **visible v1→v2→v3 iteration** (also [`Retention_Allocation.html`](Retention_Allocation.html) if you prefer not to run Python)
2. **[`WRITEUP.md`](WRITEUP.md)** — ~1–2 page narrative + iteration log
3. **`allocation.csv`** / **`holdout_scores.csv`** — required scoring deliverables

Optional stakeholder supplement (not a substitute for the notebook/writeup):  
https://flamingocalves.github.io/intandem-ds-takehome/interactive_brief/

## Submission artifacts

| File | Description |
|------|-------------|
| `allocation.csv` | `user_id,offer_arm` for scoring users (spend ≤ $40k) |
| `holdout_scores.csv` | `user_id,uplift_score` for every holdout row |
| `Retention_Allocation.ipynb` | Primary analysis with visible development path + Plotly charts |
| `Retention_Allocation.html` | Executed notebook export (open in a browser) |
| `WRITEUP.md` | ~1–2 page writeup + iteration log |
| `AI_USAGE.md` | AI assistance disclosure |
| `run_pipeline.py` / `src/` | Reproducible pipeline behind the notebook |
| `interactive_brief/` | Optional stakeholder summary (also hosted on GitHub Pages) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py       # writes CSVs + outputs/metrics.json
python build_notebook.py     # builds & executes the notebook
```

Or in Colab: paste `colab_bootstrap.txt`, then copy this repo’s `src/` + `run_pipeline.py`.

## Method (one paragraph)

Exclude the post-treatment leak `offer_window_logins`. Start from a **naive churn → concierge** baseline (shows negative holdout value). Then estimate per-arm effects (segment-shrunken ATEs, then a shrunken T-learner), convert to net value, and allocate with a **λ-search** under the budget while keeping offers off predicted sleeping dogs. Validate with **IPW + DR** on holdout (nuisances fit on train only; holdout also used to choose v3 over v2). **Shipped policy: v3** (see `WRITEUP.md`).

## Notes

- Data are **synthetic**.
- Time is open-ended; process visibility is part of the grade.
- AI assistance was used — see `AI_USAGE.md`.
