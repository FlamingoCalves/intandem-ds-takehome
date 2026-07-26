# In Tandem — Product Data Scientist take-home

**Allocate a retention budget across offers.** Decide which offer (none / $1 nudge / $5 discount / $15 concierge) to give each of ~40,000 prospective subscribers within a **$40,000** budget to maximize **net incremental retained revenue**.

Official brief: [`TASK.md`](TASK.md) · Data docs: [`data/data_dictionary.md`](data/data_dictionary.md)

## Submission artifacts

| File | Description |
|------|-------------|
| `allocation.csv` | `user_id,offer_arm` for scoring users (spend ≤ $40k) |
| `holdout_scores.csv` | `user_id,uplift_score` for every holdout row |
| `Retention_Allocation.ipynb` | Analysis with **visible v1→v2→v3 iteration** + Plotly charts |
| `Retention_Allocation.html` | Executed notebook export (open in a browser) |
| `interactive_brief/` | Stakeholder-facing interactive brief (open `index.html`) |
| `WRITEUP.md` | ~1–2 page writeup + iteration log |
| `AI_USAGE.md` | AI assistance disclosure |
| `run_pipeline.py` / `src/` | Reproducible pipeline behind the notebook |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py       # writes CSVs + outputs/metrics.json
python build_notebook.py     # builds & executes the notebook
```

Open the stakeholder brief: `interactive_brief/index.html` (double-click or serve the folder).

Or in Colab: paste `colab_bootstrap.txt`, then copy this repo’s `src/` + `run_pipeline.py`.

## Method (one paragraph)

Exclude the post-treatment leak `offer_window_logins`. Start from a **naive churn → concierge** baseline (shows negative holdout value). Then estimate per-arm effects (segment-shrunken ATEs, then a shrunken T-learner), convert to net value, and allocate with a **λ-search** under the budget while keeping offers off predicted sleeping dogs. Validate with **IPW + DR** on holdout (nuisances fit on train only; holdout also used to choose v3 over v2). **Shipped policy: v3** (see `WRITEUP.md`). Stakeholder summary: `interactive_brief/index.html`.

## Notes

- Data are **synthetic**.
- Time is open-ended; process visibility is part of the grade.
- AI assistance was used — see `AI_USAGE.md`.
