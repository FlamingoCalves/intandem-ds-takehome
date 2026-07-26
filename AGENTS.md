# Agent instructions (In Tandem DS Take-Home)

## Stack
- Python 3.11+ data science take-home (pandas, scikit-learn, plotly)
- No database / no paid APIs
- Synthetic retention experiment CSVs in `data/`

## Bootstrap checklist
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `python run_pipeline.py`
4. Optional: `python build_notebook.py` to regenerate/execute `Retention_Allocation.ipynb`

## Verification order
1. Pipeline completes and writes `allocation.csv`, `holdout_scores.csv`, `outputs/metrics.json`
2. Spend ≤ 40000; row counts match scoring/holdout
3. WRITEUP iteration log matches `outputs/iteration_log.csv`
4. Notebook executes top-to-bottom

## Cost safety
- None — local synthetic data only

## Git / GitHub
- Commit/push only when user asks
- Feature branches + PRs
- Commit author: Jonathan Evans <jevans6911@utexas.edu>

## Deploy targets
- **Now:** local / Colab / grader re-run
- **Planned:** N/A (take-home submission)

## Key docs
- [TASK.md](TASK.md) — official brief
- [WRITEUP.md](WRITEUP.md) — 1–2 page narrative
- [AI_USAGE.md](AI_USAGE.md) — AI disclosure
- [data/data_dictionary.md](data/data_dictionary.md) — includes leak column
