#!/usr/bin/env python3
"""Build and execute Retention_Allocation.ipynb with visible iteration + Plotly charts."""

from __future__ import annotations

import nbformat as nbf
from nbclient import NotebookClient
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def md(source: str):
    return nbf.v4.new_markdown_cell(source)


def code(source: str):
    return nbf.v4.new_code_cell(source)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            """# In Tandem — Allocating a Retention Budget Across Offers

**Role:** Product Data Scientist take-home  
**Objective:** Maximize **net incremental retained revenue** under a **$40,000** budget across four offer arms (none / nudge $1 / discount $5 / concierge $15).

This notebook shows the **development path** (v1 → v2 → v3), not only the final policy. Numbers are recomputed from the provided CSVs; DR/IPW nuisances are fit on **train only**.

> Company context: In Tandem builds family apps (co-parenting / family organization). We keep product framing light and stick to the experimental economics in the brief.
"""
        ),
        md("## 0. Setup"),
        code(
            """
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, HTML

ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT))

from src.config import BUDGET, HALF_BUDGET, ARM_COSTS, ARM_NAMES, LEAKY_FEATURE, RANDOM_SEED
from src.data import (
    load_splits, assert_data_contract, retention_ate_by_arm, tenure_quartile_ates,
)
from src.metrics import fit_outcome_models_on_train, summarize_policy, bootstrap_incremental
from src.models import fit_v1_naive, fit_v2_segments, fit_v3_tlearner, compare_greedy

pd.set_option('display.max_columns', 50)

# Embed Plotly via HTML+CDN so charts survive nbconvert → .html (not just live Jupyter).
class _PlotlyShow:
    def __init__(self):
        self._include_js = 'cdn'
    def __call__(self, fig):
        display(HTML(fig.to_html(include_plotlyjs=self._include_js, full_html=False)))
        self._include_js = False  # only include the JS bundle once

show_fig = _PlotlyShow()
print('seed', RANDOM_SEED, 'budget', BUDGET)
"""
        ),
        md("## 1. Data contract & the measurement trap"),
        code(
            """
train, holdout, scoring = load_splits()
contract = assert_data_contract(train, holdout, scoring)
contract
"""
        ),
        md(
            """
### Leakage note

`offer_window_logins` is measured **during the offer window** (post-treatment). In train/holdout it is strongly associated with the outcome (~8 logins for retained vs ~1 for churned). In **scoring** it is **not** an obvious all-zero placeholder — it has a realistic-looking distribution (mean ~4.5). We exclude it from every model anyway, per the data dictionary.
"""
        ),
        code(
            """
leak = (
    train.groupby('churned')[LEAKY_FEATURE]
    .agg(['mean', 'median', 'count'])
    .rename(index={0: 'retained', 1: 'churned'})
)
print('Train leak by outcome:')
display(leak)

fig = px.histogram(
    scoring, x=LEAKY_FEATURE, nbins=16,
    title='Scoring: offer_window_logins looks realistic (trap is sneaky)',
)
fig.update_layout(template='plotly_white')
show_fig(fig)
"""
        ),
        md("## 2. Look before you model — ATEs hide sleeping dogs"),
        code(
            """
ate = retention_ate_by_arm(train)
display(ate)

fig = px.bar(
    ate[ate.offer_arm > 0], x='offer_arm', y='retention_ate_vs_control',
    text=ate[ate.offer_arm > 0]['retention_ate_vs_control'].map(lambda x: f'{100*x:.2f}pp'),
    labels={'offer_arm': 'Offer arm', 'retention_ate_vs_control': 'Retention ATE vs control'},
    title='Population ATEs are small — averages hide who is helped vs hurt',
)
fig.update_traces(textposition='outside')
fig.update_layout(template='plotly_white', xaxis=dict(tickmode='array', tickvals=[1,2,3],
                  ticktext=['nudge', 'discount', 'concierge']))
show_fig(fig)
"""
        ),
        code(
            """
tq = tenure_quartile_ates(train)
display(tq)

conc = tq[tq.offer_arm == 3].copy()
fig = px.bar(
    conc, x='tenure_q', y='retention_ate',
    title='Concierge sign-flip by tenure quartile (real sleeping dogs)',
    labels={'retention_ate': 'Retention ATE', 'tenure_q': 'Tenure quartile'},
    text=conc['retention_ate'].map(lambda x: f'{100*x:+.1f}pp'),
)
fig.add_hline(y=0, line_dash='dash', line_color='gray')
fig.update_traces(textposition='outside')
fig.update_layout(template='plotly_white')
show_fig(fig)
"""
        ),
        md(
            """## 3. Evaluation rules (holdout)

- **Train / train-val:** iterate here.
- **Holdout:** final IPW/DR report card; also used to **choose** v3 over v2 (selection set). DR outcome models are fit on **train only**.
- Primary decision metric: **IPW + DR net incremental value vs all-control** under the $40k constraint.
- `holdout_scores.csv` uses `uplift_score = max_a τ̂_a · annual_value` (best-arm expected incremental revenue for ranking). Allocation still uses net value under the budget.
- Multi-arm “Qini” is diagnostic; policy value under budget is the headline.
- **v3 scoring note:** predictions for holdout/scoring come from models **refit on full train**; cross-fitting is a development diagnostic only.
"""
        ),
        code(
            """
print('Fitting DR nuisances on TRAIN only...')
outcome_models = fit_outcome_models_on_train(train)

rng = np.random.default_rng(RANDOM_SEED)
mask = rng.random(len(train)) < 0.8
train_dev = train.loc[mask].reset_index(drop=True)
train_val = train.loc[~mask].reset_index(drop=True)
len(train_dev), len(train_val)
"""
        ),
        md("## 4. Iteration v1 — Naive churn → big offer (the trap)"),
        code(
            """
v1 = fit_v1_naive(train, holdout, budget=BUDGET)
ev1 = summarize_policy(
    'v1_holdout', holdout, v1.arms, v1.uplift_score, outcome_models, budget=BUDGET,
)
ev1
"""
        ),
        md(
            """**Plumbing check.** TASK.md says the naive policy lands near **≈ −2% value capture / ≈ −0.03 Qini** on the grader’s oracle. Our verifiable OPE check: v1 must be **clearly negative** on IPW/DR incremental value and Qini. If it were positive, we would debug the metric code before trusting v2/v3.
"""
        ),
        code(
            """
print(f"v1 IPW incremental total: ${ev1.ipw_incremental_total:,.0f}")
print(f"v1 DR incremental total:  ${ev1.dr_incremental_total:,.0f}")
print(f"v1 Qini (diagnostic):     {ev1.qini:.4f}")
print(f"v1 spend / arm mix:       ${ev1.spend:,.0f}  {ev1.arm_mix}")
assert ev1.ipw_incremental < 0 and ev1.dr_incremental < 0 and ev1.qini < 0, \\
    'v1 plumbing check failed — debug OPE before continuing'
print('✓ v1 is value-destroying as expected')
"""
        ),
        md("## 5. Iteration v2 — Segment shrunken net-ATE + λ allocation"),
        code(
            """
v2 = fit_v2_segments(train, holdout, budget=BUDGET)
ev2 = summarize_policy(
    'v2_holdout', holdout, v2.arms, v2.uplift_score, outcome_models,
    budget=BUDGET, pred_tau=v2.tau,
)
greedy_arms = compare_greedy(v2.values, budget=BUDGET)
ev2g = summarize_policy(
    'v2_greedy', holdout, greedy_arms, v2.uplift_score, outcome_models,
    budget=BUDGET, pred_tau=v2.tau,
)
print('v2 λ-allocation:', ev2)
print('v2 greedy (secondary):', ev2g)
"""
        ),
        code(
            """
mix = pd.DataFrame({
    'arm': list(ev2.arm_mix.keys()),
    'count': list(ev2.arm_mix.values()),
    'name': [ARM_NAMES[a] for a in ev2.arm_mix.keys()],
})
fig = px.bar(mix, x='name', y='count', title='v2 holdout policy arm mix', text='count')
fig.update_layout(template='plotly_white')
show_fig(fig)
"""
        ),
        md("## 6. Iteration v3 — Cross-fitted T-learner + λ (promote only if clear win)"),
        code(
            """
v3 = fit_v3_tlearner(train, holdout, budget=BUDGET)
ev3 = summarize_policy(
    'v3_holdout', holdout, v3.arms, v3.uplift_score, outcome_models,
    budget=BUDGET, pred_tau=v3.tau,
)
ev3
"""
        ),
        code(
            """
cmp = pd.DataFrame([
    {'policy': 'v1 naive', 'dr_incremental_total': ev1.dr_incremental_total, 'qini': ev1.qini, 'spend': ev1.spend},
    {'policy': 'v2 segments', 'dr_incremental_total': ev2.dr_incremental_total, 'qini': ev2.qini, 'spend': ev2.spend},
    {'policy': 'v3 t-learner', 'dr_incremental_total': ev3.dr_incremental_total, 'qini': ev3.qini, 'spend': ev3.spend},
])
display(cmp)

fig = go.Figure(data=[
    go.Bar(name='DR incremental $ (holdout total)', x=cmp.policy, y=cmp.dr_incremental_total),
])
fig.update_layout(template='plotly_white', title='Iteration log — holdout DR incremental value')
show_fig(fig)

rel_ok = ev3.dr_incremental > ev2.dr_incremental * 1.10
abs_ok = (ev3.dr_incremental - ev2.dr_incremental) >= 0.25
winner = 'v3' if (rel_ok and abs_ok) else 'v2'
print(f'Promotion gate: rel_ok={rel_ok}, abs_ok={abs_ok} → ship {winner}')
"""
        ),
        md("## 7. Frozen winner — scores, half-budget, bootstrap"),
        code(
            """
# Prefer metrics.json from run_pipeline.py if present (single source of truth).
metrics_path = ROOT / 'outputs' / 'metrics.json'
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text())
    winner = metrics['winner']
    print('Loaded winner from outputs/metrics.json:', winner)
else:
    metrics = None

bundle = v3 if winner == 'v3' else v2
half = fit_v3_tlearner(train, holdout, budget=HALF_BUDGET) if winner == 'v3' else fit_v2_segments(train, holdout, budget=HALF_BUDGET)
ev_half = summarize_policy(
    f'{winner}_half', holdout, half.arms, half.uplift_score, outcome_models,
    budget=HALF_BUDGET, pred_tau=half.tau,
)
boot = bootstrap_incremental(holdout, bundle.arms, outcome_models, n_boot=100)
print('Half-budget:', ev_half)
print('Bootstrap:', boot)
"""
        ),
        code(
            """
# Policy value curve across budgets (reuse frozen scores; only re-allocate)
from src.allocation import allocate_lambda

rows = []
for b in [5000, 10000, 20000, 30000, 40000]:
    arms_b, _ = allocate_lambda(bundle.values, budget=b)
    ev = summarize_policy(
        f'b{b}', holdout, arms_b, bundle.uplift_score, outcome_models, budget=b, pred_tau=bundle.tau,
    )
    rows.append({'budget': b, 'dr_incremental_total': ev.dr_incremental_total, 'spend': ev.spend, 'n_treated': ev.n_treated})
curve = pd.DataFrame(rows)
display(curve)
fig = px.line(curve, x='budget', y='dr_incremental_total', markers=True,
              title='Holdout DR incremental value vs budget (re-allocated λ)')
fig.update_layout(template='plotly_white')
show_fig(fig)
"""
        ),
        md("## 8. Score the prospective users"),
        code(
            """
final = fit_v3_tlearner(train, scoring, budget=BUDGET) if winner == 'v3' else fit_v2_segments(train, scoring, budget=BUDGET)
allocation = pd.DataFrame({'user_id': scoring.user_id.astype(int), 'offer_arm': final.arms.astype(int)})
spend = allocation.offer_arm.map(ARM_COSTS).sum()
print('scoring spend', spend, 'mix', allocation.offer_arm.value_counts().sort_index().to_dict())
assert spend <= BUDGET + 1e-6
assert len(allocation) == len(scoring)

holdout_scores = pd.DataFrame({
    'user_id': holdout.user_id.astype(int),
    'uplift_score': bundle.uplift_score.astype(float),
})
allocation.to_csv(ROOT / 'allocation.csv', index=False)
holdout_scores.to_csv(ROOT / 'holdout_scores.csv', index=False)
(ROOT / 'outputs').mkdir(exist_ok=True)
allocation.to_csv(ROOT / 'outputs' / 'allocation.csv', index=False)
holdout_scores.to_csv(ROOT / 'outputs' / 'holdout_scores.csv', index=False)
print('Wrote allocation.csv and holdout_scores.csv')
"""
        ),
        md(
            """## 9. What we would tell Finance & Product

- **Finance (ROI):** The naive high-churn → concierge policy **destroys** value on holdout (negative IPW/DR incremental). The shipped uplift policy turns the same $40k into positive incremental retained revenue; at half budget, re-solve λ from scratch rather than truncating the $40k list.
- **Product:** Best offer differs by segment — especially tenure. Concierge helps new subscribers and **hurts** long-tenure users on average. Cheap nudges win many value-per-dollar allocations; concierge is sparse and targeted.
- **Production sketch (Databricks + AWS):** feature table (pre-treatment only) → weekly batch score job → policy table with arm + expected net value → monitor spend, arm mix, realized IPW uplift on a randomized holdout / exploration slice → retrain when effect decay or covariate shift exceeds thresholds.

See `WRITEUP.md` for the 1–2 page narrative and `AI_USAGE.md` for the AI disclosure.
"""
        ),
    ]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return nb


def main():
    import sys

    nb_path = ROOT / "Retention_Allocation.ipynb"
    nb = build()
    nbf.write(nb, nb_path)
    print("Wrote", nb_path)

    # Ensure the venv interpreter is used as the kernel.
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "ipykernel", "install", "--user", "--name", "intandem-ds", "--display-name", "intandem-ds"]
    )

    client = NotebookClient(
        nb,
        timeout=1200,
        kernel_name="intandem-ds",
        resources={"metadata": {"path": str(ROOT)}},
    )
    print("Executing notebook...")
    client.execute()
    nbf.write(nb, nb_path)
    print("Executed and saved", nb_path)


if __name__ == "__main__":
    main()
