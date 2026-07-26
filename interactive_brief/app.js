/* In Tandem retention brief — numbers from outputs/metrics.json (pipeline run) */

const DATA = {
  v1: {
    title: "v1 · Naive churn → big offer",
    blurb:
      "Predict who will cancel, then spend the budget on concierge for the highest-risk users. Sounds intuitive. It destroys value.",
    learn:
      "High churn risk ≠ high saveability. Lost causes and sleeping dogs soak up expensive outreach.",
    spend: 40000,
    drTotal: -35809,
    ipwTotal: -32491,
    qini: -0.085,
    mix: { none: 47332, nudge: 0, discount: 2, concierge: 2666 },
  },
  v2: {
    title: "v2 · Segment effects + budget math",
    blurb:
      "Estimate which offer helps which kind of subscriber (especially by tenure), skip negative effects, spend where each dollar earns.",
    learn:
      "Personalization can be simple and still beat the trap — if it is causal and cost-aware.",
    spend: 35284,
    drTotal: 108638,
    ipwTotal: 177393,
    qini: 0.024,
    mix: { none: 17876, nudge: 31334, discount: 790, concierge: 0 },
  },
  v3: {
    title: "v3 · Personal uplift scores (shipped)",
    blurb:
      "Estimate per-person, per-offer lift, convert to dollars, allocate with a budget dual (λ-search). Promoted only because it beat v2 on holdout.",
    learn:
      "More flexible modeling is worth shipping only when the honest report card improves.",
    spend: 39997,
    drTotal: 121476,
    ipwTotal: 167720,
    qini: 0.043,
    mix: { none: 22565, nudge: 24362, discount: 3046, concierge: 27 },
  },
  half: {
    drTotal: 104588,
    ipwTotal: 108945,
    spend: 20000,
    mix: { none: 31232, nudge: 18460, discount: 308, concierge: 0 },
  },
  scoring: {
    spend: 39996,
    mix: { none: 16522, nudge: 19501, discount: 3916, concierge: 61 },
  },
  tenureConcierge: [
    { q: "Newest (Q1)", pp: 6.5 },
    { q: "Q2", pp: 4.9 },
    { q: "Q3", pp: -1.3 },
    { q: "Longest (Q4)", pp: -3.9 },
  ],
};

const money = (n) =>
  (n < 0 ? "−" : "+") +
  "$" +
  Math.abs(Math.round(n)).toLocaleString("en-US");

const plotLayout = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "Figtree, sans-serif", color: "#1a1b1f", size: 13 },
  margin: { t: 28, r: 18, b: 48, l: 56 },
  hoverlabel: { bgcolor: "#512731", font: { color: "#fff" } },
};

function drawValueChart(mode) {
  const full = mode === "40k";
  const naive = DATA.v1.drTotal;
  const shipped = full ? DATA.v3.drTotal : DATA.half.drTotal;
  const labels = full
    ? ["Naive churn → concierge", "Shipped uplift policy ($40k)"]
    : ["Naive churn → concierge", "Shipped uplift policy ($20k)"];
  const values = [naive, shipped];
  const colors = ["#b33a2b", "#2f6f5e"];

  Plotly.newPlot(
    "value-chart",
    [
      {
        type: "bar",
        x: labels,
        y: values,
        marker: { color: colors },
        text: values.map(money),
        textposition: "outside",
        hovertemplate: "%{x}<br>%{text}<extra></extra>",
      },
    ],
    {
      ...plotLayout,
      title: { text: "Holdout incremental retained revenue (DR estimate)", font: { size: 14 } },
      yaxis: { title: "Incremental $ vs do-nothing", zeroline: true, zerolinecolor: "#e4d8d2" },
      height: 380,
    },
    { displayModeBar: false, responsive: true }
  );

  const panel = document.querySelector("#value-panel");
  panel.querySelector(".big").textContent = money(shipped);
  panel.querySelector(".big").className = "big good";
  panel.querySelector("[data-field='compare']").textContent = full
    ? `Versus the naive policy (${money(naive)}), the shipped approach turns the same retention budget into material positive incremental value on the honest holdout.`
    : `At half budget, re-solving the allocation still yields ${money(shipped)} — the first dollars are the most valuable; don’t just truncate the $40k list.`;
}

function drawTenureChart() {
  const xs = DATA.tenureConcierge.map((d) => d.q);
  const ys = DATA.tenureConcierge.map((d) => d.pp);
  Plotly.newPlot(
    "tenure-chart",
    [
      {
        type: "bar",
        x: xs,
        y: ys,
        marker: {
          color: ys.map((v) => (v >= 0 ? "#2f6f5e" : "#b33a2b")),
        },
        text: ys.map((v) => `${v > 0 ? "+" : ""}${v.toFixed(1)} pp`),
        textposition: "outside",
        hovertemplate: "%{x}<br>Retention change: %{text}<extra></extra>",
      },
    ],
    {
      ...plotLayout,
      title: { text: "Concierge effect on retention by tenure", font: { size: 14 } },
      yaxis: { title: "Percentage-point change vs no offer", zeroline: true, zerolinecolor: "#512731" },
      height: 400,
    },
    { displayModeBar: false, responsive: true }
  );
}

function drawMixChart(mix, title) {
  const labels = ["No offer", "Nudge $1", "Discount $5", "Concierge $15"];
  const keys = ["none", "nudge", "discount", "concierge"];
  const values = keys.map((k) => mix[k]);
  Plotly.newPlot(
    "mix-chart",
    [
      {
        type: "bar",
        orientation: "h",
        y: labels,
        x: values,
        marker: { color: ["#d1b9b4", "#0050bd", "#2f6f5e", "#512731"] },
        text: values.map((v) => v.toLocaleString()),
        textposition: "outside",
        hovertemplate: "%{y}: %{x:,}<extra></extra>",
      },
    ],
    {
      ...plotLayout,
      title: { text: title, font: { size: 14 } },
      xaxis: { title: "Subscribers" },
      margin: { t: 36, r: 48, b: 40, l: 120 },
      height: 360,
    },
    { displayModeBar: false, responsive: true }
  );
}

function setIteration(key) {
  const d = DATA[key];
  document.querySelectorAll(".iter-nav button").forEach((btn) => {
    btn.classList.toggle("is-active", btn.dataset.iter === key);
  });
  document.querySelector("#iter-title").textContent = d.title;
  document.querySelector("#iter-blurb").textContent = d.blurb;
  document.querySelector("#iter-learn").textContent = d.learn;
  document.querySelector("#iter-dr").textContent = money(d.drTotal);
  document.querySelector("#iter-dr").className = d.drTotal < 0 ? "bad" : "good";
  document.querySelector("#iter-spend").textContent = `$${d.spend.toLocaleString()}`;
  document.querySelector("#iter-qini").textContent = d.qini.toFixed(3);
  drawMixChart(d.mix, "Holdout policy arm mix for this iteration");
}

function initBudgetToggle() {
  const buttons = document.querySelectorAll("[data-budget]");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.toggle("is-active", b === btn));
      drawValueChart(btn.dataset.budget);
    });
  });
  drawValueChart("40k");
}

function initIterationNav() {
  document.querySelectorAll(".iter-nav button").forEach((btn) => {
    btn.addEventListener("click", () => setIteration(btn.dataset.iter));
  });
  setIteration("v3");
}

function initReveal() {
  const nodes = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) {
    nodes.forEach((n) => n.classList.add("is-visible"));
    return;
  }
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          io.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.16 }
  );
  nodes.forEach((n) => io.observe(n));
}

document.addEventListener("DOMContentLoaded", () => {
  initBudgetToggle();
  initIterationNav();
  drawTenureChart();
  Plotly.newPlot(
    "scoring-chart",
    [
      {
        type: "bar",
        orientation: "h",
        y: ["No offer", "Nudge $1", "Discount $5", "Concierge $15"],
        x: [
          DATA.scoring.mix.none,
          DATA.scoring.mix.nudge,
          DATA.scoring.mix.discount,
          DATA.scoring.mix.concierge,
        ],
        marker: { color: ["#d1b9b4", "#0050bd", "#2f6f5e", "#512731"] },
        text: [
          DATA.scoring.mix.none,
          DATA.scoring.mix.nudge,
          DATA.scoring.mix.discount,
          DATA.scoring.mix.concierge,
        ].map((v) => v.toLocaleString()),
        textposition: "outside",
        hovertemplate: "%{y}: %{x:,}<extra></extra>",
      },
    ],
    {
      ...plotLayout,
      title: {
        text: `Scoring-set recommendation ($${DATA.scoring.spend.toLocaleString()} spent)`,
        font: { size: 14 },
      },
      xaxis: { title: "Subscribers" },
      margin: { t: 36, r: 48, b: 40, l: 120 },
      height: 360,
    },
    { displayModeBar: false, responsive: true }
  );
  initReveal();
});
