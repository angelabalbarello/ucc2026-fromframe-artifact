"""Regenerate Fig. 2 (A-OSC stress regime) from the frozen E10 summary.

No experiment is run: the script only reads
  artifact/experiments/e10_eaml/outputs/summary_aosc.json  (recall, FAR/trip, alerts/trip)
  artifact/experiments/e10_eaml/outputs/spec_e10.json      (budget B)
and draws the same three panels as the submitted figure. Fonts are embedded
as TrueType (pdf.fonttype = 42), so the PDF contains no Type 3 fonts.

Usage: python make_fig_aosc.py <summary_aosc.json> <spec_e10.json> <out.pdf>
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

summary_path, spec_path, out_path = sys.argv[1:4]
summary = json.load(open(summary_path, encoding="utf-8"))
budget = json.load(open(spec_path, encoding="utf-8"))["budget_alerts_per_trip"]
rows = {r["arm"]: r for r in summary["arms_summary"]}
arms = ["A0", "A1", "A2", "A3"]

plt.rcParams.update({
    "pdf.fonttype": 42,          # TrueType, never Type 3
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "TeX Gyre Termes", "Times New Roman"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Serif",
    "mathtext.it": "Liberation Serif:italic",
    "font.size": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
colors = ["#b0b0b0", "#999999", "#1f4e79", "#7f9fbf"]
panels = [
    ("recall", "event recall", (0, 1.2), [0, 0.25, 0.5, 0.75, 1.0], "{:.2f}"),
    ("far_per_trip", "false alerts/trip", (0, 12), [0, 2.5, 5.0, 7.5, 10.0], "{:.2f}"),
    ("alerts_per_trip", "alerts/trip", (0, 16), [0, 5, 10, 15], "{:.2f}"),
]

fig, axes = plt.subplots(1, 3, figsize=(510.63 / 72, 179.071 / 72))
for ax, (key, label, ylim, yticks, fmt) in zip(axes, panels):
    vals = [rows[a][key] for a in arms]
    bars = ax.bar(arms, vals, color=colors, width=0.62, zorder=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + ylim[1] * 0.01, fmt.format(v),
                ha="center", va="bottom", fontsize=11.5)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel(label)
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.8, zorder=0)
    if key == "alerts_per_trip":
        ax.axhline(budget, color="#c0392b", linestyle="--", linewidth=2.2, zorder=3)
        ax.text(3.55, budget, "B", color="#c0392b", fontstyle="italic",
                va="center", ha="left", fontsize=13)
fig.tight_layout(w_pad=1.6)
fig.savefig(out_path)
print("written", out_path, {a: [rows[a][k] for k, *_ in panels] for a in arms})
