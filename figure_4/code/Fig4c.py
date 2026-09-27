# Fig 4c rod injury focal region number and size across conditions

## Load packages
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

## Load paper style conventions and the shared focal region functions
sys.path.insert(0, "../../_shared/code")
import paper_style
import _focal_regions as focal
paper_style.set_style()

## Load files
out_png = "../panels/Fig4c.png"
out_csv = "../data_plot/Fig4c.csv"
out_summary = "../data_plot/Fig4c_summary.csv"
conditions = ["WT_P21", "WT_P64", "LCA5_P21", "LCA5_P30"]
condition_reps = {c: [f"{c}_rep1", f"{c}_rep2"] for c in conditions}
x_positions = {c: i for i, c in enumerate(conditions)}
random_generator = np.random.default_rng(0)

## Collect focal region widths and counts for each retina
width_rows, count_rows = [], []
for condition, reps in condition_reps.items():
    for rep in reps:
        per_focal_region = focal.per_focal_region_table(rep)
        rep_label = "rep1" if rep.endswith("rep1") else "rep2"
        count_rows.append({"condition": condition, "rep": rep_label, "n": len(per_focal_region)})
        for _, row in per_focal_region.iterrows():
            width_rows.append({"condition": condition, "rep": rep_label,
                               "width_um": float(row["width_um"]), "truncated": bool(row["truncated"])})
widths = pd.DataFrame(width_rows)
counts = pd.DataFrame(count_rows)
widths.merge(counts, on=["condition", "rep"], validate="many_to_one").to_csv(out_csv, index=False)
y_top = widths["width_um"].max() * 1.1


## Save the median width for each condition
def rep_median(condition, rep):
    vals = widths[(widths.condition == condition) & (widths.rep == rep)]["width_um"]
    return round(float(vals.median()), 1) if len(vals) else np.nan


summary_rows = []
for condition in conditions:
    tp = widths[widths.condition == condition]
    cn = counts[counts.condition == condition].set_index("rep")["n"]
    w = tp["width_um"]
    q1, q3 = (np.percentile(w, [25, 75]) if len(w) else (np.nan, np.nan))
    summary_rows.append({"condition": condition,
                         "n_rep1": int(cn.get("rep1", 0)), "n_rep2": int(cn.get("rep2", 0)),
                         "median_width_um": round(float(w.median()), 1) if len(w) else np.nan,
                         "median_width_um_rep1": rep_median(condition, "rep1"),
                         "median_width_um_rep2": rep_median(condition, "rep2"),
                         "min_width_um": round(float(w.min()), 1) if len(w) else np.nan,
                         "max_width_um": round(float(w.max()), 1) if len(w) else np.nan,
                         "iqr_lo_um": round(float(q1), 1) if len(w) else np.nan,
                         "iqr_hi_um": round(float(q3), 1) if len(w) else np.nan})
pd.DataFrame(summary_rows).to_csv(out_summary, index=False)

## Focal region size box per condition with every region as a jittered dot rep1 open rep2 filled
## and the region count for each retina on top
figure, axes = plt.subplots(figsize=(5.6, 4.2))
for condition, x_pos in x_positions.items():
    color = paper_style.CONDITION_COLORS[condition]
    tp = widths[widths.condition == condition]
    if len(tp):
        box = axes.boxplot([tp["width_um"].to_numpy()], positions=[x_pos], widths=0.5, showfliers=False,
                           patch_artist=True, medianprops=dict(color="#222", lw=1.6),
                           whiskerprops=dict(color=color, lw=1.0), capprops=dict(color=color, lw=1.0),
                           boxprops=dict(edgecolor=color, lw=1.0))
        box["boxes"][0].set_facecolor(color)
        box["boxes"][0].set_alpha(0.20)
    for rep, filled in (("rep1", False), ("rep2", True)):
        rep_widths = tp[tp.rep == rep]["width_um"].to_numpy()
        axes.scatter(x_pos + random_generator.uniform(-0.14, 0.14, len(rep_widths)), rep_widths, s=30,
                     facecolor=(color if filled else "white"), edgecolor=color, linewidth=1.1, alpha=0.9, zorder=3)
    cn = counts[counts.condition == condition].set_index("rep")["n"]
    axes.text(x_pos, y_top, f"rep1 n={cn['rep1']}\nrep2 n={cn['rep2']}", ha="center", va="bottom",
              fontsize=9, fontweight="bold", color=color)

## Style
axes.set_ylim(0, y_top * 1.16)
axes.set_ylabel("Approximate focal region width (µm)", fontsize=11, fontweight="bold")
axes.set_xticks(list(x_positions.values()))
axes.set_xticklabels([paper_style.condition_label(c) for c in conditions], fontsize=10, rotation=20, ha="right")
for tick_label, condition in zip(axes.get_xticklabels(), conditions):
    tick_label.set_color(paper_style.CONDITION_COLORS[condition])
axes.set_xlim(-0.6, len(conditions) - 0.4)
for spine in ("top", "right"):
    axes.spines[spine].set_visible(False)
axes.legend(handles=[Line2D([0], [0], marker="o", ls="", mfc="white", mec="#555", ms=8, label="rep1"),
                     Line2D([0], [0], marker="o", ls="", mfc="#555", mec="#555", ms=8, label="rep2")],
            fontsize=9, frameon=False, loc="lower right")
figure.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(figure)

## Report
print("wrote", out_png)
for condition in conditions:
    tp = widths[widths.condition == condition]
    cn = counts[counts.condition == condition].set_index("rep")["n"]
    med = f"{tp['width_um'].median():.0f}" if len(tp) else "-"
    print(f"{condition}: n per rep = {cn['rep1']},{cn['rep2']}   median size = {med} µm")
