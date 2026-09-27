# Fig S2b raw rod and cone counts per replicate

## Load packages
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Load paper style conventions
sys.path.insert(0, "../../_shared/code")
import paper_style
paper_style.set_style()

## Load files
per_cell = "../../_shared/data_processed/per_cell_gene_counts.parquet"
output_png = "../panels/FigS2b.png"
output_table = "../data_processed/FigS2b.csv"

## Define conditions and colors
conditions = paper_style.CONDITION_ORDER
condition_reps = {
    "WT_P21":   ["WT_P21_rep1", "WT_P21_rep2"],
    "WT_P64":   ["WT_P64_rep1", "WT_P64_rep2"],
    "LCA5_P21": ["LCA5_P21_rep1", "LCA5_P21_rep2"],
    "LCA5_P30": ["LCA5_P30_rep1", "LCA5_P30_rep2"],
    "LCA5_P64": ["LCA5_P64_rep1", "LCA5_P64_rep2"],
}
rod_color = paper_style.CELLTYPE_COLORS["rod"]
cone_color = paper_style.CELLTYPE_COLORS["cone"]

## Define broken axis limits
y_limits_top = (4800, 13000)
y_limits_bottom = (0, 1900)
dodge = 0.16

## Load data and keep typed cells
data = pd.read_parquet(per_cell)
data = data[data["cell_type"] != "unassigned"]

## Count rods and cones per sample
rows = []
for condition in conditions:
    for sample in condition_reps[condition]:
        sample_cells = data[data["sample_id"] == sample]
        rows.append({"sample": sample, "condition": condition,
                     "rep": "rep1" if sample.endswith("_rep1") else "rep2",
                     "n_rod": int((sample_cells["cell_type"] == "rod").sum()),
                     "n_cone": int((sample_cells["cell_type"] == "cone").sum())})
table = pd.DataFrame(rows)
table.to_csv(output_table, index=False)

## Set up the broken axis figure
bar_positions = np.arange(len(conditions))
figure, (axes_top, axes_bottom) = plt.subplots(2, 1, sharex=True, figsize=(3.4, 3.2),
    gridspec_kw=dict(height_ratios=[1.0, 1.0], hspace=0.08, left=0.19, right=0.97, top=0.90, bottom=0.20))

## Draw both replicate dots for rods and cones
for index, condition in enumerate(conditions):
    rods = table.loc[table["condition"] == condition, "n_rod"].to_numpy(dtype=float)
    cones = table.loc[table["condition"] == condition, "n_cone"].to_numpy(dtype=float)
    for axes in (axes_top, axes_bottom):
        axes.scatter(np.full_like(rods, bar_positions[index] - dodge), rods, s=26, color=rod_color, edgecolor="white", linewidth=0.6, zorder=3)
        axes.scatter(np.full_like(cones, bar_positions[index] + dodge), cones, s=26, color=cone_color, edgecolor="white", linewidth=0.6, zorder=3)
axes_top.set_ylim(*y_limits_top)
axes_bottom.set_ylim(*y_limits_bottom)

## Hide the inner spines and add the break marks
axes_top.spines["bottom"].set_visible(False)
axes_bottom.spines["top"].set_visible(False)
axes_top.tick_params(axis="x", which="both", length=0)
for axes in (axes_top, axes_bottom):
    for spine_name in ("top", "right"):
        axes.spines[spine_name].set_visible(False)
    axes.tick_params(axis="y", labelsize=8)
break_size = 0.012
break_style = dict(transform=axes_top.transAxes, color="black", clip_on=False, lw=0.9)
axes_top.plot((-break_size, +break_size), (-break_size, +break_size), **break_style)
break_style.update(transform=axes_bottom.transAxes)
axes_bottom.plot((-break_size, +break_size), (1 - break_size, 1 + break_size), **break_style)

## Add the x axis labels
axes_bottom.set_xticks(bar_positions)
axes_bottom.set_xticklabels([paper_style.condition_label(c) for c in conditions], fontsize=8, fontweight="bold", rotation=20, ha="right")
for index, condition in enumerate(conditions):
    axes_bottom.get_xticklabels()[index].set_color(paper_style.condition_color_simple(condition))
axes_bottom.set_xlim(-0.55, len(conditions) - 0.45)
figure.text(0.035, 0.55, "Typed cells per section", rotation=90, va="center", ha="center", fontweight="bold", fontsize=9)

## Add the legend
legend_handles = [plt.Line2D([0], [0], marker="o", color=rod_color, lw=0, markeredgecolor="white", markeredgewidth=0.6, markersize=6, label="rod"),
                  plt.Line2D([0], [0], marker="o", color=cone_color, lw=0, markeredgecolor="white", markeredgewidth=0.6, markersize=6, label="cone")]
axes_top.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=8, handlelength=1.2, borderaxespad=0.2)

## Save the panel
figure.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(figure)
print("wrote", output_png)
