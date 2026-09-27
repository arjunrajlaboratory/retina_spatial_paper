# Fig S2a photoreceptor composition of rods vs cones per condition

## Load packages
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

## Load paper style conventions
sys.path.insert(0, "../../_shared/code")
import paper_style
paper_style.set_style()

## Load files
per_cell = "../../_shared/data_processed/per_cell_gene_counts.parquet"
output_png = "../panels/FigS2a.png"
output_table = "../data_processed/FigS2a.csv"

## Define conditions and photoreceptor types
condition_reps = {
    "WT_P21":   ["WT_P21_rep1", "WT_P21_rep2"],
    "WT_P64":   ["WT_P64_rep1", "WT_P64_rep2"],
    "LCA5_P21": ["LCA5_P21_rep1", "LCA5_P21_rep2"],
    "LCA5_P30": ["LCA5_P30_rep1", "LCA5_P30_rep2"],
    "LCA5_P64": ["LCA5_P64_rep1", "LCA5_P64_rep2"],
}
conditions = paper_style.CONDITION_ORDER
photoreceptor_types = ["rod", "cone"]
photoreceptor_colors = {photoreceptor_type: paper_style.CELLTYPE_COLORS[photoreceptor_type] for photoreceptor_type in photoreceptor_types}

## Load data and keep photoreceptors
cells = pd.read_parquet(per_cell, columns=["sample_id", "cell_type"])
cells = cells[cells["cell_type"].isin(photoreceptor_types)]

## Compute rod and cone fraction per sample
rows = []
for condition in conditions:
    for sample in condition_reps[condition]:
        sample_cells = cells[cells["sample_id"] == sample]
        n_rod = int((sample_cells["cell_type"] == "rod").sum())
        n_cone = int((sample_cells["cell_type"] == "cone").sum())
        n_pr = n_rod + n_cone
        rows.append({"condition": condition, "sample": sample,
                     "rep": "rep1" if sample.endswith("_rep1") else "rep2",
                     "n_rod": n_rod, "n_cone": n_cone, "n_pr": n_pr,
                     "rod_fraction": (n_rod / n_pr) if n_pr else np.nan,
                     "cone_fraction": (n_cone / n_pr) if n_pr else np.nan})
per_sample = pd.DataFrame(rows)
per_sample.to_csv(output_table, index=False)

## Average the fractions across reps per condition
by_condition = (per_sample.groupby("condition", as_index=False)[["rod_fraction", "cone_fraction"]].mean()
                .set_index("condition").reindex(conditions).reset_index())

## Draw the stacked bars
figure, axes = plt.subplots(1, 1, figsize=(2.9, 3.0),
                            gridspec_kw=dict(left=0.18, right=0.97, top=0.86, bottom=0.20))
bar_positions = np.arange(len(conditions))
bar_bottom = np.zeros(len(conditions))
for photoreceptor_type in photoreceptor_types:
    values = by_condition[f"{photoreceptor_type}_fraction"].to_numpy()
    axes.bar(bar_positions, values, bottom=bar_bottom, width=0.90, color=photoreceptor_colors[photoreceptor_type], edgecolor="none")
    bar_bottom = bar_bottom + values
axes.set_ylabel("Fraction of photoreceptors", fontweight="bold")
axes.set_xticks(bar_positions)
axes.set_xticklabels([paper_style.condition_label(c) for c in conditions], fontsize=8, fontweight="bold", rotation=20, ha="right")
for index, condition in enumerate(conditions):
    axes.get_xticklabels()[index].set_color(paper_style.condition_color_simple(condition))
axes.set_xlim(-0.55, len(conditions) - 0.45)
axes.tick_params(axis="y", labelsize=8)
for spine_name in ("top", "right"):
    axes.spines[spine_name].set_visible(False)
axes.set_ylim(0, 1)

## Add the legend
legend_handles = [mpatches.Patch(facecolor=photoreceptor_colors["rod"], edgecolor="none", label="rod"),
                  mpatches.Patch(facecolor=photoreceptor_colors["cone"], edgecolor="none", label="cone")]
legend = figure.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, fontsize=8,
                       bbox_to_anchor=(0.5, 1.005), columnspacing=1.4, handlelength=1.1)
for text in legend.get_texts():
    text.set_fontweight("bold")

## Save the panel
figure.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(figure)
print("wrote", output_png)
