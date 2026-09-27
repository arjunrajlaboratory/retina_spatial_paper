# Fig 2d cell type composition stacked bars as fraction of typed cell bodies

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
output_png = "../panels/Fig2d.png"
output_table = "../data_processed/Fig2d.csv"

## Define conditions and stack order
conditions = paper_style.CONDITION_ORDER
condition_reps = {
    "WT_P21":   ["WT_P21_rep1", "WT_P21_rep2"],
    "WT_P64":   ["WT_P64_rep1", "WT_P64_rep2"],
    "LCA5_P21": ["LCA5_P21_rep1", "LCA5_P21_rep2"],
    "LCA5_P30": ["LCA5_P30_rep1", "LCA5_P30_rep2"],
    "LCA5_P64": ["LCA5_P64_rep1", "LCA5_P64_rep2"],
}
colors = paper_style.CELLTYPE_COLORS
stack_order = ["rod", "cone", "bipolar", "amacrine_gaba", "amacrine_gly",
               "horizontal", "rgc", "muller", "microglia", "vascular", "rpe"]

## Load data and keep typed cells
data = pd.read_parquet(per_cell)
data = data[data["cell_type"] != "unassigned"]

## Compute cell type fraction per sample
rows = []
for condition in conditions:
    for sample in condition_reps[condition]:
        sample_cells = data[data["sample_id"] == sample]
        n_total = len(sample_cells)
        for cell_type in stack_order:
            cell_count = int((sample_cells["cell_type"] == cell_type).sum())
            rows.append({"sample": sample, "condition": condition,
                         "rep": "rep1" if sample.endswith("_rep1") else "rep2",
                         "celltype": cell_type, "n_cells": cell_count, "n_total_typed": n_total,
                         "fraction": cell_count / n_total})
per_sample = pd.DataFrame(rows)
per_sample.to_csv(output_table, index=False)

## Average the fractions across reps per condition
by_condition = per_sample.groupby(["condition", "celltype"], as_index=False)["fraction"].mean()
wide = by_condition.pivot(index="condition", columns="celltype", values="fraction").reindex(conditions)

## Draw the stacked bars
figure, axes = plt.subplots(1, 1, figsize=(2.9, 2.8),
                            gridspec_kw=dict(left=0.19, right=0.98, top=0.96, bottom=0.18))
bar_bottom = np.zeros(len(conditions))
bar_positions = np.arange(len(conditions))
for cell_type in stack_order:
    values = wide[cell_type].to_numpy() if cell_type in wide.columns else np.zeros(len(conditions))
    axes.bar(bar_positions, values, bottom=bar_bottom, width=0.92, color=colors[cell_type], edgecolor="none")
    bar_bottom = bar_bottom + values
axes.set_ylabel("Fraction of typed cells", fontweight="bold")
axes.set_ylim(0, 1)
axes.set_xticks(bar_positions)
axes.set_xticklabels([paper_style.condition_label(c) for c in conditions], fontsize=8, fontweight="bold", rotation=20, ha="right")
for index, condition in enumerate(conditions):
    axes.get_xticklabels()[index].set_color(paper_style.condition_color_simple(condition))
axes.set_xlim(-0.55, len(conditions) - 0.45)
axes.tick_params(axis="y", labelsize=8)
for spine_name in ("top", "right"):
    axes.spines[spine_name].set_visible(False)

## Save the panel
figure.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(figure)
print("wrote", output_png)
