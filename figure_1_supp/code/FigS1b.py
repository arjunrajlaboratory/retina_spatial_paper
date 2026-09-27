# Fig S1b cell type by layer occupancy for marker typing validation

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
out_png = "../panels/FigS1b.png"
out_table = "../data_processed/FigS1b_celltype_layer_fractions.parquet"

## Define cell type and layer order
celltype_order = ["rod", "cone", "horizontal", "bipolar", "amacrine_gaba",
                  "amacrine_gly", "muller", "rgc", "microglia", "vascular", "rpe"]
layer_order = ["ONL", "INL", "IPL", "GCL", "bg"]
layer_display = {"bg": "off-layer"}

## Load data and keep marker typed cells
data = pd.read_parquet(per_cell, columns=["sample_id", "cell_type", "argmax_layer"])
typed_cells = data[data["cell_type"] != "unassigned"].copy()
typed_cells["condition"] = (typed_cells["sample_id"].str.replace("_rep1", "", regex=False)
                                                    .str.replace("_rep2", "", regex=False))

## Compute cell type by layer fractions per condition
rows = []
for condition in paper_style.CONDITION_ORDER:
    subset = typed_cells[typed_cells["condition"] == condition]
    crosstab = pd.crosstab(subset["cell_type"], subset["argmax_layer"]).reindex(index=celltype_order, columns=layer_order, fill_value=0)
    count_per_type = crosstab.sum(axis=1)
    fractions = crosstab.div(count_per_type.where(count_per_type > 0, np.nan), axis=0)
    for cell_type in celltype_order:
        for layer in layer_order:
            rows.append({"condition": condition, "cell_type": cell_type, "layer": layer,
                         "fraction": fractions.loc[cell_type, layer], "n_cells": int(count_per_type.get(cell_type, 0))})
pd.DataFrame(rows).to_parquet(out_table, index=False)

## Compute pooled cell type by layer fractions
crosstab_all = pd.crosstab(typed_cells["cell_type"], typed_cells["argmax_layer"]).reindex(index=celltype_order, columns=layer_order, fill_value=0)
count_per_type = crosstab_all.sum(axis=1)
fractions_all = crosstab_all.div(count_per_type.where(count_per_type > 0, np.nan), axis=0)

## Draw the heatmap
celltype_labels = [paper_style.celltype_label(cell_type) for cell_type in celltype_order]
layer_labels = [layer_display.get(layer, layer) for layer in layer_order]
figure, axes = plt.subplots(figsize=(8.1, 3.5), facecolor="white")
image = axes.imshow(fractions_all.to_numpy(dtype=float).T, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
axes.set_xticks(range(len(celltype_order)))
axes.set_xticklabels(celltype_labels, rotation=45, ha="right", fontsize=13)
axes.set_yticks(range(len(layer_order)))
axes.set_yticklabels(layer_labels, fontsize=14)
for spine in axes.spines.values():
    spine.set_visible(True)
    spine.set_color("#cccccc")
    spine.set_linewidth(0.8)
axes.tick_params(length=0)

## Add the colorbar
colorbar = figure.colorbar(image, ax=axes, fraction=0.046, pad=0.03)
colorbar.set_label("Fraction of cell type in layer", fontsize=9, fontweight="bold")
colorbar.ax.tick_params(labelsize=8)
colorbar.outline.set_visible(False)

## Save the panel
figure.tight_layout()
figure.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(figure)
print("wrote", out_png)
