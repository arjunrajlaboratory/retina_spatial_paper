# Fig S4h microglia redistribute into the ONL in LCA5

## Load packages
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

## Load paper style conventions
sys.path.insert(0, "../../_shared/code")
import paper_style
paper_style.set_style()

## Load files
master = "../../_shared/data_processed/umap_coordinates.parquet"
out_png = "../panels/FigS4h.png"
out_csv = "../data_plot/FigS4h.csv"

## Define layer order
layer_order = ["ONL", "INL", "IPL", "GCL", "bg"]

## Load microglia and compute the within sample fraction per layer
data = pd.read_parquet(master, columns=["sample", "condition", "rep", "final_label", "argmax_layer", "cx_um", "cy_um"])
microglia = data[data["final_label"] == "microglia"].copy()
counts = (microglia.groupby(["condition", "sample", "rep", "argmax_layer"]).size()
          .unstack("argmax_layer", fill_value=0).reindex(columns=layer_order, fill_value=0))
counts["TOTAL"] = counts.sum(axis=1)
fractions = counts[layer_order].div(counts["TOTAL"], axis=0)
fractions.columns = [layer + "_frac" for layer in layer_order]
composition = pd.concat([counts, fractions], axis=1).reset_index()
condition_rank = {cond: index for index, cond in enumerate(paper_style.CONDITION_ORDER)}
composition = composition.sort_values(["condition", "rep"],
              key=lambda series: series.map(condition_rank) if series.name == "condition" else series).reset_index(drop=True)
composition.to_csv(out_csv, index=False)

## Draw a mean bar per condition with the two replicates as circles rep1 open and rep2 filled
figure, axes = plt.subplots(figsize=(4.2, 4.2))
for x_position, cond in enumerate(paper_style.CONDITION_ORDER):
    condition_rows = composition[composition["condition"] == cond].sort_values("rep")
    color = paper_style.condition_color_simple(cond)
    values = [v for v in condition_rows["ONL_frac"].to_numpy() if np.isfinite(v)]
    if values:
        axes.bar([x_position], [float(np.mean(values))], width=0.6, color=color, alpha=0.40, edgecolor=color, linewidth=1.3, zorder=2)
    for value, rep_face, dx in zip(condition_rows["ONL_frac"].to_numpy(), ("white", "#333333"), (-0.13, 0.13)):
        if np.isfinite(value):
            axes.scatter([x_position + dx], [value], s=85, facecolor=rep_face, edgecolor="#222222", linewidths=1.4, zorder=6)
axes.set_xticks(range(len(paper_style.CONDITION_ORDER)))
axes.set_xticklabels([paper_style.condition_label(c) for c in paper_style.CONDITION_ORDER], rotation=35, ha="right", fontsize=9)
for tick_label, cond in zip(axes.get_xticklabels(), paper_style.CONDITION_ORDER):
    tick_label.set_color(paper_style.condition_color_simple(cond))
axes.set_ylabel("Fraction of microglia in ONL", fontsize=10.5, fontweight="bold")
axes.set_ylim(0, max(0.65, float(composition["ONL_frac"].max()) * 1.10))
axes.set_xlim(-0.6, len(paper_style.CONDITION_ORDER) - 0.4)
axes.set_box_aspect(1)

## Add the replicate legend
legend_handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="#222", markersize=8, markeredgewidth=1.4, label="rep1"),
                  Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#333333", markeredgecolor="#222", markersize=8, label="rep2")]
axes.legend(handles=legend_handles, loc="upper left", fontsize=9, frameon=False)

## Save the panel
figure.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(figure)
print("wrote", out_png)
