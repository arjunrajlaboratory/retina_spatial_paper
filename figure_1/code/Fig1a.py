# Fig 1a cell type color legend for the seqFISH atlas

## Load packages
import sys
import pandas as pd
import matplotlib.pyplot as plt

## Load paper style conventions
sys.path.insert(0, "../../_shared/code")
import paper_style
paper_style.set_style()
plt.rcParams.update({"font.weight": "normal", "axes.labelweight": "normal",
                     "axes.titleweight": "normal"})

## Load files
harmony = "../../_shared/data_processed/umap_coordinates.parquet"
output = "../panels/Fig1a.png"

## Define cell types in legend order
type_order = ["rod", "cone", "bipolar", "amacrine_gaba", "amacrine_gly",
              "horizontal", "rgc", "muller", "microglia", "vascular", "rpe"]

## Assign colors
colors = dict(paper_style.CELLTYPE_COLORS)
present = set(pd.read_parquet(harmony, columns=["final_label"])["final_label"])
for cell_type in present:
    if cell_type not in colors:
        colors[cell_type] = "#cccccc"

## Keep only the types present
order = []
for cell_type in type_order:
    if cell_type in present:
        order.append(cell_type)

## Draw the legend
figure = plt.figure(figsize=(5.2, 2.4), facecolor="white")
axes = figure.add_axes([0.0, 0.0, 1.0, 1.0])
axes.axis("off")
handles = []
for cell_type in order:
    handles.append(plt.Line2D([0], [0], marker="o", color="none",
                              markerfacecolor=colors[cell_type], markeredgecolor="none",
                              markersize=13, label=paper_style.celltype_label(cell_type)))
legend = axes.legend(handles=handles, loc="center", ncol=3, frameon=True,
                     fontsize=13, handletextpad=0.3, columnspacing=1.6,
                     labelspacing=0.9, borderpad=0.9, title="Cell type", title_fontsize=16)
legend.get_title().set_fontweight("bold")
legend.get_title().set_color("black")
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(1.0)
legend.get_frame().set_facecolor("white")

## Save the panel
figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(figure)
print("wrote", output)
