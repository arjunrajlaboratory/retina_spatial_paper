# Fig 2a Harmony UMAP with a gray backbone and per condition overlay

## Load packages
import sys
import pandas as pd
import matplotlib.pyplot as plt

## Load paper style conventions
sys.path.insert(0, "../../_shared/code")
import paper_style
paper_style.set_style()

## Load files
harmony = "../../_shared/data_processed/umap_coordinates.parquet"
output = "../panels/Fig2a.png"

## Define conditions and cell type order
conditions = paper_style.CONDITION_ORDER
legend_order = ["rod", "cone", "bipolar", "amacrine_gaba", "amacrine_gly",
                "horizontal", "rgc", "muller", "microglia", "vascular", "rpe"]

## Read data
data = pd.read_parquet(harmony)

## Assign colors with gray for missing types
colors = dict(paper_style.CELLTYPE_COLORS)
for cell_type in data["final_label"].unique():
    if cell_type not in colors:
        colors[cell_type] = "#cccccc"

## Set the plot range with a small margin
umap1 = data["umap1_harmony"].to_numpy()
umap2 = data["umap2_harmony"].to_numpy()
x_limits = (umap1.min() - 0.5, umap1.max() + 0.5)
y_limits = (umap2.min() - 0.5, umap2.max() + 0.5)

## Set up the figure
n_conditions = len(conditions)
panel_width, panel_height = 2.6, 1.9
legend_height = 0.65
figure_width = panel_width * n_conditions + 0.3
figure_height = panel_height + legend_height + 0.7
figure = plt.figure(figsize=(figure_width, figure_height))
grid = figure.add_gridspec(nrows=2, ncols=n_conditions, left=0.02, right=0.99, top=0.90, bottom=0.06,
                           height_ratios=[panel_height, legend_height], wspace=0.10, hspace=0.22)

## Draw one UMAP per condition
background_style = dict(s=0.6, c="#dddddd", lw=0, alpha=0.30, rasterized=True)
overlay_style = dict(s=1.2, lw=0, alpha=0.40, rasterized=True)
for column_index, cond in enumerate(conditions):
    axes = figure.add_subplot(grid[0, column_index])
    # Draw the gray backbone of all cells
    axes.scatter(umap1, umap2, **background_style)
    # Overlay this condition colored by cell type
    condition_cells = data[data["condition"] == cond]
    axes.scatter(condition_cells["umap1_harmony"], condition_cells["umap2_harmony"],
                 c=condition_cells["final_label"].map(colors).to_numpy(), **overlay_style)
    axes.set_xlim(*x_limits)
    axes.set_ylim(*y_limits)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)

    # Draw orientation arrows on the leftmost panel only
    if column_index == 0:
        arrow_y_length = 0.20
        arrow_x_length = arrow_y_length * panel_height / panel_width
        arrow_x0, arrow_y0 = 0.02, -0.06
        arrow_props = dict(arrowstyle="-|>", color="#2A2A2A", lw=1.3, shrinkA=0, shrinkB=0)
        arrow_horizontal = axes.annotate("", xy=(arrow_x0 + arrow_x_length, arrow_y0), xytext=(arrow_x0, arrow_y0),
                                         xycoords="axes fraction", arrowprops=arrow_props, zorder=11, annotation_clip=False)
        arrow_vertical = axes.annotate("", xy=(arrow_x0, arrow_y0 + arrow_y_length), xytext=(arrow_x0, arrow_y0),
                                       xycoords="axes fraction", arrowprops=arrow_props, zorder=11, annotation_clip=False)
        arrow_horizontal.arrow_patch.set_clip_on(False)
        arrow_vertical.arrow_patch.set_clip_on(False)
        axes.text(arrow_x0 + arrow_x_length / 2, arrow_y0 - 0.045, "UMAP 1", transform=axes.transAxes, ha="center", va="top",
                  fontsize=9, fontweight="bold", color="#2A2A2A", zorder=11, clip_on=False)
        axes.text(arrow_x0 - 0.035, arrow_y0 + arrow_y_length / 2, "UMAP 2", transform=axes.transAxes, ha="right", va="center",
                  rotation=90, fontsize=9, fontweight="bold", color="#2A2A2A", zorder=11, clip_on=False)

    # Add the title above the panel with the genotype spelled out
    genotype_code, age = cond.split("_", 1)
    genotype = r"$\mathsf{Lca5}$ mutant" if genotype_code == "LCA5" else "Wild-type"
    axes.set_title(f"{genotype} {paper_style.age_label(age)}", color=paper_style.condition_color_simple(cond), fontsize=16, weight="bold", pad=14)

    # Add the cell count for each retina in the top left corner
    n_rep1 = int(((data["condition"] == cond) & (data["rep"] == "rep1")).sum())
    n_rep2 = int(((data["condition"] == cond) & (data["rep"] == "rep2")).sum())
    axes.text(0.03, 0.97, f"rep1: {n_rep1:,}\nrep2: {n_rep2:,}", transform=axes.transAxes, ha="left", va="top",
              color="black", fontsize=6.5, weight="normal", linespacing=1.15)

## Draw the shared legend row
legend_axes = figure.add_subplot(grid[1, :])
legend_axes.axis("off")
handles = []
for cell_type in legend_order:
    handles.append(plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[cell_type],
                              markeredgecolor="none", markersize=11, label=paper_style.celltype_label(cell_type)))
legend_axes.legend(handles=handles, loc="center", ncol=len(handles), frameon=False, fontsize=11,
                   handletextpad=0.4, columnspacing=1.3, labelspacing=0.4, borderaxespad=0.1)

## Save the panel
figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(figure)
print("wrote", output)
