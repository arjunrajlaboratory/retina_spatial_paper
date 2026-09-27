# Fig 1b marker gene dot plot (cell type x marker)

## Load packages
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

## Load paper style conventions
sys.path.insert(0, "../../_shared/code")
import paper_style
paper_style.set_style()

## Load files
per_cell = "../../_shared/data_processed/per_cell_gene_counts.parquet"
out_panel = "../panels/Fig1b.png"
out_mean = "../data_processed/Fig1b_dotplot_mean.csv"
out_f = "../data_processed/Fig1b_dotplot_frac.csv"

## Define cell types and marker genes
type_order = ["rod", "cone", "bipolar", "amacrine_gaba", "amacrine_gly",
              "horizontal", "rgc", "muller", "microglia", "vascular", "rpe"]
gene_groups = [
    ("rod",           ["Nrl", "Nr2e3", "Pde6b"]),
    ("cone",          ["Arr3", "Opn1mw", "Opn1sw"]),
    ("bipolar",       ["Vsx2", "Prkca", "Grm6", "Vsx1", "Scgn"]),
    ("amacrine_gaba", ["Gad1", "Slc32a1"]),
    ("amacrine_gly",  ["Slc6a9"]),
    ("horizontal",    ["Onecut2", "Calb1"]),
    ("rgc",           ["Rbpms", "Pou4f2", "Nefh"]),
    ("muller",        ["Sox9", "Slc1a3", "Glul", "Aqp4"]),
    ("microglia",     ["Csf1r", "Cx3cr1", "C1qa"]),
    ("vascular",      ["Cldn5", "Pecam1", "Rgs5"]),
    ("rpe",           ["Rpe65"]),
]
genes = []
for cell_type, group in gene_groups:
    genes += group
body_columns = ["body_" + gene for gene in genes]

## Load data
data = pd.read_parquet(per_cell, columns=["cell_type", "cell_area_um2"] + body_columns)
data = data[data["cell_type"].isin(type_order)]

## Construct dot plot matrices
# Compute mean log1p transcript density per gene per cell type
density = np.log1p(data[body_columns].div(data["cell_area_um2"], axis=0))
means = density.groupby(data["cell_type"]).mean().loc[type_order]
# Compute fraction of cells expressing each gene
fractions = (data[body_columns] > 0).groupby(data["cell_type"]).mean().loc[type_order]
# Save matrices then convert to arrays
means.columns = genes
fractions.columns = genes
means.to_csv(out_mean)
fractions.to_csv(out_f)
means = means.to_numpy()
fractions = fractions.to_numpy()

## Compute gene positions along the x axis
gene_x_positions = []
group_spans = []
x_position = 0.0
for cell_type, group in gene_groups:
    if gene_x_positions:
        x_position += 0.10
    start = x_position
    for _ in group:
        gene_x_positions.append(x_position)
        x_position += 1.0
    group_spans.append((cell_type, start, x_position - 1.0))
gene_x_positions = np.array(gene_x_positions)

## Set up the figure
figure = plt.figure(figsize=(10.71, 6.18), facecolor="white")
axes = figure.add_subplot(1, 1, 1)
axes.set_xlim(-0.7, 30.6)
axes.set_ylim(10.5, -2.6)
figure.subplots_adjust(left=0.15, right=0.83, top=0.86, bottom=0.20)

## Draw the left color strip (cell type per row)
colors = paper_style.CELLTYPE_COLORS
swatch_width = 0.25786785
for row_index, cell_type in enumerate(type_order):
    axes.add_patch(Rectangle((-0.5 - swatch_width, row_index - 0.5), swatch_width, 1.0,
                             facecolor=colors[cell_type], edgecolor="none", clip_on=False, zorder=3))

## Draw the top color strip (cell type per gene group)
bar_thickness = 0.19270374
for group_index, (cell_type, start, end) in enumerate(group_spans):
    left = (start - 0.5) if group_index == 0 else (group_spans[group_index - 1][2] + start) / 2.0
    right = (end + 0.5) if group_index == len(group_spans) - 1 else (end + group_spans[group_index + 1][1]) / 2.0
    axes.add_patch(Rectangle((left, -0.5 - bar_thickness), right - left, bar_thickness,
                             facecolor=colors[cell_type], edgecolor="none", clip_on=False, zorder=3))
    axes.text((start + end) / 2.0, -0.5 - bar_thickness - 0.22, paper_style.celltype_label(cell_type),
              ha="left", va="bottom", rotation=60, fontsize=9,
              fontweight="bold", color="#2A2A2A", clip_on=False)

## Draw the dots colored by mean density and sized by fraction
colormap = plt.get_cmap("Reds")
# Scale white to red from zero up to the maximum mean density
value_max = float(np.nanmax(means))
norm = matplotlib.colors.Normalize(vmin=0.0, vmax=value_max)
for row_index in range(len(type_order)):
    for col_index in range(len(genes)):
        if not (np.isfinite(means[row_index, col_index]) and np.isfinite(fractions[row_index, col_index])):
            continue
        fraction = np.clip(fractions[row_index, col_index], 0.0, 1.0)
        if fraction <= 0.0:
            continue
        size = 138 * fraction
        axes.scatter([gene_x_positions[col_index]], [row_index], s=float(size),
                     color=colormap(norm(means[row_index, col_index])), edgecolor="none", zorder=2)

## Add axis labels
axes.set_xticks(gene_x_positions)
axes.set_xticklabels(genes, rotation=45, ha="right", fontsize=8.5,
                     fontstyle="italic", fontweight="bold", color="#2A2A2A")
axes.set_yticks(np.arange(len(type_order)))
axes.set_yticklabels([paper_style.celltype_label(cell_type) for cell_type in type_order], fontsize=9.5,
                     color="#2A2A2A", fontweight="bold")
axes.tick_params(axis="y", length=0, pad=14)
axes.tick_params(axis="x", length=0, pad=2)
for spine in axes.spines.values():
    spine.set_visible(False)

## Add the colorbar
axes_position = axes.get_position()
colorbar_axes = figure.add_axes([axes_position.x1 + 0.02, axes_position.y0 + 0.50 * axes_position.height, 0.008, 0.34 * axes_position.height])
scalar_mappable = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
colorbar = figure.colorbar(scalar_mappable, cax=colorbar_axes)
colorbar.set_label("Mean transcript density\n(log1p)", fontsize=8, color="#2A2A2A")
colorbar.ax.tick_params(labelsize=7, colors="#2A2A2A")
colorbar.outline.set_visible(False)

## Add the size key
handles = []
for fraction in [0.10, 0.30, 0.60, 1.0]:
    size = np.sqrt(138 * fraction)
    handles.append(Line2D([0], [0], marker="o", linestyle="none",
                          markerfacecolor="#888888", markeredgecolor="none",
                          markersize=size, label=f"{int(fraction * 100)}%"))
figure.legend(handles=handles, title="fraction of cells\nexpressing gene", loc="upper left",
              bbox_to_anchor=(axes_position.x1 + 0.015, axes_position.y0 + 0.40 * axes_position.height),
              frameon=False, fontsize=8, title_fontsize=8,
              labelspacing=0.9, handletextpad=1.0)

## Save the panel
figure.savefig(out_panel, dpi=250, bbox_inches="tight")
print("wrote", out_panel)
