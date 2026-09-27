# Fig S1c replicate concordance of cell type fractions and gene totals

## Load packages
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

## Load paper style conventions
sys.path.insert(0, "../../_shared/code")
import paper_style
paper_style.set_style()

## Load files
per_cell = "../../_shared/data_processed/per_cell_gene_counts.parquet"
output_png = "../panels/FigS1c.png"
output_fraction = "../data_processed/FigS1c_fraction_per_celltype.csv"
output_gene = "../data_processed/FigS1c_gene_cellbody_density.csv"
output_rtable = "../data_processed/FigS1c_replicate_r.csv"

## Define conditions and palettes
conditions = ["WT_P21", "WT_P64", "LCA5_P21", "LCA5_P30", "LCA5_P64"]
condition_reps = {
    "WT_P21":   ["WT_P21_rep1", "WT_P21_rep2"],
    "WT_P64":   ["WT_P64_rep1", "WT_P64_rep2"],
    "LCA5_P21": ["LCA5_P21_rep1", "LCA5_P21_rep2"],
    "LCA5_P30": ["LCA5_P30_rep1", "LCA5_P30_rep2"],
    "LCA5_P64": ["LCA5_P64_rep1", "LCA5_P64_rep2"],
}
non_gene_columns = {"body_total", "body_density_per_um2", "body_source"}
celltype_colors = paper_style.CELLTYPE_COLORS
celltypes = list(celltype_colors.keys())
condition_colors = paper_style.CONDITION_COLORS
condition_labels = {condition: paper_style.condition_label(condition) for condition in conditions}

## Load data and keep marker typed cells
cells = pd.read_parquet(per_cell).copy()

## Compute cell type fraction per sample
fraction_rows = []
for condition in conditions:
    for sample in condition_reps[condition]:
        sample_cells = cells[(cells["sample_id"] == sample) & (cells["cell_type"] != "unassigned")]
        n_total = len(sample_cells)
        for cell_type in celltypes:
            cell_count = int((sample_cells["cell_type"] == cell_type).sum())
            fraction_rows.append({"sample": sample, "condition": condition,
                                  "rep": "rep1" if sample.endswith("_rep1") else "rep2",
                                  "celltype": cell_type, "fraction": cell_count / n_total if n_total else 0.0})
fraction_data = pd.DataFrame(fraction_rows)
fraction_data.to_csv(output_fraction, index=False)

## Compute gene transcript totals per sample
body_columns = [column for column in cells.columns if column.startswith("body_") and column not in non_gene_columns]
gene_rows = []
for condition in conditions:
    for sample in condition_reps[condition]:
        sample_cells = cells[cells["sample_id"] == sample]
        # Compute mean cell transcript density
        density = sample_cells[body_columns].div(sample_cells["cell_area_um2"], axis=0).mean(axis=0)
        for gene, value in density.items():
            gene_rows.append({"sample": sample, "condition": condition,
                              "rep": "rep1" if sample.endswith("_rep1") else "rep2",
                              "gene": gene.replace("body_", ""), "density": float(value)})
gene_data = pd.DataFrame(gene_rows)
gene_data.to_csv(output_gene, index=False)

## Pivot each replicate pair for the fraction panel
wide = fraction_data.assign(cr=fraction_data["condition"] + "_" + fraction_data["rep"])
wide = wide.pivot(index="celltype", columns="cr", values="fraction")
fraction_pairs = []
for condition in conditions:
    rep1_column = condition + "_rep1"
    rep2_column = condition + "_rep2"
    if rep1_column in wide.columns and rep2_column in wide.columns:
        fraction_pairs.append((condition, wide[rep1_column].to_numpy(), wide[rep2_column].to_numpy(), wide.index.tolist()))

## Pivot each replicate pair for the gene total panel
wide = gene_data.assign(cr=gene_data["condition"] + "_" + gene_data["rep"])
wide = wide.pivot(index="gene", columns="cr", values="density")
gene_pairs = []
for condition in conditions:
    rep1_column = condition + "_rep1"
    rep2_column = condition + "_rep2"
    if rep1_column in wide.columns and rep2_column in wide.columns:
        gene_pairs.append((condition, wide[rep1_column].to_numpy(), wide[rep2_column].to_numpy(), wide.index.tolist()))

## Compute per condition log10 Pearson r and rep to rep fold difference
## And report if any sample has an undetected (zero) point
r_fraction = {}
fold_fraction = {}
for condition, rep1_values, rep2_values, keys in fraction_pairs:
    if not ((rep1_values > 0).all() and (rep2_values > 0).all()):
        raise ValueError(f"{condition}: undetected cell type in a replicate; decide how to handle before correlating")
    r_fraction[condition] = float(pearsonr(np.log10(rep1_values), np.log10(rep2_values))[0])
    fold_fraction[condition] = float(np.median(np.maximum(rep1_values, rep2_values) / np.minimum(rep1_values, rep2_values)))
r_gene = {}
fold_gene = {}
for condition, rep1_values, rep2_values, keys in gene_pairs:
    if not ((rep1_values > 0).all() and (rep2_values > 0).all()):
        raise ValueError(f"{condition}: undetected gene in a replicate; decide how to handle before correlating")
    r_gene[condition] = float(pearsonr(np.log10(rep1_values), np.log10(rep2_values))[0])
    fold_gene[condition] = float(np.median(np.maximum(rep1_values, rep2_values) / np.minimum(rep1_values, rep2_values)))

## Save the concordance table
table_rows = []
for condition in r_fraction:
    table_rows.append({"panel": "cell_type_fraction", "condition": condition,
                       "r": r_fraction[condition], "median_rep_fold": fold_fraction[condition]})
for condition in r_gene:
    table_rows.append({"panel": "gene_cellbody_density", "condition": condition,
                       "r": r_gene[condition], "median_rep_fold": fold_gene[condition]})
pd.DataFrame(table_rows).to_csv(output_rtable, index=False)

## Set up the figure
figure, (axes_fraction, axes_gene) = plt.subplots(1, 2, figsize=(7.6, 3.8),
                                                  gridspec_kw=dict(left=0.10, right=0.99, top=0.93, bottom=0.18, wspace=0.32))

## Draw the cell type fraction panel
x_values = []
y_values = []
for condition, rep1_values, rep2_values, pair_keys in fraction_pairs:
    for cell_type, rep1_value, rep2_value in zip(pair_keys, rep1_values, rep2_values):
        axes_fraction.scatter(rep1_value, rep2_value, s=18, c=celltype_colors.get(cell_type, "#888"), edgecolor="none", alpha=0.85)
        x_values.append(rep1_value)
        y_values.append(rep2_value)
x_values = np.array(x_values)
y_values = np.array(y_values)
axis_lo = max(min(x_values.min(), y_values.min()) * 0.7, 1e-4)
axis_hi = max(x_values.max(), y_values.max()) * 1.4
axes_fraction.plot([axis_lo, axis_hi], [axis_lo, axis_hi], "--", color="#888", lw=0.6)
axes_fraction.set_xscale("log")
axes_fraction.set_yscale("log")
axes_fraction.set_xlim(axis_lo, axis_hi)
axes_fraction.set_ylim(axis_lo, axis_hi)
r_values = np.array(list(r_fraction.values()))
axes_fraction.set_title(f"$r$ = {np.median(r_values):.3f}  (per-condition median; {r_values.min():.3f}–{r_values.max():.3f})", fontsize=7, loc="center", pad=3)
axes_fraction.set_xlabel("Rep1 cell type fraction", fontsize=8)
axes_fraction.set_ylabel("Rep2 cell type fraction", fontsize=8)
axes_fraction.tick_params(labelsize=7)
axes_fraction.spines[["top", "right"]].set_visible(False)

## Add the cell type key
celltype_handles = []
for cell_type in celltypes:
    celltype_handles.append(plt.Line2D([0], [0], marker="o", color="none",
                                       markerfacecolor=celltype_colors.get(cell_type, "#888"),
                                       markeredgecolor="none", markersize=4, label=paper_style.celltype_label(cell_type)))
axes_fraction.legend(handles=celltype_handles, loc="upper left", ncol=2, fontsize=5, frameon=False,
                     handletextpad=0.2, columnspacing=0.6, labelspacing=0.2, borderaxespad=0.2)

## Draw the gene total panel
all_points = []
condition_handles = []
for condition, rep1_values, rep2_values, keys in gene_pairs:
    axes_gene.scatter(rep1_values, rep2_values, s=8, c=condition_colors[condition], edgecolor="none", alpha=0.70, rasterized=True)
    all_points.append(np.column_stack([rep1_values, rep2_values]))
    condition_handles.append(plt.Line2D([0], [0], marker="o", color="none",
                                        markerfacecolor=condition_colors[condition], markeredgecolor="none",
                                        markersize=5, label=condition_labels[condition]))
points = np.vstack(all_points)
axis_lo = max(points.min() * 0.7, 1e-6)
axis_hi = points.max() * 1.4
axes_gene.plot([axis_lo, axis_hi], [axis_lo, axis_hi], "--", color="#888", lw=0.6)
axes_gene.set_xscale("log")
axes_gene.set_yscale("log")
axes_gene.set_xlim(axis_lo, axis_hi)
axes_gene.set_ylim(axis_lo, axis_hi)
r_values = np.array(list(r_gene.values()))
axes_gene.set_title(f"$r$ = {np.median(r_values):.3f}  (per-condition median; {r_values.min():.3f}–{r_values.max():.3f})", fontsize=7, loc="center", pad=3)
axes_gene.set_xlabel("Rep1 cell body transcript density (transcripts/µm²)", fontsize=8)
axes_gene.set_ylabel("Rep2 cell body transcript density (transcripts/µm²)", fontsize=8)
axes_gene.tick_params(labelsize=7)
axes_gene.spines[["top", "right"]].set_visible(False)
axes_gene.legend(handles=condition_handles, loc="upper left", fontsize=6, frameon=False,
                 handletextpad=0.3, borderaxespad=0.4)

## Save the panel
figure.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(figure)
print("wrote", output_png)
