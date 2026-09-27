# Fig 3b rod superior/inferior transcriptional organization

## Load packages
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

## Load paper style conventions and the shared superior/inferior opsin field
sys.path.insert(0, "../../_shared/code")
import paper_style
from _superior_inferior_axis import smoothed_opsin_field
paper_style.set_style()

## Load files
per_cell = "../../_shared/data_processed/per_cell_gene_counts.parquet"
output_dir = "../panels"
data_dir = "../data_processed"
superior_color, inferior_color = "#2ca02c", "#c800a8"
up_color, down_color = "#d62728", "#1f77b4"
map_genes = {"Edn2", "Gadd45b", "Nxnl1"}

## Define the analysis parameters
wt_reps = ["WT_P21_rep1", "WT_P21_rep2"]
lca_reps = ["LCA5_P21_rep1", "LCA5_P21_rep2"]

## Load the changed genes from the Fig 3a response table
response = pd.read_csv(f"{data_dir}/Fig3a.csv")
changed = response[response.changed][["gene", "group_log2fc"]].reset_index(drop=True)
genes = changed.gene.tolist()

## Load the per cell transcript counts and cone opsin signal
gene_cols = [f"body_{g}" for g in genes]
columns = ["sample_id", "cell_type", "cell_area_um2", "cx_um", "cy_um", "body_Opn1mw", "body_Opn1sw"] + gene_cols
cells = pd.read_parquet(per_cell, columns=list(dict.fromkeys(columns)))

## Correlate rod transcript density with the continuous sigma 120 cone opsin field in each retina
def si_per_sample(sample):
    frame = cells[cells.sample_id == sample]
    if frame.empty:
        return {gene: np.nan for gene in genes}
    frame = frame.assign(sup=smoothed_opsin_field(frame))
    rods = frame[frame.cell_type == "rod"]
    field_rank = rankdata(rods["sup"].to_numpy(float))
    area = rods["cell_area_um2"].to_numpy(float)
    out = {}
    for gene in genes:
        density = rods[f"body_{gene}"].to_numpy(float) / area
        logged = np.log1p(density)
        out[gene] = float(np.corrcoef(rankdata(logged), field_rank)[0, 1]) if np.std(logged) > 0 else np.nan
    return out

lca = {sample: si_per_sample(sample) for sample in lca_reps}
wt = {sample: si_per_sample(sample) for sample in wt_reps}

## Assemble the ranked superior/inferior table for LCA5 with the wild-type reference
rows = []
for gene in genes:
    l1, l2 = lca[lca_reps[0]][gene], lca[lca_reps[1]][gene]
    w1, w2 = wt[wt_reps[0]][gene], wt[wt_reps[1]][gene]
    if not np.all(np.isfinite([l1, l2, w1, w2])):
        raise ValueError(f"{gene}: a biological replicate has no S/I value (lca={l1},{l2} wt={w1},{w2})")
    rows.append(dict(gene=gene, lca1=l1, lca2=l2, mean=0.5 * (l1 + l2),
                     wt1=w1, wt2=w2, wt_mean=0.5 * (w1 + w2)))
table_all = pd.DataFrame(rows).merge(changed, on="gene", validate="one_to_one")
table = table_all.sort_values("mean", ascending=False).reset_index(drop=True)
table.to_csv(f"{data_dir}/Fig3b.csv", index=False)

## Draw the ranked S/I bars with replicate dots
bar_positions = np.arange(len(table))
bar_colors = np.where(table["group_log2fc"] >= 0, up_color, down_color)
figure_width = max(10.0, 0.34 * len(table))
figure, axes = plt.subplots(figsize=(figure_width, 6.4), facecolor="white")
axes.bar(bar_positions, table["mean"], width=0.55, color=bar_colors, alpha=0.6, edgecolor="none", zorder=2)
axes.scatter(bar_positions, table["lca1"], s=12, facecolor="white", edgecolor="#333333", linewidths=0.6, zorder=4)
axes.scatter(bar_positions, table["lca2"], s=12, facecolor="#333333", edgecolor="#333333", linewidths=0.6, zorder=4)
axes.axhline(0, color="#999999", lw=0.8, zorder=1)
axes.set_xlim(-0.7, len(table) - 0.3)
axes.set_xticks(bar_positions)
axes.set_xticklabels(table["gene"], fontsize=12, fontstyle="italic", rotation=90)
axes.tick_params(axis="x", length=0)
# Box the genes shown as example maps in Fig 3c
for tick_label in axes.get_xticklabels():
    if tick_label.get_text() in map_genes:
        tick_label.set_bbox(dict(boxstyle="square,pad=0.25", facecolor="none", edgecolor="black", lw=1.0))
axes.set_ylabel("Superior–inferior bias (Spearman $\\rho$)", fontsize=12, fontweight="bold")
for tick_label in axes.get_yticklabels():
    tick_label.set_fontweight("bold")

## Add the superior and inferior anchors
axes.text(-0.10, 0.99, "superior-biased", transform=axes.transAxes, rotation=90, va="top", ha="center", fontsize=11, fontweight="bold", color=superior_color)
axes.text(-0.10, 0.01, "inferior-biased", transform=axes.transAxes, rotation=90, va="bottom", ha="center", fontsize=11, fontweight="bold", color=inferior_color)
for spine_name in ("top", "right"):
    axes.spines[spine_name].set_visible(False)

## Add the legend
legend_handles = [plt.Line2D([0], [0], marker="s", ls="", ms=8, mfc=up_color, mec="none", label=f"expression up in {paper_style.GENOTYPE}"),
                  plt.Line2D([0], [0], marker="s", ls="", ms=8, mfc=down_color, mec="none", label=f"expression down in {paper_style.GENOTYPE}"),
                  plt.Line2D([0], [0], marker="o", ls="", ms=6, mfc="white", mec="#333", label="replicate 1"),
                  plt.Line2D([0], [0], marker="o", ls="", ms=6, mfc="#333", mec="#333", label="replicate 2")]
axes.legend(handles=legend_handles, loc="upper right", fontsize=9, frameon=False)

## Save the panel
figure.subplots_adjust(left=0.135, right=0.995, top=0.97, bottom=0.22)
figure.savefig(f"{output_dir}/Fig3b.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(figure)
print("wrote", f"{output_dir}/Fig3b.png")
