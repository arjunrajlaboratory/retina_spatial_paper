# Fig S4g local Muller reactivity vs Muller abundance within each retina

## Load packages
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

## Load paper style conventions and the shared focal region functions
sys.path.insert(0, "../../_shared/code")
import paper_style
import _focal_regions as focal
paper_style.set_style()

## Load files
per_cell_path = "../../_shared/data_processed/per_cell_gene_counts.parquet"
density_path = "../../_shared/data_raw/gene_density_by_layer.parquet"
out_png = "../panels/FigS4g.png"
out_tbl = "../data_plot/FigS4g.parquet"
out_rho = "../data_plot/FigS4g_rho.csv"

## Define parameters
samples = ["LCA5_P21_rep1", "LCA5_P21_rep2", "LCA5_P30_rep1", "LCA5_P30_rep2"]


## Compute per column Müller cell body density and gliosis transcript density for one sample
def per_sample(sample):
    skeletons = pd.read_parquet(focal.FT / "skeletons.parquet")
    skeletons = skeletons[skeletons["sample"] == sample]
    columns_meta = pd.read_parquet(focal.FT / "columns_meta.parquet")
    columns_meta = columns_meta[columns_meta["sample"] == sample].sort_values("col_id_local").reset_index(drop=True)
    cells = pd.read_parquet(per_cell_path, columns=["sample_id", "cell_type", "cx_um", "cy_um"])
    muller = cells[(cells.sample_id == sample) & (cells.cell_type == "muller")].copy()
    muller = focal._project_nearest(muller, skeletons, columns_meta)
    muller_counts = muller.groupby("col_id_local").size().rename("n_mu")
    # Layer areas come from columns_meta, which defines an area for every column
    columns = columns_meta.set_index("col_id_local")[["ONL_area_um2", "INL_area_um2"]]
    valid = set(int(c) for c in focal.focal_region_columns(sample).query("valid_tissue")["col_id_local"])
    columns = columns[columns.index.isin(valid)]
    # A column missing from the count has zero Müller cell bodies
    columns["n_mu"] = muller_counts.reindex(columns.index, fill_value=0)
    density = pd.read_parquet(density_path, columns=["sample_id", "col_id_local", "gene", "layer", "transcript_count"])
    density = density[(density.sample_id == sample) & density.layer.isin(["ONL", "INL"])]
    # Müller cell body density over the INL area
    columns["mu_density"] = np.where(columns.INL_area_um2 > 0, columns.n_mu / columns.INL_area_um2, 0.0)
    # Gliosis transcript density over the ONL+INL area
    gli_count = density[density.gene.isin(["Gfap", "Serpina3n"])].groupby("col_id_local").transcript_count.sum()
    # A column missing from the count has zero gliosis transcripts
    columns["n_gli"] = gli_count.reindex(columns.index, fill_value=0)
    onl_inl_area = columns.ONL_area_um2 + columns.INL_area_um2
    columns["gli_density"] = np.where(onl_inl_area > 0, columns.n_gli / onl_inl_area, 0.0)
    return columns[["mu_density", "gli_density"]].copy()


## Spearman correlation within each retina then the median over the four retinas
per_sample_tables = {sample: per_sample(sample) for sample in samples}
rho = {sample: float(spearmanr(t.mu_density, t.gli_density)[0]) for sample, t in per_sample_tables.items()}
median_rho = float(np.median(list(rho.values())))
pd.DataFrame([{"sample": s, "n_cols": len(per_sample_tables[s]), "spearman_rho": rho[s]} for s in samples]
             + [{"sample": "median", "n_cols": np.nan, "spearman_rho": median_rho}]).to_csv(out_rho, index=False)
print("per-retina Spearman rho:", {s: round(rho[s], 3) for s in samples}, "median", round(median_rho, 3))

## Plot one correlation for each retina with the median and zero reference
figure, axes = plt.subplots(figsize=(3.6, 4.0))
x_of = {"LCA5_P21": 0.0, "LCA5_P30": 1.0}
axes.axhline(0.0, color="#bbbbbb", lw=1.0, ls="--", zorder=1)
axes.plot([-0.5, 1.5], [median_rho, median_rho], color="#333333", lw=1.8, zorder=2)
axes.text(1.45, median_rho, f"median ρ = {median_rho:+.2f}", ha="right", va="bottom", fontsize=10, fontweight="bold")
for sample in samples:
    condition = "LCA5_P21" if "P21" in sample else "LCA5_P30"
    filled = sample.endswith("rep2")
    color = paper_style.CONDITION_COLORS[condition]
    axes.scatter([x_of[condition]], [rho[sample]], s=95, facecolor=(color if filled else "white"),
                 edgecolor=color, linewidth=1.2, zorder=4)
axes.set_xticks(list(x_of.values()))
axes.set_xticklabels([paper_style.condition_label(c) for c in x_of], fontsize=10, fontweight="bold")
for tick, condition in zip(axes.get_xticklabels(), x_of):
    tick.set_color(paper_style.CONDITION_COLORS[condition])
axes.set_xlim(-0.5, 1.5)
all_rhos = list(rho.values())
axes.set_ylim(min(all_rhos) - 0.04, max(all_rhos) + 0.04)
axes.set_ylabel("Spearman ρ\n(Müller glia cell body vs gliosis marker density)", fontsize=11, fontweight="bold")
axes.legend(handles=[Line2D([0], [0], marker="o", ls="", mfc="white", mec="#555", ms=8, label="rep1"),
                     Line2D([0], [0], marker="o", ls="", mfc="#555", mec="#555", ms=8, label="rep2")],
            fontsize=9, frameon=False, loc="upper left")
for spine in ("top", "right"):
    axes.spines[spine].set_visible(False)
figure.tight_layout()

## Save the panel and the plotting tables
figure.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(figure)
table = pd.concat([per_sample_tables[sample].assign(sample=sample) for sample in samples], ignore_index=True)
table.to_parquet(out_tbl, index=False)
print("wrote", out_png)
