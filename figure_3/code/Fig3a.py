# Fig 3a rod gene response replicate concordance LCA5 vs wild-type P21

## Load packages
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

## Load paper style conventions
sys.path.insert(0, "../../_shared/code")
import paper_style
paper_style.set_style()

## Load files
per_cell = "../../_shared/data_processed/per_cell_gene_counts.parquet"
output_dir = "../panels"
data_dir = "../data_processed"
up_color, down_color, null_color = "#d62728", "#1f6fb5", "#cccccc"

## Define the response and rod expression filter parameters
mag_threshold = 0.5
relative_rod_min = 0.15
detect_min_pct = 10.0
mag_sweep = (0.3, 0.5, 1.0)
wt_reps = ["WT_P21_rep1", "WT_P21_rep2"]
lca_reps = ["LCA5_P21_rep1", "LCA5_P21_rep2"]
# Detection filter passes when both replicates of any one condition pass
gate_conditions = {
    "WT_P21":   ["WT_P21_rep1", "WT_P21_rep2"],
    "WT_P64":   ["WT_P64_rep1", "WT_P64_rep2"],
    "LCA5_P21": ["LCA5_P21_rep1", "LCA5_P21_rep2"],
    "LCA5_P30": ["LCA5_P30_rep1", "LCA5_P30_rep2"],
}
# Cell type groups for finding the peak expressing cell type per gene
cell_type_groups = {
    "rod": ["rod"], "cone": ["cone"], "bipolar": ["bipolar"],
    "amacrine_gaba": ["amacrine_gaba"], "amacrine_gly": ["amacrine_gly"],
    "horizontal": ["horizontal"],
    "muller": ["muller"], "rgc": ["rgc"], "microglia": ["microglia"],
    "vascular": ["vascular"], "rpe": ["rpe"],
}

## Load the per cell transcript counts for every panel gene
non_gene = {"body_total", "body_density_per_um2", "body_source"}
gene_cols = [c for c in pq.read_schema(per_cell).names if c.startswith("body_") and c not in non_gene]
cells = pd.read_parquet(per_cell, columns=["sample_id", "cell_type", "cell_area_um2"] + gene_cols)
gene_cols = [c for c in gene_cols if pd.api.types.is_numeric_dtype(cells[c])]
genes = np.array([c[len("body_"):] for c in gene_cols])

## Compute mean rod transcript density per gene in each retina
rods = cells[cells.cell_type == "rod"]
def rod_density_mean(sample):
    sub = rods[rods.sample_id == sample]
    density = sub[gene_cols].to_numpy(float) / sub["cell_area_um2"].to_numpy(float)[:, None]
    return density.mean(axis=0)
wt1, wt2 = rod_density_mean(wt_reps[0]), rod_density_mean(wt_reps[1])
lca1, lca2 = rod_density_mean(lca_reps[0]), rod_density_mean(lca_reps[1])
wt_mean = 0.5 * (wt1 + wt2)
lca_mean = 0.5 * (lca1 + lca2)

## Compute the group fold change and each replicate fold change versus the shared wild type mean
group_log2fc = np.log2(lca_mean / wt_mean)
lfc_lca1 = np.log2(lca1 / wt_mean)
lfc_lca2 = np.log2(lca2 / wt_mean)

## Determine complete replicate separation (both LCA5 above both wild type or both below)
up_separated = np.minimum(lca1, lca2) > np.maximum(wt1, wt2)
down_separated = np.maximum(lca1, lca2) < np.minimum(wt1, wt2)
separated = up_separated | down_separated
sep_dir = np.where(up_separated, "up", np.where(down_separated, "down", "none"))

## Compute relative rod expression as rod density over the peak expressing cell type in LCA5 P21
lca_cells = cells[cells.sample_id.isin(lca_reps)]
ct_density = {}
for name, members in cell_type_groups.items():
    grp = lca_cells[lca_cells.cell_type.isin(members)]
    density = grp[gene_cols].to_numpy(float) / grp["cell_area_um2"].to_numpy(float)[:, None]
    ct_density[name] = np.nanmean(density, axis=0) if len(grp) else np.full(len(genes), np.nan)
ct_names = list(ct_density)
density_matrix = np.vstack([ct_density[n] for n in ct_names])
peak_index = np.where(np.isnan(density_matrix), -np.inf, density_matrix).argmax(axis=0)
peak_density = density_matrix[peak_index, np.arange(len(genes))]
peak_ct = np.array(ct_names)[peak_index]
relative_rod_expression = np.where(peak_density > 0, ct_density["rod"] / peak_density, np.nan)

## Require detection in at least 10% of rods in both retinas of one condition
gate_samples = sorted({s for reps in gate_conditions.values() for s in reps})
detect_pct = {}
for sample in gate_samples:
    r = rods[rods.sample_id == sample]
    detect_pct[sample] = (r[gene_cols].to_numpy(float) >= 1).mean(axis=0) * 100.0 if len(r) else np.zeros(len(genes))
passes_detection_filter = np.zeros(len(genes), bool)
for reps in gate_conditions.values():
    condition_pass = np.ones(len(genes), bool)
    for sample in reps:
        condition_pass &= detect_pct[sample] >= detect_min_pct
    passes_detection_filter |= condition_pass

## Retain genes with substantial rod enriched signal that also pass the detection filter
retained_for_rod_analysis = (relative_rod_expression >= relative_rod_min) & passes_detection_filter
replicate_up = (lfc_lca1 >= mag_threshold) & (lfc_lca2 >= mag_threshold)
replicate_down = (lfc_lca1 <= -mag_threshold) & (lfc_lca2 <= -mag_threshold)
changed = retained_for_rod_analysis & (
    (replicate_up & up_separated) | (replicate_down & down_separated)
)

## Save the complete per gene table with the rod expression filter and the P21 response
complete = pd.DataFrame({
    "gene": genes,
    "relative_rod_expression": relative_rod_expression,
    "peak_ct": peak_ct,
    "passes_detection_filter": passes_detection_filter,
    "retained_for_rod_analysis": retained_for_rod_analysis,
    "wt1": wt1, "wt2": wt2, "lca1": lca1, "lca2": lca2,
    "group_log2fc": group_log2fc, "lfc_lca1": lfc_lca1, "lfc_lca2": lfc_lca2,
    "replicate_up": replicate_up, "replicate_down": replicate_down,
    "separated": separated, "sep_dir": sep_dir, "changed": changed,
}).sort_values("relative_rod_expression").reset_index(drop=True)
complete.to_csv(f"{data_dir}/rod_expression_filter.csv", index=False)

## Build the response table for the retained genes shown in the scatter
response = complete[complete.retained_for_rod_analysis].reset_index(drop=True)
response["dir"] = np.where(response.group_log2fc >= 0, "up", "down")
response.to_csv(f"{data_dir}/Fig3a.csv", index=False)

## Save the changed gene count across the effect size sweep
sensitivity_rows = []
for mag in mag_sweep:
    rep_up = (response.lfc_lca1 >= mag) & (response.lfc_lca2 >= mag) & response.separated & (response.sep_dir == "up")
    rep_down = (response.lfc_lca1 <= -mag) & (response.lfc_lca2 <= -mag) & response.separated & (response.sep_dir == "down")
    hit = rep_up | rep_down
    sensitivity_rows.append(dict(mag=mag, n=int(hit.sum()),
                                 up=int(rep_up.sum()), down=int(rep_down.sum())))
pd.DataFrame(sensitivity_rows).to_csv(f"{data_dir}/Fig3a_sensitivity.csv", index=False)

## Verify every changed gene passes both replicate effect threshold and complete separation
for _, row in response[response.changed].iterrows():
    if not row["separated"]:
        raise ValueError(f"{row['gene']}: called changed but not separated")
    if row["dir"] == "up":
        if row["lfc_lca1"] < mag_threshold or row["lfc_lca2"] < mag_threshold:
            raise ValueError(f"{row['gene']}: up but rep lfc below threshold")
    else:
        if row["lfc_lca1"] > -mag_threshold or row["lfc_lca2"] > -mag_threshold:
            raise ValueError(f"{row['gene']}: down but rep lfc above -threshold")

## Set a symmetric axis limit
lfc_values = pd.concat([response.lfc_lca1, response.lfc_lca2])
axis_limit = max(1.0, float(np.nanmax(np.abs(lfc_values))) * 1.05)

## Draw the concordance scatter
figure, axes = plt.subplots(figsize=(6.6, 6.6), facecolor="white")
non_changed = response[~response.changed]
axes.scatter(non_changed.lfc_lca1, non_changed.lfc_lca2, s=9, c=null_color, alpha=0.5, edgecolors="none", zorder=1)
for direction, color in [("up", up_color), ("down", down_color)]:
    direction_subset = response[response.changed & (response["dir"] == direction)]
    axes.scatter(direction_subset.lfc_lca1, direction_subset.lfc_lca2, s=42, c=color, alpha=0.9, edgecolors="white", linewidths=0.5, zorder=3)
axes.plot([-axis_limit, axis_limit], [-axis_limit, axis_limit], color="#cfcfcf", lw=0.8, zorder=0)
axes.axhline(0, color="#999", lw=0.5, ls="--", zorder=0)
axes.axvline(0, color="#999", lw=0.5, ls="--", zorder=0)

## Label the changed genes
changed_genes = response[response.changed]
label_colors = np.where(changed_genes["dir"].to_numpy() == "up", up_color, down_color)
texts = [axes.text(row.lfc_lca1, row.lfc_lca2, gene, fontsize=11, fontstyle="italic", fontweight="bold", color=color)
         for (_, row), color, gene in zip(changed_genes.iterrows(), label_colors, changed_genes.gene)]
if texts:
    adjust_text(texts, ax=axes, arrowprops=dict(arrowstyle="-", color="#999999", lw=0.4), force_text=(0.5, 0.9), expand=(1.4, 1.9))
axes.set_xlim(-axis_limit, axis_limit)
axes.set_ylim(-axis_limit, axis_limit)
axes.set_box_aspect(1.0)
axes.set_xlabel(r"Log2 FC ($\mathsf{Lca5}^{gt/gt}$ rep 1 / mean wild-type)", fontsize=13, fontweight="bold")
axes.set_ylabel(r"Log2 FC ($\mathsf{Lca5}^{gt/gt}$ rep 2 / mean wild-type)", fontsize=13, fontweight="bold")
axes.tick_params(labelsize=11)
for spine_name in ("top", "right"):
    axes.spines[spine_name].set_visible(False)

## Add the direction legend
lca_label = "$\\mathsf{Lca5}^{gt/gt}$"
legend_handles = [plt.Line2D([0], [0], marker="o", ls="", ms=8, mfc=up_color, mec="none", label=f"up in {lca_label}"),
                  plt.Line2D([0], [0], marker="o", ls="", ms=8, mfc=down_color, mec="none", label=f"down in {lca_label}")]
axes.legend(handles=legend_handles, loc="upper left", fontsize=10, frameon=False)

## Add the two tone title
title_segments = [("Wild-type 21d", paper_style.CONDITION_SIMPLE_WT), ("  vs  ", "#666666"),
                  ("$\\mathsf{Lca5}^{gt/gt}$ 21d", paper_style.CONDITION_SIMPLE_LCA5)]
title_boxes = [TextArea(text, textprops=dict(color=color, fontsize=17, fontweight="bold")) for text, color in title_segments]
axes.add_artist(AnchoredOffsetbox(loc="lower center", child=HPacker(children=title_boxes, align="baseline", pad=0, sep=0),
                pad=0, borderpad=0, frameon=False, bbox_to_anchor=(0.5, 1.0), bbox_transform=axes.transAxes))
axes.text(0.5, 1.09, "Rods", transform=axes.transAxes, ha="center", va="bottom", fontsize=20, fontweight="bold", color="black")

## Save the panel
figure.tight_layout()
figure.savefig(f"{output_dir}/Fig3a.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(figure)
print("wrote", f"{output_dir}/Fig3a.png")
