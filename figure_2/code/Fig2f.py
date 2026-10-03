# Fig 2f per gene relative transcript abundance for LCA5 vs WT

## Load packages
import sys
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

## Load paper style conventions
sys.path.insert(0, "../../_shared/code")
import paper_style
from _retina_panel import PX_DS_UM
from _paths import findpath_transcripts, findpath_unet_pred, findpath_exclude_final
paper_style.set_style()

## Load files
per_cell = "../../_shared/data_processed/per_cell_gene_counts.parquet"
output_png = "../panels/Fig2f.png"
output_csv = "../data_processed/Fig2f.csv"

## Define comparisons and thresholds
comparisons = [("21d", "WT_P21", "LCA5_P21"), ("64d", "WT_P64", "LCA5_P64")]
lfc_threshold = 0.5
up_color = "#d62728"
down_color = "#1f6fb5"
ns_color = "#cccccc"
n_label_up = 12
n_label_down = 10

## Define the full transcript panel
columns = pd.read_parquet(per_cell).head(0).columns
non_gene_columns = {"body_total", "body_density_per_um2", "body_source"}
panel_genes = {column[5:] for column in columns if column.startswith("body_") and column not in non_gene_columns}
if len(panel_genes) != 173 or "Rho" not in panel_genes:
    raise ValueError(f"expected 173 panel genes including Rho, got {len(panel_genes)}")

## Count transcripts inside the retinal tissue with paper exclusions removed
def tissue_transcript_counts(sample):
    layer = tifffile.imread(findpath_unet_pred(sample, "layer")) > 0
    height, width = layer.shape
    transcripts = pd.read_csv(findpath_transcripts(sample), usecols=["name", "x", "y"])
    col = np.rint(transcripts["x"].to_numpy() / PX_DS_UM).astype(int)
    row = np.rint(transcripts["y"].to_numpy() / PX_DS_UM).astype(int)
    in_bounds = (col >= 0) & (col < width) & (row >= 0) & (row < height)
    keep = np.zeros(len(transcripts), bool)
    keep[in_bounds] = layer[row[in_bounds], col[in_bounds]]
    exclude_path = findpath_exclude_final(sample)
    if exclude_path.exists():
        excluded = tifffile.imread(exclude_path) > 0
        eh, ew = excluded.shape
        inside = keep & (col < ew) & (row < eh)
        drop = np.zeros(len(transcripts), bool)
        drop[inside] = excluded[row[inside], col[inside]]
        keep &= ~drop
    return transcripts.loc[keep].groupby("name").size()


## Compute relative transcript abundance per sample
samples = [f"{condition}_{rep}" for _, wt, lca in comparisons for condition in (wt, lca) for rep in ("rep1", "rep2")]
counts = {sample: tissue_transcript_counts(sample) for sample in samples}
genes = sorted(set().union(*[set(count_series.index) for count_series in counts.values()]))
genes = [gene for gene in genes if gene in panel_genes]
if len(genes) != 173:
    raise ValueError(f"expected all 173 panel genes present in the tissue counts, got {len(genes)}")
for sample in samples:
    zero = [gene for gene in genes if float(counts[sample].get(gene, 0.0)) <= 0]
    if zero:
        raise ValueError(f"{sample}: panel genes with zero or absent tissue counts: {sorted(zero)}")

# Relative abundance of each gene is its tissue count over the panel total
transcript_fraction = pd.DataFrame(index=genes)
for sample in samples:
    gene_counts = counts[sample].reindex(genes).to_numpy()
    transcript_fraction[sample] = gene_counts / float(gene_counts.sum())

## Compute per rep log2 fold change and direction per comparison
panels = []
all_rows = []
for age, wt, lca in comparisons:
    wt_mean = 0.5 * (transcript_fraction[f"{wt}_rep1"] + transcript_fraction[f"{wt}_rep2"])
    stats = pd.DataFrame(index=transcript_fraction.index)
    stats["wt_rep1"] = transcript_fraction[f"{wt}_rep1"]
    stats["wt_rep2"] = transcript_fraction[f"{wt}_rep2"]
    stats["lca_rep1"] = transcript_fraction[f"{lca}_rep1"]
    stats["lca_rep2"] = transcript_fraction[f"{lca}_rep2"]
    stats["lfc_rep1"] = np.log2(transcript_fraction[f"{lca}_rep1"] / wt_mean)
    stats["lfc_rep2"] = np.log2(transcript_fraction[f"{lca}_rep2"] / wt_mean)
    stats["group_log2fc"] = np.log2(0.5 * (stats["lca_rep1"] + stats["lca_rep2"])
                                    / (0.5 * (stats["wt_rep1"] + stats["wt_rep2"])))
    up_separated = stats["lca_rep1"].gt(stats[["wt_rep1", "wt_rep2"]].max(axis=1)) & stats["lca_rep2"].gt(stats[["wt_rep1", "wt_rep2"]].max(axis=1))
    down_separated = stats["lca_rep1"].lt(stats[["wt_rep1", "wt_rep2"]].min(axis=1)) & stats["lca_rep2"].lt(stats[["wt_rep1", "wt_rep2"]].min(axis=1))
    stats["separated"] = up_separated | down_separated
    is_up = (stats["lfc_rep1"] >= lfc_threshold) & (stats["lfc_rep2"] >= lfc_threshold) & up_separated
    is_down = (stats["lfc_rep1"] <= -lfc_threshold) & (stats["lfc_rep2"] <= -lfc_threshold) & down_separated
    stats["direction"] = np.where(is_up, "up", np.where(is_down, "down", "ns"))
    panels.append((age, stats))
    record = stats.reset_index().rename(columns={"index": "gene"})
    record.insert(0, "comparison", f"{lca}_vs_{wt}")
    all_rows.append(record)
output_df = pd.concat(all_rows, ignore_index=True)
for _, row in output_df[output_df["direction"] != "ns"].iterrows():
    if not row["separated"]:
        raise ValueError(f"{row['gene']}: called changed but not separated")
    if row["direction"] == "up":
        if row["lfc_rep1"] < lfc_threshold or row["lfc_rep2"] < lfc_threshold:
            raise ValueError(f"{row['gene']}: up but rep lfc below threshold")
    else:
        if row["lfc_rep1"] > -lfc_threshold or row["lfc_rep2"] > -lfc_threshold:
            raise ValueError(f"{row['gene']}: down but rep lfc above -threshold")
output_df.to_csv(output_csv, index=False)

## Set a shared symmetric axis limit
axis_limit = max(float(np.nanmax(np.abs(stats[["lfc_rep1", "lfc_rep2"]].to_numpy()))) for _, stats in panels) * 1.15

## Draw the two scatter panels
figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.5), facecolor="white")
for axis, (age, stats) in zip(axes, panels):
    direction_colors = {"up": up_color, "down": down_color, "ns": ns_color}
    # Show genes without a consistent change in gray
    not_reproducible = stats[stats["direction"] == "ns"]
    axis.scatter(not_reproducible["lfc_rep1"], not_reproducible["lfc_rep2"], s=8, c=ns_color, edgecolors="none", alpha=0.5, zorder=1)
    # Color the reproducible genes
    reproducible = stats[stats["direction"] != "ns"].copy()
    reproducible["color"] = reproducible["direction"].map(direction_colors)
    axis.scatter(reproducible["lfc_rep1"], reproducible["lfc_rep2"], s=22, c=reproducible["color"], edgecolors="none", alpha=0.9, zorder=3)
    # Draw reference lines at zero and the diagonal
    axis.axhline(0, color="#999", lw=0.5, ls="--", zorder=0)
    axis.axvline(0, color="#999", lw=0.5, ls="--", zorder=0)
    axis.plot([-axis_limit, axis_limit], [-axis_limit, axis_limit], color="#cfcfcf", lw=0.8, ls="-", zorder=0)
    # Label the strongest reproducible genes per direction
    reproducible["mean_lfc"] = 0.5 * (reproducible["lfc_rep1"] + reproducible["lfc_rep2"])
    top_up = reproducible[reproducible["direction"] == "up"].nlargest(n_label_up, "mean_lfc")
    top_down = reproducible[reproducible["direction"] == "down"].nsmallest(n_label_down, "mean_lfc")
    texts = [axis.text(row["lfc_rep1"], row["lfc_rep2"], gene, fontsize=14, fontweight="bold",
                       fontstyle="italic", color=row["color"]) for gene, row in pd.concat([top_up, top_down]).iterrows()]
    if texts:
        adjust_text(texts, ax=axis, arrowprops=dict(arrowstyle="-", color="#999999", lw=0.4),
                    force_text=(0.5, 0.9), expand=(1.4, 2.0), min_arrow_len=3,
                    only_move={"text": "xy", "static": "xy"}, ensure_inside_axes=True)
    axis.set_xlim(-axis_limit, axis_limit)
    axis.set_ylim(-axis_limit, axis_limit)
    axis.set_box_aspect(1)
    for spine_name in ("top", "right"):
        axis.spines[spine_name].set_visible(False)
    # Add the direction legend matching Figure 3a
    lca_label = "$\\mathsf{Lca5}^{gt/gt}$"
    legend_handles = [plt.Line2D([0], [0], marker="o", ls="", ms=6, mfc=up_color, mec="none", label=f"up in {lca_label}"),
                      plt.Line2D([0], [0], marker="o", ls="", ms=6, mfc=down_color, mec="none", label=f"down in {lca_label}")]
    axis.legend(handles=legend_handles, loc="upper left", fontsize=8, frameon=False, handletextpad=0.4, labelspacing=0.3)
    # Add the two color title
    title_segments = [("Wild-type", paper_style.CONDITION_SIMPLE_WT), (" vs ", "#666666"),
                      (f"$\\mathsf{{Lca5}}^{{gt/gt}}$ {age}", paper_style.CONDITION_SIMPLE_LCA5)]
    title_boxes = [TextArea(text, textprops=dict(color=color, fontsize=19, fontweight="bold")) for text, color in title_segments]
    axis.add_artist(AnchoredOffsetbox(loc="lower center", child=HPacker(children=title_boxes, align="baseline", pad=0, sep=0),
                    pad=0, borderpad=0, frameon=False, bbox_to_anchor=(0.5, 1.0), bbox_transform=axis.transAxes))
    axis.text(0.5, 1.09, "Total retina", transform=axis.transAxes, ha="center", va="bottom",
              fontsize=20, fontweight="bold", color="black")
    axis.tick_params(labelsize=11)
axes[0].set_ylabel(r"Log2 FC ($\mathsf{Lca5}^{gt/gt}$ rep 2 / mean wild-type)", fontsize=13, fontweight="bold")
figure.supxlabel(r"Log2 FC ($\mathsf{Lca5}^{gt/gt}$ rep 1 / mean wild-type)",
                 fontsize=13, fontweight="bold")

## Save the panel
figure.tight_layout(w_pad=1.0)
figure.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(figure)
print("wrote", output_png)
