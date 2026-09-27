# Fig S3a rod expression filter threshold

## Load packages
import sys
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text

## Load paper style conventions
sys.path.insert(0, "../../_shared/code")
import paper_style
paper_style.set_style()

## Load files
rod_filter_csv = "../../figure_3/data_processed/rod_expression_filter.csv"
out_png = "../panels/FigS3a.png"
out_table = "../data_processed/FigS3a_rod_expression_filter.csv"
threshold = 0.15
drop_color = "#d9d9d9"
retained_color = "#2166ac"   # Blue for the final retained_for_rod_analysis genes

## Define marker genes for cell types other than rods
marker_genes = {
    "cone": ["Arr3", "Opn1mw", "Opn1sw"],
    "bipolar": ["Vsx2", "Prkca", "Grm6", "Vsx1", "Scgn"],
    "amacrine_gaba": ["Gad1", "Slc32a1"],
    "amacrine_gly": ["Slc6a9"],
    "horizontal": ["Onecut2", "Calb1"],
    "rgc": ["Rbpms", "Pou4f2", "Nefh"],
    "muller": ["Sox9", "Glul", "Slc1a3", "Kcnj10", "Aqp4", "Gja1"],
    "microglia": ["Csf1r", "Cx3cr1", "C1qa"],
    "vascular": ["Cldn5", "Pecam1", "Rgs5"],
    "rpe": ["Rpe65"],
}
marker_order = ["cone", "bipolar", "amacrine_gaba", "amacrine_gly", "horizontal",
                "rgc", "muller", "microglia", "vascular", "rpe"]
program_genes = ["Edn2", "Socs3", "Cebpd", "Fgf2", "Stat3", "Gadd45b", "Nrl", "Nxnl1"]

## Load the per gene rod expression filter and map genes to a marker cell type
table = pd.read_csv(rod_filter_csv)
gene_to_celltype = {gene: cell_type for cell_type, gene_list in marker_genes.items() for gene in gene_list}
table["marker_ct"] = table.gene.map(gene_to_celltype)
table["passes_relative_rod_threshold"] = table.relative_rod_expression >= threshold
markers = table[table.marker_ct.notna()].sort_values("relative_rod_expression", ascending=False)
top_marker_val = float(markers.relative_rod_expression.max())
top_marker = markers.iloc[0].gene
table.to_csv(out_table, index=False)

## Rank genes by relative rod expression
table = table.sort_values("relative_rod_expression").reset_index(drop=True)
n_retained = int(table.retained_for_rod_analysis.sum())
n_excluded = int((~table.retained_for_rod_analysis).sum())

## Draw the ranked filter scatter
# Genes that fail the detection filter remain gray even when their relative rod expression exceeds 0.15
figure, axes = plt.subplots(figsize=(9.0, 5.4), facecolor="white")
generic = table[table.marker_ct.isna()]
gen_excluded = generic[~generic.retained_for_rod_analysis]
gen_retained = generic[generic.retained_for_rod_analysis]
axes.scatter(gen_excluded.index.to_numpy(), gen_excluded.relative_rod_expression, s=15, c=drop_color, edgecolors="none", zorder=2, label=f"excluded (n={n_excluded})")
axes.scatter(gen_retained.index.to_numpy(), gen_retained.relative_rod_expression, s=24, c=retained_color, edgecolors="none", zorder=3, label=f"retained for rod analysis (n={n_retained})")
for cell_type in marker_order:
    marker_subset = table[table.marker_ct == cell_type]
    label = ("Müller glia" if cell_type == "muller" else paper_style.celltype_label(cell_type)) + " marker"
    axes.scatter(marker_subset.index.to_numpy(), marker_subset.relative_rod_expression, s=46, c=paper_style.CELLTYPE_COLORS[cell_type], edgecolors="none", alpha=0.95, zorder=4, label=label)
axes.axhline(threshold, color="#d62728", lw=1.3, zorder=3)
axes.text(len(table) - 1, threshold + 0.02, f"relative rod expression ≥ {threshold:.2f}", fontsize=10, fontweight="bold", color="#d62728", va="bottom", ha="right")
axes.annotate(f"Highest cell type marker\n({top_marker}, {top_marker_val:.2f})",
              xy=(table.index[table.gene == top_marker][0], top_marker_val),
              xytext=(len(table) * 0.52, 0.30), fontsize=8.5, color="#555", ha="left",
              arrowprops=dict(arrowstyle="->", color="#999", lw=0.8))

## Label the program and top marker genes
top_marker_genes = set(markers.head(4).gene)
label_genes = set(program_genes) | top_marker_genes | set(marker_genes["cone"])
texts = []
for row in table[table.gene.isin(label_genes)].itertuples():
    color = paper_style.CELLTYPE_COLORS.get(row.marker_ct if isinstance(row.marker_ct, str) else "", "#222")
    texts.append(axes.text(row.Index, row.relative_rod_expression, row.gene, fontsize=8.5, fontstyle="italic", fontweight="bold", color=color))
adjust_text(texts, ax=axes, arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.4), expand=(1.4, 2.0), force_text=(0.5, 1.0), only_move={"text": "xy"})
axes.set_xlabel("Genes ranked by relative rod expression", fontsize=12, fontweight="bold")
axes.set_ylabel("Rod density / peak cell type", fontsize=12, fontweight="bold")
axes.set_ylim(-0.03, 1.08)
axes.tick_params(labelsize=10)
for spine_name in ("top", "right"):
    axes.spines[spine_name].set_visible(False)
axes.legend(loc="upper left", fontsize=8, frameon=False, ncol=2,
            title="retained = relative rod expression ≥ 0.15 and detected in ≥ 10% of rods", title_fontsize=9)

## Save the panel
figure.tight_layout()
figure.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(figure)
print("wrote", out_png)
