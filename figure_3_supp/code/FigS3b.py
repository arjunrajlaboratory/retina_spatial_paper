# Fig S3b rod superior/inferior bias in wild-type versus LCA5 P21

## Load packages
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text

## Load paper style conventions
sys.path.insert(0, "../../_shared/code")
import paper_style
paper_style.set_style()

## Load files
fig3b_csv = "../../figure_3/data_processed/Fig3b.csv"
out_png = "../panels/FigS3b.png"
out_csv = "../data_processed/FigS3b_wt_vs_lca5_si.csv"
up_color, down_color = "#d62728", "#1f77b4"

## Load the changed gene superior/inferior table for WT and LCA5
table = pd.read_csv(fig3b_csv).reset_index(drop=True)
table.to_csv(out_csv, index=False)

## Draw the WT vs LCA5 scatter with replicate range bars
figure, axes = plt.subplots(figsize=(6.6, 6.4), facecolor="white")
axis_limit = float(np.nanmax(np.abs(np.r_[table["mean"], table["wt_mean"]]))) + 0.08
axes.plot([-axis_limit, axis_limit], [-axis_limit, axis_limit], color="#999999", lw=1.0, ls="--", zorder=1)
axes.axhline(0, color="#dddddd", lw=0.8, zorder=0)
axes.axvline(0, color="#dddddd", lw=0.8, zorder=0)
point_colors = np.where(table["group_log2fc"] >= 0, up_color, down_color)
axes.hlines(table["mean"], table[["wt1", "wt2"]].min(axis=1), table[["wt1", "wt2"]].max(axis=1), colors=point_colors, lw=1.0, alpha=0.5, zorder=2)
axes.vlines(table["wt_mean"], table[["lca1", "lca2"]].min(axis=1), table[["lca1", "lca2"]].max(axis=1), colors=point_colors, lw=1.0, alpha=0.5, zorder=2)
axes.scatter(table["wt_mean"], table["mean"], s=44, c=point_colors, edgecolors="none", zorder=4)

## Label the genes
texts = [axes.text(row.wt_mean, row.mean, row.gene, fontsize=8.5, fontstyle="italic", fontweight="bold", color="#222222") for row in table.itertuples()]
adjust_text(texts, ax=axes, arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.4), expand=(1.5, 1.9), force_text=(0.5, 1.0))
axes.set_xlabel("Wild-type S/I (Spearman $\\rho$)", fontsize=12, fontweight="bold")
axes.set_ylabel(f"{paper_style.GENOTYPE} S/I (Spearman $\\rho$)", fontsize=12, fontweight="bold")
axes.set_xlim(-axis_limit, axis_limit)
axes.set_ylim(-axis_limit, axis_limit)
axes.set_aspect("equal")

## Add the legend
legend_handles = [plt.Line2D([0], [0], marker="s", ls="", ms=8, mfc=up_color, mec="none", label=f"expression up in {paper_style.GENOTYPE}"),
                  plt.Line2D([0], [0], marker="s", ls="", ms=8, mfc=down_color, mec="none", label=f"expression down in {paper_style.GENOTYPE}")]
axes.legend(handles=legend_handles, loc="upper left", fontsize=9, frameon=False)
for spine_name in ("top", "right"):
    axes.spines[spine_name].set_visible(False)

## Save the panel
figure.tight_layout()
figure.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(figure)
print("wrote", out_png)
