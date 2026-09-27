# Fig S2c rod fraction of cells superior vs inferior

## Load packages
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

## Load paper style conventions
sys.path.insert(0, "../../_shared/code")
import paper_style
from _superior_inferior_axis import superior_mask
paper_style.set_style()

## Load files
per_cell = "../../_shared/data_processed/per_cell_gene_counts.parquet"
output_png = "../panels/FigS2c.png"
output_table = "../data_processed/FigS2c.csv"

## Define conditions colors and parameters
conditions = paper_style.CONDITION_ORDER
condition_reps = {
    "WT_P21": ["WT_P21_rep1", "WT_P21_rep2"], "WT_P64": ["WT_P64_rep1", "WT_P64_rep2"],
    "LCA5_P21": ["LCA5_P21_rep1", "LCA5_P21_rep2"], "LCA5_P30": ["LCA5_P30_rep1", "LCA5_P30_rep2"],
    "LCA5_P64": ["LCA5_P64_rep1", "LCA5_P64_rep2"],
}
superior_color, inferior_color = "#1b9e3b", "#c800a8"
bar_width, gap = 0.38, 0.02

## Load data and keep typed cells
data = pd.read_parquet(per_cell, columns=["sample_id", "cell_type",
                       "cx_um", "cy_um", "body_Opn1sw", "body_Opn1mw"])
data = data[data["cell_type"] != "unassigned"].copy()
# Split superior vs inferior at the per sample median of the shared sigma=120 opsin field
data["is_superior"] = superior_mask(data)

## Compute rod fraction superior vs inferior per sample
records = []
for condition in conditions:
    for sample in condition_reps[condition]:
        sample_cells = data[data["sample_id"] == sample]
        is_superior = sample_cells["is_superior"].to_numpy()
        is_rod = (sample_cells["cell_type"] == "rod").to_numpy()
        n_all_sup = int(is_superior.sum())
        n_all_inf = int((~is_superior).sum())
        n_rod_sup = int((is_rod & is_superior).sum())
        n_rod_inf = int((is_rod & ~is_superior).sum())
        superior_fraction = n_rod_sup / n_all_sup if n_all_sup else np.nan
        inferior_fraction = n_rod_inf / n_all_inf if n_all_inf else np.nan
        fold = superior_fraction / inferior_fraction if (inferior_fraction and inferior_fraction > 0 and superior_fraction > 0) else np.nan
        records.append(dict(sample_id=sample, condition=condition,
                            rep="rep1" if sample.endswith("rep1") else "rep2",
                            n_rod=n_rod_sup + n_rod_inf,
                            n_rod_superior=n_rod_sup, n_rod_inferior=n_rod_inf,
                            n_cells_superior=n_all_sup, n_cells_inferior=n_all_inf,
                            rod_frac_sup=round(superior_fraction, 4) if np.isfinite(superior_fraction) else np.nan,
                            rod_frac_inf=round(inferior_fraction, 4) if np.isfinite(inferior_fraction) else np.nan,
                            fold_sup_over_inf=round(fold, 3) if np.isfinite(fold) else np.nan))
summary_table = pd.DataFrame(records)
summary_table.to_csv(output_table, index=False)

## Draw the superior and inferior bars with replicate dots
figure, axes = plt.subplots(1, 1, figsize=(4.1, 3.1), gridspec_kw=dict(left=0.17, right=0.97, top=0.84, bottom=0.24))
bar_positions = np.arange(len(conditions), dtype=float)
for index, condition in enumerate(conditions):
    condition_rows = summary_table[summary_table["condition"] == condition]
    superior_mean, inferior_mean = condition_rows["rod_frac_sup"].mean(), condition_rows["rod_frac_inf"].mean()
    x_superior, x_inferior = index - (bar_width / 2 + gap), index + (bar_width / 2 + gap)
    if np.isfinite(superior_mean):
        axes.bar(x_superior, superior_mean, bar_width, color=superior_color, alpha=0.30, edgecolor=superior_color, linewidth=1.1)
    if np.isfinite(inferior_mean):
        axes.bar(x_inferior, inferior_mean, bar_width, color=inferior_color, alpha=0.30, edgecolor=inferior_color, linewidth=1.1)
    for _, row in condition_rows.iterrows():
        for x_position, color, column in ((x_superior, superior_color, "rod_frac_sup"), (x_inferior, inferior_color, "rod_frac_inf")):
            axes.scatter([x_position], [row[column]], s=22, zorder=4, facecolor=color, edgecolor=color, linewidth=1.1)
    mean_fold = condition_rows["fold_sup_over_inf"].mean()
    folds = condition_rows["fold_sup_over_inf"].dropna().to_numpy()
    direction_label = None
    if len(folds) >= 2 and np.sign(folds[0] - 1) == np.sign(folds[1] - 1) and folds[0] != 1:
        direction_label = "S>I" if mean_fold > 1 else "I>S"
    if np.isfinite(mean_fold):
        text = f"{mean_fold:.2f}×" + (f"\n{direction_label}" if direction_label else "")
    else:
        text = "n/a"
    axes.text(index, 0.955, text, transform=axes.get_xaxis_transform(), ha="center",
              va="top", fontsize=8, fontweight="bold", color="#222")
axes.set_xticks(bar_positions)
axes.set_xticklabels([paper_style.condition_label(c) for c in conditions], fontsize=8, fontweight="bold", rotation=20, ha="right")
for index, condition in enumerate(conditions):
    axes.get_xticklabels()[index].set_color(paper_style.condition_color_simple(condition))
axes.tick_params(axis="x", length=0)
axes.set_ylabel("Rod fraction of cells", fontsize=9.5, fontweight="bold")
axes.set_ylim(0, min(1.0, axes.get_ylim()[1] * 1.22))
axes.set_xlim(-0.7, len(conditions) - 0.3)
for spine_name in ("top", "right"):
    axes.spines[spine_name].set_visible(False)

## Add the legend above the panel
figure.legend(handles=[Patch(facecolor=superior_color, alpha=0.55, label="superior (S)"),
                       Patch(facecolor=inferior_color, alpha=0.55, label="inferior (I)")],
              fontsize=8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 0.965), ncol=2, handlelength=1.1, columnspacing=1.6)

## Save the panel
figure.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(figure)
print("wrote", output_png)
