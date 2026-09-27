# Fig S4i ONL microglia abundance in focal regions and the remaining retina

## Load packages
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

## Load paper style conventions and the shared focal region functions
sys.path.insert(0, "../../_shared/code")
import paper_style
import _focal_regions as focal
paper_style.set_style()

## Load files
per_cell_path = "../../_shared/data_processed/per_cell_gene_counts.parquet"
out_png = "../panels/FigS4i.png"
out_csv = "../data_plot/FigS4i.csv"
report_conditions = ["WT_P21", "WT_P64", "LCA5_P21", "LCA5_P30"]
plot_conditions = ["LCA5_P21", "LCA5_P30"]
x_of = {"LCA5_P21": 0.0, "LCA5_P30": 0.38}


## Compare microglia per column inside focal regions with the remaining retina
def microglia_fold(sample):
    skel = pd.read_parquet(focal.FT / "skeletons.parquet")
    skel = skel[skel["sample"] == sample]
    cm = pd.read_parquet(focal.FT / "columns_meta.parquet")
    cm = cm[cm["sample"] == sample].sort_values("col_id_local")
    cells = pd.read_parquet(per_cell_path, columns=["sample_id", "cell_type", "argmax_layer", "cx_um", "cy_um"])
    microglia = cells[(cells.sample_id == sample) & (cells.cell_type == "microglia")
                      & (cells.argmax_layer == "ONL")].copy()
    microglia = focal._project_nearest(microglia, skel, cm)
    cols = focal.focal_region_columns(sample)
    valid = cols[cols["valid_tissue"]]
    region_ids = set(int(x) for x in valid.loc[valid["in_focal_region"], "col_id_local"])
    valid_ids = set(int(x) for x in valid["col_id_local"])
    microglia = microglia[microglia["col_id_local"].isin(valid_ids)]
    n_in = int(microglia["col_id_local"].isin(region_ids).sum())
    n_out = len(microglia) - n_in
    n_cols_in = len(region_ids)
    n_cols_out = len(valid_ids - region_ids)
    rate_in = n_in / n_cols_in if n_cols_in > 0 else np.nan
    rate_out = n_out / n_cols_out if n_cols_out > 0 else np.nan
    fold = rate_in / rate_out if (rate_out and rate_out > 0) else np.nan
    return {"n_in": n_in, "n_out": n_out, "n_cols_in": n_cols_in, "n_cols_out": n_cols_out, "fold": fold}


## Compute per retina
rows = []
for condition in report_conditions:
    for rep in ("rep1", "rep2"):
        sample = f"{condition}_{rep}"
        rec = microglia_fold(sample)
        rec.update({"sample": sample, "condition": condition, "rep": rep})
        rows.append(rec)
data = pd.DataFrame(rows)
data.to_csv(out_csv, index=False)
print("FigS4i ONL microglia count per column fold (focal region / rest); counts per retina:")
print(data[["sample", "n_in", "n_out", "n_cols_in", "n_cols_out", "fold"]].to_string(index=False))

## Plot the mean and both retinas at each timepoint
figure, axes = plt.subplots(figsize=(4.2, 4.2))
axes.axhline(1.0, color="#888888", lw=1.0, ls="--", zorder=4)
for condition in plot_conditions:
    x = x_of[condition]
    color = paper_style.CONDITION_COLORS[condition]
    reps = {rep: float(data[(data.condition == condition) & (data.rep == rep)]["fold"].iloc[0]) for rep in ("rep1", "rep2")}
    finite = [v for v in reps.values() if np.isfinite(v)]
    axes.bar([x], [float(np.mean(finite))], width=0.18, color=color, alpha=0.40, edgecolor=color, linewidth=1.3, zorder=2)
    for rep, dx in (("rep1", -0.10), ("rep2", 0.10)):
        value = reps[rep]
        if np.isfinite(value):
            axes.scatter([x + dx], [value], s=85, facecolor=(color if rep == "rep2" else "white"),
                         edgecolor=color, linewidth=1.2, zorder=6)
axes.set_xticks(list(x_of.values()))
axes.set_xticklabels([paper_style.condition_label(t) for t in x_of], fontsize=11, fontweight="bold", rotation=25, ha="right")
for tick, tp in zip(axes.get_xticklabels(), x_of):
    tick.set_color(paper_style.CONDITION_COLORS[tp])
axes.set_xlim(-0.25, 0.63)
axes.set_ylim(0, 2.0)
axes.set_box_aspect(1)
axes.set_ylabel("ONL microglia (in vs out fold)", fontsize=10.5, fontweight="bold")
axes.legend(handles=[Line2D([0], [0], marker="o", ls="", mfc="white", mec="#222", ms=8, label="rep1"),
                     Line2D([0], [0], marker="o", ls="", mfc="#333333", mec="#222", ms=8, label="rep2")],
            fontsize=9, frameon=False, loc="upper right")
for spine in ("top", "right"):
    axes.spines[spine].set_visible(False)
figure.tight_layout()
figure.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(figure)
print("wrote", out_png)
