# Fig S4e rod abundance inside and outside focal injury response regions

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
out_png = "../panels/FigS4e.png"
out_csv = "../data_plot/FigS4e.csv"
conditions = ["WT_P21", "WT_P64", "LCA5_P21", "LCA5_P30"]
condition_reps = {c: [f"{c}_rep1", f"{c}_rep2"] for c in conditions}
x_of = {c: i for i, c in enumerate(conditions)}
rng = np.random.default_rng(0)


## Count rods in each 25 µm column
def rod_count(sample):
    cols = focal.focal_region_columns(sample)[["col_id_local", "n_rods"]]
    return {int(row["col_id_local"]): float(row["n_rods"]) for _, row in cols.iterrows()}


## Compare each focal region with its local reference columns
# Empty flanking columns are included as zero counts
rows = []
for tp, reps in condition_reps.items():
    for sample in reps:
        rep = "rep1" if sample.endswith("rep1") else "rep2"
        count = rod_count(sample)
        reference = focal.local_reference(sample, require_rods=False)
        for region_id, block in reference.groupby("focal_region_id"):
            if not bool(block["has_reference"].iloc[0]):
                continue
            in_vals = [count[c] for c in block.loc[block.role == "in", "col_id_local"] if c in count]
            out_vals = [count[c] for c in block.loc[block.role == "ref", "col_id_local"] if c in count]
            if in_vals and out_vals and np.mean(out_vals) > 0:
                rows.append({"sample": sample, "condition": tp, "rep": rep, "focal_region_id": int(region_id),
                             "rod_count_in": float(np.mean(in_vals)),
                             "rod_count_out": float(np.mean(out_vals)),
                             "ratio": float(np.mean(in_vals) / np.mean(out_vals))})
data = pd.DataFrame(rows)
data.to_csv(out_csv, index=False)

## Report per replicate the region count median ratio and range
print("FigS4e rod number in/out per replicate:")
for reps in condition_reps.values():
    for sample in reps:
        v = data.loc[data["sample"] == sample, "ratio"].to_numpy()
        if len(v):
            print(f"  {sample:16} n={len(v):2d}  median={np.median(v):.2f}  range={v.min():.2f}-{v.max():.2f}")

## Plot each focal region and the median for each retina
figure, axes = plt.subplots(figsize=(5.4, 3.6))
axes.axhline(1.0, color="#bbbbbb", lw=1.1, ls="--", zorder=1)
for tp in conditions:
    x = x_of[tp]
    color = paper_style.CONDITION_COLORS[tp]
    for rep, filled, dx in (("rep1", False, -0.16), ("rep2", True, 0.16)):
        v = data.loc[(data.condition == tp) & (data.rep == rep), "ratio"].to_numpy()
        if not len(v):
            continue
        axes.scatter(x + dx + rng.uniform(-0.06, 0.06, len(v)), v, s=34,
                     facecolor=(color if filled else "white"), edgecolor=color, linewidth=1.4, alpha=0.9, zorder=3)
        axes.plot([x + dx - 0.10, x + dx + 0.10], [np.median(v)] * 2, color=color, lw=2.4, zorder=4)
axes.set_xticks(list(x_of.values()))
axes.set_xticklabels([paper_style.condition_label(t) for t in x_of], fontsize=11, fontweight="bold", rotation=15, ha="right")
for tick, tp in zip(axes.get_xticklabels(), x_of):
    tick.set_color(paper_style.CONDITION_COLORS[tp])
axes.set_xlim(-0.6, 3.6)
axes.set_ylim(min(data["ratio"].min(), 0.9) - 0.05, max(data["ratio"].max(), 1.1) + 0.05)
axes.set_ylabel("Rod number (in vs out)", fontsize=11, fontweight="bold")
axes.legend(handles=[Line2D([0], [0], marker="o", ls="", mfc="white", mec="#555", ms=8, label="rep1"),
                     Line2D([0], [0], marker="o", ls="", mfc="#555", mec="#555", ms=8, label="rep2")],
            fontsize=9, frameon=False, loc="upper right")
for spine in ("top", "right"):
    axes.spines[spine].set_visible(False)
figure.tight_layout()
figure.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(figure)
print("wrote", out_png)
