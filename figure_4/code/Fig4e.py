# Fig 4e Muller gliosis local enrichment in vs out of the focal rod injury response regions (LCA5)

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
out_png = "../panels/Fig4e.png"
out_csv = "../data_plot/Fig4e.csv"
out_csv_regions = "../data_plot/Fig4e_regions.csv"
mg_genes = ["Gfap", "Serpina3n"]
mg_layers = ["ONL", "INL"]
timepoint_reps = {"LCA5_P21": ["LCA5_P21_rep1", "LCA5_P21_rep2"], "LCA5_P30": ["LCA5_P30_rep1", "LCA5_P30_rep2"]}
x_of = {"LCA5_P21": 0.0, "LCA5_P30": 1.0}
rng = np.random.default_rng(0)


## Müller gliosis transcript density (transcripts/µm²) in each column
def gliosis_density(sample):
    cm = pd.read_parquet(focal.FT / "columns_meta.parquet")
    cm = cm[cm["sample"] == sample].set_index("col_id_local")
    dens = pd.read_parquet(focal.DENS, columns=["sample_id", "col_id_local", "gene", "layer", "transcript_count"])
    dens = dens[(dens.sample_id == sample) & dens.gene.isin(mg_genes) & dens.layer.isin(mg_layers)]
    count = dens.groupby("col_id_local")["transcript_count"].sum().reindex(cm.index, fill_value=0)
    onl_inl_area = cm["ONL_area_um2"] + cm["INL_area_um2"]
    density = np.where(onl_inl_area > 0, count / onl_inl_area, 0.0)
    return {int(col): float(value) for col, value in zip(cm.index, density)}


## Calculate one ratio of means for each retina and retain each region value for display
region_rows, retina_rows = [], []
for tp, reps in timepoint_reps.items():
    for sample in reps:
        rep = "rep1" if sample.endswith("rep1") else "rep2"
        densities = gliosis_density(sample)
        reference = focal.local_reference(sample, require_rods=False)
        pooled_in, pooled_ref, n_reg = [], [], 0
        for region_id, block in reference.groupby("focal_region_id"):
            if not bool(block["has_reference"].iloc[0]):
                continue
            in_vals = [densities[c] for c in block.loc[block.role == "in", "col_id_local"] if c in densities]
            ref_vals = [densities[c] for c in block.loc[block.role == "ref", "col_id_local"] if c in densities]
            if not (in_vals and ref_vals and np.mean(ref_vals) > 0):
                continue
            pooled_in += in_vals
            pooled_ref += ref_vals
            n_reg += 1
            region_rows.append({"tp": tp, "rep": rep, "focal_region_id": int(region_id),
                                "fold": float(np.mean(in_vals) / np.mean(ref_vals))})
        retina_rows.append({"tp": tp, "rep": rep, "n_regions": n_reg,
                            "in_density": float(np.mean(pooled_in)), "out_density": float(np.mean(pooled_ref)),
                            "fold": float(np.mean(pooled_in) / np.mean(pooled_ref))})
retina = pd.DataFrame(retina_rows)
region = pd.DataFrame(region_rows)
retina.to_csv(out_csv, index=False)
region.to_csv(out_csv_regions, index=False)
print("Fig4e retina-level ratio-of-means fold (density):")
print(retina[["tp", "rep", "n_regions", "in_density", "out_density", "fold"]].to_string(index=False))

## Use a broken y axis to show the main range and the highest focal region values
break_at = 2.9
y_top = max(region["fold"].max(), retina["fold"].max()) * 1.06
region_xy = {tp: (x_of[tp] + rng.uniform(-0.13, 0.13, int((region.tp == tp).sum())),
                  region.loc[region.tp == tp, "fold"].to_numpy()) for tp in timepoint_reps}
figure, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, figsize=(4.4, 4.2),
                                        gridspec_kw={"height_ratios": [1, 3], "hspace": 0.08})


## Draw the same values on both y axis ranges
def draw(ax):
    ax.axhline(1.0, color="#bbbbbb", lw=1.1, ls="--", zorder=1)
    for tp in timepoint_reps:
        x = x_of[tp]
        color = paper_style.CONDITION_COLORS[tp]
        xs, ys = region_xy[tp]
        ax.scatter(xs, ys, s=20, color=color, alpha=0.40, edgecolor="none", zorder=2)
        vals = retina.loc[retina.tp == tp, "fold"].to_numpy()
        ax.plot([x, x], [vals.min(), vals.max()], color=color, lw=1.4, zorder=3)
        ax.plot([x - 0.14, x + 0.14], [vals.mean(), vals.mean()], color=color, lw=2.2, zorder=4)
        for rep, filled in (("rep1", False), ("rep2", True)):
            v = retina.loc[(retina.tp == tp) & (retina.rep == rep), "fold"]
            ax.scatter([x] * len(v), v, s=75, facecolor=(color if filled else "white"), edgecolor=color, linewidth=1.7, zorder=5)


draw(ax_top)
draw(ax_bot)
ax_bot.set_ylim(0, break_at)
ax_top.set_ylim(break_at, y_top)
ax_top.set_yticks(np.arange(3, int(y_top) + 1))

## Hide the inner spines and draw the diagonal break marks like Fig4a
for ax in (ax_top, ax_bot):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
ax_top.spines["bottom"].set_visible(False)
ax_bot.spines["top"].set_visible(False)
ax_top.tick_params(axis="x", length=0, labelbottom=False)
dx, dt, db = 0.012, 0.03, 0.01
mark = dict(color="#222222", clip_on=False, lw=1.2)
ax_top.plot((-dx, dx), (-dt, dt), transform=ax_top.transAxes, **mark)
ax_top.plot((1 - dx, 1 + dx), (-dt, dt), transform=ax_top.transAxes, **mark)
ax_bot.plot((-dx, dx), (1 - db, 1 + db), transform=ax_bot.transAxes, **mark)
ax_bot.plot((1 - dx, 1 + dx), (1 - db, 1 + db), transform=ax_bot.transAxes, **mark)

ax_bot.set_xticks(list(x_of.values()))
ax_bot.set_xticklabels([paper_style.condition_label(t) for t in x_of], fontsize=14, fontweight="bold")
for tick, tp in zip(ax_bot.get_xticklabels(), x_of):
    tick.set_color(paper_style.CONDITION_COLORS[tp])
ax_bot.set_xlim(-0.6, 1.6)
ax_bot.set_ylabel("Müller gliosis marker enrichment (in vs out)", fontsize=11, fontweight="bold")
ax_bot.yaxis.set_label_coords(-0.115, 0.72)
ax_top.legend(handles=[Line2D([0], [0], marker="o", ls="", mfc="white", mec="#555", ms=8, label="rep1"),
                       Line2D([0], [0], marker="o", ls="", mfc="#555", mec="#555", ms=8, label="rep2"),
                       Line2D([0], [0], marker="o", ls="", mfc="#999", mec="none", ms=5, label="focal region")],
              fontsize=9, frameon=False, loc="upper right")
figure.tight_layout()
figure.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(figure)
print("wrote", out_png)
