# Fig S4d compare each three gene focal region profile with the full four gene profile

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
pool_path = "../../figure_4/data_raw_manifest/all_rods.parquet"
out_png = "../panels/FigS4d.png"
out_csv = "../data_plot/FigS4d.csv"
program = ["Edn2", "Socs3", "Fgf2", "Stat3"]
timepoint_reps = {"LCA5_P21": ["LCA5_P21_rep1", "LCA5_P21_rep2"],
                  "LCA5_P30": ["LCA5_P30_rep1", "LCA5_P30_rep2"]}
samples = [s for reps in timepoint_reps.values() for s in reps]


## Z score zeros if constant
def zscore(values):
    sd = float(np.std(values))
    return np.zeros_like(values, dtype=float) if sd < 1e-9 else (values - float(np.mean(values))) / sd


## Standardize each score gene across the full rod pool
pool = pd.read_parquet(pool_path, columns=["sample", "label", "cx_um", "cy_um", "cell_area_um2"] + [f"body_{g}" for g in program])
area = pool["cell_area_um2"].to_numpy(float)
for gene in program:
    pool[f"z_{gene}"] = zscore(np.log1p(pool[f"body_{gene}"].to_numpy(float) / area))

## Compare each three gene profile with the full four gene profile in every retina
rows = []
full_curve = {s: focal.focal_region_columns(s)["smoothed_residual"].to_numpy(float) for s in samples}
for sample in samples:
    for held in program:
        keep = [g for g in program if g != held]
        pool["injury_response_score"] = pool[[f"z_{g}" for g in keep]].mean(axis=1)
        override = pool[pool["sample"] == sample][["label", "injury_response_score"]]
        curve = focal.focal_region_columns(sample, score_override=override)["smoothed_residual"].to_numpy(float)
        ok = np.isfinite(full_curve[sample]) & np.isfinite(curve)
        r = float(np.corrcoef(full_curve[sample][ok], curve[ok])[0, 1])
        rows.append({"sample": sample, "condition": sample.rsplit("_", 1)[0],
                     "rep": "rep" + sample.split("_rep")[-1], "held_gene": held, "correlation": r})
data = pd.DataFrame(rows)
data.to_csv(out_csv, index=False)
print(data.pivot_table(index="held_gene", values="correlation", aggfunc=["mean", "min"]).to_string())

## Plot one correlation for each omitted gene and retina
figure, axes = plt.subplots(figsize=(4.8, 4.0))
x_of = {g: i for i, g in enumerate(program)}
axes.axhline(1.0, color="#bbbbbb", lw=1.0, ls="--")
for _, r in data.iterrows():
    color = paper_style.CONDITION_COLORS[r["condition"]]
    filled = r["rep"] == "rep2"
    axes.scatter([x_of[r["held_gene"]] + (0.11 if filled else -0.11)], [r["correlation"]], s=55,
                 facecolor=(color if filled else "white"), edgecolor=color, linewidth=1.4, zorder=3)
for gene in program:
    axes.plot([x_of[gene] - 0.24, x_of[gene] + 0.24], [data.loc[data.held_gene == gene, "correlation"].mean()] * 2,
              color="#444", lw=2, zorder=2)
axes.set_xticks(list(x_of.values()))
axes.set_xticklabels([rf"$\mathit{{{g}}}$" for g in program], fontsize=11)
axes.set_xlabel("Gene omitted from the score", fontsize=10.5, fontweight="bold")
axes.set_ylabel("Spatial profile correlation\n(3-gene vs full 4-gene)", fontsize=10.5, fontweight="bold")
axes.set_ylim(0.0, 1.02)
axes.set_xlim(-0.5, len(program) - 0.5)
legend_handles = [Line2D([0], [0], marker="o", ls="", mfc="white", mec="#555", ms=8, label="rep1"),
                  Line2D([0], [0], marker="o", ls="", mfc="#555", mec="#555", ms=8, label="rep2"),
                  Line2D([0], [0], marker="s", ls="", mfc=paper_style.CONDITION_COLORS["LCA5_P21"], mec="none", ms=9, label=paper_style.condition_label("LCA5_P21")),
                  Line2D([0], [0], marker="s", ls="", mfc=paper_style.CONDITION_COLORS["LCA5_P30"], mec="none", ms=9, label=paper_style.condition_label("LCA5_P30"))]
axes.legend(handles=legend_handles, fontsize=8.5, frameon=False, loc="lower right", labelspacing=0.35)
for spine in ("top", "right"):
    axes.spines[spine].set_visible(False)
figure.tight_layout()
figure.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(figure)
print("wrote", out_png)
