# Fig S4c focal regions per retina vs the prominence threshold and vs the Gaussian smoothing sigma

## Load packages
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

## Load paper style conventions and the shared focal region functions
sys.path.insert(0, "../../_shared/code")
import paper_style
import _focal_regions as focal
paper_style.set_style()

## Load files
out_png = "../panels/FigS4c.png"
out_csv = "../data_plot/FigS4c.csv"
samples = ["WT_P21_rep1", "WT_P21_rep2", "WT_P64_rep1", "WT_P64_rep2",
           "LCA5_P21_rep1", "LCA5_P21_rep2", "LCA5_P30_rep1", "LCA5_P30_rep2"]
prominence_grid = [0.0, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50]
sigma_grid = [0.0, 12.5, 25.0, 37.5, 50.0, 62.5]


def condition(sample):
    return sample.rsplit("_", 1)[0]


## Vary one detection parameter at a time
prominence_counts, sigma_counts = {}, {}
for sample in samples:
    prominence_counts[sample] = [len(focal.per_focal_region_table(sample, prominence_min=prominence)) for prominence in prominence_grid]
    sigma_counts[sample] = [len(focal.per_focal_region_table(sample, sigma_um=sigma)) for sigma in sigma_grid]

## Save the swept counts
rows = []
for sample in samples:
    for prominence, n in zip(prominence_grid, prominence_counts[sample]):
        rows.append({"sample": sample, "sweep": "prominence", "value": prominence, "n_focal_regions": n})
    for sigma, n in zip(sigma_grid, sigma_counts[sample]):
        rows.append({"sample": sample, "sweep": "sigma", "value": sigma, "n_focal_regions": n})
pd.DataFrame(rows).to_csv(out_csv, index=False)

## Draw the two sweeps
figure, (ax_prom, ax_sigma) = plt.subplots(1, 2, figsize=(9.0, 3.6))
for sample in samples:
    color = paper_style.CONDITION_COLORS[condition(sample)]
    style = "-" if sample.endswith("rep1") else "--"
    ax_prom.plot(prominence_grid, prominence_counts[sample], color=color, ls=style, marker="o", ms=3, lw=1.4)
    ax_sigma.plot(sigma_grid, sigma_counts[sample], color=color, ls=style, marker="o", ms=3, lw=1.4)
ax_prom.axvline(0.10, color="#222", lw=1.1, ls=":")
ax_sigma.axvline(37.5, color="#222", lw=1.1, ls=":")
ax_prom.set_xlabel("Prominence threshold", fontsize=10, fontweight="bold")
ax_sigma.set_xlabel("Smoothing σ (µm)", fontsize=10, fontweight="bold")
ax_prom.set_ylabel("Focal regions per retina", fontsize=10, fontweight="bold")
ax_sigma.set_ylabel("Focal regions per retina", fontsize=10, fontweight="bold")
ax_prom.set_title("Prominence (chosen = 0.10)", fontsize=9)
ax_sigma.set_title("σ (chosen = 37.5 µm)", fontsize=9)
for ax in (ax_prom, ax_sigma):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
condition_handles = [Line2D([0], [0], color=paper_style.CONDITION_COLORS[c], lw=2, label=paper_style.condition_label(c))
                     for c in ["WT_P21", "WT_P64", "LCA5_P21", "LCA5_P30"]]
rep_handles = [Line2D([0], [0], color="#666", lw=1.4, ls="-", label="rep1"),
               Line2D([0], [0], color="#666", lw=1.4, ls="--", label="rep2")]
ax_sigma.legend(handles=condition_handles + rep_handles, fontsize=7, frameon=False, loc="upper right", labelspacing=0.3)

## Save the panel
figure.tight_layout()
figure.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(figure)
print("wrote", out_png)
