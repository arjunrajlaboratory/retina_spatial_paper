# Fig S4f Müller gliosis transcript counts by condition

## Load packages
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb

## Load paper style conventions and the shared focal region functions
sys.path.insert(0, "../../_shared/code")
import paper_style
import _focal_regions as focal
paper_style.set_style()


## Mix a color toward black by a fraction
def darken(color, fraction):
    r, g, b = to_rgb(color)
    return (r * (1 - fraction), g * (1 - fraction), b * (1 - fraction))

## Load files
out_png = "../panels/FigS4f.png"
out_csv = "../data_plot/FigS4f.csv"
mg_genes = ["Gfap", "Serpina3n"]
mg_layers = ["ONL", "INL"]
conditions = ["WT_P21", "WT_P64", "LCA5_P21", "LCA5_P30"]
condition_reps = {c: [f"{c}_rep1", f"{c}_rep2"] for c in conditions}


## Per column Müller gliosis transcript density (transcripts/µm²)
## Normalize the pooled Gfap and Serpina3n count by the combined ONL and INL area from columns_meta
## which defines an area for every column so the denominator never depends on where the marker was detected
def gliosis_density(sample):
    cm = pd.read_parquet(focal.FT / "columns_meta.parquet")
    cm = cm[cm["sample"] == sample].set_index("col_id_local")
    dens = pd.read_parquet(focal.DENS, columns=["sample_id", "col_id_local", "gene", "layer", "transcript_count"])
    dens = dens[(dens.sample_id == sample) & dens.gene.isin(mg_genes) & dens.layer.isin(mg_layers)]
    count = dens.groupby("col_id_local")["transcript_count"].sum().reindex(cm.index, fill_value=0)
    onl_inl_area = cm["ONL_area_um2"] + cm["INL_area_um2"]
    density = np.where(onl_inl_area > 0, count / onl_inl_area, 0.0)
    return {int(col): float(value) for col, value in zip(cm.index, density)}


## Collect per valid column gliosis density for every retina
rows = []
for condition in conditions:
    for sample in condition_reps[condition]:
        rep = "rep1" if sample.endswith("rep1") else "rep2"
        densities = gliosis_density(sample)
        valid = set(int(c) for c in focal.focal_region_columns(sample).query("valid_tissue")["col_id_local"])
        for col, value in densities.items():
            if col in valid:
                rows.append({"condition": condition, "rep": rep, "density": value})
data = pd.DataFrame(rows)
data.to_csv(out_csv, index=False)

## Report medians
print("FigS4f Muller gliosis density per column (median transcripts/µm²):")
print(data.groupby("condition")["density"].median().to_string())

## Plot column values and the median for each retina
row_positions = np.arange(len(conditions))[::-1]
x_hi = float(np.percentile(data["density"].to_numpy(), 99))
random_generator = np.random.default_rng(0)
figure, axes = plt.subplots(figsize=(5.2, 3.4))
for row_index, condition in enumerate(conditions):
    y_row = row_positions[row_index]
    color = paper_style.CONDITION_COLORS[condition]
    vals = data.loc[data.condition == condition, "density"].to_numpy()
    body = axes.violinplot([vals], positions=[y_row], vert=False, widths=0.72, showextrema=False)["bodies"][0]
    body.set_facecolor(color)
    body.set_edgecolor("none")
    body.set_alpha(0.35)
    # Jitter all columns in one band
    axes.scatter(vals, y_row + random_generator.uniform(-0.18, 0.18, len(vals)), s=4,
                 facecolor=color, edgecolor="none", alpha=0.16, zorder=3)
    # Draw the middle bar with IQR band plus median line
    lower_quartile, upper_quartile = np.percentile(vals, [25, 75])
    band_half = 0.05
    axes.add_patch(Rectangle((lower_quartile, y_row - band_half), upper_quartile - lower_quartile, 2 * band_half,
                             facecolor=darken(color, 0.35), edgecolor="none", alpha=0.95, zorder=4))
    axes.plot([float(np.median(vals))] * 2, [y_row - band_half, y_row + band_half], color="white", lw=1.2, zorder=5)
    # Draw replicate medians as circles offset off the bar rep1 open and rep2 filled
    for rep, filled, dy in (("rep1", False, 0.15), ("rep2", True, -0.15)):
        rep_vals = data.loc[(data.condition == condition) & (data.rep == rep), "density"].to_numpy()
        if len(rep_vals):
            axes.scatter([float(np.median(rep_vals))], [y_row + dy], s=48, facecolor=(color if filled else "white"),
                         edgecolor="white", linewidth=0.8, zorder=6)
axes.set_yticks(row_positions)
axes.set_yticklabels([paper_style.condition_label(c) for c in conditions], fontsize=9)
for tick_label, condition in zip(axes.get_yticklabels(), conditions):
    tick_label.set_color(paper_style.CONDITION_COLORS[condition])
axes.set_ylim(-0.7, len(conditions) - 0.3)
axes.set_xlim(0, x_hi)
axes.set_xlabel(r"$\mathit{Gfap}$+$\mathit{Serpina3n}$ transcript density, ONL+INL (transcripts/µm²)", fontsize=10, fontweight="bold")
axes.legend(handles=[Line2D([0], [0], marker="o", ls="", mfc="white", mec="#555", ms=8, label="rep1 median"),
                     Line2D([0], [0], marker="o", ls="", mfc="#555", mec="#555", ms=8, label="rep2 median")],
            fontsize=9, frameon=False, loc="lower right")
for spine in ("top", "right"):
    axes.spines[spine].set_visible(False)
figure.tight_layout()
figure.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(figure)
print("wrote", out_png)
