# Fig S4a per gene rod injury response violins

## Load packages
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

## Load paper style conventions
sys.path.insert(0, "../../_shared/code")
import paper_style
paper_style.set_style()

## Load files
out_panel = "../panels"
out_data = "../data_plot"
pool_path = "../../figure_4/data_raw_manifest/all_rods.parquet"

## Define genes conditions and parameters
genes = ["Edn2", "Socs3", "Fgf2", "Stat3"]
condition_order = paper_style.CONDITION_ORDER
condition_colors = paper_style.CONDITION_COLORS
max_dots = 500
jitter_width = 0.30
random_generator = np.random.default_rng(0)


## Standardize values to a z score zeros if all values equal
def zscore(values):
    standard_deviation = float(np.std(values))
    if standard_deviation < 1e-9:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.mean(values))) / standard_deviation


## Mix a color toward white by fraction
def lighten(color, fraction):
    red, green, blue = to_rgb(color)
    return (red + (1 - red) * fraction, green + (1 - green) * fraction, blue + (1 - blue) * fraction)


## Mix a color toward black by fraction
def darken(color, fraction):
    red, green, blue = to_rgb(color)
    return (red * (1 - fraction), green * (1 - fraction), blue * (1 - fraction))


## Draw the violins one row per condition
def draw_violins(axes, data, score_column, dot_size, ytick_fontsize, show_yticklabels):
    row_positions = np.arange(len(condition_order))[::-1]
    for row_index, cond in enumerate(condition_order):
        y_row = row_positions[row_index]
        cond_scores = data.loc[data["condition"] == cond, score_column].to_numpy()
        violin = axes.violinplot([cond_scores], positions=[y_row], showmeans=False, showmedians=False, showextrema=False, widths=0.62, vert=False)
        violin_body = violin["bodies"][0]
        violin_body.set_facecolor(condition_colors[cond])
        violin_body.set_edgecolor("none")
        violin_body.set_alpha(0.45)
        lower_quartile, upper_quartile = np.percentile(cond_scores, [25, 75])
        median = float(np.median(cond_scores))
        band_half = 0.045
        axes.add_patch(mpl.patches.Rectangle((lower_quartile, y_row - band_half), upper_quartile - lower_quartile, 2 * band_half,
                       facecolor=darken(condition_colors[cond], 0.35), edgecolor="none", alpha=0.95, zorder=4))
        axes.plot([median, median], [y_row - band_half, y_row + band_half], color="white", lw=1.2, zorder=5)
        rep_offsets = {"rep1": -band_half * 1.6, "rep2": +band_half * 1.6}
        for rep in ("rep1", "rep2"):
            rep_scores = data.loc[(data["condition"] == cond) & (data["rep"] == rep), score_column].to_numpy()
            if len(rep_scores) == 0:
                continue
            dot_color = lighten(condition_colors[cond], 0.50) if rep == "rep1" else darken(condition_colors[cond], 0.25)
            axes.scatter([float(np.median(rep_scores))], [y_row + rep_offsets[rep]], s=dot_size * 18, facecolor=dot_color, edgecolor="white", linewidth=0.8, zorder=7)
        for rep in ("rep1", "rep2"):
            rep_scores = data.loc[(data["condition"] == cond) & (data["rep"] == rep), score_column].to_numpy()
            if len(rep_scores) == 0:
                continue
            if len(rep_scores) > max_dots:
                rep_scores = random_generator.choice(rep_scores, size=max_dots, replace=False)
            dot_color = lighten(condition_colors[cond], 0.50) if rep == "rep1" else darken(condition_colors[cond], 0.25)
            y_jitter = y_row + random_generator.uniform(-jitter_width, jitter_width, size=len(rep_scores))
            axes.scatter(rep_scores, y_jitter, s=dot_size, c=[dot_color], edgecolor="none", alpha=0.55, zorder=3)
    axes.set_yticks(row_positions)
    if show_yticklabels:
        axes.set_yticklabels([paper_style.condition_label(c) for c in condition_order], fontsize=ytick_fontsize)
        for tick_label, cond in zip(axes.get_yticklabels(), condition_order):
            tick_label.set_color(condition_colors[cond])
    else:
        axes.set_yticklabels([])
    axes.set_ylim(-0.7, len(condition_order) - 0.3)


## Style the axes spines
def style_spines(axes):
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_visible(True)
        axes.spines[side].set_edgecolor("#222222")
        axes.spines[side].set_linewidth(0.8)


## Load the rod pool and compute per gene z scores
rods = pd.read_parquet(pool_path).reset_index(drop=True)
rods = rods[rods["condition"].isin(condition_order)].copy()
area = rods["cell_area_um2"].to_numpy()
per_gene = rods[["label", "sample", "condition", "rep"]].copy()
for gene in genes:
    per_gene["z_" + gene] = zscore(np.log1p(rods["body_" + gene].to_numpy() / area))
per_gene.to_parquet(f"{out_data}/FigS4a.parquet", index=False)

## Draw one violin panel per gene
n_columns = len(genes)
figure, axes_grid = plt.subplots(1, n_columns, figsize=(2.4 * n_columns, 2.5), sharey=False, squeeze=False)
for column_index, gene in enumerate(genes):
    axes = axes_grid[0, column_index]
    column = "z_" + gene
    x_lo = float(np.percentile(per_gene[column], 1)) - 0.3
    x_hi = float(np.percentile(per_gene[column], 99)) + 0.3
    draw_violins(axes, per_gene, column, dot_size=0.9, ytick_fontsize=8.0, show_yticklabels=(column_index == 0))
    axes.axvline(0, color="#bbbbbb", lw=0.4, zorder=1)
    axes.set_xlim(x_lo, x_hi)
    axes.set_title(gene, fontsize=9, fontstyle="italic", color="#222222")
    axes.set_xlabel("Z log1p density", fontsize=8)
    axes.tick_params(axis="x", labelsize=7)
    style_spines(axes)
figure.subplots_adjust(left=0.13, right=0.99, top=0.93, bottom=0.10, wspace=0.14, hspace=0.30)

## Save the panel
figure.savefig(f"{out_panel}/FigS4a.png", dpi=300, bbox_inches="tight")
plt.close(figure)
print("wrote", f"{out_panel}/FigS4a.png")
