# Fig 4a rod injury response score distributions

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
pool_path = "../data_raw_manifest/all_rods.parquet"
out_panel = "../panels/Fig4a.png"
# Fig4a generates the rod injury response score used by the other Figure 4 panels
out_score = "../data_raw_manifest/rod_injury_response_score.parquet"

## Define genes conditions and parameters
genes = ["Edn2", "Socs3", "Fgf2", "Stat3"]
condition_order = paper_style.CONDITION_ORDER
condition_colors = paper_style.CONDITION_COLORS
max_dots = 500
jitter_width = 0.30
dot_size = 0.9
x_break = 3.8
random_generator = np.random.default_rng(0)


## Standardize values to a z score returning zeros if all values are equal
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


## Load the rod pool
rods = pd.read_parquet(pool_path).reset_index(drop=True)
rods = rods[rods["condition"].isin(condition_order)].copy()

## Compute the injury response score per rod
area = rods["cell_area_um2"].to_numpy()
gene_zscores = []
for gene in genes:
    gene_zscores.append(zscore(np.log1p(rods["body_" + gene].to_numpy() / area)))
scores = rods[["cell_id", "sample_id", "condition", "rep"]].rename(columns={"cell_id": "label", "sample_id": "sample"})
scores["injury_response_score"] = np.mean(gene_zscores, axis=0)
scores.to_parquet(out_score, index=False)

## Set the x axis range to span every rod across a broken axis
x_lo = float(scores["injury_response_score"].min()) - 0.3
x_hi = float(scores["injury_response_score"].max()) + 0.3
row_positions = np.arange(len(condition_order))[::-1]

## Sample the rod dots once for both y axis ranges
jitter_batches = []
for row_index, cond in enumerate(condition_order):
    y_row = row_positions[row_index]
    for rep in ("rep1", "rep2"):
        rep_scores = scores.loc[(scores["condition"] == cond) & (scores["rep"] == rep), "injury_response_score"].to_numpy()
        if len(rep_scores) == 0:
            continue
        if len(rep_scores) > max_dots:
            rep_scores = random_generator.choice(rep_scores, size=max_dots, replace=False)
        dot_color = lighten(condition_colors[cond], 0.50) if rep == "rep1" else darken(condition_colors[cond], 0.25)
        y_jitter = y_row + random_generator.uniform(-jitter_width, jitter_width, size=len(rep_scores))
        jitter_batches.append((rep_scores, y_jitter, dot_color))


## Draw the distributions, quartiles, retina medians, and rod values
def draw_panel(axes):
    for row_index, cond in enumerate(condition_order):
        y_row = row_positions[row_index]
        cond_scores = scores.loc[scores["condition"] == cond, "injury_response_score"].to_numpy()
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
            rep_scores = scores.loc[(scores["condition"] == cond) & (scores["rep"] == rep), "injury_response_score"].to_numpy()
            if len(rep_scores) == 0:
                continue
            dot_color = lighten(condition_colors[cond], 0.50) if rep == "rep1" else darken(condition_colors[cond], 0.25)
            axes.scatter([float(np.median(rep_scores))], [y_row + rep_offsets[rep]], s=dot_size * 18, facecolor=dot_color, edgecolor="white", linewidth=0.8, zorder=7)
    for dot_x, dot_y, dot_color in jitter_batches:
        axes.scatter(dot_x, dot_y, s=dot_size, c=[dot_color], edgecolor="none", alpha=0.55, zorder=3)


## Draw the bulk panel and the compressed tail panel sharing the condition rows
figure, (ax_main, ax_tail) = plt.subplots(1, 2, sharey=True, figsize=(2.9, 2.35), facecolor="white",
                                          gridspec_kw={"width_ratios": [3.0, 1.0], "wspace": 0.06})
draw_panel(ax_main)
draw_panel(ax_tail)
ax_main.axvline(0, color="#bbbbbb", lw=0.4, zorder=1)
ax_main.set_xlim(x_lo, x_break)
ax_tail.set_xlim(x_break, x_hi)
ax_main.set_xticks([0, 1, 2, 3])
ax_tail.set_xticks([4, 6, 8, 10, 12])

## Label the condition rows on the shared left axis
ax_main.set_yticks(row_positions)
ax_main.set_yticklabels([paper_style.condition_label(c) for c in condition_order], fontsize=7.0)
for tick_label, cond in zip(ax_main.get_yticklabels(), condition_order):
    tick_label.set_color(condition_colors[cond])
ax_main.set_ylim(-0.7, len(condition_order) - 0.3)

## Style the axes and draw the break marks between the two panels
for axes in (ax_main, ax_tail):
    axes.tick_params(axis="x", labelsize=7)
    axes.spines["top"].set_visible(False)
ax_main.spines["right"].set_visible(False)
ax_tail.spines["left"].set_visible(False)
ax_tail.tick_params(axis="y", length=0)
for side, axes in (("left", ax_main), ("bottom", ax_main), ("bottom", ax_tail)):
    axes.spines[side].set_visible(True)
    axes.spines[side].set_edgecolor("#222222")
    axes.spines[side].set_linewidth(1.1)
break_size = 0.015
ax_main.plot((1 - break_size, 1 + break_size), (-break_size, break_size), transform=ax_main.transAxes, color="#222222", lw=1.1, clip_on=False)
ax_tail.plot((-break_size * 3.0, break_size * 3.0), (-break_size, break_size), transform=ax_tail.transAxes, color="#222222", lw=1.1, clip_on=False)

## Add the shared x axis label
figure.text(0.5, 0.01, "Rod injury response score", ha="center", fontsize=8, fontweight="bold")
figure.subplots_adjust(left=0.02, right=0.995, top=0.99, bottom=0.16)

## Save the panel
figure.savefig(out_panel, dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close(figure)
print("wrote", out_panel)
