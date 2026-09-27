# Fig S4b S/I structure of the per rod injury response score

## Load packages
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

## Load paper style conventions and the shared superior/inferior axis
sys.path.insert(0, "../../_shared/code")
import paper_style
from _superior_inferior_axis import superior_mask
paper_style.set_style()

## Load files
score_path = "../../figure_4/data_raw_manifest/rod_injury_response_score.parquet"
cells_path = "../../_shared/data_processed/per_cell_gene_counts.parquet"
out_png = "../panels/FigS4b.png"
out_csv = "../data_plot/FigS4b.csv"

## Define conditions colors and parameters
violin_conditions = {"WT_P21": ["WT_P21_rep1", "WT_P21_rep2"],
                     "WT_P64": ["WT_P64_rep1", "WT_P64_rep2"],
                     "LCA5_P21": ["LCA5_P21_rep1", "LCA5_P21_rep2"],
                     "LCA5_P30": ["LCA5_P30_rep1", "LCA5_P30_rep2"]}
condition_label_color = {cond: paper_style.CONDITION_COLORS[cond] for cond in ["WT_P21", "WT_P64", "LCA5_P21", "LCA5_P30"]}
sup_color = "#1b9e3b"
inf_color = "#c800a8"
group_step = 1.50
violin_width = 1.10
y_break = 8.0
min_count = 10

## Load the per rod score and assign the superior inferior hemisphere
score = pd.read_parquet(score_path)
cells = pd.read_parquet(cells_path, columns=["cell_id", "sample_id", "cell_type",
                        "cx_um", "cy_um", "body_Opn1mw", "body_Opn1sw"])
cells = cells.copy()
# Split superior vs inferior at the per sample median of the shared sigma=120 opsin field
cells["is_superior"] = superior_mask(cells)
rods = (cells[cells["cell_type"] == "rod"][["cell_id", "sample_id", "is_superior"]]
        .rename(columns={"cell_id": "label", "sample_id": "sample"}))
data = score.merge(rods, on=["sample", "label"], how="inner", validate="one_to_one")
per_condition = {}
for cond, reps in violin_conditions.items():
    per_condition[cond] = data[data["sample"].isin(reps)][["injury_response_score", "is_superior", "rep"]].copy()

## Build the split half violins per condition
conditions = list(violin_conditions)
centers = np.arange(len(conditions), dtype=float) * group_step
halves = []
for index, cond in enumerate(conditions):
    condition_data = per_condition[cond]
    for side, color in (("left", sup_color), ("right", inf_color)):
        subset = condition_data[condition_data["is_superior"]] if side == "left" else condition_data[~condition_data["is_superior"]]
        halves.append(dict(x=centers[index], side=side, color=color, vals=subset["injury_response_score"].to_numpy(), reps=subset[["injury_response_score", "rep"]]))

## Seed the cosmetic jitter subsample per half
random_generator = np.random.default_rng(0)
n_jitter = 180
for half in halves:
    values = half["vals"]
    if len(values):
        sample_values = values if len(values) <= n_jitter else random_generator.choice(values, n_jitter, replace=False)
        jitter = random_generator.uniform(-0.34, -0.02, len(sample_values)) if half["side"] == "left" else random_generator.uniform(0.02, 0.34, len(sample_values))
        half["samp"], half["jx"] = sample_values, half["x"] + jitter
    else:
        half["samp"], half["jx"] = np.array([]), np.array([])


## Draw the half violins on one axis
def draw(axes):
    for half in halves:
        if len(half["vals"]) < min_count:
            continue
        x, side, color = half["x"], half["side"], half["color"]
        violin = axes.violinplot([half["vals"]], positions=[x], vert=True, widths=violin_width, showmeans=False, showmedians=False, showextrema=False)
        body = violin["bodies"][0]
        vertices = body.get_paths()[0].vertices
        if side == "left":
            vertices[:, 0] = np.clip(vertices[:, 0], -np.inf, x)
        else:
            vertices[:, 0] = np.clip(vertices[:, 0], x, np.inf)
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)
        body.set_linewidth(1.0)
        axes.scatter(half["jx"], half["samp"], s=1.6, color=color, alpha=0.18, edgecolor="none", zorder=2)
        x_left, x_right = (x - 0.40, x) if side == "left" else (x, x + 0.40)
        axes.hlines(np.median(half["vals"]), x_left, x_right, color=color, lw=2.0, zorder=4)
        for rep in ("rep1", "rep2"):
            rep_scores = half["reps"].loc[half["reps"]["rep"] == rep, "injury_response_score"]
            if len(rep_scores):
                x_dot = x - 0.18 if side == "left" else x + 0.18
                axes.scatter([x_dot], [rep_scores.median()], s=16, color=color, zorder=5, edgecolor="white", linewidth=0.5)


## Set up the broken axis figure and draw
figure = plt.figure(figsize=(3.85, 3.9))
grid = figure.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.06, left=0.15, right=0.97, top=0.88, bottom=0.16)
axes_top = figure.add_subplot(grid[0])
axes_bottom = figure.add_subplot(grid[1], sharex=axes_top)
for axes in (axes_top, axes_bottom):
    draw(axes)
axes_bottom.axhline(0.0, color="#888", lw=0.6, zorder=1)
low = float(np.floor(np.concatenate([half["vals"] for half in halves if len(half["vals"])]).min() * 2) / 2)
high = float(np.ceil(max(half["vals"].max() for half in halves if len(half["vals"])) * 5) / 5) + 0.1
axes_bottom.set_ylim(low, y_break)
axes_top.set_ylim(y_break, high)
axes_top.spines["bottom"].set_visible(False)
axes_bottom.spines["top"].set_visible(False)
axes_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
break_style = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=8, linestyle="none", color="k", mec="k", mew=1, clip_on=False)
axes_top.plot([0], [0], transform=axes_top.transAxes, **break_style)
axes_bottom.plot([0], [1], transform=axes_bottom.transAxes, **break_style)
axes_bottom.set_xticks(centers)
axes_bottom.set_xticklabels([paper_style.condition_label(c) for c in conditions], fontsize=8, fontweight="bold", rotation=30, ha="right")
for tick_label, cond in zip(axes_bottom.get_xticklabels(), conditions):
    tick_label.set_color(condition_label_color[cond])
axes_bottom.tick_params(axis="x", length=0)
axes_bottom.set_xlim(centers[0] - group_step * 0.7, centers[-1] + group_step * 0.7)
for axes in (axes_top, axes_bottom):
    axes.tick_params(axis="y", labelsize=8)
axes_bottom.set_yticks([tick for tick in axes_bottom.get_yticks() if low <= tick < y_break])
axes_top.set_yticks([tick for tick in axes_top.get_yticks() if y_break <= tick <= high])
figure.text(0.035, 0.5, "Rod injury response score", rotation=90, va="center", ha="center", fontsize=11, fontweight="bold")
top_transform = axes_top.get_xaxis_transform()


## Mark the replicate concordant superior over inferior conditions
def rep_dir(subset):
    superior_values = subset.loc[subset["is_superior"], "injury_response_score"].to_numpy()
    inferior_values = subset.loc[~subset["is_superior"], "injury_response_score"].to_numpy()
    if len(superior_values) >= min_count and len(inferior_values) >= min_count:
        return float(np.sign(np.median(superior_values) - np.median(inferior_values)))
    return np.nan


records = []
for index, cond in enumerate(conditions):
    condition_data = per_condition[cond]
    scores = condition_data["injury_response_score"].to_numpy()
    sup_mask = condition_data["is_superior"].to_numpy()
    superior_values, inferior_values = scores[sup_mask], scores[~sup_mask]
    directions = [rep_dir(condition_data[condition_data["rep"] == rep]) for rep in ("rep1", "rep2")]
    directions = [direction for direction in directions if np.isfinite(direction) and direction != 0]
    concordant = len(directions) == 2 and directions[0] == directions[1]
    marker = "S>I" if (concordant and directions[0] > 0) else None
    if marker:
        axes_top.text(centers[index], 0.55, marker, transform=top_transform, ha="center", va="bottom", fontsize=9, fontweight="bold", color="#222")
    records.append(dict(condition=cond, mean_S=float(superior_values.mean()), mean_I=float(inferior_values.mean()) if len(inferior_values) else np.nan,
                        n_S=int(superior_values.size), n_I=int(inferior_values.size), rep_concordant=concordant, direction=marker))

## Add the legend and save
axes_top.legend(handles=[Patch(facecolor=sup_color, alpha=0.55, label="superior (S)"),
                         Patch(facecolor=inf_color, alpha=0.55, label="inferior (I)")],
                fontsize=8.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.05),
                ncol=2, handlelength=1.1, handleheight=1.1, columnspacing=1.4)
summary_df = pd.DataFrame(records)
summary_df.to_csv(out_csv, index=False)
figure.savefig(out_png, dpi=200, bbox_inches="tight")
plt.close(figure)
print("wrote", out_png)
