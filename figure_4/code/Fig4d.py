# Fig 4d unbiased screen for rod stress genes enriched in the focal injury response regions

## Load packages
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

## Load paper style conventions and the shared focal region functions
sys.path.insert(0, "../../_shared/code")
import paper_style
import _focal_regions as focal
paper_style.set_style()

## Load files
pool_path = "../data_raw_manifest/all_rods.parquet"
rod_filter_csv = "../../figure_3/data_processed/rod_expression_filter.csv"
out_png = "../panels/Fig4d.png"
out_csv = "../data_plot/Fig4d.csv"
red = "#CC0000"
score_color = "#21918c"   # Matches the score gene color in Fig4b
LABEL_MIN_FOLD = 1.2   # Concordant genes above this mean fold get a text label
timepoint_reps = {"LCA5_P21": ["LCA5_P21_rep1", "LCA5_P21_rep2"], "LCA5_P30": ["LCA5_P30_rep1", "LCA5_P30_rep2"]}
samples = [s for reps in timepoint_reps.values() for s in reps]
program = list(focal.PROGRAM_GENES)   # The four score genes shown as the teal control

## Gene set is all retained rod genes minus the four program genes
retained = set(pd.read_csv(rod_filter_csv).query("retained_for_rod_analysis")["gene"])
present = [c[5:] for c in pq.ParquetFile(pool_path).schema_arrow.names if c.startswith("body_")]
genes = sorted([g for g in present if g in retained and g not in set(program)])


## Standardize each score gene across the full rod pool
def zscore(values):
    sd = float(np.std(values))
    return np.zeros_like(values, dtype=float) if sd < 1e-9 else (values - float(np.mean(values))) / sd


## Build the rod table used when each score gene is omitted
prog_pool = pd.read_parquet(pool_path, columns=["sample", "label", "cx_um", "cy_um", "cell_area_um2"] + [f"body_{g}" for g in program])
parea = prog_pool["cell_area_um2"].to_numpy(float)
for g in program:
    prog_pool[f"z_{g}"] = zscore(np.log1p(prog_pool[f"body_{g}"].to_numpy(float) / parea))
    prog_pool[f"d_{g}"] = prog_pool[f"body_{g}"].to_numpy(float) / parea

## Calculate local enrichment in each retina
per_retina = {}
for sample in samples:
    pool = pd.read_parquet(pool_path, columns=["sample", "label", "cell_area_um2"] + [f"body_{g}" for g in genes])
    pool = pool[pool["sample"] == sample].merge(focal.rod_membership(sample)[["label", "col_id_local"]], on="label", how="inner", validate="one_to_one")
    for g in genes:
        pool[f"d_{g}"] = pool[f"body_{g}"] / pool["cell_area_um2"]
    colmean = pool.groupby("col_id_local")[[f"d_{g}" for g in genes]].mean()
    reference = focal.local_reference(sample)
    region_id_col = "focal_region_id"
    in_cols, ref_cols = [], []
    for _, block in reference.groupby(region_id_col):
        if not bool(block["has_reference"].iloc[0]):
            continue
        in_cols += [c for c in block.loc[block.role == "in", "col_id_local"] if c in colmean.index]
        ref_cols += [c for c in block.loc[block.role == "ref", "col_id_local"] if c in colmean.index]
    out = {}
    for g in genes:
        in_mean = colmean.loc[in_cols, f"d_{g}"].mean()
        ref_mean = colmean.loc[ref_cols, f"d_{g}"].mean()
        out[g] = float(in_mean / ref_mean) if (ref_mean and ref_mean > 0) else np.nan
    per_retina[sample] = out

screen_fold = pd.DataFrame(per_retina)
screen_fold.index.name = "gene"

## Average the two retina values at each timepoint
tp_fold = {tp: {g: float(np.nanmean([per_retina[s][g] for s in reps])) for g in genes} for tp, reps in timepoint_reps.items()}

## Identify genes enriched in all four retinas and label those with mean enrichment at least 1.2
mean_fold = {g: float(np.nanmean([per_retina[s][g] for s in samples])) for g in genes}
concordant = [g for g in genes if all(np.isfinite(per_retina[s][g]) and per_retina[s][g] > 1 for s in samples)]
labelled = sorted([g for g in concordant if mean_fold[g] >= LABEL_MIN_FOLD], key=lambda g: -mean_fold[g])
print(f"{len(concordant)} concordant genes; labelled (mean fold >= {LABEL_MIN_FOLD}): "
      + ", ".join(f"{g} {mean_fold[g]:.2f}x" for g in labelled))

## Test each score gene using regions defined by the other three genes
prog_retina = {}
for sample in samples:
    membership = focal.rod_membership(sample)[["label", "col_id_local"]]
    sub = prog_pool[prog_pool["sample"] == sample].merge(membership, on="label", how="inner", validate="one_to_one")
    colmean = sub.groupby("col_id_local")[[f"d_{g}" for g in program]].mean()
    out = {}
    for held in program:
        keep = [g for g in program if g != held]
        override = prog_pool.assign(injury_response_score=prog_pool[[f"z_{g}" for g in keep]].mean(axis=1))
        override = override[override["sample"] == sample][["label", "injury_response_score"]]
        reference = focal.local_reference(sample, score_override=override)
        rid = "focal_region_id"
        in_cols, ref_cols = [], []
        for _, block in reference.groupby(rid):
            if not bool(block["has_reference"].iloc[0]):
                continue
            in_cols += [c for c in block.loc[block.role == "in", "col_id_local"] if c in colmean.index]
            ref_cols += [c for c in block.loc[block.role == "ref", "col_id_local"] if c in colmean.index]
        in_mean = colmean.loc[in_cols, f"d_{held}"].mean()
        ref_mean = colmean.loc[ref_cols, f"d_{held}"].mean()
        out[held] = float(in_mean / ref_mean) if (ref_mean and ref_mean > 0) else np.nan
    prog_retina[sample] = out
tp_prog = {tp: {g: float(np.nanmean([prog_retina[s][g] for s in reps])) for g in program} for tp, reps in timepoint_reps.items()}
print("score genes (leave one out): " + ", ".join(f"{g} {np.nanmean([prog_retina[s][g] for s in samples]):.2f}x" for g in program))

## Save enrichment values for the screened genes and score genes
score_fold = pd.DataFrame(prog_retina)
score_fold.index.name = "gene"
screen_fold["gene_set"] = "screen"
score_fold["gene_set"] = "score_leave_one_out"
pd.concat([screen_fold, score_fold]).to_csv(out_csv)

## Plot all genes and highlight genes enriched in every retina
figure, axes = plt.subplots(figsize=(6.2, 5.8))
x_positions = {"LCA5_P21": 0.0, "LCA5_P30": 1.9}
rng = np.random.default_rng(0)
axes.axhline(1.0, color="#bbbbbb", lw=1.0, ls="--")
for tp, x in x_positions.items():
    vals = np.array([tp_fold[tp][g] for g in genes])
    vals = vals[np.isfinite(vals)]
    color = paper_style.CONDITION_COLORS[tp]
    body = axes.violinplot([vals], positions=[x], widths=0.95, showextrema=False)["bodies"][0]
    body.set_facecolor(color)
    body.set_edgecolor("none")
    body.set_alpha(0.22)
    axes.scatter(x + rng.uniform(-0.28, 0.28, len(vals)), vals, s=3.5, color="#9a9a9a", alpha=0.30, edgecolor="none", zorder=2)
    for g in concordant:
        axes.scatter([x], [tp_fold[tp][g]], s=40, color=red, edgecolor="none", zorder=5)


def label_side(tp, x_lead0, x_lead1, x_text, ha):
    ordered = sorted(labelled, key=lambda g: -tp_fold[tp][g])
    vals = [tp_fold[tp][g] for g in ordered]
    if not vals:
        return
    gap = (max(vals) - min(vals) + 0.02) * 0.16 + 0.02
    ys = list(vals)
    for i in range(1, len(ys)):
        if ys[i - 1] - ys[i] < gap:
            ys[i] = ys[i - 1] - gap
    for g, y in zip(ordered, ys):
        axes.plot([x_lead0, x_lead1], [tp_fold[tp][g], y], color=red, lw=0.6, zorder=4)
        axes.text(x_text, y, f"{g}  {tp_fold[tp][g]:.2f}×", va="center", ha=ha, fontsize=10, fontweight="bold", fontstyle="italic", color=red, zorder=6)


label_side("LCA5_P21", -0.16, -0.32, -0.35, "right")
label_side("LCA5_P30", 2.06, 2.20, 2.23, "left")

## Plot the score genes in teal at both timepoints
for tp, x in x_positions.items():
    for g in program:
        axes.scatter([x], [tp_prog[tp][g]], s=45, color=score_color, edgecolor="white", linewidth=0.6, zorder=7)


## Name the score genes at both timepoints with a leader from each dot into the inner gap
def label_scores(tp, x_lead0, x_lead1, x_text, ha):
    ordered = sorted(program, key=lambda g: -tp_prog[tp][g])
    vals = [tp_prog[tp][g] for g in ordered]
    gap = (max(vals) - min(vals) + 0.02) * 0.28 + 0.06
    ys = list(vals)
    for i in range(1, len(ys)):
        if ys[i - 1] - ys[i] < gap:
            ys[i] = ys[i - 1] - gap
    for g, y in zip(ordered, ys):
        axes.plot([x_lead0, x_lead1], [tp_prog[tp][g], y], color=score_color, lw=0.6, zorder=6)
        axes.text(x_text, y, rf"$\mathit{{{g}}}$", va="center", ha=ha, fontsize=9.5, fontweight="bold", color=score_color, zorder=7)


label_scores("LCA5_P21", 0.16, 0.55, 0.62, "left")
label_scores("LCA5_P30", 1.74, 1.35, 1.28, "right")

axes.set_xticks(list(x_positions.values()))
axes.set_xticklabels([paper_style.condition_label(t) for t in x_positions], fontsize=16, fontweight="bold")
for tick, tp in zip(axes.get_xticklabels(), x_positions):
    tick.set_color(paper_style.CONDITION_COLORS[tp])
axes.set_xlim(-1.6, 3.6)
axes.set_ylabel("Focal region enrichment (in vs out)", fontsize=16, fontweight="bold")
axes.tick_params(axis="y", labelsize=13)
axes.legend(handles=[Line2D([0], [0], marker="o", ls="", mfc=score_color, mec="white", ms=8, label="score genes (leave one out)"),
                     Line2D([0], [0], marker="o", ls="", mfc=red, mec="none", ms=8, label="replicate concordant"),
                     Line2D([0], [0], marker="o", ls="", mfc="#9a9a9a", mec="none", ms=5, label="all rod genes")],
            fontsize=9, frameon=False, loc="upper right")
for spine in ("top", "right"):
    axes.spines[spine].set_visible(False)
figure.tight_layout()
figure.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(figure)
print("wrote", out_png)
