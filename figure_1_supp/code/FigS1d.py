# Fig S1d opsin transcript maps for superior and inferior calibration

## Load packages
import sys
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

## Load shared imaging functions
sys.path.insert(0, "../../_shared/code")
from _retina_panel import load_panel, project_xy_to_panel, PX_DS_UM, _orient_to_panel as orient_to_panel
from _paths import findpath_exclude_final, ROI_TO_SAMPLE
from _superior_up_orientation import install_s1c_orientations
import paper_style

## Load files
calibration_tx = "../../_shared/data_raw/calibration_tx"
out_png = "../panels/FigS1d.png"

## Define conditions and colors
display_reps = [
    ("WT_P21",   "WT_P21_rep2"),
    ("WT_P64",   "WT_P64_rep1"),
    ("LCA5_P21", "LCA5_P21_rep2"),
    ("LCA5_P30", "LCA5_P30_rep2"),
    ("LCA5_P64", "LCA5_P64_rep1"),
]
opsin = {"Opn1mw": ("#1b9e3b", "Opn1mw"), "Opn1sw": ("#c800a8", "Opn1sw")}
dot_size = 0.7
muted = "#2A2A2A"
sb_margin_frac = 0.13
infl = 0.06
gene2 = "Aldh1a1"
gene2_color = "#0F9D9A"

## Define drawing functions
def render_dapi18s(axes, dapi, s18, excl_panel=None, valid=None):
    # Draw the DAPI and 18S rRNA grayscale underlay on white
    if valid is None:
        valid = np.ones_like(dapi, dtype=bool)
    d = dapi.astype(np.float32)
    s = s18.astype(np.float32)
    dp = d[(d > 0) & valid]
    dlo, dhi = np.percentile(dp, (2, 98)) if dp.size else (0.0, 1.0)
    dn = np.clip((d - dlo) / max(dhi - dlo, 1e-6), 0, 1) ** 0.55
    sp = s[(s > 0) & valid]
    slo, shi = np.percentile(sp, (2, 98)) if sp.size else (0.0, 1.0)
    sn = np.clip((s - slo) / max(shi - slo, 1e-6), 0, 1) ** 0.7
    light = 1.0 - (dn * 0.15 + sn * 0.45)
    light = np.clip(light, 0, 1)
    light[~valid] = 1.0
    if excl_panel is not None:
        light[excl_panel] = 1.0
    axes.imshow(light, cmap="gray", vmin=0, vmax=1, zorder=0)


def render_map(axes, sample, panel, excl_panel, mode, bbox):
    # Draw one retina map with faint DAPI and transcript dots
    dapi = panel["dapi"]
    H, W = dapi.shape
    render_dapi18s(axes, dapi, panel["s18"], excl_panel=excl_panel)
    df = pd.read_parquet(f"{calibration_tx}/{sample}_tx.parquet")
    if mode == "opsin":
        tx = df[df["name"].isin(opsin)].copy()
    else:
        tx = df[df["name"] == gene2].copy()
    xr, yr = project_xy_to_panel(panel, tx["x"].to_numpy(), tx["y"].to_numpy())
    tx = tx.assign(xr=xr, yr=yr)
    # Drop transcripts inside the excluded polygon
    xr_a, yr_a = tx["xr"].to_numpy(), tx["yr"].to_numpy()
    in_bounds = (xr_a >= 0) & (xr_a < W) & (yr_a >= 0) & (yr_a < H)
    xi = np.clip(np.round(xr_a).astype(int), 0, W - 1)
    yi = np.clip(np.round(yr_a).astype(int), 0, H - 1)
    in_exclude = excl_panel[yi, xi] if excl_panel is not None else np.zeros(len(tx), bool)
    tx = tx[in_bounds & ~in_exclude]
    print(f"  {sample} [{mode}]: dots={len(tx)}")
    if mode == "opsin":
        for g, (color, _lab) in opsin.items():
            sub = tx[tx["name"] == g]
            axes.scatter(sub["xr"], sub["yr"], s=dot_size, c=color, edgecolor="none", alpha=0.85, zorder=3, rasterized=True)
    else:
        axes.scatter(tx["xr"], tx["yr"], s=dot_size, c=gene2_color, edgecolor="none", alpha=0.85, zorder=3, rasterized=True)
    x0_t, x1_t, y0_t, y1_t = bbox[sample]
    Wbb, Hbb = x1_t - x0_t, y1_t - y0_t
    margin = sb_margin_frac * Hbb
    x0d, x1d = x0_t - infl * Wbb, x1_t + infl * Wbb
    y0d, ylo = y0_t - infl * Hbb, y1_t + infl * Hbb + margin
    axes.set_xlim(x0d, x1d)
    axes.set_ylim(ylo, y0d)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)
    return x0d, x1d, y0d, ylo, y1_t


def add_scalebar(axes, ext):
    # Draw a 500 um scale bar in the blank band below the tissue
    x0d, x1d, y0d, ylo, y1_t = ext
    bar_px = 500.0 / PX_DS_UM
    x1b = x1d - 0.03 * (x1d - x0d)
    x0b = x1b - bar_px
    y_b = y1_t + 0.60 * (ylo - y1_t)
    halo = [pe.withStroke(linewidth=2.2, foreground="white")]
    axes.plot([x0b, x1b], [y_b, y_b], color="black", lw=2.0, solid_capstyle="butt", zorder=6, path_effects=halo)


def si_arrows(axes):
    # Draw superior and inferior arrows to the left of the panel
    for txt, fy, fs in [("S", 0.94, 12), ("↑", 0.86, 15), ("↓", 0.22, 15), ("I", 0.16, 12)]:
        axes.text(-0.045, fy, txt, transform=axes.transAxes, ha="center", va="center", fontsize=fs, color=muted, fontweight="bold", clip_on=False)


## Set the paper style and panel orientation
paper_style.set_style()
install_s1c_orientations()

## Load panels and their exclude masks
loaded = {}
tissue = {}
excl = {}
for cond, sample in display_reps:
    panel = load_panel(sample)
    loaded[sample] = panel
    tissue[sample] = panel["tissue"]
    _ep = findpath_exclude_final(ROI_TO_SAMPLE[panel['roi']])
    raw = tifffile.imread(_ep) > 0 if _ep.exists() else np.zeros((panel['orig_H'], panel['orig_W']), dtype=bool)
    excl[sample] = orient_to_panel(raw.astype(np.uint8), panel, order=0).astype(bool)

## Compute the full tissue bounding box and column widths
aspects = []
bbox = {}
for cond, sample in display_reps:
    ys, xs = np.where(tissue[sample])
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    bbox[sample] = (x0, x1, y0, y1)
    Wbb, Hbb = x1 - x0, y1 - y0
    yr = Hbb * (1 + 2 * infl) + sb_margin_frac * Hbb
    aspects.append(Wbb * (1 + 2 * infl) / yr)
wr = aspects

## Set up the figure with two rows
top, bottom, hspace = 0.94, 0.03, 0.18
row_h = 3.5
fig_h = (2 * row_h) / (top - bottom)
fig_w = row_h * sum(wr) / 0.965
figure = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
grid = gridspec.GridSpec(2, len(display_reps), figure=figure, width_ratios=wr, height_ratios=[1, 1],
                       left=0.03, right=0.995, top=top, bottom=bottom, wspace=0.03, hspace=hspace)

## Draw each column with opsin on top and Aldh1a1 below
for j, (cond, sample) in enumerate(display_reps):
    panel = loaded[sample]
    excl_panel = excl[sample]

    # Top row opsin map
    ax0 = figure.add_subplot(grid[0, j])
    ext0 = render_map(ax0, sample, panel, excl_panel, "opsin", bbox)
    ax0.set_title(paper_style.condition_label(cond), fontsize=12, fontweight="bold", color=paper_style.condition_color_simple(cond), pad=6)
    add_scalebar(ax0, ext0)
    if j == 0:
        si_arrows(ax0)
        # Draw the opsin dot key
        halo = [pe.withStroke(linewidth=2.0, foreground="white")]
        for i, g in enumerate(["Opn1mw", "Opn1sw"]):
            color, label = opsin[g]
            y = 0.965 - 0.085 * i
            ax0.scatter([0.045], [y], s=26, c=color, transform=ax0.transAxes, edgecolor="none", clip_on=False, zorder=6)
            ax0.text(0.085, y, label, transform=ax0.transAxes, ha="left", va="center", fontsize=9, fontstyle="italic", fontweight="bold", color=color, path_effects=halo, zorder=6)

    # Bottom row Aldh1a1 map
    ax1 = figure.add_subplot(grid[1, j])
    ext1 = render_map(ax1, sample, panel, excl_panel, "aldh1a1", bbox)
    add_scalebar(ax1, ext1)
    if j == 0:
        si_arrows(ax1)
        # Draw the single gene dot key
        halo = [pe.withStroke(linewidth=2.0, foreground="white")]
        ax1.scatter([0.045], [0.965], s=26, c=gene2_color, transform=ax1.transAxes, edgecolor="none", clip_on=False, zorder=6)
        ax1.text(0.085, 0.965, gene2, transform=ax1.transAxes, ha="left", va="center", fontsize=9, fontstyle="italic", fontweight="bold", color=gene2_color, path_effects=halo, zorder=6)

## Add the DAPI and 18S rRNA square key between the rows
handles = [
    Line2D([0], [0], marker="s", color="none", markerfacecolor="#b0b0b0", markeredgecolor="none", markersize=9, label="DAPI"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor="#606060", markeredgecolor="none", markersize=9, label="18S rRNA"),
]
legend = figure.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, bottom + (top - bottom) * 0.5), ncol=2, frameon=False, fontsize=11, handletextpad=0.4, columnspacing=1.8)
for t in legend.get_texts():
    t.set_fontweight("bold")

## Save the panel
figure.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(figure)
print("wrote", out_png)
