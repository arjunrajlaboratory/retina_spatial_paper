# Fig S5a rod injury response spatial maps and 1D arc heatmaps across all five conditions

## Load packages
import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
from matplotlib.colors import Normalize

## Load paper style conventions and shared spatial functions
sys.path.insert(0, ".")
sys.path.insert(0, "../../_shared/code")
import paper_style
import _focal_regions as focal
import _retina_panel as paper_figure_functions
from _retina_panel import load_panel, PX_DS_UM
from _retina_panel import load_unet_tissue_oriented, load_exclude_final_oriented, load_cellbody_mask_oriented
warnings.filterwarnings("ignore", category=FutureWarning)

import _opsin_common as opsin_common

## Load files
cells_path = "../../_shared/data_processed/per_cell_gene_counts.parquet"
SCORE_PARQ = "../../figure_4/data_raw_manifest/rod_injury_response_score.parquet"
panel_png = "../panels/FigS5a.png"
data_plot_dir = "../data_plot"

## Point the maps at the five conditions and their representative replicates
CONDITIONS = ["WT_P21", "WT_P64", "LCA5_P21", "LCA5_P30", "LCA5_P64"]
DISPLAY_REP = {"WT_P21": "WT_P21_rep2", "WT_P64": "WT_P64_rep1",
                 "LCA5_P21": "LCA5_P21_rep1", "LCA5_P30": "LCA5_P30_rep1",
                 "LCA5_P64": "LCA5_P64_rep1"}
LOW_N_CONDITIONS = {"LCA5_P64"}   # Flag the low rod count condition

## Define map parameters
SCORE_CMAP = mpl.colormaps["viridis"].copy()
SCORE_CMAP.set_bad(SCORE_CMAP(0.0))


## Define drawing functions

# Set the bold display font for the composites
def set_style():
    mpl.rcParams.update({
        "font.family": "Helvetica Neue", "font.weight": "bold",
        "axes.labelweight": "bold", "axes.titleweight": "bold",
        "savefig.dpi": 200, "figure.dpi": 120,
    })


# Read the S1C dorsal up orientation for a sample from its cone opsin gradient
def s1c_orient_override(sample):
    o = opsin_common.fit_gradient(opsin_common.load_cones(sample))
    return {"rotation_deg": float(-np.degrees(o["rotation_rad"])),
            "flip_x": bool(o["flip_x"]), "flip_y": bool(o["flip_y"])}


# Install the S1C orientations so panels render dorsal up optic nerve left
def install_s1c_orientations():
    for sample in DISPLAY_REP.values():
        paper_figure_functions.SAMPLE_ORIENT_OVERRIDE[sample] = s1c_orient_override(sample)


# Load the per rod injury response scores for the displayed replicates
def load_scores():
    d = pd.read_parquet(SCORE_PARQ)
    return d[d["sample"].isin(list(DISPLAY_REP.values()))].copy()


# Grayscale DAPI and 18S base with rod fills colored by rod injury response score
def composite_rods(panel, body_mask, rod_scores, vmin, vmax):
    dapi = panel["dapi"].astype(np.float32)
    s18 = panel["s18"].astype(np.float32)
    d_pos = dapi[dapi > 0]
    d_lo, d_hi = np.percentile(d_pos, (2, 98)) if d_pos.size else (0, 1)
    d_norm = np.clip((dapi - d_lo) / max(d_hi - d_lo, 1e-6), 0, 1) ** 0.55
    s_pos = s18[s18 > 0]
    s_lo, s_hi = np.percentile(s_pos, (2, 98)) if s_pos.size else (0, 1)
    s_norm = np.clip((s18 - s_lo) / max(s_hi - s_lo, 1e-6), 0, 1) ** 0.7
    # Make DAPI light and 18S darker
    inv = np.clip(1.0 - (d_norm * 0.15 + s_norm * 0.45), 0, 1)
    base = np.stack([inv] * 3, axis=-1)

    H, W = body_mask.shape

    def _fit(m):
        mm = np.zeros((H, W), dtype=bool)
        Hm, Wm = m.shape
        mm[:min(H, Hm), :min(W, Wm)] = m[:H, :W]
        return mm

    # Mask excluded pixels
    not_excl = np.ones((H, W), dtype=bool)
    excl = panel.get("excl_mask")
    if excl is not None:
        not_excl &= ~_fit(excl)
    base = np.where(not_excl[..., None], base, 1.0)

    max_label = int(body_mask.max())
    lut = np.zeros((max_label + 1, 4), dtype=np.float32)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    for lab, val in rod_scores.items():
        if 0 < lab <= max_label:
            lut[lab] = SCORE_CMAP(norm(float(val)))
    color = lut[body_mask]
    has_cell = (color[..., 3] > 0) & not_excl
    out = base.copy()
    out[has_cell] = color[has_cell, :3]
    return out


## Set the style and load the injury response scores
set_style()
paper_style.set_style()
install_s1c_orientations()
scores = load_scores()

## Set the map color scale from the 2nd to 98th percentile of the displayed scores
display_scores = scores[scores["sample"].isin([DISPLAY_REP[c] for c in CONDITIONS])]
vmin, vmax = np.percentile(display_scores["injury_response_score"].to_numpy(), [2, 98])
vmin, vmax = float(vmin), float(vmax)
print(f"map vmin/vmax = {vmin:.3f}/{vmax:.3f}")

## Build the oriented composites and arc profiles per condition
composites, widths_um, heights_um, profiles = {}, {}, {}, {}
n_rods_display = {}
for cond in CONDITIONS:
    sample = DISPLAY_REP[cond]
    panel = load_panel(sample)
    panel["tissue"] = load_unet_tissue_oriented(panel)
    body_mask = load_cellbody_mask_oriented(sample, panel)
    panel["excl_mask"] = load_exclude_final_oriented(panel)
    sample_scores = scores[scores["sample"] == sample]
    n_rods_display[cond] = len(sample_scores)
    rod_scores = dict(zip(sample_scores["label"].astype(int), sample_scores["injury_response_score"]))
    composites[cond] = composite_rods(panel, body_mask, rod_scores, vmin, vmax)
    widths_um[cond] = composites[cond].shape[1] * PX_DS_UM
    heights_um[cond] = composites[cond].shape[0] * PX_DS_UM
    profiles[cond] = focal.arc_strip(sample)

## Set the arc strip color scale
all_means = np.concatenate([means[~np.isnan(means)] for _, means in profiles.values()])
strip_vmin, strip_vmax = np.percentile(all_means, [2, 98])
print(f"strip vmin/vmax = {strip_vmin:.3f}/{strip_vmax:.3f}")

## Set up the figure layout
H_OV = 4.6
STRIP_H = 0.21
SG = 0.14
SB_BAND = 0.36
GG = 0.55
LM, RM, TM, BM = 0.42, 1.05, 0.62, 0.55
SCALE_BAR_UM = 500.0
overview_widths = {c: H_OV * widths_um[c] / heights_um[c] for c in CONDITIONS}
STRIP_W = min(overview_widths.values())
strip_cmap = SCORE_CMAP.copy()
strip_cmap.set_bad("white")
figure_width = LM + sum(overview_widths.values()) + GG * (len(CONDITIONS) - 1) + RM
figure_height = TM + H_OV + SB_BAND + SG + STRIP_H + BM
figure = plt.figure(figsize=(figure_width, figure_height))

def ax_at(x, y, w, h):
    return figure.add_axes([x / figure_width, y / figure_height, w / figure_width, h / figure_height])

top_in = figure_height - TM
strip_bottom = BM
map_bottom = BM + STRIP_H + SG + SB_BAND

## Draw each condition map with its arc strip
cursor_x = LM
for cond in CONDITIONS:
    image = composites[cond]
    image_width = image.shape[1]
    axes = ax_at(cursor_x, map_bottom, overview_widths[cond], H_OV)
    axes.imshow(image, interpolation="nearest", aspect="equal")
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)
    figure.text((cursor_x + overview_widths[cond] / 2) / figure_width, (top_in + 0.10) / figure_height,
                paper_style.condition_label(cond), ha="center", va="bottom", fontsize=20,
                fontweight="bold", color=paper_style.CONDITION_COLORS[cond])
    # Flag the low rod count condition in a subtitle
    if cond in LOW_N_CONDITIONS:
        figure.text((cursor_x + overview_widths[cond] / 2) / figure_width, (top_in + 0.02) / figure_height,
                    f"few surviving rods (n≈{n_rods_display[cond]:d})", ha="center", va="top",
                    fontsize=11, style="italic", color="#8E2430")
    # Draw one scale bar per map in the blank band below the tissue
    px_to_inch = overview_widths[cond] / image_width
    bar_length_in = (SCALE_BAR_UM / PX_DS_UM) * px_to_inch
    x_right_in = cursor_x + overview_widths[cond] - 0.02
    x_left_in = x_right_in - bar_length_in
    bar_y_in = map_bottom - SB_BAND * 0.55
    figure.add_artist(mlines.Line2D([x_left_in / figure_width, x_right_in / figure_width], [bar_y_in / figure_height, bar_y_in / figure_height],
                      color="black", lw=3.2, solid_capstyle="butt", zorder=20, transform=figure.transFigure))
    # Draw the horizontal arc strip beneath with S left and I right
    edges, means = profiles[cond]
    strip_x = cursor_x + (overview_widths[cond] - STRIP_W) / 2
    strip_axes = ax_at(strip_x, strip_bottom, STRIP_W, STRIP_H)
    strip_axes.imshow(np.ma.masked_invalid(means[::-1])[np.newaxis, :], aspect="auto", interpolation="nearest",
                      cmap=strip_cmap, vmin=strip_vmin, vmax=strip_vmax, extent=[0, 1, 0, 1])
    strip_axes.set_yticks([])
    strip_axes.set_xticks([0, 1])
    strip_axes.set_xticklabels(["S", "I"], fontsize=15, fontweight="bold")
    for tick_label, color in zip(strip_axes.get_xticklabels(), ("#1b9e3b", "#c800a8")):
        tick_label.set_color(color)
    strip_axes.tick_params(axis="x", length=0)
    for spine in strip_axes.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#444444")
        spine.set_linewidth(0.7)
    strip_axes.text(0.5, -0.7, "arc position", transform=strip_axes.transAxes, ha="center", va="top", fontsize=16)
    cursor_x += overview_widths[cond] + GG

## Add the shared score colorbar
colorbar_height = H_OV * 0.55
colorbar_axes = ax_at(figure_width - RM + 0.12, map_bottom + (H_OV - colorbar_height) / 2, 0.14, colorbar_height)
scalar_mappable = cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=SCORE_CMAP)
colorbar = plt.colorbar(scalar_mappable, cax=colorbar_axes)
colorbar.set_label("Rod injury response score", fontsize=11, fontweight="bold")
colorbar.ax.tick_params(labelsize=8)
colorbar.outline.set_linewidth(0.3)

## Add the DAPI and 18S channel key
channel_key = [mlines.Line2D([0], [0], marker="s", linestyle="none", markersize=11, markerfacecolor="#b0b0b0", markeredgecolor="none", label="DAPI"),
               mlines.Line2D([0], [0], marker="s", linestyle="none", markersize=11, markerfacecolor="#606060", markeredgecolor="none", label="18S rRNA")]
figure.legend(handles=channel_key, loc="upper left",
              bbox_to_anchor=((figure_width - RM + 0.02) / figure_width, (map_bottom + (H_OV - colorbar_height) / 2 - 0.12) / figure_height),
              frameon=False, fontsize=9, handletextpad=0.4, labelspacing=0.5, borderaxespad=0.0)

## Add the arc strip colorbar
strip_colorbar_axes = ax_at(figure_width - RM + 0.12, strip_bottom, 0.14, STRIP_H)
strip_scalar_mappable = cm.ScalarMappable(norm=Normalize(strip_vmin, strip_vmax), cmap=SCORE_CMAP)
strip_colorbar = plt.colorbar(strip_scalar_mappable, cax=strip_colorbar_axes, orientation="vertical")
strip_colorbar.set_label("Arc-mean", fontsize=14, fontweight="bold")
strip_colorbar.ax.tick_params(labelsize=11)
strip_colorbar.outline.set_linewidth(0.3)

## Add the superior and inferior arrows on the left margin
x_si = (LM - 0.055) / figure_width
map_height_in = top_in - map_bottom
figure.text(x_si, (map_bottom + map_height_in * 0.83) / figure_height, "↑", ha="center", va="center", fontsize=24, fontweight="bold", color="#1b9e3b")
figure.text(x_si, (map_bottom + map_height_in * 0.93) / figure_height, "S", ha="center", va="center", fontsize=16, fontweight="bold", color="#1b9e3b")
figure.text(x_si, (map_bottom + map_height_in * 0.17) / figure_height, "↓", ha="center", va="center", fontsize=24, fontweight="bold", color="#c800a8")
figure.text(x_si, (map_bottom + map_height_in * 0.07) / figure_height, "I", ha="center", va="center", fontsize=16, fontweight="bold", color="#c800a8")

## Save the panel
os.makedirs("../panels", exist_ok=True)
figure.savefig(panel_png, dpi=200, bbox_inches="tight")
plt.close(figure)
print("wrote", panel_png)

## Export the per bin arc mean table
os.makedirs(data_plot_dir, exist_ok=True)
rows = []
for cond in CONDITIONS:
    sample = DISPLAY_REP[cond]
    edges, means = profiles[cond]
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    rows.append(pd.DataFrame({"condition": cond, "sample": sample,
                              "arc_bin_center_um": bin_centers, "arc_mean_score": means}))
profile_df = pd.concat(rows, ignore_index=True)
profile_df.to_csv(f"{data_plot_dir}/FigS5a.csv", index=False)
print("wrote", f"{data_plot_dir}/FigS5a.csv")
