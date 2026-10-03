# Fig S1a segmentation and U-Net layer QC

## Load packages
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
import tifffile
from skimage.filters import threshold_otsu

## Load shared imaging functions
sys.path.insert(0, "../../_shared/code")
import paper_style
from _retina_panel import (load_panel, PX_DS_UM, SAMPLE_TO_ROI as sample_to_roi_all,
                           _orient_to_panel as orient_to_panel,
                           load_cellbody_mask_oriented,
                           zoom_straighten_geom, zoom_straighten_apply)
from _paths import SEGMENTATION_EXPORT_ROOT, findpath_dapi, findpath_unet_pred, findpath_exclude_final
from _superior_up_orientation import install_s1c_orientations

## Set fonts to match the other panels
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial:italic",
    "mathtext.default": "it",
    "savefig.dpi": 220,
    "figure.dpi": 130,
})

## Load files
out_crops = "../panels/FigS1a_1.png"
out_bars = "../panels/FigS1a_2.png"
out_table = Path("../data_plot/FigS1a_coverage.parquet")
export_root = SEGMENTATION_EXPORT_ROOT

## Define samples layers and zoom regions
condition_order = ["WT_P21", "WT_P64", "LCA5_P21", "LCA5_P30", "LCA5_P64"]
# Define the 5 representative reps and their zoom centers
display_reps = [
    ("WT_P21",   "WT_P21_rep2"),
    ("WT_P64",   "WT_P64_rep1"),
    ("LCA5_P21", "LCA5_P21_rep2"),
    ("LCA5_P30", "LCA5_P30_rep2"),
    ("LCA5_P64", "LCA5_P64_rep1"),
]
sample_zoom_center_frac = {
    "WT_P21_rep2":   (0.50, 0.14),
    "WT_P64_rep1":   (0.45, 0.14),
    "LCA5_P21_rep2": (0.18, 0.25),
    "LCA5_P30_rep2": (0.39, 0.105),
    "LCA5_P64_rep1": (0.23, 0.15),
}
zoom_h_um = 300.0        # Retinal depth window
zoom_aspect = 1.35       # Landscape crop width
onl_top_frac = 0.25      # ONL apical edge at 25% of crop height
layer_ids = {"ONL": 1, "INL": 2, "IPL": 3, "GCL": 4}
layer_order = ["ONL", "INL", "IPL", "GCL"]
layer_colors = {"ONL": "#ff6a00", "INL": "#00b312", "IPL": "#1560ff", "GCL": "#b500d6"}
layer_alpha = 0.55


## Define imaging functions
# Normalize DAPI or 18S to 0..1 by the 2nd/98th percentile of nonzero pixels
def normalize_channel(img, gamma):
    arr = img.astype(np.float32)
    pos = arr[arr > 0]
    if pos.size == 0:
        return np.zeros_like(arr)
    p_lo, p_hi = np.percentile(pos, [2, 98])
    if p_hi <= p_lo:
        return np.zeros_like(arr)
    return np.clip((arr - p_lo) / (p_hi - p_lo), 0, 1) ** gamma


# Merge DAPI and 18S with gray 18S and blue DAPI nuclei on top
def composite_rgb(dapi_raw, s18_raw):
    dapi = normalize_channel(dapi_raw, 0.55)
    s18 = normalize_channel(s18_raw, 0.7)
    rgb = np.zeros((*dapi.shape, 3), dtype=np.float32)
    rgb[..., 0] = s18
    rgb[..., 1] = s18
    rgb[..., 2] = s18 + dapi * 0.9
    return np.clip(rgb, 0, 1)


def condition_of(sample):
    return sample.rsplit("_rep", 1)[0]


# Compute fraction of DAPI positive area captured by a cellpose body mask
def dapi_assigned_fraction(sample, roi):
    dapi = tifffile.imread(findpath_dapi(sample))
    mask = tifffile.imread(export_root / sample / "cellbody_masks.tif")
    if dapi.shape != mask.shape:
        raise ValueError(f"{sample}: DAPI {dapi.shape} != mask {mask.shape}")
    mask_pos = mask > 0
    del mask

    pos_vals = dapi[dapi > 0]
    if pos_vals.size < 1000:
        raise ValueError(f"{sample}: <1000 positive DAPI pixels; image suspect")
    thr = float(threshold_otsu(pos_vals))
    dapi_pos = dapi > thr
    denom = int(dapi_pos.sum())
    mask_px = int(mask_pos.sum())
    assigned = int((dapi_pos & mask_pos).sum())
    if denom == 0 or mask_px == 0:
        raise ValueError(f"{sample}: zero DAPI-positive or zero mask pixels")
    frac = assigned / denom
    print(f"  {sample:14s} roi{roi:<2d} otsu={thr:8.1f} "
          f"recall={frac:.3f} (denom={denom:,})", flush=True)
    return dict(sample=sample, condition=condition_of(sample), roi=roi,
                otsu_thr=thr, dapi_pos_px=denom, mask_px=mask_px,
                assigned_px=assigned, frac_assigned=frac)


# Compute DAPI coverage within cell masks for all 10 retinas
def compute_coverage():
    print("computing DAPI-in-mask coverage for all 10 reps ...", flush=True)
    rows = [dapi_assigned_fraction(sample, roi) for sample, roi in sample_to_roi_all.items()]
    data = pd.DataFrame(rows)
    data.to_parquet(out_table, index=False)
    print(f"wrote {out_table}", flush=True)
    return data


# Draw per condition bars with min max whiskers and per rep dots
def render_coverage_panel(axes, coverage, col="frac_assigned", ylabel="DAPI area in cell mask", xlabels=True):
    x_positions = np.arange(len(condition_order))
    means, lows, highs = [], [], []
    for condition in condition_order:
        values = coverage.loc[coverage["condition"] == condition, col].to_numpy()
        means.append(values.mean())
        lows.append(values.min())
        highs.append(values.max())
    means = np.array(means)
    lows = np.array(lows)
    highs = np.array(highs)

    colors = [paper_style.condition_color_simple(c) for c in condition_order]
    axes.bar(x_positions, means, width=0.66, color=colors, alpha=0.30, edgecolor=colors, linewidth=1.4)
    axes.vlines(x_positions, lows, highs, colors=colors, linewidth=1.6)
    for index, condition in enumerate(condition_order):
        values = coverage.loc[coverage["condition"] == condition, col].to_numpy()
        axes.scatter(np.full(values.shape, index), values, s=26,
                     color=paper_style.condition_color_simple(condition), zorder=3,
                     edgecolor="white", linewidth=0.6)

    axes.set_xticks(x_positions)
    if xlabels:
        axes.set_xticklabels([paper_style.condition_label(c) for c in condition_order], rotation=40,
                             ha="right", fontsize=13, fontweight="bold")
        for tick_label, condition in zip(axes.get_xticklabels(), condition_order):
            tick_label.set_color(paper_style.condition_color_simple(condition))
    else:
        axes.set_xticklabels([])
    axes.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    axes.set_ylim(0, 1)
    axes.set_yticks(np.arange(0, 1.01, 0.2))
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.tick_params(labelsize=12)
    for tick_label in axes.get_yticklabels():
        tick_label.set_fontweight("bold")


# Load the U-Net layer prediction oriented to the panel frame
def load_layer6_oriented(panel):
    from _paths import ROI_TO_SAMPLE
    raw = tifffile.imread(findpath_unet_pred(ROI_TO_SAMPLE[panel['roi']], "layer"))
    return orient_to_panel(raw, panel, order=0)


def load_exclude_final_oriented(panel):
    from _paths import ROI_TO_SAMPLE
    p = findpath_exclude_final(ROI_TO_SAMPLE[panel['roi']])
    raw = tifffile.imread(p) > 0 if p.exists() else np.zeros((panel['orig_H'], panel['orig_W']), dtype=bool)
    return orient_to_panel(raw.astype(np.uint8), panel, order=0).astype(bool)


# Straighten the ONL band apical up then crop a rectangle
def apical_up_crop(dapi, s18, body, layer6, final_h, final_w):
    geom = zoom_straighten_geom(layer6, final_h, final_w,
                                onl_code=1, gcl_code=4, onl_top_frac=onl_top_frac)
    d = zoom_straighten_apply(dapi.astype(np.float32), geom, order=1, cval=0.0)
    s = zoom_straighten_apply(s18.astype(np.float32), geom, order=1, cval=0.0)
    b = zoom_straighten_apply(body, geom, order=0, cval=0)
    lay = zoom_straighten_apply(layer6.astype(np.int16), geom, order=0, cval=0)
    return d, s, b, lay


# Crop the Fig1c zoom region apical up
def crop_zoom(sample):
    panel = load_panel(sample)
    body = load_cellbody_mask_oriented(sample, panel)
    layer6 = load_layer6_oriented(panel)
    dapi = panel["dapi"].astype(np.float32)
    s18 = panel["s18"].astype(np.float32)
    H, W = dapi.shape
    fx, fy = sample_zoom_center_frac[sample]
    cx, cy = fx * W, fy * H
    half_h = (zoom_h_um / PX_DS_UM) / 2.0
    half_w = half_h * zoom_aspect
    pre_half = 0.5 * float(np.hypot(2 * half_w, 2 * half_h)) + 6
    xp0, yp0 = int(max(0, cx - pre_half)), int(max(0, cy - pre_half))
    xp1, yp1 = int(min(W, cx + pre_half)), int(min(H, cy + pre_half))

    # Flag any excluded overlap in the crop region
    excl = load_exclude_final_oriented(panel)
    n_excl = int(excl[yp0:yp1, xp0:xp1].sum())
    if n_excl:
        print(f"  [warn] {sample}: {n_excl} exclude_final px inside the crop region", flush=True)

    final_h = int(round(zoom_h_um / PX_DS_UM))
    final_w = int(round(half_w * 2))
    d, s, b, lay = apical_up_crop(
        dapi[yp0:yp1, xp0:xp1], s18[yp0:yp1, xp0:xp1],
        body[yp0:yp1, xp0:xp1], layer6[yp0:yp1, xp0:xp1], final_h, final_w)
    print(f"  {sample}: crop {b.shape[1]}x{b.shape[0]} px "
          f"({b.shape[1] * PX_DS_UM:.0f}x{b.shape[0] * PX_DS_UM:.0f} um), "
          f"{int((b > 0).any() and len(np.unique(b)) - 1)} cell labels", flush=True)
    return dict(sample=sample, dapi=d, s18=s, body=b, layer=lay)


# Draw channel key text for the composite
def channel_key(axes):
    halo = [pe.withStroke(linewidth=2.0, foreground="black")]
    axes.text(0.035, 0.965, "DAPI", transform=axes.transAxes, ha="left", va="top",
              fontsize=13, fontweight="bold", color="#4da6ff", path_effects=halo, zorder=6)
    axes.text(0.035, 0.875, "18S rRNA", transform=axes.transAxes, ha="left", va="top",
              fontsize=13, fontweight="bold", color="#d0d0d0", path_effects=halo, zorder=6)


# Draw 1 px per label boundaries with no dilation
def thin_outlines(mask):
    m = mask
    edge = np.zeros(m.shape, dtype=bool)
    edge[:, :-1] |= (m[:, :-1] != m[:, 1:]) & ((m[:, :-1] > 0) | (m[:, 1:] > 0))
    edge[:-1, :] |= (m[:-1, :] != m[1:, :]) & ((m[:-1, :] > 0) | (m[1:, :] > 0))
    return edge


# Draw a scale bar sized for the ds8 crop
def add_scalebar(axes, length_um=50.0, color="white"):
    H, W = axes.images[0].get_array().shape[:2]
    bar_frac = (length_um / PX_DS_UM) / W
    x0 = 0.95 - bar_frac
    y_bar = 0.09
    halo = [pe.withStroke(linewidth=3.0, foreground="black")]
    axes.plot([x0, x0 + bar_frac], [y_bar, y_bar], transform=axes.transAxes,
              color=color, lw=2.2, solid_capstyle="butt", clip_on=False, path_effects=halo, zorder=7)


# Draw the composite with thin white per cell mask outlines
def render_masks_row(panel_axes, crops):
    for axes, crop in zip(panel_axes, crops):
        rgb = composite_rgb(crop["dapi"], crop["s18"])
        axes.imshow(rgb, interpolation="bilinear")
        edges = thin_outlines(crop["body"])
        ov = np.zeros((*edges.shape, 4), dtype=np.float32)
        ov[edges] = (1.0, 1.0, 1.0, 0.9)
        axes.imshow(ov, interpolation="nearest")
        axes.set_xticks([])
        axes.set_yticks([])
        for spine in axes.spines.values():
            spine.set_visible(False)
        axes.set_aspect("equal")


# Draw the same crops with the U-Net layer call painted on top
def render_layer_row(panel_axes, crops):
    for axes, crop in zip(panel_axes, crops):
        rgb = composite_rgb(crop["dapi"], crop["s18"])
        axes.imshow(rgb, interpolation="bilinear")
        layer = crop["layer"]
        ov = np.zeros((*layer.shape, 4), dtype=np.float32)
        for name in layer_order:
            rgba = mpl.colors.to_rgba(layer_colors[name])
            m = layer == layer_ids[name]
            ov[m, 0], ov[m, 1], ov[m, 2] = rgba[0], rgba[1], rgba[2]
            ov[m, 3] = layer_alpha
        axes.imshow(ov, interpolation="nearest")
        axes.set_xticks([])
        axes.set_yticks([])
        for spine in axes.spines.values():
            spine.set_visible(False)
        axes.set_aspect("equal")


## Compute DAPI coverage
coverage = compute_coverage()

## Install the Fig1c orientations and build the 5 zoom crops
install_s1c_orientations()
print("building 5 representative zoom crops (Fig1c regions) ...", flush=True)
crops = [crop_zoom(sample) for _cond, sample in display_reps]

## Draw tile 1 crop grid of masks and U-Net layers
figure_width, left, right, top, bottom, width_space, height_space = 15.0, 0.055, 0.99, 0.945, 0.03, 0.035, 0.03
cell_width_in = figure_width * (right - left) / (5 + 4 * width_space)
cell_height_in = cell_width_in / zoom_aspect
figure_height = cell_height_in * (2 + height_space) / (top - bottom)
figure = plt.figure(figsize=(figure_width, figure_height), facecolor="white")
grid = figure.add_gridspec(2, 5, wspace=width_space, hspace=height_space, left=left, right=right, top=top, bottom=bottom)
mask_axes = [figure.add_subplot(grid[0, col_index]) for col_index in range(5)]
layer_axes = [figure.add_subplot(grid[1, col_index]) for col_index in range(5)]

render_masks_row(mask_axes, crops)
render_layer_row(layer_axes, crops)
for axes in mask_axes + layer_axes:
    axes.set_anchor("N")
for col_index, (condition, _sample) in enumerate(display_reps):
    mask_axes[col_index].set_title(paper_style.condition_label(condition), fontsize=17, fontweight="bold",
                                   color=paper_style.condition_color_simple(condition), pad=6)
for label, axes in [("Cellpose masks", mask_axes[0]), ("U-Net layer", layer_axes[0])]:
    axes.text(-0.055, 0.5, label, transform=axes.transAxes, rotation=90, va="center", ha="right",
              fontsize=13, fontweight="bold", color="#222222")
channel_key(mask_axes[0])
for axes in mask_axes + layer_axes:
    add_scalebar(axes, length_um=50, color="white")
legend_handles = [Patch(facecolor=layer_colors[name], edgecolor="none",
                        alpha=min(1.0, layer_alpha + 0.30), label=name) for name in layer_order]
legend = layer_axes[0].legend(handles=legend_handles, loc="upper left", fontsize=8.5, frameon=True,
                              handlelength=1.1, handletextpad=0.4, labelspacing=0.3,
                              borderpad=0.35, borderaxespad=0.4)
legend.get_frame().set_facecolor("white")
legend.get_frame().set_alpha(0.78)
legend.get_frame().set_edgecolor("none")
for text in legend.get_texts():
    text.set_fontweight("bold")
figure.savefig(out_crops, dpi=400, bbox_inches="tight")
plt.close(figure)
print(f"wrote {out_crops}")

## Draw tile 2 segmentation QC bars
bar_figure, coverage_axes = plt.subplots(figsize=(5.0, 4.1), facecolor="white")
render_coverage_panel(coverage_axes, coverage, col="frac_assigned", ylabel="DAPI area in cell mask")
bar_figure.subplots_adjust(left=0.19, right=0.96, top=0.97, bottom=0.26)
bar_figure.savefig(out_bars, dpi=300, bbox_inches="tight")
plt.close(bar_figure)
print(f"wrote {out_bars}")

## Print coverage summary
for condition in condition_order:
    values = coverage.loc[coverage["condition"] == condition, "frac_assigned"].to_numpy()
    print(f"{condition}: n_reps={len(values)} median_frac={np.median(values):.3f} "
          f"range=[{values.min():.3f}, {values.max():.3f}]")
