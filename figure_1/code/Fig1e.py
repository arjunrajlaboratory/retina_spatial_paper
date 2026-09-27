# Fig 1e WT P21 seqFISH tissue composite with cell type key inset

## Load packages
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy import ndimage as ndi

## Load paper style conventions and shared imaging functions
sys.path.insert(0, "../../_shared/code")
from _retina_panel import (load_panel, project_xy_to_panel, PX_DS_UM, load_cellbody_mask_oriented,
                           zoom_straighten_geom, zoom_straighten_apply)
from _paths import findpath_exclude_final, findpath_unet_pred, ROI_TO_SAMPLE
import paper_style

## Load the shared superior-up orientation (cone opsin gradient fit)
from _superior_up_orientation import install_s1c_orientations

## Load files
per_cell = "../../_shared/data_processed/per_cell_gene_counts.parquet"
opsin_tx_dir = "../../_shared/data_raw/opsin_tx"
marker_tx_dir = "../../_shared/data_raw/marker_tx"
out_png = "../panels/Fig1e.png"

## Define the sample and markers
sample = "WT_P21_rep2"
zoom_center_frac = (0.728, 0.93)
markers = [
    ("Nrl", "rod"), ("Arr3", "cone"), ("Prkca", "bipolar"),
    ("Gad1", "amacrine_gaba"), ("Glul", "muller"), ("Rbpms", "rgc"),
]
celltype_stack = ["rod", "cone", "bipolar", "amacrine_gaba", "amacrine_gly",
                  "horizontal", "rgc", "muller", "microglia", "vascular", "rpe"]
zoom_rot_delta_deg = -2.3
med_aspect = 1.45
onl_top_frac = 0.28
zoom_h_um = 300


## Define layout in inches
h_fig = 9.5
m_l = 0.45
m_r = m_t = m_b = 0.12
g_ab = 0.50
gap_b = 0.09
sb_band_frac = 0.11


## Reorient a mask into the panel frame (pad rotate flip crop)
def orient_to_panel(raster, panel, order=0):
    hd, wd = panel["orig_H"], panel["orig_W"]
    hr, wr = raster.shape
    pad = np.zeros((hd, wd), dtype=raster.dtype)
    pad[:min(hr, hd), :min(wr, wd)] = raster[:hd, :wd]
    angle = panel["angle"]
    rot = (ndi.rotate(pad, angle, reshape=True, order=order, mode="constant", cval=0)
           if abs(angle) > 0.1 else pad)
    if panel["flipped_v"]:
        rot = rot[::-1, :]
    if panel["flipped_h"]:
        rot = rot[:, ::-1]
    h, w = panel["H"], panel["W"]
    cx0, cy0 = panel["crop_x0"], panel["crop_y0"]
    rot = rot[cy0:cy0 + h, cx0:cx0 + w]
    out = np.zeros((h, w), dtype=raster.dtype)
    hf, wf = rot.shape
    out[:min(h, hf), :min(w, wf)] = rot[:h, :w]
    return out


def load_unet_tissue_oriented(panel):
    raw = tifffile.imread(findpath_unet_pred(ROI_TO_SAMPLE[panel['roi']], "layer")) > 0
    return orient_to_panel(raw.astype(np.uint8), panel, order=0).astype(bool)


## Load the oriented exclude mask
def load_exclude_final_oriented(panel):
    p = findpath_exclude_final(ROI_TO_SAMPLE[panel['roi']])
    raw = tifffile.imread(p) > 0 if p.exists() else np.zeros((panel['orig_H'], panel['orig_W']), dtype=bool)
    return orient_to_panel(raw.astype(np.uint8), panel, order=0).astype(bool)


## Compose the panel image on a white background (DAPI and 18S gray plus cell colors)
def build_composite(panel, body_mask, label_to_color):
    dapi = panel["dapi"].astype(np.float32)
    s18 = panel["s18"].astype(np.float32)

    # Normalize with a percentile clip and gamma
    d_pos = dapi[dapi > 0]
    d_lo, d_hi = np.percentile(d_pos, (2, 98)) if d_pos.size else (0, 1)
    d_norm = np.clip((dapi - d_lo) / max(d_hi - d_lo, 1e-6), 0, 1)
    d_norm = d_norm ** 0.55
    s_pos = s18[s18 > 0]
    s_lo, s_hi = np.percentile(s_pos, (2, 98)) if s_pos.size else (0, 1)
    s_norm = np.clip((s18 - s_lo) / max(s_hi - s_lo, 1e-6), 0, 1)
    s_norm = s_norm ** 0.7

    # Build white background grayscale with DAPI light and 18S darker
    inv = 1.0 - (d_norm * 0.18 + s_norm * 0.55)
    inv = np.clip(inv, 0, 1)
    base = np.stack([inv] * 3, axis=-1)

    # Mask excluded pixels in the background and cell colors
    excl = panel.get("excl_mask")
    h, w = body_mask.shape
    not_excl = np.ones((h, w), dtype=bool)
    if excl is not None:
        e = np.zeros((h, w), dtype=bool)
        he, we = excl.shape
        e[:min(h, he), :min(w, we)] = excl[:h, :w]
        not_excl &= ~e
    base = np.where(not_excl[..., None], base, 1.0)

    # Build the cell color image leaving background cells white
    max_label = int(body_mask.max())
    lut = np.zeros((max_label + 1, 3), dtype=np.float32)
    for lab, rgb in label_to_color.items():
        if 0 < lab <= max_label:
            lut[lab] = rgb
    color = lut[body_mask]
    typed_mask = np.any(color > 0, axis=-1) & not_excl
    out = np.where(typed_mask[..., None], color, base)
    return out


## Draw a micron accurate scale bar inside the axes at the bottom right
def add_scalebar(axes, length_um, label, pad_frac=0.04, thickness_frac=0.014, color="black"):
    x0, x1 = axes.get_xlim()
    w_data = abs(x1 - x0)
    length_data = length_um / PX_DS_UM
    length_frac = length_data / w_data
    x_right = 1.0 - pad_frac
    x_left = x_right - length_frac
    y_bar_bot = pad_frac
    y_bar_top = pad_frac + thickness_frac
    label_y = y_bar_top + 0.005
    bar = mpatches.Rectangle((x_left, y_bar_bot), length_frac, thickness_frac,
                             linewidth=0, facecolor=color, zorder=10,
                             transform=axes.transAxes, clip_on=False)
    axes.add_patch(bar)


## Compute geometry for the medium zoom (rotate apical up then crop)
def med_geom(zl, fh, fw):
    return zoom_straighten_geom(zl, fh, fw, onl_code=1, gcl_code=4,
                                onl_top_frac=onl_top_frac, extra_rot_deg=zoom_rot_delta_deg)


## Project full panel px coords into the med_geom frame
def med_proj(g, xp0, yp0, xp1, yp1, xr_arr, yr_arr):
    a = np.radians(g["rot_deg"])
    cosa, sina = np.cos(a), np.sin(a)
    cxp, cyp = (xp1 - xp0) / 2.0, (yp1 - yp0) / 2.0
    xx = xr_arr - xp0 - cxp
    yy = yr_arr - yp0 - cyp
    xf = cosa * xx + sina * yy + g["Wr"] / 2.0
    yf = -sina * xx + cosa * yy + g["Hr"] / 2.0
    if g["flip"]:
        yf = g["Hr"] - 1 - yf
    return xf - g["x0"], yf - g["y0"]


## Map a point in the cropped zoom frame back to full panel px
def med_unproj(g, xp0, yp0, xp1, yp1, mx, my):
    a = np.radians(g["rot_deg"])
    cosa, sina = np.cos(a), np.sin(a)
    cxp, cyp = (xp1 - xp0) / 2.0, (yp1 - yp0) / 2.0
    xf = mx + g["x0"]
    yf = my + g["y0"]
    yf0 = (g["Hr"] - 1 - yf) if g["flip"] else yf
    u = xf - g["Wr"] / 2.0
    v = yf0 - g["Hr"] / 2.0
    xx = cosa * u - sina * v
    yy = sina * u + cosa * v
    return xx + xp0 + cxp, yy + yp0 + cyp


## Draw a 500 um scale bar in the band below the tissue
def scalebar(axes, x0d, x1d, y_band_top, y_band_bot):
    bar = 500.0 / PX_DS_UM
    x1 = x1d - 0.02 * (x1d - x0d)
    x0 = x1 - bar
    band = y_band_bot - y_band_top
    yb = y_band_top + 0.38 * band
    axes.plot([x0, x1], [yb, yb], color="black", lw=4.5,
            solid_capstyle="butt", zorder=8, clip_on=False)


## Draw the cell type key inside the zoom row
def celltype_key(axes, celltypes):
    handles = [mpatches.Patch(facecolor=paper_style.CELLTYPE_COLORS[ct], edgecolor="none",
                              label=paper_style.celltype_label(ct)) for ct in celltypes]
    legend = axes.legend(handles=handles, title="Cell type", loc="upper right",
                    fontsize=8, title_fontsize=9, frameon=True, framealpha=0.85,
                    edgecolor="none", borderpad=0.35, labelspacing=0.25,
                    handlelength=1.0, handletextpad=0.5)
    legend.get_frame().set_facecolor("white")
    legend.get_title().set_fontweight("bold")
    for text in legend.get_texts():
        text.set_fontweight("bold")
    legend.set_zorder(30)


## Draw the DAPI and 18S channel key inside the zoom row
def channel_key(axes):
    handles = [mpatches.Patch(facecolor="#b0b0b0", edgecolor="none", label="DAPI"),
               mpatches.Patch(facecolor="#606060", edgecolor="none", label="18S rRNA")]
    legend = axes.legend(handles=handles, loc="upper right", fontsize=8, frameon=True,
                    framealpha=0.85, edgecolor="none", borderpad=0.35,
                    labelspacing=0.25, handlelength=1.0, handletextpad=0.5)
    legend.get_frame().set_facecolor("white")
    for text in legend.get_texts():
        text.set_fontweight("bold")
        text.set_color("#404040")
    legend.set_zorder(30)


## Draw the marker gene key inside the zoom row
def marker_key(axes, marker_list):
    handles = [plt.Line2D([0], [0], marker="o", lw=0, markersize=6,
                          markerfacecolor=paper_style.CELLTYPE_COLORS[ct],
                          markeredgecolor="none", label=g) for g, ct in marker_list]
    legend = axes.legend(handles=handles, title="Gene", loc="upper right",
                    fontsize=8, title_fontsize=9, frameon=True, framealpha=0.85,
                    edgecolor="none", borderpad=0.35, labelspacing=0.25,
                    handlelength=1.0, handletextpad=0.5)
    legend.get_frame().set_facecolor("white")
    legend.get_title().set_fontweight("bold")
    for text, (_g, ct) in zip(legend.get_texts(), marker_list):
        text.set_fontstyle("italic")
        text.set_fontweight("bold")
        text.set_color(paper_style.CELLTYPE_COLORS[ct])
    legend.set_zorder(30)


## Style a zoom row axis with a title
def style_zoom(axes, shape, title):
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_aspect("equal")
    axes.set_xlim(-0.5, shape[1] - 0.5)
    axes.set_ylim(shape[0] - 0.5, -0.5)
    for spine in axes.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    axes.text(0.015, 0.965, title, transform=axes.transAxes, ha="left", va="top",
            fontsize=16, fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))


## Set paper style and panel orientations
paper_style.set_style()
install_s1c_orientations()

## Load the panel and its oriented masks
panel = load_panel(sample)
panel["tissue"] = load_unet_tissue_oriented(panel)
panel["excl_mask"] = load_exclude_final_oriented(panel)
body = load_cellbody_mask_oriented(sample, panel)
H, W = panel["H"], panel["W"]
layer = panel["layer"]
tissue = panel["tissue"]
excl = panel["excl_mask"]

## Map each cell label to its cell type color
per_cell_data = pd.read_parquet(per_cell, columns=["sample_id", "cell_id", "cell_type"])
per_cell_data = per_cell_data[(per_cell_data["sample_id"] == sample) & (per_cell_data["cell_type"] != "unassigned")]
label_to_color = {}
label_to_ct = {}
for label, ct in zip(per_cell_data["cell_id"].to_numpy(), per_cell_data["cell_type"].to_numpy()):
    if ct in paper_style.CELLTYPE_COLORS:
        label_to_color[int(label)] = mcolors.to_rgb(paper_style.CELLTYPE_COLORS[ct])
        label_to_ct[int(label)] = ct

## Build the shared DAPI and 18S base composite plus the cell body composite
base_full = build_composite(panel, body, {})
img_dapi = base_full
img_cells = build_composite(panel, body, label_to_color)

## Project the marker transcripts into the panel frame
transcripts = pd.read_parquet(f"{marker_tx_dir}/{sample}_tx.parquet")
xr, yr = project_xy_to_panel(panel, transcripts["x"].to_numpy(), transcripts["y"].to_numpy())
transcripts = transcripts.assign(xr=xr, yr=yr)
inb = (transcripts["xr"] >= 0) & (transcripts["xr"] < W) & (transcripts["yr"] >= 0) & (transcripts["yr"] < H)
transcripts = transcripts[inb].copy()
ix = transcripts["xr"].to_numpy().astype(int)
iy = transcripts["yr"].to_numpy().astype(int)
transcripts["ok"] = tissue[iy, ix] & (~excl[iy, ix])

## Crop the medium zoom apical up on the lower arm of the C
fx, fy = zoom_center_frac
cx, cy = fx * W, fy * H
half_h = (zoom_h_um / PX_DS_UM) / 2.0
half_w = half_h * med_aspect
cx -= 0.5 * half_w
cy -= 0.4 * half_h
pre = 0.5 * float(np.hypot(2 * half_w, 2 * half_h)) + 6
xp0, yp0 = int(max(0, cx - pre)), int(max(0, cy - pre))
xp1, yp1 = int(min(W, cx + pre)), int(min(H, cy + pre))
zl = layer[yp0:yp1, xp0:xp1]
fh, fw = int(2 * half_h), int(2 * half_w)
g = med_geom(zl, fh, fw)
med_h, med_w = g["side_h"], g["side_w"]

## Rotate flip and crop the DAPI cell and body crops through the same geometry
med_dapi = zoom_straighten_apply(img_dapi[yp0:yp1, xp0:xp1], g, order=1, cval=1.0)
med_cells = zoom_straighten_apply(img_cells[yp0:yp1, xp0:xp1], g, order=1, cval=1.0)
med_body = zoom_straighten_apply(body[yp0:yp1, xp0:xp1].astype(np.int32), g, order=0, cval=0)
cts_shown = {label_to_ct[int(label)] for label in np.unique(med_body) if int(label) in label_to_ct}
zoom_celltypes = [ct for ct in celltype_stack if ct in cts_shown]

## Compute the figure layout
usable_h = h_fig - m_t - m_b
hB = (usable_h - 2 * gap_b) / 3.0
wB = hB * med_aspect
ys_t, xs_t = np.where(tissue)
ty0, ty1 = int(ys_t.min()), int(ys_t.max())
x0d, x1d = xs_t.min() - 0.03 * W, xs_t.max() + 0.03 * W
y0d = ty0 - 0.03 * H
y_band_top = ty1 + 0.03 * H
y1d = y_band_top + sb_band_frac * (ty1 - ty0)
map_aspect = (x1d - x0d) / (y1d - y0d)
wA = usable_h * map_aspect
W_FIG = m_l + wA + g_ab + wB + m_r


## Convert an inches rectangle to figure fraction
def rect(x_in, y_in, w_in, h_in):
    return [x_in / W_FIG, y_in / h_fig, w_in / W_FIG, h_in / h_fig]


## Set up the figure and axes
figure = plt.figure(figsize=(W_FIG, h_fig), facecolor="white")
xA = m_l
xB = xA + wA + g_ab
main_axes = figure.add_axes(rect(xA, m_b, wA, usable_h))
zoom_axes = [figure.add_axes(rect(xB, m_b + usable_h - hB - index * (hB + gap_b), wB, hB)) for index in range(3)]

## Draw the marker map in the main column
main_axes.imshow(base_full, origin="upper", interpolation="nearest")
for gene, ct in markers:
    subset = transcripts[(transcripts["name"] == gene) & transcripts["ok"]]
    main_axes.scatter(subset["xr"], subset["yr"], s=0.4,
                    c=[mcolors.to_rgb(paper_style.CELLTYPE_COLORS[ct])],
                    linewidths=0, alpha=0.85, rasterized=True, zorder=3)
main_axes.set_xlim(x0d, x1d)
main_axes.set_ylim(y1d, y0d)
main_axes.set_aspect("equal")
main_axes.set_xticks([])
main_axes.set_yticks([])
for spine in main_axes.spines.values():
    spine.set_visible(False)

## Draw the true rotated zoom box on the marker map
box_poly = [med_unproj(g, xp0, yp0, xp1, yp1, mxc, myc)
            for mxc, myc in [(0, 0), (med_w, 0), (med_w, med_h), (0, med_h)]]
main_axes.add_patch(plt.Polygon([(float(a), float(b)) for a, b in box_poly],
                              closed=True, fill=False, edgecolor="black",
                              lw=1.6, zorder=20))

## Draw the gene and channel key centered in the C opening
keys = [(g_, paper_style.CELLTYPE_COLORS[ct], True) for g_, ct in markers]
keys += [("DAPI", "#b0b0b0", False), ("18S rRNA", "#606060", False)]
x_dot, x_lab, dy = 0.775, 0.825, 0.040
y_top = 0.56 + (len(keys) - 1) * dy / 2.0
for index, (label, color, is_italic) in enumerate(keys):
    yy = y_top - index * dy
    main_axes.scatter([x_dot], [yy], s=95, c=[mcolors.to_rgb(color)],
                    marker="o" if is_italic else "s", edgecolor="none",
                    transform=main_axes.transAxes, clip_on=False, zorder=25)
    main_axes.text(x_lab, yy, label, transform=main_axes.transAxes, ha="left",
                 va="center", fontsize=16,
                 fontstyle="italic" if is_italic else "normal",
                 fontweight="bold", color=color if is_italic else "#404040", zorder=25)
scalebar(main_axes, x0d, x1d, y_band_top, y1d)

## Draw zoom row 1 with DAPI and 18S
zoom_axes[0].imshow(med_dapi, origin="upper", interpolation="nearest")
style_zoom(zoom_axes[0], (med_h, med_w), "DAPI + 18S rRNA")
channel_key(zoom_axes[0])

## Draw zoom row 2 with cell types
zoom_axes[1].imshow(med_cells, origin="upper", interpolation="nearest")
style_zoom(zoom_axes[1], med_cells.shape, "+ cell types")
celltype_key(zoom_axes[1], zoom_celltypes)

## Draw zoom row 3 with marker transcripts
marker_axes = zoom_axes[2]
marker_axes.imshow(med_dapi, origin="upper", interpolation="nearest")
for gene, ct in markers:
    subset = transcripts[(transcripts["name"] == gene) & transcripts["ok"]]
    mx, my = med_proj(g, xp0, yp0, xp1, yp1, subset["xr"].to_numpy(), subset["yr"].to_numpy())
    keep = (mx >= 0) & (mx < med_w) & (my >= 0) & (my < med_h)
    marker_axes.scatter(mx[keep], my[keep], s=2.5,
                c=[mcolors.to_rgb(paper_style.CELLTYPE_COLORS[ct])], linewidths=0,
                alpha=0.9, zorder=3)
style_zoom(marker_axes, (med_h, med_w), "+ marker transcripts")
marker_key(marker_axes, markers)

## Add a 50 um scale bar to every zoom row
for zoom_axis in zoom_axes:
    add_scalebar(zoom_axis, 50, "50 µm")

## Save the panel
figure.savefig(out_png, dpi=250, facecolor="white")
plt.close(figure)
print("wrote", out_png)
