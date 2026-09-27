# Fig 1d WT P21 retina opsin transcript map

## Load packages
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

## Load paper style conventions
sys.path.insert(0, "../../_shared/code")
import paper_style
paper_style.set_style()

## Load shared imaging functions
from _retina_panel import load_panel, project_xy_to_panel, PX_DS_UM
from _paths import findpath_exclude_final, findpath_unet_pred, ROI_TO_SAMPLE
from _superior_up_orientation import install_s1c_orientations
install_s1c_orientations()

## Load files
opsin_tx_dir = "../../_shared/data_raw/opsin_tx"
out_png = "../panels/Fig1d.png"

## Define the sample and opsin colors
sample = "WT_P21_rep2"
opsin = {"Opn1mw": "#1b9e3b", "Opn1sw": "#c800a8"}
muted = "#2A2A2A"
dot_size = 0.7
sb_band_frac = 0.11   # Scale-bar band below the tissue

## Load the panel
panel = load_panel(sample)
dapi = panel["dapi"]
H, W = dapi.shape

## Orient the tissue and exclude masks into the panel frame
oriented = {}
_sample_name = ROI_TO_SAMPLE[panel['roi']]
_excl_p = findpath_exclude_final(_sample_name)
_unet_p = findpath_unet_pred(_sample_name, "layer")
for name, path in [("tissue", str(_unet_p)),
                   ("excl", str(_excl_p))]:
    raster = (tifffile.imread(path) > 0 if Path(path).exists() else np.zeros((panel["orig_H"], panel["orig_W"]), dtype=np.uint8)).astype(np.uint8)
    Hd, Wd = panel["orig_H"], panel["orig_W"]
    Hr, Wr = raster.shape
    pad = np.zeros((Hd, Wd), dtype=raster.dtype)
    pad[:min(Hr, Hd), :min(Wr, Wd)] = raster[:Hd, :Wd]
    angle = panel["angle"]
    rot = ndi.rotate(pad, angle, reshape=True, order=0, mode="constant", cval=0) if abs(angle) > 0.1 else pad
    if panel["flipped_v"]:
        rot = rot[::-1, :]
    if panel["flipped_h"]:
        rot = rot[:, ::-1]
    ph, pw = panel["H"], panel["W"]
    cx0, cy0 = panel["crop_x0"], panel["crop_y0"]
    rot = rot[cy0:cy0 + ph, cx0:cx0 + pw]
    out = np.zeros((ph, pw), dtype=raster.dtype)
    hf, wf = rot.shape
    out[:min(ph, hf), :min(pw, wf)] = rot[:ph, :pw]
    oriented[name] = out.astype(bool)
tissue = oriented["tissue"]
excl_panel = oriented["excl"]

## Measure the tissue bounding box
ys, xs = np.where(tissue)
x0_t, x1_t = int(xs.min()), int(xs.max())
y0_t, y1_t = int(ys.min()), int(ys.max())

## Frame the panel like the Fig1e composite column
x0d, x1d = x0_t - 0.03 * W, x1_t + 0.03 * W
y0d = y0_t - 0.03 * H
y_band_top = y1_t + 0.03 * H
y1d = y_band_top + sb_band_frac * (y1_t - y0_t)
map_aspect = (x1d - x0d) / (y1d - y0d)

## Set up the figure at the Fig1e composite panel height
fig_h = 9.5
margin = 0.12
usable_h = fig_h - 2 * margin
wA = usable_h * map_aspect
m_l = 0.45   # Room for the S and I arrows on the left
fig_w = m_l + wA + margin
figure = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
axes = figure.add_axes([m_l / fig_w, margin / fig_h, wA / fig_w, usable_h / fig_h])

## Draw the DAPI background as light gray
d = dapi.astype(np.float32)
dpos = d[d > 0]
lo, hi = np.percentile(dpos, (2, 98)) if dpos.size else (0.0, 1.0)
dn = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.55
light = 1.0 - dn * 0.34
light[excl_panel] = 1.0
axes.imshow(light, cmap="gray", vmin=0, vmax=1, zorder=0)

## Draw the opsin transcripts
tx = pd.read_parquet(f"{opsin_tx_dir}/{sample}_opsin_tx.parquet")
tx = tx[tx["name"].isin(opsin)].copy()
xr, yr = project_xy_to_panel(panel, tx["x"].to_numpy(), tx["y"].to_numpy())
tx = tx.assign(xr=xr, yr=yr)
tx = tx[(tx["xr"] >= 0) & (tx["xr"] < W) & (tx["yr"] >= 0) & (tx["yr"] < H)]
ixr = tx["xr"].to_numpy().astype(int)
iyr = tx["yr"].to_numpy().astype(int)
in_excl = excl_panel[iyr, ixr]
tx = tx[~in_excl]
ixr = ixr[~in_excl]
iyr = iyr[~in_excl]
on_tissue = tissue[iyr, ixr]
n_off = int((~on_tissue).sum())
tx = tx[on_tissue]
print(f"  {sample}: opsin dots={len(tx)} (dropped {int(in_excl.sum())} in exclude_final, {n_off} off-tissue)")
for g, color in opsin.items():
    sub = tx[tx["name"] == g]
    axes.scatter(sub["xr"], sub["yr"], s=dot_size, c=color, edgecolor="none",
               alpha=0.85, zorder=3, rasterized=True)

## Set the axis limits to the framing computed above
axes.set_xlim(x0d, x1d)
axes.set_ylim(y1d, y0d)
axes.set_aspect("equal")
axes.set_xticks([])
axes.set_yticks([])
for spine in axes.spines.values():
    spine.set_visible(False)

## Add the 500 um scale bar in the band below the tissue
bar = 500.0 / PX_DS_UM
x1b = x1d - 0.02 * (x1d - x0d)
x0b = x1b - bar
band = y1d - y_band_top
y_b = y_band_top + 0.38 * band
axes.plot([x0b, x1b], [y_b, y_b], color="black", lw=4.5, solid_capstyle="butt",
        zorder=8, clip_on=False)

## Add the superior inferior arrows
for txt, fy, fs in [("S", 0.95, 26), ("↑", 0.86, 36), ("↓", 0.15, 36), ("I", 0.06, 26)]:
    axes.text(-0.055, fy, txt, transform=axes.transAxes, ha="center", va="center",
            fontsize=fs, color=muted, fontweight="bold", clip_on=False)

## Add the opsin key
halo = [pe.withStroke(linewidth=2.2, foreground="white")]
for i, g in enumerate(["Opn1mw", "Opn1sw"]):
    color = opsin[g]
    y = 0.965 - 0.045 * i
    axes.scatter([0.05], [y], s=85, c=color, transform=axes.transAxes,
               edgecolor="none", clip_on=False, zorder=6)
    axes.text(0.105, y, g, transform=axes.transAxes, ha="left", va="center",
            fontsize=16, fontstyle="italic", fontweight="bold", color=color,
            path_effects=halo, zorder=6)

## Save the panel
figure.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(figure)
print("wrote", out_png)
