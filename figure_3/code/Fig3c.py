# Fig 3c rod spatial maps with superior and inferior vignettes

## Load packages
import sys
import numpy as np
import pandas as pd
import tifffile
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi
from scipy.ndimage import rotate as nd_rotate, binary_closing, binary_fill_holes, gaussian_filter
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects

## Load shared calibration and the superior/inferior split from the retina panel module
sys.path.insert(0, "../../_shared/code")
from _retina_panel import DS, PX_DS_UM
from _paths import (SEGMENTATION_EXPORT_ROOT, findpath_unet_pred, findpath_exclude_final,
                    SAMPLE_TO_ROI)
from _superior_inferior_axis import superior_mask

## Load files and parameters
per_cell = "../../_shared/data_processed/per_cell_gene_counts.parquet"
out_png = "../panels/Fig3c.png"
DS8_CACHE = Path("../../_shared/data_raw/ds8_cache")
PER = Path("../../preprocessing/outputs/per_cell_typed")
EXPORT_SEG = SEGMENTATION_EXPORT_ROOT
SAMPLE = "LCA5_P21_rep2"
CROP_PAD_UM = 80
# Display only horizontal flip for this sample so the optic nerve head faces left
SAMPLE_FLIP_X = {"LCA5_P21_rep2": True}

SUP_COLOR, INF_COLOR = "#1b9e3b", "#c800a8"     # Green superior, magenta inferior
ZOOM_W_UM, ZOOM_H_UM = 480.0, 240.0            # Vignette crop is a 2:1 box
SUP_TOP_FRAC, INF_TOP_FRAC = 0.03, 0.05        # ONL apical edge from crop top
SUP_VIG_TOP_FRAC = 0.30                         # Superior vignette ONL band position
INF_VIG_TOP_FRAC = 0.16                         # Inferior vignette ONL band position
SCALE_UM = 50.0
MAP_SCALE_UM = 500.0
TITLE_COLOR = "#222222"
MAPS = ["Edn2", "Gadd45b", "Nxnl1"]            # Three rod example genes boxed in Fig3b
MIN_CROP_FILL = 0.50
POLE_Q = 0.15


## Set the paper style
def set_style():
    mpl.rcParams.update({
        "font.family": "Arial", "font.weight": "bold",
        "axes.labelweight": "bold", "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 200, "figure.dpi": 120,
    })


## Cone opsin loaders and orientation

# Load cones with opsin counts and positions
def load_cones(sample):
    t = pd.read_parquet(PER / f"{sample}_typed.parquet")
    t = t[t["keep_for_analysis"] & (t["final_label"] == "cone")].copy()
    opsin = pd.read_parquet(per_cell, columns=["sample_id", "cell_id", "body_Opn1sw", "body_Opn1mw"])
    opsin = opsin[opsin["sample_id"] == sample].rename(columns={"cell_id": "label"})
    return t.merge(opsin[["label", "body_Opn1sw", "body_Opn1mw"]], on="label", validate="one_to_one")


# Display only orientation from cone positions and the opsin gradient
def fit_display_orientation(cones):
    cx = cones["cx_px"].to_numpy(dtype=float)
    cy = cones["cy_px"].to_numpy(dtype=float)
    sw = cones["body_Opn1sw"].to_numpy(dtype=float)
    mw = cones["body_Opn1mw"].to_numpy(dtype=float)
    ms = np.log1p(mw) - np.log1p(sw)

    xc, yc = cx - cx.mean(), cy - cy.mean()
    cov = np.cov(np.vstack([xc, yc]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = eigvecs[:, -1]
    rotation = float(np.arctan2(-1, 0)) - float(np.arctan2(pc1[1], pc1[0]))

    cos_a, sin_a = np.cos(rotation), np.sin(rotation)
    yr = sin_a * xc + cos_a * yc
    top = yr < 0
    bot = yr > 0
    ms_top = float(ms[top].mean()) if top.any() else 0.0
    ms_bot = float(ms[bot].mean()) if bot.any() else 0.0

    return {"rotation_rad": float(rotation), "flip_y": bool(ms_top < ms_bot)}


## Tissue and layer mask loaders oriented to the panel

# Reorient a native mask into the panel frame (pad, rotate, flip, then crop)
def _orient_to_panel(raster, panel, order=0):
    Hd, Wd = panel["orig_H"], panel["orig_W"]
    Hr, Wr = raster.shape
    pad = np.zeros((Hd, Wd), dtype=raster.dtype)
    pad[:min(Hr, Hd), :min(Wr, Wd)] = raster[:Hd, :Wd]
    angle = panel["angle"]
    if abs(angle) > 0.1:
        rot = ndi.rotate(pad, angle, reshape=True, order=order, mode="constant", cval=0)
    else:
        rot = pad
    if panel["flipped_v"]:
        rot = rot[::-1, :]
    if panel["flipped_h"]:
        rot = rot[:, ::-1]
    H, W = panel["H"], panel["W"]
    cx0, cy0 = panel["crop_x0"], panel["crop_y0"]
    rot = rot[cy0:cy0 + H, cx0:cx0 + W]
    out = np.zeros((H, W), dtype=raster.dtype)
    Hf, Wf = rot.shape
    out[:min(H, Hf), :min(W, Wf)] = rot[:H, :W]
    return out


# Load the excluded region mask oriented to the panel
def load_exclude_final_oriented(panel):
    from _paths import ROI_TO_SAMPLE
    p = findpath_exclude_final(ROI_TO_SAMPLE[panel['roi']])
    raw = tifffile.imread(p) > 0 if p.exists() else np.zeros((panel['orig_H'], panel['orig_W']), dtype=bool)
    return _orient_to_panel(raw.astype(np.uint8), panel, order=0).astype(bool)


# Display only straightening angle for horizontal ONL views
def _onl_pca_angle(layer_arr):
    ys, xs = np.where(layer_arr == 1)
    if len(ys) < 50:
        return 0.0
    pts = np.column_stack([xs, ys]).astype(np.float64)
    pts -= pts.mean(axis=0, keepdims=True)
    eigvals, eigvecs = np.linalg.eigh(np.cov(pts.T))
    pc1 = eigvecs[:, -1]
    if pc1[0] < 0:
        pc1 = -pc1
    return float(np.degrees(np.arctan2(pc1[1], pc1[0])))


## Load the panel orient the ROI then crop to the tissue

def _derive_tissue_mask(dapi):
    smoothed = gaussian_filter(dapi.astype(np.float32), sigma=8.0 / PX_DS_UM)
    thr = threshold_otsu(smoothed)
    mask = smoothed > thr * 0.5
    mask = binary_closing(mask, iterations=int(round(25 / PX_DS_UM)))
    mask = remove_small_objects(mask, min_size=int(round((300 / PX_DS_UM) ** 2)))
    mask = binary_fill_holes(mask)
    return mask


def _rotate_xy(x, y, cx, cy, angle_deg):
    a = np.radians(angle_deg)
    cosa, sina = np.cos(a), np.sin(a)
    dx, dy = x - cx, y - cy
    xr =  cosa * dx + sina * dy + cx
    yr = -sina * dx + cosa * dy + cy
    return xr, yr


def _flip_horizontal(layer, cells, dapi, excl_mask):
    H, W = layer.shape
    layer = layer[:, ::-1].copy()
    if dapi is not None:
        dapi = dapi[:, ::-1].copy()
    if excl_mask is not None:
        excl_mask = excl_mask[:, ::-1].copy()
    cells = cells.assign(xr=W - 1 - cells["xr"])
    return layer, cells, dapi, excl_mask


def _flip_vertical(layer, cells, dapi, excl_mask):
    H, W = layer.shape
    layer = layer[::-1, :].copy()
    if dapi is not None:
        dapi = dapi[::-1, :].copy()
    if excl_mask is not None:
        excl_mask = excl_mask[::-1, :].copy()
    cells = cells.assign(yr=H - 1 - cells["yr"])
    return layer, cells, dapi, excl_mask


# Load and align the final exclusion mask on the ds8 grid
def load_panel(sample):
    roi = SAMPLE_TO_ROI[sample]
    npz = np.load(DS8_CACHE / f"roi{roi}_dapi18s_ds8.npz")
    dapi = npz["arr"][0]
    s18 = npz["arr"][1]
    layer = tifffile.imread(findpath_unet_pred(sample, "layer"))
    Hd, Wd = dapi.shape
    Hl, Wl = layer.shape
    H_pad = max(Hd, Hl)
    W_pad = max(Wd, Wl)
    layer_full = np.zeros((H_pad, W_pad), dtype=layer.dtype)
    layer_full[:Hl, :Wl] = layer
    dapi_full = np.zeros((H_pad, W_pad), dtype=dapi.dtype) + dapi.min()
    dapi_full[:Hd, :Wd] = dapi
    s18_full = np.zeros((H_pad, W_pad), dtype=np.float32)
    Hs18, Ws18 = s18.shape
    s18_full[:min(Hs18, H_pad), :min(Ws18, W_pad)] = s18[:H_pad, :W_pad]
    dapi, layer, s18 = dapi_full, layer_full, s18_full
    H, W = dapi.shape

    typed = pd.read_parquet(PER / f"{sample}_typed.parquet")
    _ep = findpath_exclude_final(sample)
    raw_excl = tifffile.imread(_ep) > 0 if _ep.exists() else np.zeros((H, W), dtype=bool)
    excl_mask = np.zeros((H, W), dtype=bool)
    excl_mask[:min(H, raw_excl.shape[0]), :min(W, raw_excl.shape[1])] = raw_excl[:H, :W]
    cells = typed[typed["keep_for_analysis"]].copy()
    cell_r = np.clip(np.rint(cells["cy_px"].to_numpy() / DS).astype(int), 0, H - 1)
    cell_c = np.clip(np.rint(cells["cx_px"].to_numpy() / DS).astype(int), 0, W - 1)
    cells = cells.loc[~excl_mask[cell_r, cell_c]].copy()
    cells["xr"] = cells["cx_px"] / DS
    cells["yr"] = cells["cy_px"] / DS

    tissue_full = _derive_tissue_mask(dapi) & (~excl_mask)

    display_fit = fit_display_orientation(load_cones(sample))
    angle = float(-np.degrees(display_fit["rotation_rad"]))
    flipped_v = bool(display_fit["flip_y"])
    flipped_h = bool(SAMPLE_FLIP_X.get(sample, False))

    orig_H, orig_W = H, W
    cx_orig_save, cy_orig_save = W / 2.0, H / 2.0
    rot_Hn, rot_Wn = H, W
    dx_shift_save, dy_shift_save = 0.0, 0.0

    if abs(angle) > 0.1:
        dapi_r = nd_rotate(dapi, angle, reshape=True, order=1, mode="constant", cval=float(dapi.min()))
        s18_r = nd_rotate(s18, angle, reshape=True, order=1, mode="constant", cval=0.0)
        layer_r = nd_rotate(layer, angle, reshape=True, order=0, mode="constant", cval=0)
        excl_r = nd_rotate(excl_mask.astype(np.uint8), angle, reshape=True, order=0, mode="constant", cval=0).astype(bool)
        tissue_r = nd_rotate(tissue_full.astype(np.uint8), angle, reshape=True, order=0, mode="constant", cval=0).astype(bool)
        cy_orig = H / 2.0
        cx_orig = W / 2.0
        Hn, Wn = layer_r.shape
        dx_shift = Wn / 2.0 - cx_orig
        dy_shift = Hn / 2.0 - cy_orig
        xr, yr = _rotate_xy(cells["xr"].to_numpy(), cells["yr"].to_numpy(), cx_orig, cy_orig, angle)
        cells = cells.assign(xr=xr + dx_shift, yr=yr + dy_shift)
        dapi, layer, excl_mask, tissue_full, s18 = (dapi_r, layer_r, excl_r, tissue_r, s18_r)
        H, W = tissue_full.shape
        cx_orig_save, cy_orig_save = cx_orig, cy_orig
        rot_Hn, rot_Wn = Hn, Wn
        dx_shift_save, dy_shift_save = dx_shift, dy_shift

    if flipped_v:
        layer, cells, dapi, excl_mask = _flip_vertical(layer, cells, dapi, excl_mask)
        tissue_full = tissue_full[::-1, :].copy()
        s18 = s18[::-1, :].copy()
    if flipped_h:
        layer, cells, dapi, excl_mask = _flip_horizontal(layer, cells, dapi, excl_mask)
        tissue_full = tissue_full[:, ::-1].copy()
        s18 = s18[:, ::-1].copy()

    crop_x0, crop_y0 = 0, 0
    ys, xs = np.where(tissue_full)
    if len(ys):
        pad = int(CROP_PAD_UM / PX_DS_UM)
        y0 = max(0, ys.min() - pad)
        y1 = min(H, ys.max() + pad)
        x0 = max(0, xs.min() - pad)
        x1 = min(W, xs.max() + pad)
        dapi = dapi[y0:y1, x0:x1]
        s18 = s18[y0:y1, x0:x1]
        layer = layer[y0:y1, x0:x1]
        excl_mask = excl_mask[y0:y1, x0:x1]
        tissue_full = tissue_full[y0:y1, x0:x1]
        cells = cells.assign(xr=cells["xr"] - x0, yr=cells["yr"] - y0)
        H, W = tissue_full.shape
        crop_x0, crop_y0 = int(x0), int(y0)
    return dict(sample=sample, roi=roi, dapi=dapi, s18=s18, layer=layer,
                tissue=tissue_full, excl_mask=excl_mask, cells=cells,
                H=H, W=W, angle=angle,
                flipped_v=flipped_v, flipped_h=flipped_h,
                orig_H=int(orig_H), orig_W=int(orig_W),
                cx_orig=float(cx_orig_save), cy_orig=float(cy_orig_save),
                rot_Hn=int(rot_Hn), rot_Wn=int(rot_Wn),
                dx_shift=float(dx_shift_save), dy_shift=float(dy_shift_save),
                crop_x0=crop_x0, crop_y0=crop_y0)


## Panel specific functions

# Load the cellbody label mask and orient it to the panel
def load_body_mask_oriented(sample, panel):
    native = tifffile.imread(EXPORT_SEG / sample / "cellbody_masks.tif")
    body = native[::DS, ::DS]
    Hd, Wd = panel["orig_H"], panel["orig_W"]
    Hb, Wb = body.shape
    pad = np.zeros((Hd, Wd), dtype=body.dtype)
    pad[:min(Hb, Hd), :min(Wb, Wd)] = body[:Hd, :Wd]
    angle = panel["angle"]
    rot = ndi.rotate(pad, angle, reshape=True, order=0, mode="constant", cval=0) if abs(angle) > 0.1 else pad
    if panel["flipped_v"]:
        rot = rot[::-1, :]
    if panel["flipped_h"]:
        rot = rot[:, ::-1]
    H, W = panel["H"], panel["W"]
    cx0, cy0 = panel["crop_x0"], panel["crop_y0"]
    rot = rot[cy0:cy0 + H, cx0:cx0 + W]
    out = np.zeros((H, W), dtype=rot.dtype)
    Hf, Wf = rot.shape
    out[:min(H, Hf), :min(W, Wf)] = rot[:H, :W]
    return out


# Truncate a named colormap to the range lo to hi
def _trunc(name, lo, hi):
    c = LinearSegmentedColormap.from_list(f"{name}_t", mpl.colormaps[name](np.linspace(lo, hi, 256)))
    c.set_bad(c(0.0))
    return c


# One monotonic purple density colormap for all maps
ONE_CMAP = _trunc("Purples", 0.15, 1.0)


# Build a pale tissue background and color rod cell bodies by expression
def composite(panel, body, vals, cmap, norm):
    dapi = panel["dapi"].astype(np.float32)
    s18 = panel["s18"].astype(np.float32)
    d_pos = dapi[dapi > 0]
    d_lo, d_hi = np.percentile(d_pos, (2, 98)) if d_pos.size else (0, 1)
    d_norm = np.clip((dapi - d_lo) / max(d_hi - d_lo, 1e-6), 0, 1) ** 0.55
    s_pos = s18[s18 > 0]
    s_lo, s_hi = np.percentile(s_pos, (2, 98)) if s_pos.size else (0, 1)
    s_norm = np.clip((s18 - s_lo) / max(s_hi - s_lo, 1e-6), 0, 1) ** 0.7
    inv = np.clip(1.0 - (d_norm * 0.15 + s_norm * 0.45), 0, 1)
    base = np.stack([inv] * 3, axis=-1)

    H, W = body.shape

    def _fit(m):
        mm = np.zeros((H, W), dtype=bool)
        Hm, Wm = m.shape
        mm[:min(H, Hm), :min(W, Wm)] = m[:H, :W]
        return mm

    not_excl = np.ones((H, W), dtype=bool)
    excl = panel.get("excl_mask")
    if excl is not None:
        not_excl &= ~_fit(excl)
    # Mask excluded pixels
    base = np.where(not_excl[..., None], base, 1.0)

    out = base.copy()
    if cmap is None:
        return out, not_excl
    maxlab = int(body.max())
    lut = np.zeros((maxlab + 1, 4), dtype=np.float32)
    for lab, v in vals.items():
        if 0 < lab <= maxlab and np.isfinite(v):
            lut[lab] = cmap(norm(float(v)))
    col = lut[body]
    has = (col[..., 3] > 0) & not_excl
    out[has] = col[has, :3]
    return out, not_excl


# Rotate the ONL horizontally with the apical side up, then crop a fixed box
def straighten(rgb, layer, cx, cy, half, halfv, top_frac=0.12):
    pre = int(round(np.hypot(half, halfv))) + 8
    H, W = layer.shape
    y0, y1 = max(0, cy - pre), min(H, cy + pre)
    x0, x1 = max(0, cx - pre), min(W, cx + pre)
    sr, sl = rgb[y0:y1, x0:x1], layer[y0:y1, x0:x1]
    ang = _onl_pca_angle(sl)
    rot = ndi.rotate(sr, ang, axes=(0, 1), reshape=True, order=1, cval=1.0, mode="constant")
    rl = ndi.rotate(sl.astype(np.int16), ang, reshape=True, order=0, cval=0)
    yo, yg = np.where(rl == 1)[0], np.where(rl == 4)[0]
    if len(yo) > 20 and len(yg) > 20 and yo.mean() > yg.mean():
        rot, rl = rot[::-1], rl[::-1]
    Hn, Wn = rot.shape[:2]
    onl_rows = np.where((rl == 1).any(axis=1))[0]
    top_margin = int(round(top_frac * 2 * halfv))
    r0 = max(0, onl_rows.min() - top_margin) if onl_rows.size else max(0, Hn // 2 - halfv)
    r0 = min(r0, max(0, Hn - 2 * halfv))
    c0 = max(0, Wn // 2 - half)
    crop = rot[r0:r0 + 2 * halfv, c0:c0 + 2 * half]
    out = np.ones((2 * halfv, 2 * half, crop.shape[2]), dtype=crop.dtype)
    sh, sw = crop.shape[:2]
    out[:sh, :sw] = crop
    return out


# True if the rotation window spans ONL to GCL
def _has_full_span(layer, cx, cy, half, halfv):
    pre = int(round(np.hypot(half, halfv))) + 8
    H, W = layer.shape
    sub = layer[max(0, cy - pre):min(H, cy + pre), max(0, cx - pre):min(W, cx + pre)]
    return int((sub == 1).sum()) > 40 and int((sub == 4).sum()) > 40


# Fraction of the straightened crop that is tissue
def _crop_fill(layer, cx, cy, half, halfv, top_frac):
    pre = int(round(np.hypot(half, halfv))) + 8
    H, W = layer.shape
    sl = layer[max(0, cy - pre):min(H, cy + pre), max(0, cx - pre):min(W, cx + pre)]
    ang = _onl_pca_angle(sl)
    rl = ndi.rotate(sl.astype(np.int16), ang, reshape=True, order=0, cval=0)
    yo, yg = np.where(rl == 1)[0], np.where(rl == 4)[0]
    if len(yo) > 20 and len(yg) > 20 and yo.mean() > yg.mean():
        rl = rl[::-1]
    Hn, Wn = rl.shape
    onl_rows = np.where((rl == 1).any(axis=1))[0]
    tm = int(round(top_frac * 2 * halfv))
    r0 = max(0, onl_rows.min() - tm) if onl_rows.size else max(0, Hn // 2 - halfv)
    r0 = min(r0, max(0, Hn - 2 * halfv))
    c0 = max(0, Wn // 2 - half)
    return float((rl[r0:r0 + 2 * halfv, c0:c0 + 2 * half] > 0).mean())


# Pick a representative full thickness ONL view for one hemisphere
def pick_region(cand, half, halfv, W, H, tiss, layer, top_frac, topmost):
    pre = int(round(np.hypot(half, halfv))) + 8
    for fill_min in (MIN_CROP_FILL, 0.35):
        pts = []
        for a, b in zip(cand.xr.to_numpy(), cand.yr.to_numpy()):
            a, b = int(round(a)), int(round(b))
            if pre <= a < W - pre and pre <= b < H - pre and tiss[b, a] \
                    and _has_full_span(layer, a, b, half, halfv) \
                    and _crop_fill(layer, a, b, half, halfv, top_frac) >= fill_min:
                pts.append((a, b))
        if pts:
            pts = np.asarray(pts, dtype=float)
            q = POLE_Q if topmost else (1.0 - POLE_Q)
            target_y = np.quantile(pts[:, 1], q)
            i = int(np.abs(pts[:, 1] - target_y).argmin())
            return int(pts[i, 0]), int(pts[i, 1])
    return None


# Draw a micron accurate scale bar in the bottom right of a vignette
def scalebar(ax, crop_shape, um):
    H, W = crop_shape[:2]
    bar_px = um / PX_DS_UM
    x1 = W - 0.05 * W
    x0 = x1 - bar_px
    y = H - 0.09 * H
    ax.plot([x0, x1], [y, y], color="black", lw=2.2, solid_capstyle="butt", zorder=11)


## Load the panel where orientation is fit inside load_panel from the cone positions
set_style()
panel = load_panel(SAMPLE)
panel["excl_mask"] = load_exclude_final_oriented(panel)
body = load_body_mask_oriented(SAMPLE, panel)

## Load the per cell table and score each cell superior or inferior
cells_all = pd.read_parquet(
    per_cell,
    columns=["sample_id", "cell_id", "cell_type", "cell_area_um2",
             "cx_um", "cy_um", "body_Opn1mw", "body_Opn1sw",
             *[f"body_{gene}" for gene in MAPS]],
)
cells = cells_all[cells_all.sample_id == SAMPLE].copy()
cells["is_superior"] = superior_mask(cells)
rods = cells[cells.cell_type == "rod"].copy()
rod_cells = (panel["cells"][["label", "xr", "yr"]].rename(columns={"label": "cell_id"})
             .merge(rods[["cell_id", "cell_type", "is_superior"]], on="cell_id", how="inner", validate="one_to_one"))

half_width = int(round(ZOOM_W_UM / 2 / PX_DS_UM))
half_height = int(round(ZOOM_H_UM / 2 / PX_DS_UM))

## Define the tissue bounds and the superior and inferior regions
_, keep_mask = composite(panel, body, {}, None, Normalize(0, 1))
keep_rows, keep_cols = np.where(keep_mask)
bbox_pad = 20
y0, y1 = max(0, keep_rows.min() - bbox_pad), min(keep_mask.shape[0], keep_rows.max() + bbox_pad)
x0, x1 = max(0, keep_cols.min() - bbox_pad), min(keep_mask.shape[1], keep_cols.max() + bbox_pad)
tissue_crop = keep_mask[y0:y1, x0:x1]
crop_height, crop_width = tissue_crop.shape
layer_crop = panel["layer"][y0:y1, x0:x1]
placed_cells = rod_cells.assign(xr=rod_cells.xr - x0, yr=rod_cells.yr - y0)
superior_center = pick_region(placed_cells[placed_cells.is_superior], half_width, half_height, crop_width, crop_height, tissue_crop, layer_crop, SUP_TOP_FRAC, True)
inferior_center = pick_region(placed_cells[~placed_cells.is_superior], half_width, half_height, crop_width, crop_height, tissue_crop, layer_crop, INF_TOP_FRAC, False)
box_width, box_height = 2 * half_width, 2 * half_height


# Shift a region center by whole box widths and heights
def _shift(center, dx_boxes, dy_boxes):
    if center is None:
        return None
    cx = int(np.clip(center[0] + dx_boxes * box_width, half_width, crop_width - half_width))
    cy = int(np.clip(center[1] + dy_boxes * box_height, half_height, crop_height - half_height))
    return (cx, cy)


# Shift the vignette placements superior right and up one box and inferior right and up
superior_center = _shift(superior_center, 1.0 + 1 / 6 + 1 / 8, -1.0)
inferior_center = _shift(inferior_center, +1.0 / 3, +0.5)

## Build the per gene composites
built_maps = []
for gene in MAPS:
    density = rods[f"body_{gene}"].to_numpy(float) / rods["cell_area_um2"].to_numpy(float)
    density_vmax = float(np.nanpercentile(density, 97))
    image, _ = composite(panel, body, dict(zip(rods.cell_id.astype(int), density)), ONE_CMAP, Normalize(0.0, density_vmax))
    built_maps.append(dict(g=gene, cmap=ONE_CMAP, vmax=density_vmax, img=image[y0:y1, x0:x1]))

## Lay out the figure
n_columns = len(MAPS)
image_height, image_width = built_maps[0]["img"].shape[:2]
left_margin, right_margin, top_margin, bottom_margin = 0.90, 0.95, 0.75, 0.20
map_height = 6.6
scalebar_band = 0.22
vig_gap, vig_gap2 = 0.05, 0.16
colorbar_gap, colorbar_width = 0.14, 0.16
column_gap = 0.85
overview_width = map_height * image_width / image_height
vignette_width = overview_width
vignette_height = vignette_width * (half_height / half_width)
colorbar_height = 0.42 * map_height

block_width = overview_width + colorbar_gap + colorbar_width
figure_width = left_margin + n_columns * block_width + (n_columns - 1) * column_gap + right_margin
figure_height = (top_margin + map_height + scalebar_band + vig_gap + vignette_height + vig_gap2 + vignette_height + bottom_margin)
figure = plt.figure(figsize=(figure_width, figure_height))


# Add an axes at figure coordinates in inches
def ax_at(x, y, w, h):
    return figure.add_axes([x / figure_width, y / figure_height, w / figure_width, h / figure_height])


inf_bottom = bottom_margin
sup_bottom = inf_bottom + vignette_height + vig_gap2
map_bottom = sup_bottom + vignette_height + vig_gap + scalebar_band
map_top = map_bottom + map_height

## Draw each gene column with its map colorbar and vignettes
column_x = left_margin
for built_map in built_maps:
    # Draw the map
    map_axes = ax_at(column_x, map_bottom, overview_width, map_height)
    map_axes.imshow(built_map["img"], interpolation="nearest", aspect="equal")
    map_axes.set_anchor("N")
    map_axes.set_xticks([])
    map_axes.set_yticks([])
    for spine in map_axes.spines.values():
        spine.set_visible(False)
    figure.text((column_x + overview_width / 2) / figure_width, (map_top + 0.06) / figure_height,
                built_map["g"], ha="center", va="bottom", fontsize=28, fontstyle="italic",
                fontweight="bold", color=TITLE_COLOR)
    for center, color in ((superior_center, SUP_COLOR), (inferior_center, INF_COLOR)):
        if center is None:
            continue
        cx, cy = center
        map_axes.add_patch(Rectangle((cx - box_width / 2, cy - box_height / 2), box_width, box_height,
                           lw=1.6, edgecolor=color, facecolor="none", zorder=11))
    image_h, image_w = built_map["img"].shape[:2]
    bar_fraction = (MAP_SCALE_UM / PX_DS_UM) / image_w
    bar_x1 = 0.98
    bar_x0 = bar_x1 - bar_fraction
    map_axes.plot([bar_x0, bar_x1], [-0.028, -0.028], transform=map_axes.transAxes, color="black",
                  lw=4.5, solid_capstyle="butt", zorder=12, clip_on=False)
    # Draw the colorbar
    colorbar_axes = ax_at(column_x + overview_width + colorbar_gap, map_bottom + (map_height - colorbar_height) / 2, colorbar_width, colorbar_height)
    scalar_mappable = cm.ScalarMappable(norm=Normalize(0.0, built_map["vmax"]), cmap=built_map["cmap"])
    colorbar = figure.colorbar(scalar_mappable, cax=colorbar_axes)
    colorbar.set_label("Rod transcripts/µm²", fontsize=10, fontweight="bold", color=TITLE_COLOR)
    colorbar.ax.tick_params(labelsize=8)
    colorbar.outline.set_visible(False)

    # Draw the superior and inferior vignettes
    for center, color, hemisphere_name, vignette_y, top_fraction in [(superior_center, SUP_COLOR, "superior", sup_bottom, SUP_VIG_TOP_FRAC),
                                                                     (inferior_center, INF_COLOR, "inferior", inf_bottom, INF_VIG_TOP_FRAC)]:
        vignette_axes = ax_at(column_x, vignette_y, vignette_width, vignette_height)
        vignette_axes.set_xticks([])
        vignette_axes.set_yticks([])
        if center is not None:
            crop = straighten(built_map["img"], layer_crop, center[0], center[1], half_width, half_height, top_frac=top_fraction)
            vignette_axes.imshow(crop, interpolation="nearest", aspect="equal")
            scalebar(vignette_axes, crop.shape, SCALE_UM)
        for spine in vignette_axes.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
            spine.set_linewidth(1.2)
        if built_map is built_maps[0]:
            figure.text((left_margin - 0.16) / figure_width, (vignette_y + vignette_height / 2) / figure_height, hemisphere_name, rotation=90,
                        ha="center", va="center", fontsize=18, fontweight="bold", color=color)
    column_x += block_width + column_gap

## Add the S and I arrows in the left margin
si_arrow_x = (left_margin - 0.44) / figure_width
for fraction_y, symbol, font_size, color in [(0.93, "↑", 34, "#444"), (0.80, "S", 24, SUP_COLOR),
                                              (0.20, "I", 24, INF_COLOR), (0.07, "↓", 34, "#444")]:
    figure.text(si_arrow_x, (map_bottom + map_height * fraction_y) / figure_height, symbol, ha="center", va="center",
                fontsize=font_size, fontweight="bold", color=color)

## Add the DAPI and 18S channel key
channel_key = [mlines.Line2D([0], [0], marker="s", ls="", markerfacecolor="#c8c8c8",
                             markeredgecolor="none", markersize=12, label="DAPI"),
               mlines.Line2D([0], [0], marker="s", ls="", markerfacecolor="#8a8a8a",
                             markeredgecolor="none", markersize=12, label="18S rRNA")]
channel_legend = figure.legend(handles=channel_key, loc="upper left", bbox_to_anchor=(0.008, 0.995),
                               ncol=1, frameon=False, fontsize=13, handletextpad=0.5, labelspacing=0.5)
for text_label in channel_legend.get_texts():
    text_label.set_fontweight("bold")

## Save the panel
figure.savefig(out_png, dpi=300, facecolor="white")
plt.close(figure)
print("wrote", out_png)
