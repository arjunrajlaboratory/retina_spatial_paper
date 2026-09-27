# Orient images and masks for retina display panels
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import (
    rotate as nd_rotate, binary_closing,
    binary_fill_holes, gaussian_filter,
)
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects

import _paths

ROOT = Path(__file__).resolve().parents[2]
EXPORT_ROOT = _paths.SEGMENTATION_EXPORT_ROOT
PER = ROOT / "preprocessing" / "outputs" / "per_cell_typed"

DS = _paths.DS
PX_UM = _paths.PX_UM
PX_DS_UM = _paths.PX_DS_UM
CROP_PAD_UM = 80

SAMPLE_TO_ROI = _paths.SAMPLE_TO_ROI

# Optional per-sample display orientation overrides
SAMPLE_ORIENT_OVERRIDE = {}


## Orientation

# Derive the display tissue mask from the DAPI image
def _derive_tissue_mask(dapi):
    smoothed = gaussian_filter(dapi.astype(np.float32), sigma=8.0 / PX_DS_UM)
    thr = threshold_otsu(smoothed)
    mask = smoothed > thr * 0.5
    mask = binary_closing(mask, iterations=int(round(25 / PX_DS_UM)))
    mask = remove_small_objects(mask, min_size=int(round((300 / PX_DS_UM) ** 2)))
    mask = binary_fill_holes(mask)
    return mask


# Fit the display orientation from tissue shape and the cone opsin signal
def _fit_orientation(tissue_mask, cells, gene_df):
    ys, xs = np.where(tissue_mask)
    if len(ys) < 100:
        return {"rotation_deg": 0.0, "flip_x": False, "flip_y": False}

    # Subsample the tissue pixels for speed
    if len(ys) > 50000:
        idx = np.random.default_rng(0).choice(len(ys), 50000, replace=False)
        ys = ys[idx]
        xs = xs[idx]
    xc = xs - xs.mean()
    yc = ys - ys.mean()

    cov = np.cov(np.vstack([xc, yc]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = eigvecs[:, -1]
    angle_pc1 = float(np.arctan2(pc1[1], pc1[0]))
    angle_target = float(np.arctan2(-1, 0))   # Point vertically up in image coordinates
    rotation_rad = angle_target - angle_pc1
    rotation_deg = float(np.degrees(rotation_rad))

    # Decide the vertical flip from the opsin gradient across the rotated cones
    flip_y = False
    cones = cells[cells["final_label"] == "cone"]
    if len(cones) >= 30:
        g = gene_df.set_index("label")[["body_Opn1sw", "body_Opn1mw"]]
        sub = cones.join(g, on="label", how="inner")
        if len(sub) >= 30:
            cxc = sub["cx_px"].to_numpy(dtype=float) - xs.mean()
            cyc = sub["cy_px"].to_numpy(dtype=float) - ys.mean()
            cos_a, sin_a = np.cos(rotation_rad), np.sin(rotation_rad)
            yr_c = sin_a * cxc + cos_a * cyc
            ms = (np.log1p(sub["body_Opn1mw"].to_numpy(dtype=float))
                  - np.log1p(sub["body_Opn1sw"].to_numpy(dtype=float)))
            top_mask = yr_c < 0
            bot_mask = yr_c > 0
            ms_top = float(ms[top_mask].mean()) if top_mask.any() else 0.0
            ms_bot = float(ms[bot_mask].mean()) if bot_mask.any() else 0.0
            flip_y = bool(ms_top < ms_bot)

    # Flip horizontally when the tissue mass sits on the right
    cos_a, sin_a = np.cos(rotation_rad), np.sin(rotation_rad)
    xr_t = cos_a * xc - sin_a * yc
    flip_x = bool(np.median(xr_t) > 0)

    return {"rotation_deg": rotation_deg,
            "flip_x": flip_x, "flip_y": flip_y}


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


## Load and render

def load_panel(sample):
    roi = SAMPLE_TO_ROI[sample]
    cache_path = ROOT / "_shared" / "data_raw" / "ds8_cache" / f"roi{roi}_dapi18s_ds8.npz"
    try:
        dapi = tifffile.imread(_paths.findpath_dapi(sample))[::DS, ::DS]
        # Load the 18S channel from the downsampled cache
        s18 = np.load(cache_path)["arr"][1]
    except ValueError:
        # Fall back to the cached DAPI and 18S images when the source image cannot be decoded
        arr = np.load(cache_path)["arr"]
        dapi = arr[0]
        s18 = arr[1]
    layer = tifffile.imread(_paths.findpath_unet_pred(sample, "layer"))
    Hd, Wd = dapi.shape
    Hl, Wl = layer.shape
    H_pad = max(Hd, Hl)
    W_pad = max(Wd, Wl)
    layer_full = np.zeros((H_pad, W_pad), dtype=layer.dtype)
    layer_full[:Hl, :Wl] = layer
    dapi_full = np.zeros((H_pad, W_pad), dtype=dapi.dtype) + dapi.min()
    dapi_full[:Hd, :Wd] = dapi
    # Pad the 18S image onto the same canvas
    s18_full = np.zeros((H_pad, W_pad), dtype=np.float32)
    Hs18, Ws18 = s18.shape
    s18_full[:min(Hs18, H_pad), :min(Ws18, W_pad)] = s18[:H_pad, :W_pad]
    dapi, layer, s18 = dapi_full, layer_full, s18_full
    H, W = dapi.shape

    typed = pd.read_parquet(PER / f"{sample}_typed.parquet")
    _excl_path = _paths.findpath_exclude_final(sample)
    raw_excl = tifffile.imread(_excl_path) > 0 if _excl_path.exists() else np.zeros((H, W), dtype=bool)
    excl_mask = np.zeros((H, W), dtype=bool)
    excl_mask[:min(H, raw_excl.shape[0]), :min(W, raw_excl.shape[1])] = raw_excl[:H, :W]
    cells = typed[typed["keep_for_analysis"]].copy()
    cell_r = np.clip(np.rint(cells["cy_px"].to_numpy() / DS).astype(int), 0, H - 1)
    cell_c = np.clip(np.rint(cells["cx_px"].to_numpy() / DS).astype(int), 0, W - 1)
    cells = cells.loc[~excl_mask[cell_r, cell_c]].copy()
    cells["xr"] = cells["cx_px"] / DS
    cells["yr"] = cells["cy_px"] / DS

    gene_df = pd.read_parquet(EXPORT_ROOT / sample / f"cells_{sample}.parquet",
                              columns=["label", "body_Opn1sw", "body_Opn1mw"])

    # Drop excluded pixels from the display tissue mask
    tissue_full = _derive_tissue_mask(dapi) & (~excl_mask)

    orient = _fit_orientation(tissue_full, cells, gene_df)
    if sample in SAMPLE_ORIENT_OVERRIDE:
        orient = {**orient, **SAMPLE_ORIENT_OVERRIDE[sample]}
    angle = orient["rotation_deg"]
    flipped_v = orient["flip_y"]
    flipped_h = orient["flip_x"]

    orig_H, orig_W = H, W
    cx_orig_save, cy_orig_save = W / 2.0, H / 2.0
    rot_Hn, rot_Wn = H, W   # Default frame size when there is no rotation
    dx_shift_save, dy_shift_save = 0.0, 0.0

    if abs(angle) > 0.1:
        dapi_r = nd_rotate(dapi, angle, reshape=True, order=1,
                           mode="constant", cval=float(dapi.min()))
        s18_r = nd_rotate(s18, angle, reshape=True, order=1,
                          mode="constant", cval=0.0)
        layer_r = nd_rotate(layer, angle, reshape=True, order=0,
                            mode="constant", cval=0)
        excl_r = nd_rotate(excl_mask.astype(np.uint8), angle, reshape=True,
                            order=0, mode="constant", cval=0).astype(bool)
        tissue_r = nd_rotate(tissue_full.astype(np.uint8), angle, reshape=True,
                              order=0, mode="constant", cval=0).astype(bool)
        cy_orig = H / 2.0
        cx_orig = W / 2.0
        Hn, Wn = layer_r.shape
        dx_shift = Wn / 2.0 - cx_orig
        dy_shift = Hn / 2.0 - cy_orig
        xr, yr = _rotate_xy(cells["xr"].to_numpy(), cells["yr"].to_numpy(),
                            cx_orig, cy_orig, angle)
        cells = cells.assign(xr=xr + dx_shift, yr=yr + dy_shift)
        dapi, layer, excl_mask, tissue_full, s18 = (
            dapi_r, layer_r, excl_r, tissue_r, s18_r)
        H, W = tissue_full.shape
        cx_orig_save, cy_orig_save = cx_orig, cy_orig
        rot_Hn, rot_Wn = Hn, Wn
        dx_shift_save, dy_shift_save = dx_shift, dy_shift

    if flipped_v:
        layer, cells, dapi, excl_mask = _flip_vertical(
            layer, cells, dapi, excl_mask)
        tissue_full = tissue_full[::-1, :].copy()
        s18 = s18[::-1, :].copy()
    if flipped_h:
        layer, cells, dapi, excl_mask = _flip_horizontal(
            layer, cells, dapi, excl_mask)
        tissue_full = tissue_full[:, ::-1].copy()
        s18 = s18[:, ::-1].copy()

    # Crop to the tissue bounding box with padding
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


# Project micron coordinates into the oriented and cropped panel frame
def project_xy_to_panel(panel, x_um, y_um):
    x = np.asarray(x_um, dtype=float) / PX_DS_UM
    y = np.asarray(y_um, dtype=float) / PX_DS_UM
    if abs(panel.get("angle", 0.0)) > 0.1:
        x, y = _rotate_xy(x, y, panel["cx_orig"], panel["cy_orig"], panel["angle"])
        x = x + panel["dx_shift"]
        y = y + panel["dy_shift"]
    # Use the frame size after rotation
    Hn, Wn = panel["rot_Hn"], panel["rot_Wn"]
    if panel.get("flipped_v"):
        y = Hn - 1 - y
    if panel.get("flipped_h"):
        x = Wn - 1 - x
    x = x - panel["crop_x0"]
    y = y - panel["crop_y0"]
    return x, y


# Reorient a native mask into the panel frame by padding, rotating, flipping, then cropping
def _orient_to_panel(raster, panel, order=0):
    Hd, Wd = panel["orig_H"], panel["orig_W"]
    Hr, Wr = raster.shape
    pad = np.zeros((Hd, Wd), dtype=raster.dtype)
    pad[:min(Hr, Hd), :min(Wr, Wd)] = raster[:Hd, :Wd]
    angle = panel["angle"]
    if abs(angle) > 0.1:
        rot = ndi.rotate(pad, angle, reshape=True, order=order,
                         mode="constant", cval=0)
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


# Build the tissue mask from the layer U-Net prediction, keeping any predicted class
def load_unet_tissue_oriented(panel):
    raw = tifffile.imread(_paths.findpath_unet_pred(_paths.ROI_TO_SAMPLE[panel['roi']], "layer")) > 0
    lbl, n = ndi.label(raw)
    if n > 1:
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        tot = int(raw.sum())
        largest = int(sizes.max())
        print(f"    UNet tissue roi{panel['roi']}: {n} components, {tot} px; "
              f"largest {100 * largest / tot:.2f}%, all components kept")
    return _orient_to_panel(raw.astype(np.uint8), panel, order=0).astype(bool)


# Load the excluded region masks for the ROI and orient it to the panel
def load_exclude_final_oriented(panel):
    p = _paths.findpath_exclude_final(_paths.ROI_TO_SAMPLE[panel['roi']])
    raw = tifffile.imread(p) > 0 if p.exists() else np.zeros((panel['orig_H'], panel['orig_W']), dtype=bool)
    return _orient_to_panel(raw.astype(np.uint8), panel, order=0).astype(bool)


# Orient the cell body label mask onto the panel
def load_cellbody_mask_oriented(sample, panel):
    native = tifffile.imread(EXPORT_ROOT / sample / "cellbody_masks.tif")
    return _orient_to_panel(native[::DS, ::DS], panel, order=0)


# Return the angle in degrees that aligns the ONL band PC1 with the x axis
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


# Compute the geometry that straightens a zoom so the ONL band runs horizontal with apical up
def zoom_straighten_geom(zoom_layer, final_h_px, final_w_px, onl_code=1, gcl_code=4,
                         onl_top_frac=0.25, extra_rot_deg=0.0):
    rot_deg = _onl_pca_angle(zoom_layer) + extra_rot_deg
    if abs(rot_deg) > 0.1:
        rot_layer = ndi.rotate(zoom_layer.astype(np.int16), rot_deg, axes=(0, 1),
                               reshape=True, order=0, mode="constant", cval=0)
    else:
        rot_layer = zoom_layer.astype(np.int16)
    Hr, Wr = rot_layer.shape
    ys_o, _ = np.where(rot_layer == onl_code)
    ys_g, _ = np.where(rot_layer == gcl_code)
    flip = len(ys_o) > 30 and len(ys_g) > 30 and ys_o.mean() > ys_g.mean()
    if flip:
        rot_layer = rot_layer[::-1]
    side_h = int(min(final_h_px, Hr))
    side_w = int(min(final_w_px, Wr))
    x0 = max(0, (Wr - side_w) // 2)
    ys_top, _ = np.where(rot_layer == onl_code)
    if ys_top.size > 30:
        onl_top = int(np.percentile(ys_top, 5))
        y0 = max(0, min(Hr - side_h, onl_top - int(round(onl_top_frac * side_h))))
    else:
        y0 = max(0, (Hr - side_h) // 2)
    return dict(rot_deg=rot_deg, Hr=Hr, Wr=Wr, flip=flip,
                x0=x0, y0=y0, side_h=side_h, side_w=side_w)


# Rotate, flip, and crop one array into the frame from zoom_straighten_geom
def zoom_straighten_apply(arr, geom, order=1, cval=0.0):
    rot_deg = geom["rot_deg"]
    if abs(rot_deg) > 0.1:
        rot = ndi.rotate(arr, rot_deg, axes=(0, 1), reshape=True, order=order,
                         mode="constant", cval=cval)
    else:
        rot = arr
    if geom["flip"]:
        rot = rot[::-1]
    return rot[geom["y0"]:geom["y0"] + geom["side_h"], geom["x0"]:geom["x0"] + geom["side_w"]]
