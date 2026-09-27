# Straightened vignette crop geometry shared by the Fig4b full section box and the Fig4b_2 vignette

## Load packages
import numpy as np
from scipy.ndimage import rotate as nd_rotate

from _paths import PX_DS_UM

## Define the vignette window and the perpendicular anchoring parameters
WIN_W_UM = 480.0                                 # Vignette width along the ONL band
WIN_H_UM = 240.0                                 # Vignette height across the retinal layers
MAX_PERP_SHIFT_UM = 80                           # Largest perpendicular move of the crop center
MIN_ONL_FRAC = 0.6                               # Require most of the ONL to stay in frame after the shift
TARGET_ONL_TOP_FRAC = 0.12                       # Seat the ONL this far below the top of the crop
GCL_BOTTOM_MARGIN_ROWS = 6                       # Keep the GCL clear of the bottom edge
REQUIRED_LAYER_LABELS = frozenset({1, 2, 3, 4})  # The ONL, INL, IPL, and GCL must all appear


# Return the angle in degrees that lays the ONL band's long axis along x
def onl_pca_angle(layer_arr):
    ys, xs = np.where(layer_arr == 1)
    if len(ys) < 50:
        return 0.0
    pts = np.column_stack([xs, ys]).astype(np.float64)
    pts -= pts.mean(axis=0, keepdims=True)
    _eigvals, eigvecs = np.linalg.eigh(np.cov(pts.T))
    pc1 = eigvecs[:, -1]
    if pc1[0] < 0:
        pc1 = -pc1
    return float(np.degrees(np.arctan2(pc1[1], pc1[0])))


# Crop a square around the center, straighten the ONL, and return the final rectangle of layer codes
def crop_layer_rect(layer, cx, cy):
    H, W = layer.shape
    diag_um = float(np.hypot(WIN_W_UM, WIN_H_UM))
    half_px = int(round(diag_um / 2 / PX_DS_UM)) + 8
    cxi, cyi = int(round(cx)), int(round(cy))
    sub = layer[max(0, cyi - half_px):min(H, cyi + half_px), max(0, cxi - half_px):min(W, cxi + half_px)]
    rot = nd_rotate(sub.astype(np.int16), onl_pca_angle(sub), reshape=True, order=0, cval=0)
    ys_o, _ = np.where(rot == 1)
    ys_g, _ = np.where(rot == 4)
    if len(ys_o) > 30 and len(ys_g) > 30 and ys_o.mean() > ys_g.mean():
        rot = rot[::-1]
    Hn, Wn = rot.shape
    half_w = int(round(WIN_W_UM / PX_DS_UM)) // 2
    half_h = int(round(WIN_H_UM / PX_DS_UM)) // 2
    cxr, cyr = Wn // 2, Hn // 2
    return rot[max(0, cyr - half_h):min(Hn, cyr + half_h), max(0, cxr - half_w):min(Wn, cxr + half_w)]


# Shift a center perpendicular to the local ONL band by a distance in microns
def perp_shifted_center(layer, cx, cy, shift_um):
    if not shift_um:
        return cx, cy
    H, W = layer.shape
    diag_um = float(np.hypot(WIN_W_UM, WIN_H_UM))
    half_px = int(round(diag_um / 2 / PX_DS_UM)) + 8
    cxi, cyi = int(round(cx)), int(round(cy))
    sub = layer[max(0, cyi - half_px):min(H, cyi + half_px), max(0, cxi - half_px):min(W, cxi + half_px)]
    a = np.radians(onl_pca_angle(sub))
    d_px = shift_um / PX_DS_UM
    return cx - np.sin(a) * d_px, cy + np.cos(a) * d_px


# Nudge the crop center perpendicular to the ONL so the full ONL to GCL span sits in frame
def anchor_onl_top_center(layer, cx, cy, tag=""):
    lim = int(round(MAX_PERP_SHIFT_UM))
    scan = []
    for shift in range(-lim, lim + 1, 5):
        ccx, ccy = perp_shifted_center(layer, cx, cy, shift)
        rect = crop_layer_rect(layer, ccx, ccy)
        full = REQUIRED_LAYER_LABELS.issubset({int(v) for v in np.unique(rect)})
        Hr = rect.shape[0]
        onl = np.where((rect == 1).any(axis=1))[0]
        gcl = np.where((rect == 4).any(axis=1))[0]
        scan.append(dict(shift=float(shift), ccx=ccx, ccy=ccy, Hr=Hr, full=full,
                         n_onl=int((rect == 1).sum()), n_gcl=int((rect == 4).sum()),
                         onl_top=int(onl.min()) if onl.size else -1,
                         gcl_bot=int(gcl.max()) if gcl.size else -1))
    n_onl_max = max(s["n_onl"] for s in scan)
    ok = [s for s in scan if s["full"] and s["n_onl"] >= MIN_ONL_FRAC * n_onl_max and s["n_gcl"] > 0]
    unclipped = [s for s in ok if s["gcl_bot"] <= s["Hr"] - GCL_BOTTOM_MARGIN_ROWS]
    pool = unclipped or ok
    if not pool:
        raise RuntimeError(f"{tag}: no perpendicular shift within +/-{lim} um shows a full ONL to GCL span")
    best = min(pool, key=lambda s: abs(s["onl_top"] - TARGET_ONL_TOP_FRAC * s["Hr"]))
    return best["ccx"], best["ccy"], best["shift"], best["onl_top"], best["gcl_bot"], best["Hr"]


# Resolve the straightened vignette crop as an ONL-anchored center and a straightening angle
# The Fig4b_2 vignette and the Fig4b full-section box both build from this single geometry
def vignette_crop_geometry(layer, cx, cy, tag=""):
    ccx, ccy, _shift, _onl_top, _gcl_bot, _hr = anchor_onl_top_center(layer, cx, cy, tag=tag)
    H, W = layer.shape
    diag_um = float(np.hypot(WIN_W_UM, WIN_H_UM))
    half_px = int(round(diag_um / 2 / PX_DS_UM)) + 8
    cxi, cyi = int(round(ccx)), int(round(ccy))
    sub = layer[max(0, cyi - half_px):min(H, cyi + half_px), max(0, cxi - half_px):min(W, cxi + half_px)]
    return ccx, ccy, onl_pca_angle(sub)
