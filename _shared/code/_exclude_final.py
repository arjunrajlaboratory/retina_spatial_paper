# Apply the excluded region filter to cells and to layer images
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import findpath_exclude_final, PX_DS_UM


# Load the excluded region mask for a sample, or None when none was drawn
def load_exclude_mask(sample):
    path = findpath_exclude_final(sample)
    if not path.exists():
        return None
    return tifffile.imread(path).astype(bool)


# Flag which cell positions fall inside the excluded region
def points_excluded(sample, cx_um, cy_um):
    cx_um = np.asarray(cx_um, dtype=float)
    cy_um = np.asarray(cy_um, dtype=float)
    out = np.zeros(cx_um.shape[0], dtype=bool)
    mask = load_exclude_mask(sample)
    if mask is None:
        return out
    h, w = mask.shape
    rows = np.round(cy_um / PX_DS_UM).astype(int)
    cols = np.round(cx_um / PX_DS_UM).astype(int)
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    out[inside] = mask[rows[inside], cols[inside]]
    return out


# Drop cells whose position falls inside the excluded region
def retain_cells(sample, cells):
    excluded = points_excluded(sample, cells["cx_um"].to_numpy(), cells["cy_um"].to_numpy())
    return cells.loc[~excluded].copy(), int(excluded.sum())


# Clear the excluded region from a tissue mask and a layer image
def apply_to_layer(sample, tissue, layer):
    mask = load_exclude_mask(sample)
    if mask is None:
        return tissue, layer, 0
    H, W = layer.shape
    eh, ew = mask.shape
    hh, ww = min(H, eh), min(W, ew)
    crop = mask[:hh, :ww]
    tissue = tissue.copy()
    layer = layer.copy()
    tissue[:hh, :ww] &= ~crop
    layer[:hh, :ww] = np.where(crop, 0, layer[:hh, :ww])
    return tissue, layer, int(crop.sum())
