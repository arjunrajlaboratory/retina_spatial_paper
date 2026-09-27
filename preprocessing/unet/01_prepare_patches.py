# Sample 256 x 256 training patches from the annotated layer regions
import os
import json
import sys
from pathlib import Path
# Make the shared preprocessing config importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _preprocessing_config as C

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.transform import downscale_local_mean

from _paths import findpath_dapi, findpath_18s, SAMPLE_TO_ROI

DS = C.DS
# Set the patch side length and how many jittered patches to draw per polygon
PATCH = 256
N_PER_POLY = 8
JITTER_PX_DS = 32
LABELS = C.UNET_LABELS
# Invert the sample-to-ROI map so each ROI resolves back to its sample
ROI_TO_SAMPLE = {v: k for k, v in SAMPLE_TO_ROI.items()}
OUT = C.UNET_PATCHES
OUT.mkdir(parents=True, exist_ok=True)

# Assign label IDs 1..5 to the retinal layers plus background
LAYER_ORDER = ["ONL", "INL", "IPL", "GCL", "bg"]
LAYER_TO_ID = {n: i + 1 for i, n in enumerate(LAYER_ORDER)}

## Hold out one annotated retina for validation
VAL_ROI = int(os.environ.get("VAL_ROI", "9"))


def stretch(img, lo=1.0, hi=99.5):
    # Rescale intensities to 0..1 between the given percentiles
    p_lo, p_hi = np.percentile(img, [lo, hi])
    # Return a blank image when the intensity range is degenerate
    if p_hi <= p_lo:
        return np.zeros_like(img, dtype=np.float32)
    out = (img.astype(np.float32) - p_lo) / (p_hi - p_lo)
    return np.clip(out, 0, 1)


def load_stack(roi, shape_ds):
    # Load the DAPI and 18S channels for an ROI, aligned to the label shape
    sample = ROI_TO_SAMPLE[roi]
    dapi = tifffile.imread(findpath_dapi(sample)).astype(np.float32)
    s18 = tifffile.imread(findpath_18s(sample)).astype(np.float32)
    # Downsample each channel to the ds8 grid
    dapi_ds = downscale_local_mean(dapi, (DS, DS))
    s18_ds = downscale_local_mean(s18, (DS, DS))
    # Place the downsampled channels into arrays matching the label shape
    out_d = np.zeros(shape_ds, dtype=np.float32)
    out_s = np.zeros(shape_ds, dtype=np.float32)
    h = min(shape_ds[0], dapi_ds.shape[0], s18_ds.shape[0])
    w = min(shape_ds[1], dapi_ds.shape[1], s18_ds.shape[1])
    out_d[:h, :w] = dapi_ds[:h, :w]
    out_s[:h, :w] = s18_ds[:h, :w]
    return np.stack([stretch(out_d), stretch(out_s)], axis=0)


def sample_polygon_patches(stack, label, roi, rng):
    # Draw jittered patches centered on each annotated layer polygon
    H, W = label.shape
    patches_X, patches_Y, patches_meta = [], [], []
    for lay_name, lay_id in LAYER_TO_ID.items():
        # Select the pixels belonging to this layer, skipping if absent
        lab = (label == lay_id)
        if not lab.any():
            continue
        # Split the layer mask into connected polygon components
        cc, n_cc = ndi.label(lab)
        for cid in range(1, n_cc + 1):
            # Take the centroid of each component as the patch anchor
            ys, xs = np.where(cc == cid)
            cy, cx = int(ys.mean()), int(xs.mean())
            for _ in range(N_PER_POLY):
                # Jitter the anchor and clamp the patch inside the image bounds
                jy = cy + int(rng.integers(-JITTER_PX_DS, JITTER_PX_DS + 1))
                jx = cx + int(rng.integers(-JITTER_PX_DS, JITTER_PX_DS + 1))
                y0 = max(0, min(H - PATCH, jy - PATCH // 2))
                x0 = max(0, min(W - PATCH, jx - PATCH // 2))
                p_X = stack[:, y0:y0 + PATCH, x0:x0 + PATCH]
                p_Y = label[y0:y0 + PATCH, x0:x0 + PATCH]
                # Drop patches that fall short of the full size at the edge
                if p_X.shape[1] != PATCH or p_X.shape[2] != PATCH:
                    continue
                patches_X.append(p_X.astype(np.float32))
                patches_Y.append(p_Y.astype(np.uint8))
                patches_meta.append((roi, lay_name))
    return patches_X, patches_Y, patches_meta


## Build the training and validation patch sets
idx_path = LABELS / "INDEX.json"
if not idx_path.exists():
    print(f"missing {idx_path}")
    raise SystemExit(1)
index = json.loads(idx_path.read_text())

# Keep only ROIs that carry at least one annotated layer polygon
labeled_rois = [int(r) for r, s in index.items()
                if any(s[t]["n_polys"] > 0 for t in LAYER_ORDER)]
if not labeled_rois:
    print("no labeled ROIs in training-label index")
    raise SystemExit(2)
print(f"labeled ROIs: {labeled_rois}; val held out = ROI{VAL_ROI}")

rng = np.random.default_rng(0)
train_X, train_Y, train_meta = [], [], []
val_X, val_Y, val_meta = [], [], []
for roi in labeled_rois:
    # Load the layer label map and the matching image stack
    lab_tif = LABELS / f"roi{roi}_layer_polygons_ds8.tif"
    label = tifffile.imread(lab_tif)
    stack = load_stack(roi, label.shape)
    Xs, Ys, Ms = sample_polygon_patches(stack, label, roi, rng)
    # Route the held-out ROI to validation and all others to training
    if roi == VAL_ROI:
        val_X.extend(Xs)
        val_Y.extend(Ys)
        val_meta.extend(Ms)
    else:
        train_X.extend(Xs)
        train_Y.extend(Ys)
        train_meta.extend(Ms)
    print(f"  roi{roi:2d}: {len(Xs)} patches")


def stack_save(name, Xs, Ys, Ms):
    # Stack the collected patches and write them to a compressed .npz
    if not Xs:
        print(f"[{name}] empty")
        return
    X = np.stack(Xs, axis=0)
    Y = np.stack(Ys, axis=0)
    M = np.array(Ms, dtype=object)
    np.savez_compressed(OUT / f"{name}.npz", X=X, Y=Y, meta=M, val_roi=VAL_ROI)
    print(f"[{name}] X={X.shape} Y={Y.shape}  -> {OUT / f'{name}.npz'}")


stack_save("train", train_X, train_Y, train_meta)
stack_save("val", val_X, val_Y, val_meta)
