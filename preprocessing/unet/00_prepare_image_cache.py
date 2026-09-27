# Cache downsampled, contrast-stretched DAPI and 18S images for the U-Net
from pathlib import Path
import sys
# Make the shared preprocessing config importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _preprocessing_config as C

import numpy as np
import tifffile
from skimage.transform import downscale_local_mean

from _paths import findpath_dapi, findpath_18s, SAMPLE_TO_ROI

DS = C.DS
OUT = C.UNET_CACHE
OUT.mkdir(parents=True, exist_ok=True)
# Invert the sample-to-ROI map so each ROI resolves back to its sample
ROI_TO_SAMPLE = {v: k for k, v in SAMPLE_TO_ROI.items()}


def stretch(img, lo=1.0, hi=99.5):
    # Rescale intensities to 0..1 between the given percentiles
    p_lo, p_hi = np.percentile(img, [lo, hi])
    # Return a blank image when the intensity range is degenerate
    if p_hi <= p_lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - p_lo) / (p_hi - p_lo), 0, 1)


# Build one cached stack per ROI
for roi in range(1, 11):
    sample = ROI_TO_SAMPLE[roi]
    out_path = OUT / f"roi{roi}_dapi18s_ds8.npy"
    # Skip ROIs that are already cached
    if out_path.exists():
        print(f"[roi{roi}] cached -> skip")
        continue
    # Load the full-resolution DAPI and 18S images
    try:
        dapi = tifffile.imread(findpath_dapi(sample)).astype(np.float32)
        s18 = tifffile.imread(findpath_18s(sample)).astype(np.float32)
    except FileNotFoundError as e:
        print(f"[roi{roi}] missing: {e}")
        continue
    # Downsample each channel by block-mean pooling to the ds8 grid
    dapi_ds = downscale_local_mean(dapi, (DS, DS))
    s18_ds = downscale_local_mean(s18, (DS, DS))
    # Crop both channels to their common size before stacking
    h = min(dapi_ds.shape[0], s18_ds.shape[0])
    w = min(dapi_ds.shape[1], s18_ds.shape[1])
    stack = np.stack([stretch(dapi_ds[:h, :w]), stretch(s18_ds[:h, :w])], axis=0)
    np.save(out_path, stack.astype(np.float32))
    print(f"[roi{roi}] -> {out_path}  shape={stack.shape}")
