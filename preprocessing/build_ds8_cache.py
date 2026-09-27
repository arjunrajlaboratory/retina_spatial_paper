# Build the downsampled DAPI and 18S image cache used for display
# Output: _shared/data_raw/ds8_cache/roi{N}_dapi18s_ds8.npz
# Each npz contains a single array "arr" of shape (2, H, W) float32:
#   channel 0 = DAPI, channel 1 = 18S, both downsampled eightfold
import sys
from pathlib import Path

import numpy as np
import tifffile

# Load the shared sample to ROI map and the image file locations
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _preprocessing_config as C
from _paths import SAMPLE_TO_ROI, DS, findpath_dapi, findpath_18s

# Create the output cache directory
ROOT = C.ROOT
OUT = ROOT / "_shared" / "data_raw" / "ds8_cache"
OUT.mkdir(parents=True, exist_ok=True)
# Process the samples given on the command line, or all samples by default
samples = sys.argv[1:] if len(sys.argv) > 1 else list(SAMPLE_TO_ROI)
for sample in samples:
    roi = SAMPLE_TO_ROI[sample]
    print(f"[{sample}] roi {roi}", flush=True)
    # Read the DAPI and 18S images and downsample eightfold via ds8 striding
    dapi = tifffile.imread(findpath_dapi(sample))[::DS, ::DS].astype(np.float32)
    s18 = tifffile.imread(findpath_18s(sample))[::DS, ::DS].astype(np.float32)
    # Stack DAPI and 18S into a two-channel array and save it
    arr = np.stack([dapi, s18], axis=0)
    out_path = OUT / f"roi{roi}_dapi18s_ds8.npz"
    np.savez_compressed(out_path, arr=arr)
    print(f"  wrote {out_path.name}  shape={arr.shape}")
print("\ndone")
