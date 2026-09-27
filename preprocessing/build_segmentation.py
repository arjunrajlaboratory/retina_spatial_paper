# Segment nuclei and cell bodies from DAPI and 18S images with Cellpose-SAM
# This script requires a GPU and Cellpose 4.1.1 or later
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
from skimage import exposure
from skimage.measure import regionprops_table
from skimage.segmentation import watershed

# Load the shared config, sample map, and image file locations
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _preprocessing_config as C
from _paths import findpath_dapi, findpath_18s

SAMPLE_TO_ROI = C.SAMPLE_TO_ROI
SEGMENTATION_EXPORT_ROOT = C.SEGMENTATION_DIR
PX_UM = C.PX_UM

from cellpose import models

## Image calibration
UM_PER_PX = PX_UM

## Nucleus segmentation parameters
NUCLEI_DIAMETER_PX = 57
NUCLEI_FLOW_THRESHOLD = 0.4
NUCLEI_CELLPROB_THRESHOLD = 0.0

## Cell body segmentation parameters
CELLBODY_DIAMETER_PX = None
CELLBODY_FLOW_THRESHOLD = 0.0
CELLBODY_CELLPROB_THRESHOLD = 0.0
CELLBODY_MIN_SIZE = 15

## CLAHE normalization for 18S
CLAHE_TILE_UM = 30.0
CLAHE_CLIP_LIMIT = 2.0

ALL_SAMPLES = list(SAMPLE_TO_ROI)


## Normalize the 18S channel with CLAHE
def normalize_18s(s18):
    # Size the CLAHE tile grid so each tile spans about CLAHE_TILE_UM
    tile_px = max(1, int(round(CLAHE_TILE_UM / UM_PER_PX)))
    gx = max(1, round(s18.shape[1] / tile_px))
    gy = max(1, round(s18.shape[0] / tile_px))
    # Apply CLAHE and rescale the result to the unit range
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(gx, gy))
    eq = clahe.apply(s18.astype(np.uint16))
    return exposure.rescale_intensity(eq.astype(np.float32), out_range=(0.0, 1.0))


## Match each Cellpose cell body to its nucleus
def reconcile_to_nuclei(cell_raw, nuc_masks, nuc_df, cyto):
    # Clip the nucleus centroids to valid pixel indices
    ry = nuc_df.cy_px.to_numpy().astype(int).clip(0, nuc_masks.shape[0] - 1)
    rx = nuc_df.cx_px.to_numpy().astype(int).clip(0, nuc_masks.shape[1] - 1)
    nlab = nuc_df.label.to_numpy().astype(np.int64)

    # Find the Cellpose object under each nucleus centroid
    cps_at_nuc = cell_raw[ry, rx]
    # Count how many nuclei fall inside each Cellpose object
    nuc_per_cps = np.bincount(
        cps_at_nuc[cps_at_nuc > 0],
        minlength=int(cell_raw.max()) + 1
    )

    # Relabel cell bodies containing one nucleus
    relabel = np.zeros(int(cell_raw.max()) + 1, dtype=np.uint32)
    for L, c in zip(nlab, cps_at_nuc):
        if c > 0 and nuc_per_cps[c] == 1:
            relabel[c] = L
    cellbody = relabel[cell_raw]

    # Split cell bodies containing multiple nuclei by watershed
    multi_cps = np.where(nuc_per_cps >= 2)[0]
    if len(multi_cps):
        mask_multi = np.isin(cell_raw, multi_cps)
        markers = np.where(mask_multi, nuc_masks, 0).astype(np.int32)
        ws = watershed(-cyto, markers=markers, mask=mask_multi)
        cellbody[mask_multi] = ws[mask_multi].astype(np.uint32)

    # Use the nucleus mask when no cell body was found
    no_body = nlab[cps_at_nuc == 0]  # 0 means no Cellpose body under the nucleus
    if len(no_body):
        cellbody[np.isin(nuc_masks, no_body)] = nuc_masks[np.isin(nuc_masks, no_body)].astype(np.uint32)

    return cellbody.astype(np.uint32)


## Segment and export the nuclei and cell bodies for one sample
def process_sample(sample):
    outdir = SEGMENTATION_EXPORT_ROOT / sample
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{sample}]", flush=True)

    # Read the DAPI and 18S images for this sample
    dapi = tifffile.imread(findpath_dapi(sample))
    s18 = tifffile.imread(findpath_18s(sample))
    print(f"  DAPI {dapi.shape} {dapi.dtype}  |  18S {s18.shape} {s18.dtype}", flush=True)

    ## Step 1: segment nuclei from DAPI
    model = models.CellposeModel(gpu=True)
    t0 = time.time()
    nuc_masks, _, _ = model.eval(
        dapi,
        diameter=NUCLEI_DIAMETER_PX,
        flow_threshold=NUCLEI_FLOW_THRESHOLD,
        cellprob_threshold=NUCLEI_CELLPROB_THRESHOLD,
    )
    n_nuc = int(nuc_masks.max())
    print(f"  nuclei: {n_nuc} in {time.time() - t0:.0f}s", flush=True)

    tifffile.imwrite(outdir / "nuclei_masks.tif", nuc_masks.astype(np.uint32), compression="zlib")

    # Measure nucleus centroids and areas and convert them to micrometers
    props = regionprops_table(nuc_masks, properties=["label", "centroid", "area"])
    nuc_df = pd.DataFrame(props).rename(columns={
        "centroid-0": "cy_px", "centroid-1": "cx_px", "area": "area_px",
    })
    nuc_df["cx_um"] = nuc_df.cx_px * UM_PER_PX
    nuc_df["cy_um"] = nuc_df.cy_px * UM_PER_PX
    nuc_df["area_um2"] = nuc_df.area_px * UM_PER_PX ** 2
    nuc_df.to_parquet(outdir / "nuclei.parquet")

    ## Step 2: segment cell bodies from 18S + DAPI
    # Stack the CLAHE-normalized 18S and normalized DAPI as the two input channels
    cyto = normalize_18s(s18)
    dapi_norm = exposure.rescale_intensity(dapi.astype(np.float32), out_range=(0.0, 1.0))
    img = np.stack([cyto, dapi_norm], axis=-1)

    t0 = time.time()
    cell_raw, _, _ = model.eval(
        img,
        channel_axis=2,
        diameter=CELLBODY_DIAMETER_PX,
        flow_threshold=CELLBODY_FLOW_THRESHOLD,
        cellprob_threshold=CELLBODY_CELLPROB_THRESHOLD,
        min_size=CELLBODY_MIN_SIZE,
        normalize=True,
    )
    cell_raw = cell_raw.astype(np.int64)
    n_raw = int(cell_raw.max())
    print(f"  cell bodies: {n_raw} raw cpsam objects in {time.time() - t0:.0f}s", flush=True)

    ## Step 3: reconcile cell bodies to nuclei
    cellbody = reconcile_to_nuclei(cell_raw, nuc_masks, nuc_df, cyto)
    tifffile.imwrite(outdir / "cellbody_masks.tif", cellbody, compression="zlib")

    # Report the median cell-body area in square micrometers
    area_px = np.bincount(cellbody.ravel(), minlength=n_nuc + 1)
    median_area = float(np.median(area_px[area_px > 0])) * UM_PER_PX ** 2
    print(f"  saved nuclei_masks.tif + cellbody_masks.tif  (median body {median_area:.0f} um2)", flush=True)


## Process the samples given on the command line, or all samples by default
samples = sys.argv[1:] if len(sys.argv) > 1 else ALL_SAMPLES
for sample in samples:
    process_sample(sample)
print("\ndone")
