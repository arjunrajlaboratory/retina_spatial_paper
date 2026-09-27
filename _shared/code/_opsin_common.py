# Shared functions and orientation for the figures using the opsin gradient
from pathlib import Path
import numpy as np
import pandas as pd

# Per cell table of retained cells, keeping the background layer cells
PER_CELL = (Path(__file__).resolve().parent.parent
            / "data_processed/per_cell_gene_counts.parquet")
_PER_CELL_CACHE = None


# Load the cones with their opsin counts and positions
def load_cones(sample):
    global _PER_CELL_CACHE
    if _PER_CELL_CACHE is None:
        _PER_CELL_CACHE = pd.read_parquet(
            PER_CELL,
            columns=["sample_id", "cell_type",
                     "cx_px", "cy_px",
                     "body_Opn1sw", "body_Opn1mw"])
    t = _PER_CELL_CACHE
    return t[(t["sample_id"] == sample)
             & (t["cell_type"] == "cone")].copy()


# Fit the display orientation from the cone positions and the opsin gradient
def fit_gradient(cones):
    cx = cones["cx_px"].to_numpy(dtype=float)
    cy = cones["cy_px"].to_numpy(dtype=float)
    sw = cones["body_Opn1sw"].to_numpy(dtype=float)
    mw = cones["body_Opn1mw"].to_numpy(dtype=float)
    ms = np.log1p(mw) - np.log1p(sw)   # Positive values mark M-dominant dorsal cones

    cx0, cy0 = cx.mean(), cy.mean()
    xc, yc = cx - cx0, cy - cy0

    # Find the PCA principal axis of the cone positions
    cov = np.cov(np.vstack([xc, yc]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = eigvecs[:, -1]  # Axis of largest variance
    # Rotate so PC1 points up in image coordinates
    angle_pc1 = float(np.arctan2(pc1[1], pc1[0]))
    angle_target = float(np.arctan2(-1, 0))  # Points up in image coordinates
    rotation = angle_target - angle_pc1

    # Apply the rotation
    cos_a, sin_a = np.cos(rotation), np.sin(rotation)
    xr = cos_a * xc - sin_a * yc
    yr = sin_a * xc + cos_a * yc

    # Pick the vertical end where M-opsin dominates as superior
    top_mask = yr < 0
    bot_mask = yr > 0
    ms_top = float(ms[top_mask].mean()) if top_mask.any() else 0.0
    ms_bot = float(ms[bot_mask].mean()) if bot_mask.any() else 0.0
    flip_y = bool(ms_top < ms_bot)  # Put the M-opsin side on top

    # Place the retinal mass on the left
    xr_check = xr
    flip_x = bool(np.median(xr_check) > 0)

    return {"centroid": (float(cx0), float(cy0)),
            "rotation_rad": float(rotation),
            "flip_x": flip_x,
            "flip_y": flip_y,
            "ms_score": ms}
