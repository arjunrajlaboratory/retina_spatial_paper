# Assign a per-cell superior-inferior axis from the smoothed sigma 120 µm cone-opsin field
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SIGMA_UM = 120.0     # Gaussian smoothing length for the cone-opsin field in µm
KMAX_CONES = 500     # Maximum number of cones summed per cell


# Compute the sigma 120 µm Gaussian-smoothed cone-opsin S/I field aligned to df row order
def smoothed_opsin_field(df):
    out = pd.Series(np.nan, index=df.index)
    for s, g in df.groupby("sample_id"):
        cones = g[g["cell_type"] == "cone"]
        if len(cones) < 10:
            raise ValueError(f"{s}: too few cones ({len(cones)}) for a cone-opsin field")
        ms = (np.log1p(cones["body_Opn1mw"].to_numpy(float))
              - np.log1p(cones["body_Opn1sw"].to_numpy(float)))
        tree = cKDTree(np.column_stack([cones["cx_um"].to_numpy(float),
                                        cones["cy_um"].to_numpy(float)]))
        k = min(KMAX_CONES, len(ms))
        dist, idx = tree.query(np.column_stack([g["cx_um"].to_numpy(float),
                                                g["cy_um"].to_numpy(float)]), k=k)
        if k == 1:
            out.loc[g.index] = ms[np.atleast_1d(idx)]
        else:
            w = np.exp(-(dist ** 2) / (2.0 * SIGMA_UM ** 2))
            out.loc[g.index] = (w * ms[idx]).sum(axis=1) / w.sum(axis=1)
    if out.isna().any():
        raise ValueError("cell without a smoothed opsin field")
    return out.to_numpy()


# Split each retina at the median smoothed value and classify higher values as superior
def superior_mask(df):
    field = smoothed_opsin_field(df)
    samples = df["sample_id"].to_numpy()
    out = np.zeros(len(df), dtype=bool)
    for s in np.unique(samples):
        m = samples == s
        out[m] = field[m] > np.median(field[m])
    return out
