# Render spatial maps of the rod injury response score
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _retina_panel as _pfh
import _opsin_common
_load_cones_opsin = _opsin_common.load_cones
_fit_gradient_opsin = _opsin_common.fit_gradient

ROOT = Path(__file__).resolve().parents[2]   # analysis/retina_paper_final
# Injury response scores for typed rods
SCORE_PARQ = ROOT / "figure_4" / "data_raw_manifest" / "rod_injury_response_score.parquet"

CONDITIONS = ["LCA5_P21", "LCA5_P30"]
DISPLAY_REP = {"LCA5_P21": "LCA5_P21_rep2",
                 "LCA5_P30": "LCA5_P30_rep2"}

SCORE_CMAP = mpl.cm.get_cmap("viridis").copy()
SCORE_CMAP.set_bad(SCORE_CMAP(0.0))


# Convert the cone opsin gradient orientation into the display convention
def s1c_orient_override(sample):
    o = _fit_gradient_opsin(_load_cones_opsin(sample))
    return {"rotation_deg": float(-np.degrees(o["rotation_rad"])),
            "flip_x": bool(o["flip_x"]), "flip_y": bool(o["flip_y"])}


# Install the display orientation for each shown retina
def install_s1c_orientations():
    for sample in DISPLAY_REP.values():
        _pfh.SAMPLE_ORIENT_OVERRIDE[sample] = s1c_orient_override(sample)


# Load the injury response scores for the displayed retinas
def load_scores():
    d = pd.read_parquet(SCORE_PARQ)
    return d[d["sample"].isin(list(DISPLAY_REP.values()))].copy()


# Build a grayscale tissue image with rods colored by injury response score
def composite_rods(panel, body_mask, rod_scores, vmin, vmax):
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

    H, W = body_mask.shape

    def _fit(m):
        mm = np.zeros((H, W), dtype=bool)
        Hm, Wm = m.shape
        mm[:min(H, Hm), :min(W, Wm)] = m[:H, :W]
        return mm

    not_excl = np.ones((H, W), dtype=bool)
    excl = panel.get("excl_mask")
    if excl is not None:
        not_excl &= ~_fit(excl)
    # Whiten the excluded pixels only
    base = np.where(not_excl[..., None], base, 1.0)

    max_label = int(body_mask.max())
    lut = np.zeros((max_label + 1, 4), dtype=np.float32)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    for lab, val in rod_scores.items():
        if 0 < lab <= max_label:
            lut[lab] = SCORE_CMAP(norm(float(val)))
    color = lut[body_mask]
    has_cell = (color[..., 3] > 0) & not_excl
    out = base.copy()
    out[has_cell] = color[has_cell, :3]
    return out
