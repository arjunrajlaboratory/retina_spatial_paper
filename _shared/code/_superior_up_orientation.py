# Orient panels by the cone-opsin gradient for the main and supplementary spatial figures
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _retina_panel as _pfh
import _opsin_common
_load_cones_opsin = _opsin_common.load_cones
_fit_gradient_opsin = _opsin_common.fit_gradient

# Representative replicate per condition matching Fig1c
DISPLAY_REP = {
    "WT_P21":   "WT_P21_rep2",
    "WT_P64":   "WT_P64_rep1",
    "LCA5_P21": "LCA5_P21_rep2",
    "LCA5_P30": "LCA5_P30_rep2",
    "LCA5_P64": "LCA5_P64_rep1",
}


# Express the cone-opsin gradient orientation in the load_panel rotation and flip convention
def s1c_orient_override(sample):
    o = _fit_gradient_opsin(_load_cones_opsin(sample))
    return {
        "rotation_deg": float(-np.degrees(o["rotation_rad"])),
        "flip_x": bool(o["flip_x"]),
        "flip_y": bool(o["flip_y"]),
    }


# Point the load_panel orientation override at the opsin-gradient fit for every displayed replicate
def install_s1c_orientations():
    for sample in DISPLAY_REP.values():
        _pfh.SAMPLE_ORIENT_OVERRIDE[sample] = s1c_orient_override(sample)
