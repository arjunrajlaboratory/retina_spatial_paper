# Define the shared paths and constants for paper preprocessing
from pathlib import Path
import os
import sys
import pandas as pd

# Resolve the paper root and make the shared code importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_shared/code"))
from _paths import (
    SEGMENTATION_EXPORT_ROOT, SAMPLE_TO_ROI as _SAMPLE_TO_ROI,
    UNET_PRED_DIR as _UNET_PRED_DIR,
    DS as _DS, PX_UM as _PX_UM, PX_DS_UM as _PX_DS_UM,
)

# Allow the data, prepared, and output roots to be overridden by environment variables
DATA_ROOT = Path(os.environ.get("DATA_DIR", ROOT))
PREPARED = Path(os.environ.get("RETINA_PREPARED_ROOT", ROOT))
OUTPUT = Path(os.environ.get("RETINA_PREPROCESSING_OUTPUT", ROOT / "preprocessing/outputs"))

# Resolve a path from an environment variable, falling back to the given default
def data_path(variable, default):
    return Path(os.environ.get(variable, default)).expanduser().resolve()

# Resolve the input and output locations used across the preprocessing scripts
SEGMENTATION_DIR = SEGMENTATION_EXPORT_ROOT
TYPED_DIR = data_path("RETINA_TYPED_DIR", OUTPUT / "per_cell_typed")
SAMPLES_CSV = data_path("RETINA_SAMPLES_CSV", DATA_ROOT / "metadata/samples.csv")
BATCH_CSV = data_path("RETINA_BATCH_CSV", DATA_ROOT / "metadata/sample_to_batch.csv")
ARC_TABLE_DIR = data_path("RETINA_ARC_TABLE_DIR", PREPARED / "_shared/data_raw/full_thickness_tables")
EXCLUDE_FINAL_DIR = data_path("RETINA_EXCLUDE_FINAL_DIR", PREPARED / "_shared/data/exclude_final_rasters")
OUT_TAB = ARC_TABLE_DIR
OUT_FIG = OUTPUT / "panels"
UNET_LABELS = data_path("RETINA_UNET_LABELS", OUTPUT / "unet/labels_5class")
UNET_CACHE = data_path("RETINA_UNET_CACHE", OUTPUT / "unet/ds8_cache")
UNET_PATCHES = OUTPUT / "unet/patches"
UNET_MODELS = data_path("RETINA_UNET_MODELS", OUTPUT / "unet/models")
UNET_OUTPUT = _UNET_PRED_DIR

# Calibration constants
MASK_UM_PER_PX = 0.831
DS = _DS
PX_UM = _PX_UM
PX_DS_UM = _PX_DS_UM
# List every sample in the canonical genotype and age order
SAMPLE_ORDER = ["WT_P21_rep1", "WT_P21_rep2", "WT_P64_rep1", "WT_P64_rep2",
                "LCA5_P21_rep1", "LCA5_P21_rep2", "LCA5_P30_rep1", "LCA5_P30_rep2",
                "LCA5_P64_rep1", "LCA5_P64_rep2"]
ALL_SAMPLES = SAMPLE_ORDER
SAMPLE_TO_ROI = _SAMPLE_TO_ROI
# Map each retinal layer to its integer raster code
LAYER_CODE = {"ONL": 1, "INL": 2, "IPL": 3, "GCL": 4}
LAYER_NAMES = list(LAYER_CODE)
# Drop the eGFP and tdTomato reporter channels from gene analyses
DROP_GENES = {"eGFP", "tdTomato"}
# Set the default arc-column width and lateral shift
DEFAULT_COL_WIDTH_UM = 25.0
DEFAULT_SHIFT_FRAC = 0.0


# Strip the replicate suffix to get the condition label
def sample_to_condition(sample):
    return sample.rsplit("_", 1)[0]


# Apply the paper-wide matplotlib style
def set_paper_style():
    import matplotlib as mpl
    mpl.rcParams.update({"font.family": "Arial", "font.weight": "bold",
                        "axes.labelweight": "bold", "axes.titleweight": "bold",
                        "axes.spines.top": False, "axes.spines.right": False,
                        "savefig.dpi": 200, "figure.dpi": 120})


# Load the sample metadata and return it in the canonical sample order
def load_samples():
    s = pd.read_csv(SAMPLES_CSV)
    # Fail if the metadata sample ids do not match the expected set
    if set(s["sample_id"]) != set(SAMPLE_ORDER):
        raise ValueError(
            f"samples.csv sample_ids {sorted(s['sample_id'])} != expected {sorted(SAMPLE_ORDER)}"
        )
    s["sample_id"] = pd.Categorical(s["sample_id"], categories=SAMPLE_ORDER, ordered=True)
    return s.sort_values("sample_id").reset_index(drop=True)


# Load the gene channel definition and verify it is identical across samples
def gene_channels():
    found = {}
    # Collect every per-sample gene channel table that exists
    for s in SAMPLE_ORDER:
        f = SEGMENTATION_DIR / s / "gene_channels.csv"
        if f.exists():
            found[s] = pd.read_csv(f)
    if not found:
        raise FileNotFoundError(
            f"no gene_channels.csv under {SEGMENTATION_DIR}"
        )
    # Check every sample against the first one as the reference panel
    ref_key = next(iter(found))
    ref = found[ref_key]
    for s, df in found.items():
        if not df.equals(ref):
            raise ValueError(
                f"gene panel definition differs between {ref_key} and {s}"
            )
    return ref
