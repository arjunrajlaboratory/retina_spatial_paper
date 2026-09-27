# Shared data locations and image calibration constants
# _local_paths.py overrides DATA_DIR when it is present
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

DS = 8
PX_UM = 0.103937
PX_DS_UM = PX_UM * DS

SAMPLE_TO_ROI = {
    "LCA5_P30_rep2": 1, "LCA5_P30_rep1": 2, "LCA5_P21_rep2": 3,
    "WT_P64_rep2": 4, "WT_P21_rep1": 5, "LCA5_P64_rep1": 6,
    "LCA5_P21_rep1": 7, "WT_P64_rep1": 8, "WT_P21_rep2": 9, "LCA5_P64_rep2": 10,
}
ROI_TO_SAMPLE = {v: k for k, v in SAMPLE_TO_ROI.items()}

try:
    import _local_paths
    _have_local = True
except ImportError:
    _have_local = False

_USE_LOCAL = _have_local and not os.environ.get("DATA_DIR")

if _USE_LOCAL:
    RAW_SEQFISH_ROOT = Path(_local_paths.RAW_SEQFISH_ROOT)
    RAW_18S_ROOT = Path(_local_paths.RAW_18S_ROOT)
    SEGMENTATION_EXPORT_ROOT = Path(_local_paths.SEGMENTATION_EXPORT_ROOT)
    EXCLUDE_FINAL_DIR = _REPO_ROOT / "_shared/data/exclude_final_rasters"
    UNET_MODELS_DIR = _REPO_ROOT / "preprocessing/outputs/unet/models"
    UNET_PRED_DIR = _REPO_ROOT / "_shared/data_raw/unet_pred"
else:
    _data_dir = os.environ.get("DATA_DIR")
    if not _data_dir:
        raise RuntimeError(
            "no _shared/code/_local_paths.py and DATA_DIR is not set; "
            "set DATA_DIR to the Zenodo data root (contains raw/, segmentation/, etc.)"
        )
    _data_dir = Path(_data_dir)
    RAW_SEQFISH_ROOT = _data_dir / "raw"
    RAW_18S_ROOT = _data_dir / "raw"
    SEGMENTATION_EXPORT_ROOT = _data_dir / "segmentation"
    EXCLUDE_FINAL_DIR = _data_dir / "exclude_rasters"
    UNET_MODELS_DIR = _data_dir / "unet/models"
    UNET_PRED_DIR = _data_dir / "unet/predictions"


def findpath_dapi(sample):
    roi = SAMPLE_TO_ROI[sample]
    if _USE_LOCAL:
        return RAW_SEQFISH_ROOT / f"roi_{roi}" / f"roi_{roi}_DAPI.tiff"
    return RAW_SEQFISH_ROOT / sample / "DAPI.tiff"


def findpath_transcripts(sample):
    roi = SAMPLE_TO_ROI[sample]
    if _USE_LOCAL:
        return RAW_SEQFISH_ROOT / f"roi_{roi}" / f"roi_{roi}_TranscriptList.csv"
    return RAW_SEQFISH_ROOT / sample / "TranscriptList.csv"


def findpath_18s(sample):
    roi = SAMPLE_TO_ROI[sample]
    if _USE_LOCAL:
        return RAW_18S_ROOT / f"roi{roi}_18S.tiff"
    return RAW_18S_ROOT / sample / "18S.tiff"


def findpath_exclude_final(sample):
    roi = SAMPLE_TO_ROI[sample]
    if _USE_LOCAL:
        return EXCLUDE_FINAL_DIR / f"roi{roi}_exclude_final_ds8.tif"
    return EXCLUDE_FINAL_DIR / f"{sample}_exclude_final.tif"


def findpath_unet_pred(sample, kind):
    roi = SAMPLE_TO_ROI[sample]
    if _USE_LOCAL:
        return UNET_PRED_DIR / f"roi{roi}_{kind}_ds8.tif"
    return UNET_PRED_DIR / f"{sample}_{kind}.tif"
