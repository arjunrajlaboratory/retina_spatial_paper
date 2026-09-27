# Train the five class layer U-Net (ONL, INL, IPL, GCL, background)
# Annotation codes 1..5 map to model indices 0..4; code 0 is masked

from pathlib import Path
import sys
# Make the shared preprocessing config importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _preprocessing_config as C

# Make the local U-Net module importable
sys.path.insert(0, str(Path(__file__).parent))
from _unet import TRAIN_DEFAULTS, train

PATCHES = C.UNET_PATCHES
MODELS = C.UNET_MODELS
MODELS.mkdir(parents=True, exist_ok=True)

# Override the defaults with the settings used for the layer model
cfg = {**TRAIN_DEFAULTS,
       "epochs": 80,
       "batch": 16,
       "n_classes": 5,
       "base_channels": 16,
       "save_path": MODELS / "layer_unet.pt",
       "log_path": MODELS / "layer_unet.log.json"}
train(cfg, PATCHES / "train.npz", PATCHES / "val.npz")
