# Apply the five class layer U-Net to all retinal sections
# Average overlapping 256 x 256 tiles before assigning each pixel a layer
from pathlib import Path
import sys

# Make the shared preprocessing config importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _preprocessing_config as C

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

# Make the local U-Net module importable
sys.path.insert(0, str(Path(__file__).parent))
from _unet import UNet, device

DS = C.DS
# Set the tile size and overlap that define the sliding-window stride
PATCH = 256
OVERLAP = 64
STRIDE = PATCH - 2 * OVERLAP
MODELS = C.UNET_MODELS
OUT = C.UNET_OUTPUT
OUT.mkdir(parents=True, exist_ok=True)
CACHE = C.UNET_CACHE
ROIS = list(range(1, 11))
LAYER_NAMES = ["ONL", "INL", "IPL", "GCL"]


## Load the cached DAPI and 18S image stack for one ROI
def load_stack(roi):
    p = CACHE / f"roi{roi}_dapi18s_ds8.npy"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run unet/00_prepare_image_cache.py first.")
    return np.load(p)


## Predict the layer map by averaging softmax probabilities over overlapping tiles
def tile_predict(model, stack, n_classes, dev):
    ch, H, W = stack.shape
    # Reflect-pad so the sliding window covers every pixel
    pad_h = (PATCH - H % STRIDE) % STRIDE
    pad_w = (PATCH - W % STRIDE) % STRIDE
    padded = np.pad(stack, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    Hp, Wp = padded.shape[1], padded.shape[2]
    # Accumulate summed probabilities and the overlap count per pixel
    prob = np.zeros((n_classes, Hp, Wp), dtype=np.float32)
    cnt = np.zeros((Hp, Wp), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        # Slide the window across the padded image and predict each tile
        for y in range(0, Hp - PATCH + 1, STRIDE):
            for x in range(0, Wp - PATCH + 1, STRIDE):
                tile = padded[:, y:y + PATCH, x:x + PATCH]
                t = torch.from_numpy(tile).unsqueeze(0).to(dev)
                p = F.softmax(model(t), dim=1)[0].cpu().numpy()
                prob[:, y:y + PATCH, x:x + PATCH] += p
                cnt[y:y + PATCH, x:x + PATCH] += 1
    # Average the overlapping predictions and crop back to the original size
    prob /= cnt.clip(min=1)
    return prob[:, :H, :W]


## Predict all ROIs
if __name__ == "__main__":
    dev = device()
    print(f"device: {dev}")

    # Require the trained layer model before predicting
    layer_path = MODELS / "layer_unet.pt"
    if not layer_path.exists():
        print("layer_unet.pt not found; run 03_train_layer_unet.py first")
        raise SystemExit(1)

    # Build the model and load the trained weights
    model = UNet(c_in=2, c_out=5, base=16).to(dev)
    model.load_state_dict(torch.load(layer_path, map_location=dev)["state_dict"])

    for roi in ROIS:
        # Skip ROIs whose cached image stack is missing
        try:
            stack = load_stack(roi)
        except FileNotFoundError as e:
            print(f"[roi{roi}] {e} -> skip")
            continue

        # Predict per-pixel probabilities and take the argmax layer, codes 1..5
        prob = tile_predict(model, stack, 5, dev)
        layer = (prob.argmax(axis=0) + 1).astype(np.uint8)
        layer[layer == 5] = 0  # 5 is the background class, remapped to 0

        # Write the layer map and report the pixel count per layer
        tifffile.imwrite(OUT / f"roi{roi}_layer_ds8.tif", layer, compression="zlib")
        print(f"[roi{roi}] " + "  ".join(f"{n}={(layer == i + 1).sum()}" for i, n in enumerate(LAYER_NAMES)))
