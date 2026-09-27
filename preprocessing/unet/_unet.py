# Define the U-Net model and its training utilities

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def device():
    # Pick the best available compute device, preferring Apple MPS then CUDA
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ConvBlock(nn.Module):
    # Apply two 3x3 conv, batch norm, and ReLU layers in sequence
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    # Define a standard four level U-Net with skip connections
    def __init__(self, c_in=2, c_out=5, base=32):
        super().__init__()
        # Build the encoder path that downsamples while doubling channels
        self.enc1 = ConvBlock(c_in, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        # Define the bottleneck at the lowest resolution
        self.bott = ConvBlock(base * 8, base * 16)
        # Build the decoder path that upsamples and merges skip features
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = ConvBlock(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        # Map the final features to per-class logits
        self.head = nn.Conv2d(base, c_out, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encode the input, halving resolution at each level
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bott(self.pool(e4))
        # Decode, concatenating each encoder skip connection before convolving
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def dice_loss_multiclass(logits, targets, valid=None, eps=1e-6):
    # Compute soft Dice loss averaged over classes, ignoring masked pixels
    n_cls = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    # One-hot encode the targets to channel-first probability format
    targets_oh = F.one_hot(targets.long(), num_classes=n_cls).permute(0, 3, 1, 2).float()
    # Weight every pixel equally when no validity mask is given
    if valid is None:
        valid_f = torch.ones_like(targets_oh[:, :1])
    else:
        valid_f = valid.float().unsqueeze(1)
    # Accumulate per-class intersection and denominator over the batch
    inter = (probs * targets_oh * valid_f).sum(dim=(0, 2, 3))
    denom = (probs * valid_f).sum(dim=(0, 2, 3)) + (targets_oh * valid_f).sum(dim=(0, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


def augment(X, Y):
    # Apply a random 90 degree rotation, flips, and brightness jitter to a patch
    k = int(torch.randint(0, 4, (1,)).item())
    X = torch.rot90(X, k, dims=(-2, -1))
    Y = torch.rot90(Y, k, dims=(-2, -1))
    # Flip horizontally with probability one half
    if torch.rand(1).item() < 0.5:
        X = X.flip(-1)
        Y = Y.flip(-1)
    # Flip vertically with probability one half
    if torch.rand(1).item() < 0.5:
        X = X.flip(-2)
        Y = Y.flip(-2)
    # Scale image brightness only, leaving the label map unchanged
    if torch.rand(1).item() < 0.5:
        X = X * (0.85 + 0.30 * torch.rand(1).item())
        X = X.clamp(0, 1)
    return X, Y


## Set the default training hyperparameters
TRAIN_DEFAULTS = {
    "epochs": 80,
    "batch": 8,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "n_classes": 5,
    "base_channels": 32,
    "log_every": 10,
    "save_path": Path("model.pt"),
    "log_path": Path("model.log.json"),
}


def train(cfg, train_npz, val_npz, remap_target=None):
    # Train a U-Net from the train and validation patch .npz files
    dev = device()
    print(f"device: {dev}")

    # Load the patch arrays into tensors
    tr = np.load(train_npz, allow_pickle=True)
    va = np.load(val_npz,   allow_pickle=True)
    X_tr = torch.from_numpy(tr["X"]).float()
    Y_tr = torch.from_numpy(tr["Y"]).long()
    X_va = torch.from_numpy(va["X"]).float()
    Y_va = torch.from_numpy(va["Y"]).long()
    # Optionally remap label codes before training
    if remap_target is not None:
        Y_tr = remap_target(Y_tr)
        Y_va = remap_target(Y_va)
    print(f"train X={tuple(X_tr.shape)}  val X={tuple(X_va.shape)}")

    # Build the model, optimizer, and cosine learning-rate schedule
    model = UNet(c_in=2, c_out=cfg["n_classes"], base=cfg["base_channels"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    n = X_tr.shape[0]

    log = {"train_loss": [], "val_iou_per_class": [], "val_iou_mean": []}
    best_iou = -1.0
    for ep in range(cfg["epochs"]):
        t0 = time.time()
        model.train()
        # Shuffle the training patches for this epoch
        perm = torch.randperm(n)
        ep_loss = 0.0
        n_batches = 0
        for i in range(0, n, cfg["batch"]):
            # Gather and augment the next mini-batch
            idx = perm[i:i + cfg["batch"]]
            X = X_tr[idx]
            Y = Y_tr[idx]
            X, Y = augment(X, Y)
            X = X.to(dev)
            Y = Y.to(dev)
            logits = model(X)
            # Treat label 0 as unlabeled and shift codes 1..5 down to 0..4
            valid = Y > 0
            Y_eff = (Y - 1).clamp(min=0)
            # Compute cross-entropy over labeled pixels only
            ce = F.cross_entropy(logits, Y_eff, reduction="none")
            ce = (ce * valid.float()).sum() / valid.float().sum().clamp(min=1)
            # Combine cross-entropy with soft Dice loss
            dice = dice_loss_multiclass(logits, Y_eff, valid=valid)
            loss = ce + dice
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batches += 1
        sched.step()
        ep_loss /= max(1, n_batches)
        log["train_loss"].append(ep_loss)

        ## Evaluate per-class IoU on the validation set
        model.eval()
        with torch.no_grad():
            inter = torch.zeros(cfg["n_classes"])
            union = torch.zeros(cfg["n_classes"])
            for i in range(0, X_va.shape[0], cfg["batch"]):
                X = X_va[i:i + cfg["batch"]].to(dev)
                Y = Y_va[i:i + cfg["batch"]]
                logits = model(X)
                pred = logits.argmax(dim=1).cpu()
                valid = Y > 0
                Y_eff = (Y - 1).clamp(min=0)
                # Accumulate intersection and union per class over labeled pixels
                for c in range(cfg["n_classes"]):
                    p = (pred == c) & valid
                    t = (Y_eff == c) & valid
                    inter[c] += (p & t).sum()
                    union[c] += (p | t).sum()
            iou = (inter / union.clamp(min=1)).numpy().tolist()
            # Average IoU across classes that appear in the validation set
            mean_iou = float(np.nanmean([i for i in iou if i > 0]))
        log["val_iou_per_class"].append(iou)
        log["val_iou_mean"].append(mean_iou)

        dt = time.time() - t0
        print(f"ep {ep:3d}  loss={ep_loss:.4f}  val_iou={[round(v,3) for v in iou]}  mean={mean_iou:.3f}  ({dt:.1f}s)", flush=True)
        # Checkpoint the model whenever validation mean IoU improves
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save({"state_dict": model.state_dict(),
                        "cfg": cfg, "epoch": ep, "val_iou": iou},
                       cfg["save_path"])

    # Write the per-epoch training log
    Path(cfg["log_path"]).write_text(json.dumps(log, indent=2))
    print(f"best val mean IoU: {best_iou:.3f} -> {cfg['save_path']}")
    return best_iou
