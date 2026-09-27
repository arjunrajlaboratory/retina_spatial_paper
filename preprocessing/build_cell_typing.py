# Assign a retinal cell type to every segmented cell from marker gene modules
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared/code"))
from _paths import SEGMENTATION_EXPORT_ROOT, findpath_unet_pred
import _exclude_final as excl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _preprocessing_config as C

EXPORT = SEGMENTATION_EXPORT_ROOT
OUT_DIR = C.TYPED_DIR
SUMMARY_DIR = C.OUTPUT
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

DS = C.DS
UM_PER_PX_DS = C.MASK_UM_PER_PX
TIE_TOL = 0.15  # Standardized-score gap below which the top two calls count as a tie

CONDITIONS = ["WT_P21", "WT_P64", "LCA5_P21", "LCA5_P30", "LCA5_P64"]
COND_REPS = {
    "WT_P21":   ["WT_P21_rep1", "WT_P21_rep2"],
    "WT_P64":   ["WT_P64_rep1", "WT_P64_rep2"],
    "LCA5_P21": ["LCA5_P21_rep1", "LCA5_P21_rep2"],
    "LCA5_P30": ["LCA5_P30_rep1", "LCA5_P30_rep2"],
    "LCA5_P64": ["LCA5_P64_rep1", "LCA5_P64_rep2"],
}
SAMPLE_TO_ROI = C.SAMPLE_TO_ROI

## Define the marker gene set for each cell type module
MARKERS = {
    "rod":           ["Nrl", "Nr2e3", "Pde6b"],
    "cone":          ["Arr3", "Opn1mw", "Opn1sw"],
    "bp_vsx2":       ["Vsx2"],
    "bp_vsx2_prkca": ["Vsx2", "Prkca"],
    "bp_vsx2_grm6":  ["Vsx2", "Grm6"],
    "bp_vsx2_vsx1":  ["Vsx2", "Vsx1"],
    "bp_vsx2_scgn":  ["Vsx2", "Scgn"],
    "gaba":          ["Gad1", "Slc32a1"],
    "gly":           ["Slc6a9"],
    "hc":            ["Onecut2", "Calb1"],
    "rgc":           ["Rbpms", "Pou4f2", "Nefh"],
    "muller":        ["Sox9", "Slc1a3", "Glul", "Aqp4"],
    "microglia":     ["Csf1r", "Cx3cr1", "C1qa"],
    "vascular":      ["Cldn5", "Pecam1", "Rgs5"],
    "rpe":           ["Rpe65"],
}

# List the cell types that compete for each cell's final assignment
COMPETING_TYPES = ["rod", "cone", "bipolar", "amacrine_gaba", "amacrine_gly",
                   "horizontal", "rgc", "rpe", "muller", "microglia", "vascular"]

# Map each cell type to the retinal layers where it is expected to reside
EXPECTED_LAYER = {
    "rod": {"ONL"}, "cone": {"ONL"},
    "bipolar": {"INL"}, "horizontal": {"INL"}, "muller": {"INL"},
    "amacrine_gaba": {"INL", "GCL"}, "amacrine_gly": {"INL", "GCL"},
    "rgc": {"GCL"},
    "rpe": {"bg", "ONL"},
    "microglia": None, "vascular": None,
}


def module_score(df, genes):
    """Sum the log1p body transcript counts of a module's marker genes per cell"""
    # Keep only the marker genes that exist as body count columns
    cols = [f"body_{g}" for g in genes if f"body_{g}" in df.columns]
    if not cols:
        return np.zeros(len(df), dtype=np.float32)
    return np.log1p(df[cols].to_numpy(np.float32)).sum(axis=1).astype(np.float32)


def otsu_bright(score, fit_mask):
    """Split cells into dim and bright for a module using an Otsu threshold"""
    # Fit the threshold on positive scores among the included cells only
    pos = score[fit_mask]
    pos = pos[pos > 0]
    if pos.size < 30:
        raise ValueError(f"cannot Otsu threshold a module with only {pos.size} positive cells")
    thr = float(threshold_otsu(pos))
    if thr <= 0:
        raise ValueError(f"Otsu returned a non-positive threshold ({thr:.4g})")
    return thr, score > thr


def zscore_clean(x, fit_mask):
    """Standardize a score to mean and standard deviation of the included cells"""
    pop = x[fit_mask]
    mu, sd = float(np.nanmean(pop)), float(np.nanstd(pop))
    if sd < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mu) / sd).astype(np.float32)


def dist_to_ONL_um(cells, roi, *, sample=None):
    """Return each cell's distance in µm to the nearest ONL pixel"""
    sample = sample or next(s for s, r in C.SAMPLE_TO_ROI.items() if r == roi)
    layer = tifffile.imread(findpath_unet_pred(sample, "layer"))
    H, W = layer.shape
    onl_mask = (layer == 1)
    if not onl_mask.any():
        print(f"  [warn] roi{roi}: layer==1 empty; using zero-dist fallback")
        return np.full(len(cells), np.inf, dtype=np.float32)
    # Distance transform of every pixel to the ONL, in downsampled pixels
    dist_ds = ndi.distance_transform_edt(~onl_mask).astype(np.float32)
    # Map each cell's centroid onto the downsampled layer grid
    r_ds = np.clip(np.rint(cells["cy_px"].to_numpy() / DS).astype(np.int64), 0, H - 1)
    c_ds = np.clip(np.rint(cells["cx_px"].to_numpy() / DS).astype(np.int64), 0, W - 1)
    return (dist_ds[r_ds, c_ds] * UM_PER_PX_DS).astype(np.float32)


LAYER_NAMES = ["bg", "ONL", "INL", "IPL", "GCL"]


def compute_layer_fractions(sample):
    """Return the fraction of each cell body mask that falls in every retinal layer"""
    masks = tifffile.imread(EXPORT / sample / "cellbody_masks.tif")
    layer = tifffile.imread(findpath_unet_pred(sample, "layer"))
    H, W = layer.shape
    # Downsample the masks to the layer grid and flatten both to per-pixel arrays
    masks_ds = masks[::DS, ::DS][:H, :W]
    flat_lab = masks_ds.ravel()
    flat_cls = layer.ravel()
    n_labels = int(flat_lab.max()) + 1
    # Keep only pixels that belong to a cell body
    cell_mask = flat_lab > 0
    lab_f = flat_lab[cell_mask]
    cls_f = flat_cls[cell_mask]
    # Count total pixels per cell body
    totals = np.bincount(lab_f, minlength=n_labels).astype(np.float64)
    # Count pixels per cell body in each layer
    fracs = {}
    for code, name in enumerate(LAYER_NAMES):
        fracs[f"p_{name}"] = np.bincount(lab_f[cls_f == code], minlength=n_labels).astype(np.float64)
    # Convert per-layer pixel counts to fractions of the cell body
    for name in LAYER_NAMES:
        safe = np.where(totals > 0, totals, 1.0)
        fracs[f"p_{name}"] = fracs[f"p_{name}"] / safe
    present = np.where(totals > 0)[0]
    argmax_arr = np.array(LAYER_NAMES)
    # Pick the layer holding the largest fraction of each cell body
    best = np.stack([fracs[f"p_{n}"] for n in LAYER_NAMES], axis=1).argmax(axis=1)
    rows = []
    for lab in present:
        row = {"label": int(lab), "argmax_layer": argmax_arr[best[lab]]}
        for n in LAYER_NAMES:
            row[f"p_{n}"] = float(fracs[f"p_{n}"][lab])
        rows.append(row)
    return pd.DataFrame(rows)


def process_sample(sample, batch_map):
    """Type every cell in one sample and write its per-cell typing table"""
    roi = SAMPLE_TO_ROI[sample]
    raw = pd.read_parquet(EXPORT / sample / f"cells_{sample}.parquet")

    # Drop cells in the excluded region before any typing
    raw, n_excluded = excl.retain_cells(sample, raw)

    # Keep the geometry columns plus every body transcript count column
    keep_cols = ["label", "cx_px", "cy_px", "cx_um", "cy_um", "area_um2"]
    blocked = {"body_source"}
    df = raw[keep_cols + [c for c in raw.columns
                          if c.startswith("body_") and c not in blocked]].copy()

    # Attach each cell's per-layer fractions and its dominant layer
    layer_fracs = compute_layer_fractions(sample)
    df = df.merge(layer_fracs, on="label", how="left", validate="one_to_one")
    for n in LAYER_NAMES:
        df[f"p_{n}"] = df[f"p_{n}"].fillna(0.0)
    df["argmax_layer"] = df["argmax_layer"].fillna("bg")

    n = len(df)
    # All retained cells enter threshold and standardization fits
    clean = np.ones(n, dtype=bool)
    df["dist_to_ONL_um"] = dist_to_ONL_um(df, roi)

    ## Compute the marker module score for every cell type
    mods = {}
    for k, genes in MARKERS.items():
        mods[k] = module_score(df, genes)

    ## Mark bright cells per module with an Otsu threshold
    bright = {}
    for k, m in mods.items():
        _, br = otsu_bright(m, clean)
        bright[k] = br

    ## Standardize module scores across included cells
    z_scores = {}
    for k, m in mods.items():
        z_scores[k] = zscore_clean(m, clean)

    ## Build the passes and z tables for the 11 competing types
    passes = {}
    z_compete = {}

    passes["rod"] = bright["rod"]
    z_compete["rod"] = z_scores["rod"]

    passes["cone"] = bright["cone"]
    z_compete["cone"] = z_scores["cone"]

    # Call bipolar if any Vsx2-based module is bright, and take the best of those scores
    bp_keys = [k for k in MARKERS if k.startswith("bp_")]
    passes["bipolar"] = np.any(np.stack([bright[k] for k in bp_keys], axis=1), axis=1)
    z_compete["bipolar"] = np.maximum.reduce([z_scores[k] for k in bp_keys])

    passes["amacrine_gaba"] = bright["gaba"]
    z_compete["amacrine_gaba"] = z_scores["gaba"]

    passes["amacrine_gly"] = bright["gly"]
    z_compete["amacrine_gly"] = z_scores["gly"]

    passes["horizontal"] = bright["hc"]
    z_compete["horizontal"] = z_scores["hc"]

    passes["rgc"] = bright["rgc"]
    z_compete["rgc"] = z_scores["rgc"]

    passes["rpe"] = bright["rpe"]
    z_compete["rpe"] = z_scores["rpe"]

    passes["muller"] = bright["muller"]
    z_compete["muller"] = z_scores["muller"]

    # Call microglia only when the module is bright and Ccr2 is not, excluding infiltrating monocytes
    ccr2_score = (np.log1p(df["body_Ccr2"].to_numpy(np.float32))
                  if "body_Ccr2" in df.columns else np.zeros(n, dtype=np.float32))
    _, ccr2_bright = otsu_bright(ccr2_score, clean)
    passes["microglia"] = bright["microglia"] & (~ccr2_bright)
    z_compete["microglia"] = z_scores["microglia"]

    passes["vascular"] = bright["vascular"]
    z_compete["vascular"] = z_scores["vascular"]

    ## Assign the passing cell type with the highest standardized score
    pass_stack = np.stack([passes[c] for c in COMPETING_TYPES], axis=1)
    z_stack = np.stack([z_compete[c] for c in COMPETING_TYPES], axis=1)
    # Restrict the competition to types the cell passed
    z_stack_masked = np.where(pass_stack, z_stack, -np.inf)

    # Take the winning type and its score for each cell
    n_pass = pass_stack.sum(axis=1)
    top_idx = np.argmax(z_stack_masked, axis=1)
    top_z = z_stack_masked[np.arange(n), top_idx]

    # Flag cells whose runner-up scores within TIE_TOL of the winner
    z2 = z_stack_masked.copy()
    z2[np.arange(n), top_idx] = -np.inf
    second_z = np.max(z2, axis=1)
    second_z = np.where(np.isfinite(second_z), second_z, -np.inf)
    close_call = (n_pass >= 2) & np.isfinite(second_z) & ((top_z - second_z) < TIE_TOL)

    ## Assign a final label to every cell, defaulting to unassigned
    final = np.array(["unassigned"] * n, dtype=object)
    path = np.array([""] * n, dtype=object)

    # Label cells that passed at least one type with the competition winner
    has_call = n_pass >= 1
    top_label = np.array(COMPETING_TYPES, dtype=object)[top_idx]
    final[has_call] = top_label[has_call]
    path[has_call] = "competition"
    path[has_call & close_call] = "competition_close"

    ## Record whether each cell sits in a layer expected for its type
    argmax_layer = df["argmax_layer"].to_numpy()
    in_expected = np.zeros(n, dtype=bool)
    for ct, layers in EXPECTED_LAYER.items():
        m = final == ct
        if layers is None:
            in_expected[m] = True
        else:
            in_expected[m] = np.isin(argmax_layer[m], list(layers))

    ## Drop cells that passed no marker module and record why
    drop = final == "unassigned"
    final_save = np.where(drop, "unassigned", final)
    drop_reason = np.full(n, "", dtype=object)
    drop_reason[drop] = "unassigned_no_call"

    ## Assemble the per-cell output table
    out = df[["label", "cx_px", "cy_px", "cx_um", "cy_um", "area_um2",
              "p_ONL", "p_INL", "p_IPL", "p_GCL", "p_bg", "argmax_layer",
              "dist_to_ONL_um"]].copy()
    out["sample"] = sample
    out["rep"] = "rep1" if sample.endswith("_rep1") else "rep2"
    out["condition"] = sample.rsplit("_", 1)[0]
    out["batch"] = batch_map[sample]

    # Store each module's score, its bright flag, and its standardized score
    for k, m in mods.items():
        out[f"mod_{k}"] = m
        if k in bright:
            out[f"bright_{k}"] = bright[k]
    for k in z_compete:
        out[f"z_{k}"] = z_compete[k]

    out["final_label"] = final_save
    out["final_label_all"] = final
    out["decision_path"] = path
    out["drop_reason"] = drop_reason
    out["keep_for_analysis"] = ~drop
    out["in_expected_layer"] = in_expected
    out["close_call"] = close_call
    out["ccr2_bright"] = ccr2_bright

    out_path = OUT_DIR / f"{sample}_typed.parquet"
    out.to_parquet(out_path, index=False)

    # Tally the kept, dropped, and excluded cell counts for the summary
    counts = pd.Series(final_save[~drop]).value_counts().to_dict()
    counts["DROPPED"] = int(drop.sum())
    counts["KEPT"] = int((~drop).sum())
    counts["TOTAL"] = int(len(out))
    counts["EXCLUDED_REGION"] = int(n_excluded)
    counts["sample"] = sample
    return pd.DataFrame([counts])


if __name__ == "__main__":
    # Map each sample to its imaging batch
    batch_csv = C.BATCH_CSV
    batch_map = pd.read_csv(batch_csv).set_index("sample")["batch"].to_dict()

    # Type every replicate of every condition and collect the per-sample tallies
    summary_rows = []
    for cond in CONDITIONS:
        for sample in COND_REPS[cond]:
            row = process_sample(sample, batch_map)
            summary_rows.append(row)
            cnt = row.iloc[0].to_dict()
            kept = cnt.get("KEPT", 0)
            total = cnt.get("TOTAL", 0)
            print(f"  {sample}: kept={kept}/{total}", flush=True)

    # Combine the per-sample tallies into one summary table and write it
    summary = pd.concat(summary_rows, ignore_index=True).fillna(0)
    cols_order = (["sample", "TOTAL", "KEPT", "DROPPED"]
                  + [c for c in summary.columns
                     if c not in ("sample", "TOTAL", "KEPT", "DROPPED")])
    summary = summary[cols_order]
    summary.to_csv(SUMMARY_DIR / "cell_typing_summary.csv", index=False)
    print(f"\nWROTE {SUMMARY_DIR / 'cell_typing_summary.csv'}")
    print(summary.to_string(index=False))
