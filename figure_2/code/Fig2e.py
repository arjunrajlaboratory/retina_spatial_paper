# Fig 2e estimated ONL and INL thickness from DAPI positive area within each U-Net layer
# divided by summed component centerline length per replicate

## Load packages
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tifffile
from skimage.filters import threshold_otsu

## Load paper style conventions and shared calibration
sys.path.insert(0, "../../_shared/code")
import paper_style
from _retina_panel import PX_DS_UM as px_ds_um
from _paths import SAMPLE_TO_ROI
from _paths import findpath_unet_pred, findpath_exclude_final
paper_style.set_style()

## Load files
dapi_cache = "../../_shared/data_raw/ds8_cache"
columns_path = "../../_shared/data_raw/full_thickness_tables/columns_meta.parquet"
out_png = "../panels/Fig2e.png"
out_tbl = "../data_processed/Fig2e_per_sample.csv"

## Define constants
px_ds_area_um2 = px_ds_um ** 2
onl_id, inl_id = 1, 2
onl_color, inl_color = "#1f77b4", "#2ca02c"
conditions = paper_style.CONDITION_ORDER
cond_reps = {
    "WT_P21":   ["WT_P21_rep1", "WT_P21_rep2"],
    "WT_P64":   ["WT_P64_rep1", "WT_P64_rep2"],
    "LCA5_P21": ["LCA5_P21_rep1", "LCA5_P21_rep2"],
    "LCA5_P30": ["LCA5_P30_rep1", "LCA5_P30_rep2"],
    "LCA5_P64": ["LCA5_P64_rep1", "LCA5_P64_rep2"],
}
roi_to_sample = {v: k for k, v in SAMPLE_TO_ROI.items()}
columns_meta = pd.read_parquet(columns_path)


## Measure ONL and INL thickness for one retina
def measure_thickness(roi, sample):
    layer = tifffile.imread(findpath_unet_pred(sample, "layer"))
    excl_p = findpath_exclude_final(sample)
    excl = tifffile.imread(excl_p) > 0 if excl_p.exists() else np.zeros_like(layer, dtype=bool)
    dapi = np.load(f"{dapi_cache}/roi{roi}_dapi18s_ds8.npz")["arr"][0].astype(np.float32)

    height = min(layer.shape[0], dapi.shape[0], excl.shape[0])
    width = min(layer.shape[1], dapi.shape[1], excl.shape[1])
    layer, excl, dapi = layer[:height, :width], excl[:height, :width], dapi[:height, :width]

    # Otsu threshold on nonzero DAPI intensities for this section
    dapi_thr = float(threshold_otsu(dapi[dapi > 0]))
    dapi_pos = dapi > dapi_thr

    # Measure DAPI positive area within each U-Net layer outside the exclusion mask
    onl_area = float(((layer == onl_id) & dapi_pos & ~excl).sum()) * px_ds_area_um2
    inl_area = float(((layer == inl_id) & dapi_pos & ~excl).sum()) * px_ds_area_um2

    # Sum the widths of columns containing tissue
    cols = columns_meta[columns_meta["sample"] == sample]
    arc_cols = cols[cols["tissue_area_um2"] > 0]
    if arc_cols.empty:
        raise ValueError(f"{sample}: no retained arc columns")
    arc_um = float((arc_cols["s_hi_um"] - arc_cols["s_lo_um"]).sum())

    print(f"  {sample}: dapi_thr={dapi_thr:.1f}  arc={arc_um:.0f} um  "
          f"ONL={onl_area / arc_um:.2f} um  INL={inl_area / arc_um:.2f} um", flush=True)
    return dict(sample=sample, dapi_threshold=dapi_thr, arc_length_um=arc_um,
                onl_area_um2=onl_area, inl_area_um2=inl_area,
                onl_thickness_um=onl_area / arc_um, inl_thickness_um=inl_area / arc_um)


## Run every replicate independently
records = []
for cond in conditions:
    for sample in cond_reps[cond]:
        roi = next(r for r, s in roi_to_sample.items() if s == sample)
        record = measure_thickness(roi, sample)
        record["condition"] = cond
        record["rep"] = "rep1" if sample.endswith("_rep1") else "rep2"
        records.append(record)
summary_df = pd.DataFrame(records)
summary_df.to_csv(out_tbl, index=False)


## Draw one mean bar with range whisker and per replicate dots
def draw_grouped_bar(axes, xc, vals, color):
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return
    axes.bar(xc, float(np.mean(vals)), width=0.44, color=color, edgecolor="none")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    axes.plot([xc, xc], [lo, hi], color="black", linewidth=1.0)
    axes.plot([xc - 0.09, xc + 0.09], [lo, lo], color="black", linewidth=1.0)
    axes.plot([xc - 0.09, xc + 0.09], [hi, hi], color="black", linewidth=1.0)
    for v in vals:
        axes.plot(xc, v, marker="o", color="black", markersize=3.4, markeredgecolor="none", alpha=0.85)


## Draw the grouped thickness bars
figure, axes = plt.subplots(figsize=(2.7, 2.7), gridspec_kw=dict(left=0.21, right=0.97, top=0.95, bottom=0.22))
xpos = np.arange(len(conditions))
off = 0.24
for i, cond in enumerate(conditions):
    sub = summary_df[summary_df["condition"] == cond]
    draw_grouped_bar(axes, xpos[i] - off, sub["onl_thickness_um"].to_numpy(), onl_color)
    draw_grouped_bar(axes, xpos[i] + off, sub["inl_thickness_um"].to_numpy(), inl_color)
axes.set_xticks(xpos)
axes.set_xticklabels([paper_style.condition_label(c) for c in conditions], fontsize=8, fontweight="bold", rotation=20, ha="right")
for i, c in enumerate(conditions):
    axes.get_xticklabels()[i].set_color(paper_style.condition_color_simple(c))
axes.set_xlim(-0.5, len(conditions) - 0.5)
axes.set_ylabel("Estimated layer thickness (µm)", fontweight="bold", fontsize=9)
axes.tick_params(axis="y", labelsize=8)
for sp in ("top", "right"):
    axes.spines[sp].set_visible(False)
axes.set_axisbelow(True)
handles = [mpl.patches.Patch(facecolor=onl_color, edgecolor="none", label="ONL"),
           mpl.patches.Patch(facecolor=inl_color, edgecolor="none", label="INL")]
axes.legend(handles=handles, loc="upper right", frameon=False, fontsize=8, handlelength=1.1, handletextpad=0.5)
figure.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(figure)
print("wrote", out_png)
