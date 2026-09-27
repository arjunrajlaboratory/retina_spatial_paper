# Fig 2bc per condition spatial ROI maps and superior inferior orientation

## Load packages
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

## Load paper style conventions and shared imaging functions
sys.path.insert(0, "../../_shared/code")
from _retina_panel import (load_panel, PX_DS_UM,
                           load_exclude_final_oriented, load_cellbody_mask_oriented,
                           zoom_straighten_geom, zoom_straighten_apply)
import paper_style

## Load the shared superior-up orientation (cone opsin gradient fit)
from _superior_up_orientation import install_s1c_orientations

## Load files
per_cell = Path("../../_shared/data_processed/per_cell_gene_counts.parquet")
out_dir = Path("../panels")
data_plot_dir = Path("../data_processed")
out_png = out_dir / "Fig2bc.png"

## Define conditions and constants
conditions = ["WT_P21", "WT_P64", "LCA5_P21", "LCA5_P30", "LCA5_P64"]
display_rep = {
    "WT_P21":   "WT_P21_rep2",
    "WT_P64":   "WT_P64_rep1",
    "LCA5_P21": "LCA5_P21_rep2",
    "LCA5_P30": "LCA5_P30_rep2",
    "LCA5_P64": "LCA5_P64_rep1",
}
celltype_colors = paper_style.CELLTYPE_COLORS
bg_color = "#cccccc"

# Final zoom height in microns for the retinal depth window ONL to GCL
zoom_h_um = 300
# Blank band below the tissue so the scale bar sits in a clean margin
scalebar_margin_frac = 0.07
# Breathing room around each retina
panel_infl = 0.05

# Explicit zoom center override as x_frac and y_frac of panel width and height
sample_zoom_center_frac = {
    "WT_P21_rep2":   (0.50, 0.14),
    "WT_P64_rep1":   (0.45, 0.14),
    "LCA5_P21_rep2": (0.18, 0.25),
    "LCA5_P30_rep2": (0.39, 0.105),
    "LCA5_P64_rep1": (0.23, 0.15),
}
# Layer labels required in the pre rotation zoom crop 1=ONL 2=INL 3=IPL 4=GCL
required_layer_labels = (1, 2, 3, 4)

# Cell type legend order matching Fig1c
legend_order = ["rod", "cone", "bipolar", "amacrine_gaba", "amacrine_gly",
                "horizontal", "rgc", "muller", "microglia", "vascular", "rpe"]

# ONL apical top sits at 25 percent of the crop height so zooms align by ONL top
onl_top_frac = 0.25
# Per sample GCL nudges
sample_zoom_gcl_nudge_frac = {}
# Layer bands to label in the leftmost zoom
layer_label_bands = [(1, "ONL"), (2, "INL"), (4, "GCL")]


## Define functions

# Convert a hex color to an rgb triple in 0 to 1
def hex_to_rgb01(hx):
    hx = hx.lstrip("#")
    return (int(hx[0:2], 16) / 255, int(hx[2:4], 16) / 255, int(hx[4:6], 16) / 255)




# Compose the panel image on a white background with DAPI and 18S gray plus cell colors
def build_composite(panel, body_mask, label_to_color):
    dapi = panel["dapi"].astype(np.float32)
    s18 = panel["s18"].astype(np.float32)

    # Normalize with a percentile clip and gamma
    d_pos = dapi[dapi > 0]
    d_lo, d_hi = np.percentile(d_pos, (2, 98)) if d_pos.size else (0, 1)
    d_norm = np.clip((dapi - d_lo) / max(d_hi - d_lo, 1e-6), 0, 1)
    d_norm = d_norm ** 0.55
    s_pos = s18[s18 > 0]
    s_lo, s_hi = np.percentile(s_pos, (2, 98)) if s_pos.size else (0, 1)
    s_norm = np.clip((s18 - s_lo) / max(s_hi - s_lo, 1e-6), 0, 1)
    s_norm = s_norm ** 0.7

    # White background grayscale with DAPI light and 18S darker
    inv = 1.0 - (d_norm * 0.18 + s_norm * 0.55)
    inv = np.clip(inv, 0, 1)
    base = np.stack([inv] * 3, axis=-1)

    # Mask excluded pixels in the background and cell colors
    excl = panel.get("excl_mask")
    H, W = body_mask.shape
    not_excl = np.ones((H, W), dtype=bool)
    if excl is not None:
        e = np.zeros((H, W), dtype=bool)
        He, We = excl.shape
        e[:min(H, He), :min(W, We)] = excl[:H, :W]
        not_excl &= ~e
    base = np.where(not_excl[..., None], base, 1.0)

    # Build the cell color image with background cells white
    max_label = int(body_mask.max())
    lut = np.zeros((max_label + 1, 3), dtype=np.float32)
    for lab, rgb in label_to_color.items():
        if 0 < lab <= max_label:
            lut[lab] = rgb
    color = lut[body_mask]
    typed_mask = np.any(color > 0, axis=-1) & not_excl
    out = np.where(typed_mask[..., None], color, base)
    return out


# Pick a representative superior hemisphere zoom center at median density
def pick_representative_superior_center(panel, cells):
    H = panel["H"]
    W = panel["W"]
    xr_all = cells["xr"].to_numpy()
    yr_all = cells["yr"].to_numpy()
    if xr_all.size == 0:
        return W / 2, H / 4
    x_lo, x_hi = np.quantile(xr_all, [0.25, 0.75])
    y_lo, y_hi = np.quantile(yr_all, [0.25, 0.50])
    eligible_pts = ((xr_all >= x_lo) & (xr_all <= x_hi) & (yr_all >= y_lo) & (yr_all <= y_hi))
    nb = 22
    xe = np.linspace(0, W, nb + 1)
    ye = np.linspace(0, H, nb + 1)
    h, _, _ = np.histogram2d(xr_all[eligible_pts], yr_all[eligible_pts], bins=(xe, ye))
    nz_idx = np.argwhere(h > 0)
    if nz_idx.size == 0:
        return float(0.5 * (x_lo + x_hi)), float(0.5 * (y_lo + y_hi))
    med = float(np.median(h[h > 0]))
    bx_c = 0.5 * (xe[:-1] + xe[1:])
    by_c = 0.5 * (ye[:-1] + ye[1:])
    ctr_x, ctr_y = 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)
    best, best_key = (W / 2.0, H / 4.0), None
    for ix, iy in nz_idx:
        cx, cy = float(bx_c[ix]), float(by_c[iy])
        key = (abs(float(h[ix, iy]) - med), (cx - ctr_x) ** 2 + (cy - ctr_y) ** 2)
        if best_key is None or key < best_key:
            best_key, best = key, (cx, cy)
    return best


# Straighten and apical up flip the zoom then crop a rectangle anchored by ONL top
def rotate_zoom_apical_up(zoom_img, zoom_layer, final_h_px, final_w_px, extra_down_frac=0.0):
    geom = zoom_straighten_geom(zoom_layer, final_h_px, final_w_px,
                                onl_code=1, gcl_code=4, onl_top_frac=onl_top_frac)
    crop = zoom_straighten_apply(zoom_img, geom, order=1, cval=1.0)
    crop_layer = zoom_straighten_apply(zoom_layer, geom, order=0, cval=0)

    # Apply the optional downward nudge
    sh = int(round(extra_down_frac * geom["side_h"]))
    if sh > 0:
        Hc = crop.shape[0]
        shifted = np.full_like(crop, 1.0)
        shifted[sh:Hc, :] = crop[0:Hc - sh, :]
        crop = shifted
        shifted_l = np.zeros_like(crop_layer)
        shifted_l[sh:Hc, :] = crop_layer[0:Hc - sh, :]
        crop_layer = shifted_l
    return crop, crop_layer


# Draw a scale bar in the bottom right corner
def _add_scalebar(axes, length_um, label, pad_frac=0.04, thickness_frac=0.014, color="black"):
    x0, x1 = axes.get_xlim()
    W_data = abs(x1 - x0)
    length_data = length_um / PX_DS_UM
    length_frac = length_data / W_data
    x_right = 1.0 - pad_frac
    x_left = x_right - length_frac
    y_bar_bot = pad_frac
    y_bar_top = pad_frac + thickness_frac
    label_y = y_bar_top + 0.005
    bar = mpatches.Rectangle((x_left, y_bar_bot), length_frac, thickness_frac,
                              linewidth=0, facecolor=color, zorder=10,
                              transform=axes.transAxes, clip_on=False)
    axes.add_patch(bar)


# Draw the full retina scale bar in the blank margin below the tissue
def _add_scalebar_full(axes, length_um, label, thickness_frac=0.012, hpad_frac=0.02, color="black"):
    x0d, x1d = axes.get_xlim()
    W_data = abs(x1d - x0d)
    length_frac = (length_um / PX_DS_UM) / W_data
    x_right = 1.0 - hpad_frac
    x_left = x_right - length_frac
    y_bar_bot = 0.030
    y_bar_top = y_bar_bot + thickness_frac
    label_y = y_bar_top + 0.004
    bar = mpatches.Rectangle((x_left, y_bar_bot), length_frac, thickness_frac,
                              linewidth=0, facecolor=color, zorder=10,
                              transform=axes.transAxes, clip_on=False)
    axes.add_patch(bar)


# Place ONL INL GCL labels in the left margin of the leftmost zoom
def _add_layer_labels(axes, layer_rot):
    H = int(layer_rot.shape[0])
    for lab, name in layer_label_bands:
        ys = np.where(layer_rot == lab)[0]
        if ys.size < 10:
            continue
        r = float(np.median(ys))
        y_frac = 1.0 - (r + 0.5) / H
        axes.text(-0.05, y_frac, name, transform=axes.transAxes, ha="right", va="center",
                fontsize=15, fontweight="bold", color="#2A2A2A", clip_on=False)


# Draw the DAPI and 18S channel key on the leftmost zoom
def _add_channel_legend(axes):
    handles = [
        mpatches.Patch(facecolor=(0.69, 0.69, 0.69), edgecolor="none", label="DAPI"),
        mpatches.Patch(facecolor=(0.376, 0.376, 0.376), edgecolor="none", label="18S rRNA"),
    ]
    legend = axes.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, -0.02),
                    frameon=True, fontsize=13, ncol=2, handlelength=1.4, borderpad=0.4,
                    columnspacing=1.0, handletextpad=0.4)
    fr = legend.get_frame()
    fr.set_facecolor("white")
    fr.set_edgecolor("none")
    fr.set_alpha(0.80)
    for txt in legend.get_texts():
        txt.set_fontweight("bold")
        txt.set_color("#2A2A2A")


# Render the full ROI plus DAPI zoom plus cell type zoom for one sample
def render_one(sample, ax_full, ax_zoom_dapi, ax_zoom, label_layers=False):
    panel = load_panel(sample)
    panel["excl_mask"] = load_exclude_final_oriented(panel)
    body = load_cellbody_mask_oriented(sample, panel)

    # Load the kept typed cells and build the label to color map
    typed = pd.read_parquet(per_cell, columns=["cell_id", "sample_id", "cell_type"])
    typed = typed[typed["sample_id"] == sample]
    typed_kept = (typed[typed["cell_type"] != "unassigned"]
                  [["cell_id", "cell_type"]]
                  .rename(columns={"cell_id": "label", "cell_type": "typed_cell_type"}))
    label_to_color = {}
    bg_rgb = hex_to_rgb01(bg_color)
    for lab, ct in zip(typed_kept["label"].to_numpy(), typed_kept["typed_cell_type"].to_numpy()):
        if ct in celltype_colors:
            label_to_color[int(lab)] = hex_to_rgb01(celltype_colors[ct])
        else:
            label_to_color[int(lab)] = bg_rgb
    cells = panel["cells"].merge(typed_kept, on="label", how="inner", validate="one_to_one")

    # Build the full composite and the DAPI only zoom base
    img = build_composite(panel, body, label_to_color)
    img_zoom_base = build_composite(panel, body, label_to_color)
    img_zoom_dapi = build_composite(panel, body, {})

    # Draw the full ROI panel
    ax_full.imshow(img, origin="upper", interpolation="nearest")
    ax_full.set_xticks([])
    ax_full.set_yticks([])
    ax_full.set_facecolor("white")
    for s in ax_full.spines.values():
        s.set_visible(False)
    tissue = panel.get("tissue")
    if tissue is not None and tissue.any():
        ys_t, xs_t = np.where(tissue)
        y0_t, y1_t = int(ys_t.min()), int(ys_t.max())
        x0_t, x1_t = int(xs_t.min()), int(xs_t.max())
        Wbb, Hbb = x1_t - x0_t, y1_t - y0_t
        margin_px = scalebar_margin_frac * Hbb
        x0d, x1d = x0_t - panel_infl * Wbb, x1_t + panel_infl * Wbb
        y0d, ylo = y0_t - panel_infl * Hbb, y1_t + panel_infl * Hbb + margin_px
        ax_full.set_xlim(x0d, x1d)
        ax_full.set_ylim(ylo, y0d)
    ax_full.set_aspect("equal", adjustable="box")
    ax_full.set_anchor("S")
    _add_scalebar_full(ax_full, 500, "500 µm")

    # Pick the zoom center from a manual fraction for hand placed samples else representative
    if sample in sample_zoom_center_frac:
        fx, fy = sample_zoom_center_frac[sample]
        cx, cy = fx * panel["W"], fy * panel["H"]
    else:
        cx, cy = pick_representative_superior_center(panel, cells)
    # Size the zoom rectangle so it fills the landscape cell at true 1 to 1
    bb = ax_zoom.get_position()
    fig_w_in, fig_h_in = ax_zoom.figure.get_size_inches()
    cell_aspect = (bb.width * fig_w_in) / (bb.height * fig_h_in)
    half_h = (zoom_h_um / PX_DS_UM) / 2.0
    half_w = half_h * cell_aspect
    pre_half = 0.5 * float(np.hypot(2 * half_w, 2 * half_h)) + 6
    rect = mpatches.Rectangle((cx - half_w, cy - half_h), 2 * half_w, 2 * half_h,
                              linewidth=1.6, edgecolor="black", facecolor="none",
                              linestyle="-", clip_on=False, zorder=20)
    ax_full.add_patch(rect)

    # Pre crop wider than final so the apical up rotation can fit inside
    H, W = panel["H"], panel["W"]
    xp0, yp0 = int(max(0, cx - pre_half)), int(max(0, cy - pre_half))
    xp1, yp1 = int(min(W, cx + pre_half)), int(min(H, cy + pre_half))
    zoom = img_zoom_base[yp0:yp1, xp0:xp1]
    zoom_layer = panel["layer"][yp0:yp1, xp0:xp1]

    # Snap to a nearby center if the window misses any of the four laminae
    def _layer_ok(ccx, ccy):
        x0c = int(max(0, ccx - half_w))
        y0c = int(max(0, ccy - half_h))
        x1c = int(min(W, ccx + half_w))
        y1c = int(min(H, ccy + half_h))
        present = set(np.unique(panel["layer"][y0c:y1c, x0c:x1c]).tolist())
        return all(lab in present for lab in required_layer_labels), present
    ok, present = _layer_ok(cx, cy)
    if not ok:
        step = half_h * 0.25
        max_r = 6
        best = None
        for r in range(1, max_r + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    ccx, ccy = cx + dx * step, cy + dy * step
                    okk, _ = _layer_ok(ccx, ccy)
                    if okk:
                        best = (ccx, ccy)
                        break
                if best: break
            if best: break
        if best is None:
            raise RuntimeError(f"{sample}: no valid zoom within {max_r} steps of ({cx:.0f},{cy:.0f})")
        cx, cy = best
        xp0, yp0 = int(max(0, cx - pre_half)), int(max(0, cy - pre_half))
        xp1, yp1 = int(min(W, cx + pre_half)), int(min(H, cy + pre_half))
        zoom = img_zoom_base[yp0:yp1, xp0:xp1]
        zoom_layer = panel["layer"][yp0:yp1, xp0:xp1]
        rect.set_xy((cx - half_w, cy - half_h))

    # Run the identical apical up rotate on the cell type and DAPI zooms
    zoom_dapi = img_zoom_dapi[yp0:yp1, xp0:xp1]
    nudge = sample_zoom_gcl_nudge_frac.get(sample, 0.0)
    zoom, zoom_layer_rot = rotate_zoom_apical_up(zoom, zoom_layer, int(2 * half_h), int(2 * half_w), extra_down_frac=nudge)
    zoom_dapi, _ = rotate_zoom_apical_up(zoom_dapi, zoom_layer, int(2 * half_h), int(2 * half_w), extra_down_frac=nudge)

    def _style_zoom_axis(axes, img):
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_facecolor("white")
        for s in axes.spines.values():
            s.set_visible(True)
            s.set_color("black")
            s.set_linestyle("-")
            s.set_linewidth(1.2)
        axes.set_aspect("equal")
        axes.set_xlim(-0.5, img.shape[1] - 0.5)
        axes.set_ylim(img.shape[0] - 0.5, -0.5)
        axes.set_anchor("N")

    # Draw the DAPI only zoom where the leftmost column adds layer labels and channel key
    ax_zoom_dapi.imshow(zoom_dapi, origin="upper", interpolation="nearest")
    _style_zoom_axis(ax_zoom_dapi, zoom_dapi)
    _add_scalebar(ax_zoom_dapi, 50, "50 µm")
    if label_layers:
        _add_layer_labels(ax_zoom_dapi, zoom_layer_rot)
        _add_channel_legend(ax_zoom_dapi)

    # Draw the cell type zoom
    ax_zoom.imshow(zoom, origin="upper", interpolation="nearest")
    _style_zoom_axis(ax_zoom, zoom)
    _add_scalebar(ax_zoom, 50, "50 µm")
    if label_layers:
        _add_layer_labels(ax_zoom, zoom_layer_rot)


# Annotate the leftmost full ROI panel with superior and inferior arrows
def add_si_arrows(figure, ax_first_full):
    bb = ax_first_full.get_position()
    x = bb.x0 - 0.006
    figure.text(x, bb.y0 + (bb.y1 - bb.y0) * 0.83, "↑", ha="center", va="center", fontsize=22, color="#444444", fontweight="bold")
    figure.text(x, bb.y0 + (bb.y1 - bb.y0) * 0.93, "S", ha="center", va="center", fontsize=16, color="#444444", fontweight="bold")
    figure.text(x, bb.y0 + (bb.y1 - bb.y0) * 0.17, "↓", ha="center", va="center", fontsize=22, color="#444444", fontweight="bold")
    figure.text(x, bb.y0 + (bb.y1 - bb.y0) * 0.07, "I", ha="center", va="center", fontsize=16, color="#444444", fontweight="bold")


## Set the paper style and orient every panel via the opsin gradient fit
paper_style.set_style()
install_s1c_orientations()

## Size the top row columns to each arc native aspect
n_cond = len(conditions)
samples = [display_rep[c] for c in conditions]
arc_aspects = []
for sample in samples:
    p_tmp = load_panel(sample)
    t = p_tmp["tissue"]
    ys_t, xs_t = np.where(t)
    Wbb = float(xs_t.max() - xs_t.min())
    Hbb = float(ys_t.max() - ys_t.min())
    arc_aspects.append(Wbb * (1.0 + 2 * panel_infl)
                       / (Hbb * (1.0 + 2 * panel_infl + scalebar_margin_frac)))
target_avg_cell_w_in = 3.6
fig_w = target_avg_cell_w_in * n_cond + 0.4

## Build the figure with a full ROI row and two zoom rows
figure = plt.figure(figsize=(fig_w, 13.0), facecolor="white")
outer = figure.add_gridspec(nrows=4, ncols=1, left=0.05, right=0.997, top=0.955, bottom=0.012,
                         height_ratios=[3.4, 1.4, 1.4, 0.30], hspace=0.06)
top_gs = outer[0].subgridspec(nrows=1, ncols=n_cond, width_ratios=arc_aspects, wspace=0.03)
dapi_gs = outer[1].subgridspec(nrows=1, ncols=n_cond, wspace=0.03)
cell_gs = outer[2].subgridspec(nrows=1, ncols=n_cond, wspace=0.03)

## Render each condition
full_axes = []
for j, cond in enumerate(conditions):
    sample = display_rep[cond]
    ax_full = figure.add_subplot(top_gs[0, j])
    ax_dapi = figure.add_subplot(dapi_gs[0, j])
    ax_zoom = figure.add_subplot(cell_gs[0, j])
    render_one(sample, ax_full, ax_dapi, ax_zoom, label_layers=(j == 0))
    full_axes.append(ax_full)

add_si_arrows(figure, full_axes[0])

## Add the cell type legend row
legend_ax = figure.add_subplot(outer[3])
legend_ax.axis("off")
handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=celltype_colors[k],
                  markeredgecolor="none", markersize=15, label=paper_style.celltype_label(k))
           for k in legend_order]
legend_ax.legend(handles=handles, loc="center", ncol=len(handles), frameon=False,
                 fontsize=15, handletextpad=0.4, columnspacing=1.3, labelspacing=0.4, borderaxespad=0.1)

## Save the panel
out_dir.mkdir(parents=True, exist_ok=True)
figure.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(figure)
print("wrote", out_png)

## Write the per condition cell count table
data_plot_dir.mkdir(parents=True, exist_ok=True)
dual = pd.read_parquet(per_cell, columns=["sample_id", "cell_type"])
rows = []
for cond in conditions:
    sample = display_rep[cond]
    sub = dual[dual["sample_id"] == sample]
    vc = sub["cell_type"].value_counts()
    rows.append({
        "condition": cond,
        "sample_id": sample,
        "n_cells_kept": int(len(sub)),
        "n_called": int((sub["cell_type"] != "unassigned").sum()),
        "n_unassigned": int((sub["cell_type"] == "unassigned").sum()),
        **{f"n_{ct}": int(vc.get(ct, 0)) for ct in celltype_colors},
    })
counts = pd.DataFrame(rows)
counts.to_csv(data_plot_dir / "Fig2bc.csv", index=False)
print("wrote", data_plot_dir / "Fig2bc.csv")
