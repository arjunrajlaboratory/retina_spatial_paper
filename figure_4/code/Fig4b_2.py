# Fig 4b wide views of the injury response score and Müller gliosis

## Load packages
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import rotate as nd_rotate

## Load paper style conventions and shared imaging functions
sys.path.insert(0, ".")
sys.path.insert(0, "../../_shared/code")
import paper_style
warnings.filterwarnings("ignore", category=FutureWarning)
from _retina_panel import load_panel, load_cellbody_mask_oriented, project_xy_to_panel, PX_DS_UM
from _vignette import (WIN_W_UM, WIN_H_UM, onl_pca_angle, crop_layer_rect,
                       anchor_onl_top_center)

## Load files
data_raw = "../data_raw_manifest"
score_path = f"{data_raw}/rod_injury_response_score.parquet"
pool_path = f"{data_raw}/all_rods.parquet"
cells_path = "../../_shared/data_processed/per_cell_gene_counts.parquet"
out_dir = "../panels"
data_plot = "../data_plot"
regions = json.loads(open(f"{data_raw}/vignette_crop_regions.json").read())
MG_TX_PATH = "../data_raw_manifest/muller_gliosis_transcripts.parquet"

## Define gene sets and colors
PROGRAM_GENES = ["Edn2", "Socs3", "Fgf2", "Stat3"]
COORD_GENES = ["Cebpd", "Gadd45b", "Nr4a1", "Stat1"]
SUPP_GENES = PROGRAM_GENES + ["Cd47", "Cebpd", "Gadd45b", "Nr4a1"]
GENES = list(dict.fromkeys(PROGRAM_GENES + COORD_GENES + SUPP_GENES))
GENE_COLOR = {"rod injury response": "#222222"}   # The loops below set the program and stress gene colors and Cd47 falls to the default red
for _pg in ("Edn2", "Socs3", "Fgf2", "Stat3"):       # Rod injury response teal matching Fig4d
    GENE_COLOR[_pg] = "#21918c"
for _cg in ("Cebpd", "Gadd45b", "Nr4a1", "Stat1"):   # Other stress genes paper red matching Fig4d
    GENE_COLOR[_cg] = "#CC0000"
MG_GENES = ["Gfap", "Serpina3n"]
MG_COLOR = {"Gfap": "#1b7f4b", "Serpina3n": "#b24bff"}   # Dark green and bright purple
TIMEPOINTS = [("LCA5 P21", "LCA5_P21_rep2"), ("LCA5 P30", "LCA5_P30_rep2")]
CMAP = mpl.colormaps["viridis"].copy()
CMAP.set_bad(CMAP(0.0))
RAW_VMAX_Q = 0.97
MG_ROW = "Müller gliosis"
MULLER_RGB = mcolors.to_rgb(paper_style.CELLTYPE_COLORS["muller"])   # Teal fill for Muller cell bodies
BAR_UM = 50.0
MAIN_ROWS = ["rod injury response", "Edn2", "Socs3", "Cebpd", MG_ROW]
DS = 8
SCORE_VMIN = None
SCORE_VMAX = None


## Vignette rendering functions

# Load the cellbody mask oriented to match the panel
def load_body_mask_panel(sample, panel):
    return load_cellbody_mask_oriented(sample, panel)


# Crop a square, rotate the ONL horizontally, and take the final wide crop
def straightened_rect_crop(dapi, s18, layer, body, tx, rod,
                           win_w_um, win_h_um, cx_px, cy_px,
                           return_valid=False):
    H, W = dapi.shape
    diag_um = float(np.hypot(win_w_um, win_h_um))
    half_px = int(round(diag_um / 2 / PX_DS_UM)) + 8
    cx_i, cy_i = int(round(cx_px)), int(round(cy_px))
    y0 = max(0, cy_i - half_px)
    y1 = min(H, cy_i + half_px)
    x0 = max(0, cx_i - half_px)
    x1 = min(W, cx_i + half_px)
    sub_dapi  = dapi[y0:y1, x0:x1].astype(np.float32)
    sub_s18   = s18[y0:y1, x0:x1].astype(np.float32)
    sub_layer = layer[y0:y1, x0:x1]
    sub_body  = body[y0:y1, x0:x1]
    Hc, Wc = sub_dapi.shape

    rot_deg = onl_pca_angle(sub_layer)
    rot_dapi  = nd_rotate(sub_dapi, rot_deg, reshape=True, order=1, cval=0)
    rot_s18   = nd_rotate(sub_s18, rot_deg, reshape=True, order=1, cval=0)
    rot_layer = nd_rotate(sub_layer.astype(np.int16),
                          rot_deg, reshape=True, order=0, cval=0)
    rot_body  = nd_rotate(sub_body.astype(np.int32),
                          rot_deg, reshape=True, order=0, cval=0)
    rot_valid = nd_rotate(np.ones_like(sub_dapi, dtype=np.float32),
                           rot_deg, reshape=True, order=0, cval=0) > 0.5
    Hn, Wn = rot_dapi.shape

    ys_o, _ = np.where(rot_layer == 1)
    ys_g, _ = np.where(rot_layer == 4)
    flip_v = False
    if len(ys_o) > 30 and len(ys_g) > 30 and ys_o.mean() > ys_g.mean():
        rot_dapi = rot_dapi[::-1]
        rot_s18 = rot_s18[::-1]
        rot_layer = rot_layer[::-1]
        rot_body = rot_body[::-1]
        rot_valid = rot_valid[::-1]
        flip_v = True

    def _project(p_df):
        p = p_df.copy()
        p["xr"] = p["xr"] - x0
        p["yr"] = p["yr"] - y0
        in_win = ((p["xr"] >= 0) & (p["xr"] < Wc)
                  & (p["yr"] >= 0) & (p["yr"] < Hc))
        p = p[in_win].copy()
        a = np.radians(rot_deg)
        cosa, sina = np.cos(a), np.sin(a)
        cx0, cy0 = Wc / 2, Hc / 2
        xo = p["xr"].to_numpy() - cx0
        yo = p["yr"].to_numpy() - cy0
        p["xr"] =  cosa * xo + sina * yo + Wn / 2
        p["yr"] = -sina * xo + cosa * yo + Hn / 2
        if flip_v:
            p["yr"] = (Hn - 1) - p["yr"].to_numpy()
        return p

    proj_tx  = _project(tx)
    proj_rod = _project(rod)

    win_w_px = int(round(win_w_um / PX_DS_UM))
    win_h_px = int(round(win_h_um / PX_DS_UM))
    half_w = win_w_px // 2
    half_h = win_h_px // 2
    cxr, cyr = Wn // 2, Hn // 2
    rx0, rx1 = max(0, cxr - half_w), min(Wn, cxr + half_w)
    ry0, ry1 = max(0, cyr - half_h), min(Hn, cyr + half_h)
    rect_dapi  = rot_dapi[ry0:ry1, rx0:rx1]
    rect_s18   = rot_s18[ry0:ry1, rx0:rx1]
    rect_body  = rot_body[ry0:ry1, rx0:rx1]
    rect_valid = rot_valid[ry0:ry1, rx0:rx1]
    for p in (proj_tx, proj_rod):
        p["xr"] = p["xr"] - rx0
        p["yr"] = p["yr"] - ry0
    rect_tx  = proj_tx[(proj_tx["xr"] >= 0) & (proj_tx["xr"] < rx1 - rx0)
                        & (proj_tx["yr"] >= 0) & (proj_tx["yr"] < ry1 - ry0)
                        ].copy()
    rect_rod = proj_rod[(proj_rod["xr"] >= 0) & (proj_rod["xr"] < rx1 - rx0)
                          & (proj_rod["yr"] >= 0)
                          & (proj_rod["yr"] < ry1 - ry0)].copy()
    return rect_dapi, rect_s18, rect_body, rect_valid, rect_tx, rect_rod


# Light underlay of DAPI and 18S with percentiles over valid pixels only
def render_dapi(ax, dapi, s18=None, valid=None):
    H, W = dapi.shape
    if valid is None:
        valid = np.ones_like(dapi, dtype=bool)
    d_pos = dapi[(dapi > 0) & valid]
    d_lo, d_hi = np.percentile(d_pos, (2, 98)) if d_pos.size else (0, 1)
    d_norm = np.clip((dapi - d_lo) / max(d_hi - d_lo, 1e-6), 0, 1) ** 0.55
    if s18 is not None:
        s_pos = s18[(s18 > 0) & valid]
        s_lo, s_hi = np.percentile(s_pos, (2, 98)) if s_pos.size else (0, 1)
        s_norm = np.clip((s18 - s_lo) / max(s_hi - s_lo, 1e-6), 0, 1) ** 0.7
    else:
        s_norm = np.zeros_like(d_norm)
    light = 1.0 - (d_norm * 0.4125 + s_norm * 0.2125)
    light = np.clip(light, 0, 1)
    light[~valid] = 1.0
    ax.imshow(light, cmap="gray", vmin=0, vmax=1)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("black")
        sp.set_linewidth(1.0)


# Thin white outline at every rod body boundary
def _rod_outline_rgba(body, rod_labels):
    H, W = body.shape
    if not rod_labels:
        return np.zeros((H, W, 4), dtype=np.float32)
    rod_arr = np.array(sorted(rod_labels), dtype=body.dtype)
    rod_mask = np.isin(body, rod_arr)
    edges = np.zeros((H, W), dtype=bool)
    edges[:-1, :] |= body[:-1, :] != body[1:, :]
    edges[1:,  :] |= body[:-1, :] != body[1:, :]
    edges[:, :-1] |= body[:, :-1] != body[:, 1:]
    edges[:, 1:]  |= body[:, :-1] != body[:, 1:]
    outline = edges & rod_mask
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[outline] = (1.0, 1.0, 1.0, 0.55)
    return rgba


# Right corner italic label
def render_label(ax, text, color):
    ax.text(0.985, 0.96, text, transform=ax.transAxes, ha="right", va="top",
             fontsize=10, fontstyle="italic", fontweight="bold", color=color, zorder=6)


# The scale bar is drawn separately by place_scalebar
def _scalebar_if(ax, dapi, draw=False):
    return None


# DAPI and 18S underlay with rod bodies filled by value and white outlines
def render_value_overlay(ax, dapi, s18, valid, body, lab_to_val, vmin, vmax,
                         label, label_color, scalebar=False):
    render_dapi(ax, dapi, s18=s18, valid=valid)
    max_label = int(body.max())
    lut = np.full(max_label + 1, np.nan, dtype=np.float32)
    for L, v in lab_to_val.items():
        if 0 < L <= max_label and np.isfinite(v):
            lut[L] = v
    val_img = lut[body]
    rod_mask = ~np.isnan(val_img)
    rgba = np.zeros(val_img.shape + (4,), dtype=np.float32)
    norm = (np.clip(val_img, vmin, vmax) - vmin) / max(vmax - vmin, 1e-9)
    # Colour only the rod pixels; the rest stay fully transparent
    rgba[rod_mask] = CMAP(norm[rod_mask])
    rgba[..., 3] = rod_mask.astype(np.float32) * 0.95
    ax.imshow(rgba, interpolation="nearest", zorder=3)
    ax.imshow(_rod_outline_rgba(body, set(lab_to_val.keys())), interpolation="nearest", zorder=4)
    render_label(ax, label, label_color)
    _scalebar_if(ax, dapi, scalebar)


# Muller gliosis transcripts for the sample
def load_mg_tx(sample):
    df = pd.read_parquet(MG_TX_PATH)
    df = df[(df["sample"] == sample) & df["name"].isin(MG_GENES)].copy()
    return df[["name", "x", "y"]]


## Define functions

# Italicize gene symbols inside a row or axis label
def ital_label(text):
    out = text
    for g in ("Edn2", "Socs3", "Fgf2", "Stat3", "Cebpd", "Gadd45b", "Nr4a1", "Stat1"):
        out = out.replace(g, rf"$\mathit{{{g}}}$")
    return out


# Fill Müller cell bodies in teal
def muller_fill_rgba(body, labels):
    H, W = body.shape
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    if not labels:
        return rgba
    mask = np.isin(body, np.array(sorted(labels), dtype=body.dtype))
    rgba[mask] = (*MULLER_RGB, 0.6)
    return rgba


# Set of layer labels present in the final straightened crop
def _crop_layer_labels(layer, cx, cy):
    return {int(v) for v in np.unique(crop_layer_rect(layer, cx, cy))}


# Draw one 50 µm bar in the bottom right corner
def place_scalebar(axes, dapi, body, valid, panel_tag="", edges=("bottom",)):
    del valid, edges
    H, W = dapi.shape
    bar_px = BAR_UM / PX_DS_UM
    x1 = W - 0.03 * W
    x0 = x1 - bar_px
    y_bar = H - 0.055 * H
    txt_y = y_bar - 0.02 * H
    axes.plot([x0, x1], [y_bar, y_bar], color="black", lw=1.6, solid_capstyle="butt", zorder=11)
    frac = float((body[int(y_bar - 0.06 * H):int(y_bar + 2), int(x0):int(x1)] > 0).mean())
    print(f"  SCALEBAR[{panel_tag}] x=[{int(x0)},{int(x1)}] y~{int(y_bar)} crop=({W}x{H}) body_frac={frac:.3f}")
    return frac > 0.02


# Rod injury response score and rod transcript density
def load_per_rod(sample):
    sc = pd.read_parquet(score_path, columns=["sample", "label", "injury_response_score"])
    sc = sc[sc["sample"] == sample][["label", "injury_response_score"]]
    pool = pd.read_parquet(pool_path, columns=["cell_id", "sample_id", "cell_area_um2"] + [f"body_{g}" for g in GENES])
    pool = pool[pool["sample_id"] == sample].rename(columns={"cell_id": "label"}).copy()
    for g in GENES:
        pool[f"tpa_{g}"] = pool[f"body_{g}"].to_numpy() / pool["cell_area_um2"].to_numpy()
    rel = pd.read_parquet(cells_path, columns=["cell_id", "sample_id", "cell_type", "arc_um"])
    rel = rel[(rel["cell_type"] == "rod") & (rel["sample_id"] == sample)]
    rel = rel.rename(columns={"cell_id": "label"})[["label", "arc_um"]]
    out = sc.merge(pool[["label"] + [f"tpa_{g}" for g in GENES]], on="label", how="inner", validate="one_to_one")
    out = out.merge(rel, on="label", how="left", validate="one_to_one")
    return out


# Compute the vignette crops plus the transcript scales
def build_crops():
    global SCORE_VMIN, SCORE_VMAX
    paper_style.set_style()
    per = {s: load_per_rod(s) for _t, s in TIMEPOINTS}
    pool = np.concatenate([per[s]["injury_response_score"].to_numpy() for _t, s in TIMEPOINTS])
    # Match the full-section map (Fig4b_1) which scales the score by its 2nd-98th percentile
    SCORE_VMIN, SCORE_VMAX = (float(v) for v in np.percentile(pool, (2, 98)))
    tpa_vmax = {g: max(float(np.quantile(per[s][f"tpa_{g}"].to_numpy(), RAW_VMAX_Q)) for _t, s in TIMEPOINTS) for g in GENES}

    # Muller cell body labels to fill in the gliosis row
    relm = pd.read_parquet(cells_path, columns=["cell_id", "sample_id", "cell_type", "cx_um", "cy_um"])
    relm = relm[relm.cell_type == "muller"]
    muller_labels = {s: set(relm.loc[relm.sample_id == s, "cell_id"].astype(int)) for _t, s in TIMEPOINTS}

    crops = {}
    for tp, sample in TIMEPOINTS:
        panel = load_panel(sample)
        dapi, s18, layer = panel["dapi"], panel["s18"], panel["layer"]
        body = load_body_mask_panel(sample, panel)
        cells = panel["cells"].merge(per[sample], on="label", how="inner", validate="one_to_one")
        rod = cells.copy()
        print(f"  {sample}: rods={len(per[sample])} panel joined rods={len(rod)} median score={per[sample]['injury_response_score'].median():.3f}")
        mtx = load_mg_tx(sample)
        xr, yr = project_xy_to_panel(panel, mtx["x"].to_numpy(), mtx["y"].to_numpy())
        mtx = mtx.assign(xr=xr, yr=yr)
        cx, cy = regions[sample]["cx"], regions[sample]["cy"]
        span_before = _crop_layer_labels(layer, cx, cy)
        cx, cy, dperp, onl_top, gcl_bot, Hr = anchor_onl_top_center(layer, cx, cy, tag=sample)
        print(f"  {sample}: ONL-top anchor {dperp:+.0f}um center=({cx:.0f},{cy:.0f}) laminae {sorted(span_before)} -> {sorted(_crop_layer_labels(layer, cx, cy))} ONL_top={onl_top} GCL_bot={gcl_bot}/{Hr}")
        rd, rs, rbody, rvalid, rtx, rrod = straightened_rect_crop(dapi, s18, layer, body, mtx, rod, WIN_W_UM, WIN_H_UM, cx, cy)
        print(f"  {sample}: crop rods={len(rrod)} tx dots={len(rtx)} valid_frac={rvalid.mean():.2f}")
        labs = rrod["label"].astype(int)
        crops[tp] = dict(dapi=rd, s18=rs, body=rbody, valid=rvalid, tx=rtx,
                         score=dict(zip(labs, rrod["injury_response_score"])),
                         tpa={g: dict(zip(labs, rrod[f"tpa_{g}"])) for g in GENES},
                         muller_labels=muller_labels[sample])
    return crops, tpa_vmax


# Render one vignette grid with rows by timepoint columns from the shared crops
def draw_grid(crops, ROWS, out_png, tpa_vmax, show_titles=True):
    n_tp = len(TIMEPOINTS)
    figure = plt.figure(figsize=(2.4 * n_tp + 0.9, 1.30 * len(ROWS) + 0.5))
    gs = figure.add_gridspec(len(ROWS), n_tp, left=0.11, right=0.88, top=0.93, bottom=0.03, wspace=0.05, hspace=0.06)
    last_c = n_tp - 1
    for c, (tp, sample) in enumerate(TIMEPOINTS):
        crop = crops[tp]
        for r, row in enumerate(ROWS):
            axes = figure.add_subplot(gs[r, c])
            tag = f"{sample}:{row}"
            if row == MG_ROW:
                # Plain tissue background under the Muller cell body and gliosis dots
                render_dapi(axes, crop["dapi"], s18=crop["s18"], valid=crop["valid"])
                axes.imshow(muller_fill_rgba(crop["body"], crop["muller_labels"]), interpolation="nearest", zorder=6)
                for g in MG_GENES:
                    sub = crop["tx"][crop["tx"]["name"] == g]
                    axes.scatter(sub["xr"], sub["yr"], s=1.4, c=MG_COLOR[g], edgecolor="none", alpha=0.95, zorder=8)
                place_scalebar(axes, crop["dapi"], crop["body"], crop["valid"], tag)
                if c == 0:
                    axes.set_ylabel("Müller gliosis markers", fontsize=9, color="#1b7f4b")
                if c == last_c:
                    figure.canvas.draw()
                    bb = axes.get_position()
                    ymid = 0.5 * (bb.y0 + bb.y1)
                    for i, g in enumerate(MG_GENES):
                        figure.text(0.908 + i * 0.020, ymid, g, rotation=90, ha="center", va="center", fontsize=9, fontstyle="italic", fontweight="bold", color=MG_COLOR[g])
                    figure.text(0.908 + len(MG_GENES) * 0.020, ymid, "Müller glia cell body", rotation=90, ha="center", va="center", fontsize=9, fontweight="bold", color=MULLER_RGB)
                continue
            if row == "rod injury response":
                lut, vmin, vmax = crop["score"], SCORE_VMIN, SCORE_VMAX
            else:
                lut, vmin, vmax = crop["tpa"][row], 0.0, tpa_vmax[row]
            render_value_overlay(axes, crop["dapi"], crop["s18"], crop["valid"], crop["body"], lut, vmin, vmax, row, GENE_COLOR[row])
            place_scalebar(axes, crop["dapi"], crop["body"], crop["valid"], tag)
            if c == 0:
                axes.set_ylabel(ital_label(row), fontsize=9, color=GENE_COLOR[row])
            if c == last_c:
                figure.canvas.draw()
                bb = axes.get_position()
                cax = figure.add_axes([0.895, bb.y0, 0.011, bb.y1 - bb.y0])
                sm = mpl.cm.ScalarMappable(cmap=CMAP, norm=mcolors.Normalize(vmin, vmax))
                cb = figure.colorbar(sm, cax=cax)
                cb.ax.tick_params(labelsize=6)
                cb.outline.set_visible(False)
                cb.set_label("Rod injury response score" if row == "rod injury response" else "tx/µm²", fontsize=6.5, fontweight="bold", color=GENE_COLOR[row], labelpad=1)
    figure.canvas.draw()
    if show_titles:
        for c, (tp, _s) in enumerate(TIMEPOINTS):
            cell = gs[0, c].get_position(figure)
            title = paper_style.condition_label(tp.replace(" ", "_"))
            tcol = paper_style.CONDITION_COLORS[tp.replace(" ", "_")]
            figure.text(0.5 * (cell.x0 + cell.x1), cell.y1 + 0.008, title, ha="center", va="bottom", fontsize=18, fontweight="bold", color=tcol)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    figure.savefig(out_png, bbox_inches="tight", pad_inches=0.1)
    plt.close(figure)
    print("wrote", out_png)


# Rod views for the genes highlighted in red in Fig4d
def draw_gene_grid_rod(crops, out_png, tpa_vmax):
    cols = list(TIMEPOINTS)
    genes = SUPP_GENES
    ncol, nrow = len(cols), len(genes)
    figure = plt.figure(figsize=(2.4 * ncol + 0.9, 1.30 * nrow + 0.5))
    gs = figure.add_gridspec(nrow, ncol, left=0.13, right=0.90, top=0.93, bottom=0.03, wspace=0.05, hspace=0.06)
    for r, g in enumerate(genes):
        col = GENE_COLOR.get(g, "#CC0000")
        for c, (tp, sample) in enumerate(cols):
            axes = figure.add_subplot(gs[r, c])
            crop = crops[tp]
            render_value_overlay(axes, crop["dapi"], crop["s18"], crop["valid"], crop["body"], crop["tpa"][g], 0.0, tpa_vmax[g], "", col)
            place_scalebar(axes, crop["dapi"], crop["body"], crop["valid"], f"{sample}:rod:{g}")
            if c == 0:
                axes.set_ylabel(g, fontsize=10, color=col, fontstyle="italic")
            if c == ncol - 1:
                figure.canvas.draw()
                bb = axes.get_position()
                cax = figure.add_axes([0.915, bb.y0, 0.011, bb.y1 - bb.y0])
                sm = mpl.cm.ScalarMappable(cmap=CMAP, norm=mcolors.Normalize(0.0, tpa_vmax[g]))
                cb = figure.colorbar(sm, cax=cax)
                cb.ax.tick_params(labelsize=6)
                cb.outline.set_visible(False)
                cb.set_label("tx/µm²", fontsize=8, fontweight="bold", color="#222")
    figure.canvas.draw()
    for c, (tp, _s) in enumerate(cols):
        cell = gs[0, c].get_position(figure)
        title = paper_style.condition_label(tp.replace(" ", "_"))
        tcol = paper_style.CONDITION_COLORS[tp.replace(" ", "_")]
        figure.text(0.5 * (cell.x0 + cell.x1), cell.y1 + 0.012, title, ha="center", va="bottom", fontsize=16, fontweight="bold", color=tcol)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    figure.savefig(out_png, bbox_inches="tight", pad_inches=0.1)
    plt.close(figure)
    print("wrote", out_png)


## Build the crops and draw only the main Fig4b vignette grid
## FigS5b imports the crop and gene grid functions below
if __name__ == "__main__":
    crops, tpa_vmax = build_crops()

    ## Export the rod plotting table
    os.makedirs(data_plot, exist_ok=True)
    plot_rows = []
    for tp, sample in TIMEPOINTS:
        cr = crops[tp]
        for lab, sc_val in cr["score"].items():
            row = {"sample": sample, "timepoint": tp, "label": int(lab), "injury_response_score": float(sc_val)}
            for g in GENES:
                row[f"tpa_{g}"] = float(cr["tpa"][g].get(lab, np.nan))
            plot_rows.append(row)
    plot_df = pd.DataFrame(plot_rows)
    plot_df.to_parquet(f"{data_plot}/Fig4b_2.parquet", index=False)

    draw_grid(crops, MAIN_ROWS, f"{out_dir}/Fig4b_2.png", tpa_vmax, show_titles=False)
