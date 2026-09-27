# Fig 4b rod injury response maps and arc strips at LCA5 P21 and P30

## Load packages
import sys
import os
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle, Polygon
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

## Load paper style conventions and shared imaging functions
sys.path.insert(0, ".")
sys.path.insert(0, "../../_shared/code")
warnings.filterwarnings("ignore", category=FutureWarning)
import _spatial_maps as spatial_maps
import _retina_panel as retina_panel
from _retina_panel import (
    load_panel, load_unet_tissue_oriented, load_exclude_final_oriented,
    load_cellbody_mask_oriented, project_xy_to_panel, _rotate_xy,
)
import _focal_regions as focal
from _vignette import WIN_W_UM, WIN_H_UM, vignette_crop_geometry
import paper_style

## Load files
cells_path = "../../_shared/data_processed/per_cell_gene_counts.parquet"
out_panel = "../panels/Fig4b_1.png"
out_plot = "../data_plot"
# The vignette crop centers, recorded as pixels in the default panel orientation
regions_json = "../data_raw_manifest/vignette_crop_regions.json"

## Pull shared settings from the spatial maps function
CONDITIONS = spatial_maps.CONDITIONS
DISPLAY_REP = spatial_maps.DISPLAY_REP
SCORE_CMAP = spatial_maps.SCORE_CMAP
from _paths import PX_DS_UM

## Define arc strip parameters
ARC_BIN_UM = 40.0        # Arc bin width for the map-edge arrow backbone display placement
SUP_COLOR = "#1b9e3b"    # Superior M opsin green
INF_COLOR = "#c800a8"    # Inferior S opsin magenta

## Define focal region core marker parameters
SHOW_ARC_EXTREMA = True
SHOW_MAP_ARC_EXTREMA = True
MAP_TRI_SIZE_UM = 34.0
MAP_EDGE_GAP_UM = 34.0
OFFARC_CAP_UM = 50.0
BOX_W_UM = 480.0


## Define functions

# Use the shared 25 µm arc strip
def shared_strip(sample):
    return focal.arc_strip(sample)


# Mark each focal region core with a triangle on the arc strip
def draw_arc_extrema(sax, core_s, smax_s, span, tag=""):
    extrema_y = 1.55
    for s in core_s:
        x = float(np.clip((smax_s - s) / span, 0.0, 1.0))
        sax.scatter([x], [extrema_y], marker="^", s=130, color="black",
                    transform=sax.get_xaxis_transform(), clip_on=False, zorder=9)
    print(f"  {tag}: {len(core_s)} focal region centers")


# Build the map edge curve from rod centroids along the arc
def arc_backbone(sub, body_mask, edges, min_rods=5):
    centers = 0.5 * (edges[:-1] + edges[1:])
    cx = np.full(len(centers), np.nan)
    cy = np.full(len(centers), np.nan)
    maxlab = int(body_mask.max())
    for k in range(len(centers)):
        win = sub[(sub["arc_um"] >= edges[k]) & (sub["arc_um"] < edges[k + 1])]
        labs = [int(v) for v in win["label"] if 0 < int(v) <= maxlab]
        if len(labs) < min_rods:
            continue
        ys, xs = np.where(np.isin(body_mask, labs))
        if len(xs):
            cx[k] = xs.mean()
            cy[k] = ys.mean()
    fin = np.isfinite(cx)
    if fin.sum() >= 3:
        idx = np.arange(len(centers))
        cx[~fin] = np.interp(idx[~fin], idx[fin], cx[fin])
        cy[~fin] = np.interp(idx[~fin], idx[fin], cy[fin])
        cx = ndi.gaussian_filter1d(cx, 1.5, mode="nearest")
        cy = ndi.gaussian_filter1d(cy, 1.5, mode="nearest")
    return centers, cx, cy


# Draw a curve following triangle at each focal region core just outside the retina edge
def draw_map_arc_extrema(axes, sub, body_mask, tissue, edges, core_s, tag=""):
    from matplotlib.patches import Polygon
    centers, bx, by = arc_backbone(sub, body_mask, edges)
    H, W = tissue.shape
    trow, tcol = np.where(tissue > 0)
    tyc, txc = float(trow.mean()), float(tcol.mean())
    tri = MAP_TRI_SIZE_UM / PX_DS_UM
    gap = MAP_EDGE_GAP_UM / PX_DS_UM

    def place(i, point_inward):
        if not (np.isfinite(bx[i]) and np.isfinite(by[i])):
            return
        j0, j1 = max(0, i - 1), min(len(centers) - 1, i + 1)
        tang = np.array([bx[j1] - bx[j0], by[j1] - by[j0]], float)
        if np.hypot(*tang) < 1e-6:
            return
        nrm = np.array([-tang[1], tang[0]], float)
        nrm /= np.hypot(*nrm)
        if np.dot(nrm, [bx[i] - txc, by[i] - tyc]) < 0:
            nrm = -nrm
        p = np.array([bx[i], by[i]], float)
        for _ in range(600):
            xi, yi = int(round(p[0])), int(round(p[1]))
            if not (0 <= xi < W and 0 <= yi < H) or tissue[yi, xi] == 0:
                break
            p += nrm * 2.0
        near = p + nrm * gap
        far = p + nrm * (gap + tri)
        t = np.array([nrm[1], -nrm[0]])
        apex, base_c = (near, far) if point_inward else (far, near)
        b1, b2 = base_c + t * tri * 0.72, base_c - t * tri * 0.72
        axes.add_patch(Polygon([apex, b1, b2], closed=True, facecolor="black",
                             edgecolor="black", linewidth=1.3, zorder=13, clip_on=False))

    for s in core_s:
        place(int(np.argmin(np.abs(centers - s))), True)
    print(f"  {tag} map-arc: {len(core_s)} focal region centers")


# Invert a default-orientation panel pixel back to physical microns
# The vignette crop centers are stored as default-orientation pixels; this recovers their microns
def panel_px_to_um(panel, cx, cy):
    x = cx + panel["crop_x0"]
    y = cy + panel["crop_y0"]
    if panel.get("flipped_h"):
        x = panel["rot_Wn"] - 1 - x
    if panel.get("flipped_v"):
        y = panel["rot_Hn"] - 1 - y
    if abs(panel.get("angle", 0.0)) > 0.1:
        x = x - panel["dx_shift"]
        y = y - panel["dy_shift"]
        x, y = _rotate_xy(x, y, panel["cx_orig"], panel["cy_orig"], -panel["angle"])
    return x * PX_DS_UM, y * PX_DS_UM


# Build the four corners of the straightened vignette rectangle in microns
# Rotate a WIN_W_UM by WIN_H_UM box to the local ONL angle so the box matches the vignette footprint
def vignette_box_corners_um(panel, cx, cy, rot_deg):
    a = np.radians(rot_deg)
    along = np.array([np.cos(a), np.sin(a)])     # Along the ONL band
    across = np.array([-np.sin(a), np.cos(a)])   # Across the retinal layers
    half_w = (WIN_W_UM / 2) / PX_DS_UM
    half_h = (WIN_H_UM / 2) / PX_DS_UM
    center = np.array([cx, cy], dtype=float)
    corners = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        px, py = center + sx * half_w * along + sy * half_h * across
        corners.append(panel_px_to_um(panel, float(px), float(py)))
    return corners


# Strip display x fractions covering the vignette crop center (matches the core triangle placement)
def crop_strip_frac(sample, x_um, y_um, smax_s, span):
    skel = pd.read_parquet(focal.FT / "skeletons.parquet")
    skel = skel[skel["sample"] == sample]
    columns_meta = pd.read_parquet(focal.FT / "columns_meta.parquet")
    columns_meta = columns_meta[columns_meta["sample"] == sample].sort_values("col_id_local").reset_index(drop=True)
    point = pd.DataFrame({"cx_um": [x_um], "cy_um": [y_um]})
    col = int(focal._project_nearest(point, skel, columns_meta)["col_id_local"].iloc[0])
    prof = focal.strip_profile(sample)
    row = prof[prof["col_id_local"] == col]
    if row.empty:
        return None
    arc = float(row["arc_col_um"].iloc[0])
    valid = prof[prof["raw_program_score"].notna()].sort_values("arc_col_um")
    arc_all = valid["arc_col_um"].to_numpy(float)
    # Apply the same superior-inferior flip used for the core triangles
    if focal._opsin_orientation(valid["opsin_field"].to_numpy(float)):
        arc = arc_all.min() + arc_all.max() - arc
    x_center = (smax_s - arc) / span
    half = (BOX_W_UM / 2) / span
    return float(np.clip(x_center - half, 0, 1)), float(np.clip(x_center + half, 0, 1))


# Assign remaining rods the position of the nearest rod on the arc
def assign_offarc_arc_position(df):
    out = df.copy()
    for s, g in df.groupby("sample"):
        have = g[g["arc_um"].notna()]
        miss = g[g["arc_um"].isna() & g["cx_um"].notna()]
        if len(miss) == 0 or len(have) < 5:
            continue
        tree = cKDTree(have[["cx_um", "cy_um"]].to_numpy())
        d, idx = tree.query(miss[["cx_um", "cy_um"]].to_numpy(), k=1)
        assigned = have["arc_um"].to_numpy()[idx]
        assigned[d > OFFARC_CAP_UM] = np.nan
        out.loc[miss.index, "arc_um"] = assigned
        n_assigned = int(np.isfinite(assigned).sum())
        print(f"  off-arc rod placement {s}: {n_assigned}/{len(miss)} placed; {len(miss) - n_assigned} still detached")
    return out


## Set the paper style
paper_style.set_style()

## Recover the vignette crop centers in microns from the default panel orientation
## Fig4b_2 records the crop centers as pixels in the default orientation, so read them
## back to microns here (before the superior-up display orientation is installed)
regions = json.loads(open(regions_json).read())
crop_center_um = {}
crop_box_um = {}
for cond in CONDITIONS:
    sample = DISPLAY_REP[cond]
    reg = regions.get(sample)
    if reg:
        default_panel = load_panel(sample)
        crop_center_um[sample] = panel_px_to_um(default_panel, reg["cx"], reg["cy"])
        # Anchor and straighten the crop the same way Fig4b_2 does, then store the box corners
        ccx, ccy, rot_deg = vignette_crop_geometry(default_panel["layer"], reg["cx"], reg["cy"], tag=sample)
        crop_box_um[sample] = vignette_box_corners_um(default_panel, ccx, ccy, rot_deg)
retina_panel.SAMPLE_ORIENT_OVERRIDE.clear()

## Install the superior-up display orientation and load the injury response scores
spatial_maps.install_s1c_orientations()
scores = spatial_maps.load_scores()

## Join typed rod positions and assign arc positions
rel = pd.read_parquet(cells_path, columns=["cell_id", "sample_id", "cell_type", "arc_um", "cx_um", "cy_um"])
rel = rel[rel["cell_type"] == "rod"].rename(columns={"cell_id": "label", "sample_id": "sample"})
scores = scores.merge(rel[["sample", "label", "arc_um", "cx_um", "cy_um"]], on=["sample", "label"], how="left", validate="one_to_one")
scores = assign_offarc_arc_position(scores)
for cond in CONDITIONS:
    s = DISPLAY_REP[cond]
    joined = int(scores.loc[scores["sample"] == s, "arc_um"].notna().sum())
    n_rod = int((scores["sample"] == s).sum())
    print(f"rod->arc placement {s}: {joined}/{n_rod} rods on the arc strip")

## Set the map and arc color scales
disp = scores[scores["sample"].isin([DISPLAY_REP[c] for c in CONDITIONS])]
vmin, vmax = np.percentile(disp["injury_response_score"].to_numpy(), (2, 98))
print(f"map vmin/vmax = {vmin:.3f}/{vmax:.3f}")

## Build the oriented composites and arc profiles per condition
composites, widths_um, heights_um, profiles = {}, {}, {}, {}
bodies, subs, tissues, backbone_edges, panels = {}, {}, {}, {}, {}
for cond in CONDITIONS:
    sample = DISPLAY_REP[cond]
    panel = load_panel(sample)
    panel["tissue"] = load_unet_tissue_oriented(panel)
    panel["excl_mask"] = load_exclude_final_oriented(panel)
    body = load_cellbody_mask_oriented(sample, panel)
    sub = scores[scores["sample"] == sample]
    bodies[cond], subs[cond], tissues[cond] = body, sub, panel["tissue"]
    panels[cond] = panel
    rod_scores = dict(zip(sub["label"].astype(int), sub["injury_response_score"]))
    composites[cond] = spatial_maps.composite_rods(panel, body, rod_scores, vmin, vmax)
    widths_um[cond] = composites[cond].shape[1] * PX_DS_UM
    heights_um[cond] = composites[cond].shape[0] * PX_DS_UM
    au = sub.loc[sub["arc_um"].notna(), "arc_um"].to_numpy()
    # Use 40 µm bins for the map arrows and 25 µm columns for the strip
    backbone_edges[cond] = np.arange(float(np.floor(au.min())), float(np.ceil(au.max())) + ARC_BIN_UM, ARC_BIN_UM)
    profiles[cond] = shared_strip(sample)

all_means = np.concatenate([m[~np.isnan(m)] for _e, m in profiles.values()])
strip_vmin, strip_vmax = np.percentile(all_means, [2, 98])
print(f"strip vmin/vmax = {strip_vmin:.3f}/{strip_vmax:.3f}")

## Set up the figure layout
H_OV = 9.6
STRIP_H = 0.26
SG = 0.14
SB_BAND = 0.36
GG = 0.20
LM, RM, TM, BM = 0.42, 1.05, 0.62, 0.55
ov_w = {c: H_OV * widths_um[c] / heights_um[c] for c in CONDITIONS}
STRIP_W = min(ov_w.values())
strip_cmap = SCORE_CMAP.copy()
strip_cmap.set_bad("white")
fig_w = LM + sum(ov_w.values()) + GG * (len(CONDITIONS) - 1) + RM
fig_h = TM + H_OV + SB_BAND + SG + STRIP_H + BM
figure = plt.figure(figsize=(fig_w, fig_h))

def ax_at(x, y, w, h):
    return figure.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h])

top_in = fig_h - TM
strip_bottom = BM
map_bottom = BM + STRIP_H + SG + SB_BAND

## Draw each condition map with its arc strip
SCALE_BAR_UM = 500.0
cursor_x = LM
for cond in CONDITIONS:
    img = composites[cond]
    W_img = img.shape[1]
    axes = ax_at(cursor_x, map_bottom, ov_w[cond], H_OV)
    axes.imshow(img, interpolation="nearest", aspect="equal")
    axes.set_xticks([])
    axes.set_yticks([])
    for sp in axes.spines.values():
        sp.set_visible(False)
    # Outline the exact vignette footprint by projecting its four straightened corners into this orientation
    if DISPLAY_REP[cond] in crop_box_um:
        corners = []
        for x_um, y_um in crop_box_um[DISPLAY_REP[cond]]:
            px, py = project_xy_to_panel(panels[cond], np.array([x_um]), np.array([y_um]))
            corners.append((float(px[0]), float(py[0])))
        axes.add_patch(Polygon(corners, closed=True, fill=False, edgecolor="black", lw=1.8, zorder=15))
        print(f"  {cond} map box corners: {[(round(x), round(y)) for x, y in corners]}")
    # Nudge each condition title right over its map
    TITLE_DX_IN = 0.6
    figure.text((cursor_x + ov_w[cond] / 2 + TITLE_DX_IN) / fig_w, (top_in + 0.10) / fig_h,
             paper_style.condition_label(cond), ha="center", va="bottom", fontsize=32,
             fontweight="bold", color=paper_style.CONDITION_COLORS[cond])
    # One scale bar per map in the blank band below the tissue
    px_to_in = ov_w[cond] / W_img
    bar_len_in = (SCALE_BAR_UM / PX_DS_UM) * px_to_in
    x_right_in = cursor_x + ov_w[cond] - 0.02
    x_left_in = x_right_in - bar_len_in
    bar_y_in = map_bottom - SB_BAND * 0.55
    figure.add_artist(mlines.Line2D([x_left_in / fig_w, x_right_in / fig_w], [bar_y_in / fig_h, bar_y_in / fig_h],
                   color="black", lw=3.2, solid_capstyle="butt", zorder=20, transform=figure.transFigure))
    # Horizontal arc strip beneath with S left and I right
    edges, means = profiles[cond]
    strip_x = cursor_x + (ov_w[cond] - STRIP_W) / 2
    sax = ax_at(strip_x, strip_bottom, STRIP_W, STRIP_H)
    sax.imshow(np.ma.masked_invalid(means[::-1])[np.newaxis, :], aspect="auto", interpolation="nearest",
               cmap=strip_cmap, vmin=strip_vmin, vmax=strip_vmax, extent=[0, 1, 0, 1])
    sax.set_xlim(0, 1)
    sax.set_ylim(0, 1)
    smin_s, smax_s = float(edges[0]), float(edges[-1])
    span = smax_s - smin_s
    # Focal region cores come from the shared 25 um focal region table
    # The strip is oriented superior-left, so it uses the flipped positions; the map is physical tissue, so it uses raw arc positions
    core_s = focal.core_arc_positions(DISPLAY_REP[cond])
    core_cols = focal.focal_region_columns(DISPLAY_REP[cond])
    core_s_raw = core_cols[core_cols["focal_region_core"]]["arc_col_um"].to_numpy(float)
    # Box the same vignette crop center on the strip
    if DISPLAY_REP[cond] in crop_center_um:
        x_um, y_um = crop_center_um[DISPLAY_REP[cond]]
        box = crop_strip_frac(DISPLAY_REP[cond], x_um, y_um, smax_s, span)
        if box:
            bx0, bx1 = box
            sax.add_patch(Rectangle((bx0, -0.35), bx1 - bx0, 1.70, fill=False, edgecolor="black", lw=1.6, linestyle="-", zorder=8, clip_on=False))
    sax.set_yticks([])
    sax.set_xticks([0, 1])
    sax.set_xticklabels(["S", "I"], fontsize=26, fontweight="bold")
    for tl, col in zip(sax.get_xticklabels(), (SUP_COLOR, INF_COLOR)):
        tl.set_color(col)
    sax.tick_params(axis="x", length=0)
    for sp in sax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("#444444")
        sp.set_linewidth(0.7)
    sax.text(0.5, -0.7, "Arc position", transform=sax.transAxes, ha="center", va="top", fontsize=16)
    # Mark the focal region cores on the strip
    if SHOW_ARC_EXTREMA:
        draw_arc_extrema(sax, core_s, smax_s, span, tag=cond)
    # Mark the same cores along the map outer edge
    if SHOW_MAP_ARC_EXTREMA:
        draw_map_arc_extrema(axes, subs[cond], bodies[cond], tissues[cond], backbone_edges[cond], core_s_raw, tag=cond)
    cursor_x += ov_w[cond] + GG

## Add the shared score colorbar
cb_h = H_OV * 0.38
cax = ax_at(fig_w - RM + 0.12, map_bottom + (H_OV - cb_h) / 2, 0.14, cb_h)
sm = cm.ScalarMappable(norm=Normalize(vmin, vmax), cmap=SCORE_CMAP)
cb = plt.colorbar(sm, cax=cax)
cb.set_label("Rod injury response score", fontsize=15, fontweight="bold")
cb.ax.tick_params(labelsize=10)
cb.outline.set_linewidth(0.3)

## Add the peak legend and channel key
if SHOW_ARC_EXTREMA:
    leg_handles = [mlines.Line2D([0], [0], marker="^", linestyle="none", markersize=13,
                                 markerfacecolor="black", markeredgecolor="black", label="focal region center")]
    leg_handles.append(mlines.Line2D([0], [0], marker="s", linestyle="none", markersize=12,
                                     markerfacecolor="#b0b0b0", markeredgecolor="none", label="DAPI"))
    leg_handles.append(mlines.Line2D([0], [0], marker="s", linestyle="none", markersize=12,
                                     markerfacecolor="#606060", markeredgecolor="none", label="18S rRNA"))
    leg_x = (fig_w - RM - 0.55) / fig_w
    leg_y = (map_bottom + (H_OV + cb_h) / 2 + 0.18) / fig_h
    figure.legend(handles=leg_handles, loc="lower left", bbox_to_anchor=(leg_x, leg_y),
               frameon=False, fontsize=12, handletextpad=0.4, labelspacing=0.5, borderaxespad=0.0)

## Add the arc strip colorbar
cax_s = ax_at(fig_w - RM + 0.12, strip_bottom, 0.14, STRIP_H)
sm_s = cm.ScalarMappable(norm=Normalize(strip_vmin, strip_vmax), cmap=SCORE_CMAP)
cb_s = plt.colorbar(sm_s, cax=cax_s, orientation="vertical")
cb_s.set_label("Rod injury\nscore", fontsize=12, fontweight="bold")
cb_s.ax.tick_params(labelsize=11)
cb_s.outline.set_linewidth(0.3)

## Add the superior and inferior arrows on the left margin
x_si = (LM - 0.055) / fig_w
map_h_in = top_in - map_bottom
figure.text(x_si, (map_bottom + map_h_in * 0.83) / fig_h, "↑", ha="center", va="center", fontsize=30, fontweight="bold", color=SUP_COLOR)
figure.text(x_si, (map_bottom + map_h_in * 0.93) / fig_h, "S", ha="center", va="center", fontsize=22, fontweight="bold", color=SUP_COLOR)
figure.text(x_si, (map_bottom + map_h_in * 0.17) / fig_h, "↓", ha="center", va="center", fontsize=30, fontweight="bold", color=INF_COLOR)
figure.text(x_si, (map_bottom + map_h_in * 0.07) / fig_h, "I", ha="center", va="center", fontsize=22, fontweight="bold", color=INF_COLOR)

## Export the per bin arc mean table
os.makedirs(out_plot, exist_ok=True)
rows = []
for cond in CONDITIONS:
    edges, means = profiles[cond]
    centers = (edges[:-1] + edges[1:]) / 2.0
    rows.append(pd.DataFrame({"condition": cond, "sample": DISPLAY_REP[cond],
                              "arc_bin_center_um": centers, "arc_mean_score": means}))
tbl = pd.concat(rows, ignore_index=True)
tbl.to_parquet(f"{out_plot}/Fig4b_1.parquet", index=False)

## Save the panel
figure.savefig(out_panel, dpi=200, bbox_inches="tight")
plt.close(figure)
print("wrote", out_panel)
