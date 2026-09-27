# Build retinal arc skeletons and bin the tissue into fixed-width columns along the arc
# Each connected tissue piece gets its own arc coordinate system
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter1d, label as ndi_label
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from skimage.morphology import binary_opening, disk, skeletonize

sys.path.insert(0, str(Path(__file__).parent))
import _preprocessing_config as C
from _preprocessing_config import (
    ALL_SAMPLES, SAMPLE_TO_ROI, DROP_GENES, LAYER_CODE,
    LAYER_NAMES, OUT_TAB, OUT_FIG,
    PX_DS_UM, sample_to_condition,
    set_paper_style, DEFAULT_COL_WIDTH_UM, DEFAULT_SHIFT_FRAC,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared/code"))
from _paths import findpath_transcripts, findpath_unet_pred
import _exclude_final as excl

SKEL_SMOOTH_SIGMA_UM = 40.0  # Gaussian smoothing applied along the skeleton centerline

# Offsets to the eight neighbors of a skeleton pixel
NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),          (0, 1),
             (1, -1),  (1, 0),  (1, 1)]


def _component_arc(comp_mask):
    """Return the smoothed centerline of one tissue piece as arc coordinate arrays"""
    # Opening only stabilizes the centerline and never changes the analysis mask
    skel_input = binary_opening(comp_mask, disk(2))
    if not skel_input.any():
        skel_input = comp_mask
    # Reduce the tissue piece to a one-pixel-wide skeleton
    skel = skeletonize(skel_input)
    rs_pix, cs_pix = np.where(skel)
    if len(rs_pix) < 2:
        return None
    # Index every skeleton pixel and record its position in the pixel grid
    coords = np.stack([rs_pix, cs_pix], axis=1)
    idx_map = -np.ones(skel.shape, dtype=np.int64)
    idx_map[rs_pix, cs_pix] = np.arange(len(coords))
    # Build a graph edge between each skeleton pixel and its skeleton neighbors
    rows, cols, data = [], [], []
    for k, (r, c) in enumerate(coords):
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < skel.shape[0] and 0 <= nc < skel.shape[1] and skel[nr, nc]:
                nk = idx_map[nr, nc]
                if nk > k:
                    rows.append(k)
                    cols.append(nk)
                    data.append(float(np.hypot(dr, dc)))
    # Assemble a symmetric sparse adjacency matrix over the skeleton pixels
    N = len(coords)
    graph = csr_matrix((data + data, (rows + cols, cols + rows)), shape=(N, N))

    # Keep the largest connected run of skeleton pixels for the centerline
    n_sk, sk_labels = connected_components(graph, directed=False)
    if n_sk > 1:
        main = int(np.argmax(np.bincount(sk_labels)))
        sub = np.where(sk_labels == main)[0]
        if len(sub) < 2:
            return None
        coords = coords[sub]
        graph = graph[sub][:, sub]

    # Take the longest path across the skeleton with two farthest point searches
    dists, _ = dijkstra(graph, indices=0, return_predecessors=True)
    a = int(np.argmax(np.where(np.isinf(dists), -1, dists)))
    dists, preds = dijkstra(graph, indices=a, return_predecessors=True)
    b = int(np.argmax(np.where(np.isinf(dists), -1, dists)))
    # Walk the predecessor chain from the far endpoint back to the start
    path = [b]
    while path[-1] != a and preds[path[-1]] != -9999:
        path.append(int(preds[path[-1]]))
    path = path[::-1]
    if len(path) < 2:
        return None
    # Smooth the path coordinates along the arc to stabilize the centerline
    pts = coords[path]
    rs = pts[:, 0].astype(float)
    cs = pts[:, 1].astype(float)
    sigma_idx = max(1.0, min(SKEL_SMOOTH_SIGMA_UM / PX_DS_UM, len(rs) / 4))
    rs_s = gaussian_filter1d(rs, sigma=sigma_idx, mode="nearest")
    cs_s = gaussian_filter1d(cs, sigma=sigma_idx, mode="nearest")
    return rs_s, cs_s


def build_one_sample(sample, col_width_um, shift_frac):
    """Build arc columns, per-column layer areas, and gene densities for one sample"""
    roi = SAMPLE_TO_ROI[sample]
    print(f"  {sample} (roi{roi})")

    # Load the U-Net layer prediction and check its codes are as expected
    layer_path = findpath_unet_pred(sample, "layer")
    layer = tifffile.imread(layer_path)
    layer_codes = set(np.unique(layer).tolist())
    expected_codes = {0, *LAYER_CODE.values()}
    if not layer_codes <= expected_codes:
        raise ValueError(
            f"{sample}: expected layer codes {sorted(expected_codes)}, "
            f"found {sorted(layer_codes)} in {layer_path}"
        )
    # Treat every non-background layer pixel as tissue
    tissue = layer > 0
    H, W = layer.shape

    # Clear the excluded region from tissue and layer before anything else
    tissue, layer, n_excl = excl.apply_to_layer(sample, tissue, layer)
    if n_excl:
        print(f"    cleared {n_excl} excluded pixels from tissue and layer")

    if tissue.sum() < 500:
        raise RuntimeError(f"{sample}: tissue too small")
    print(f"    layer shape {H}x{W}, tissue px {int(tissue.sum())}")

    # Label every contiguous tissue piece and build an arc for each one
    comp_labels, n_comp = ndi_label(tissue, structure=np.ones((3, 3), dtype=int))
    arcs, flagged = [], []
    for lab in range(1, n_comp + 1):
        comp_mask = comp_labels == lab
        arc = _component_arc(comp_mask)
        if arc is None:
            flagged.append((lab, int(comp_mask.sum())))
            continue
        rs_s, cs_s = arc
        # A piece must span at least one column to be binned along the arc
        arc_len_um = float(np.sum(np.hypot(np.diff(rs_s), np.diff(cs_s)))) * PX_DS_UM
        if arc_len_um < col_width_um:
            flagged.append((lab, int(comp_mask.sum())))
            continue
        # Orient the arc so it runs from the superior end to the inferior end
        if rs_s[0] > rs_s[-1]:
            rs_s, cs_s = rs_s[::-1], cs_s[::-1]
        arcs.append((lab, rs_s, cs_s))
    if flagged:
        detail = ", ".join(f"label {lab} ({px} px)" for lab, px in flagged)
        print(f"    FLAG {len(flagged)} tissue piece(s) too small for an arc, retained but not analyzed: {detail}")
    if not arcs:
        raise RuntimeError(f"{sample}: no tissue piece supports arc construction")
    # Order tissue pieces superior first
    arcs.sort(key=lambda a: a[1][0])
    print(f"    {len(arcs)} tissue piece(s)")

    # Load transcripts and place them on the layer grid
    tlist = pd.read_csv(findpath_transcripts(sample))
    print(f"    transcripts {len(tlist):,}")
    # Drop transcripts of the blocklisted genes
    keep_tx = ~tlist["name"].isin(DROP_GENES)
    tlist = tlist.loc[keep_tx].reset_index(drop=True)
    # Map each transcript onto the downsampled layer grid
    xy_um = tlist[["x", "y"]].to_numpy(dtype=np.float64)
    px_tx = np.clip(np.rint(xy_um[:, 0] / PX_DS_UM).astype(np.int64), 0, W - 1)
    py_tx = np.clip(np.rint(xy_um[:, 1] / PX_DS_UM).astype(np.int64), 0, H - 1)
    # Flag transcripts that fall within the grid bounds
    inb = ((xy_um[:, 0] >= 0) & (xy_um[:, 0] < W * PX_DS_UM)
           & (xy_um[:, 1] >= 0) & (xy_um[:, 1] < H * PX_DS_UM))
    # Look up each transcript's layer code and tissue piece
    tx_layer = np.where(inb, layer[py_tx, px_tx], 0)
    tx_comp = np.where(inb, comp_labels[py_tx, px_tx], 0)
    tx_names = tlist["name"].to_numpy()
    layered_codes = [LAYER_CODE[n] for n in LAYER_NAMES]

    px_area_um2 = PX_DS_UM ** 2
    col_metas, long_dfs, skel_dfs = [], [], []
    next_col = 0

    for ci, (lab, rs, cs) in enumerate(arcs):
        # Cumulative arc length in µm at each skeleton point
        seg_px = np.hypot(np.diff(rs), np.diff(cs))
        s_um = np.concatenate([[0.0], np.cumsum(seg_px)]) * PX_DS_UM
        # Spatial index over this piece's arc points for nearest-point lookups
        comp_tree = cKDTree(np.column_stack([rs, cs]))

        # Column edges anchored within this tissue piece
        s_lo_comp = float(s_um.min())
        s_hi_comp = float(s_um.max())
        anchor = s_lo_comp + shift_frac * col_width_um
        n_left = int(np.ceil((anchor - s_lo_comp) / col_width_um))
        n_right = int(np.ceil((s_hi_comp - anchor) / col_width_um))
        edges = anchor + np.arange(-n_left, n_right + 1) * col_width_um
        n_cols = len(edges) - 1
        if n_cols < 1:
            continue

        # Tissue pixels in this piece assigned to the nearest arc point
        ys, xs = np.where(comp_labels == lab)
        _, pix_idx = comp_tree.query(np.column_stack([ys.astype(float), xs.astype(float)]))
        comp_pix_s = s_um[pix_idx]
        comp_pix_layer = layer[ys, xs]
        comp_pix_col = np.clip(np.digitize(comp_pix_s, edges, right=False) - 1, 0, n_cols - 1)

        # Accumulate per-column layer area and total tissue area from the assigned pixels
        layer_area_um2 = np.zeros((n_cols, len(LAYER_NAMES)), dtype=np.float64)
        col_tissue_area = np.zeros(n_cols, dtype=np.float64)
        for li, lname in enumerate(LAYER_NAMES):
            code = LAYER_CODE[lname]
            m = comp_pix_layer == code
            if m.any():
                np.add.at(layer_area_um2[:, li], comp_pix_col[m], px_area_um2)
        np.add.at(col_tissue_area, comp_pix_col, px_area_um2)

        # Transcripts in this piece within the annotated layers assigned to the nearest arc point
        piece_tx = inb & (tx_comp == lab) & np.isin(tx_layer, layered_codes)
        _, tx_idx = comp_tree.query(np.column_stack([py_tx[piece_tx].astype(float), px_tx[piece_tx].astype(float)]))
        comp_tx_s = s_um[tx_idx]
        comp_tx_col = np.clip(np.digitize(comp_tx_s, edges, right=False) - 1, 0, n_cols - 1)
        comp_tx_names = tx_names[piece_tx]
        comp_tx_layer_codes = tx_layer[piece_tx]

        # Column IDs unique within sample
        col_ids = np.arange(n_cols) + next_col
        # Column midpoints along the arc and transcript count per column
        s_mid = 0.5 * (edges[:-1] + edges[1:])
        tx_per_col = np.bincount(comp_tx_col, minlength=n_cols)

        # One row per column with its geometry, per-layer areas, and transcript count
        col_meta = pd.DataFrame({
            "sample": sample,
            "condition": sample_to_condition(sample),
            "component_id": ci,
            "col_id_local": col_ids,
            "s_mid_um": s_mid,
            "s_lo_um": edges[:-1],
            "s_hi_um": edges[1:],
            "tissue_area_um2": col_tissue_area,
            "ONL_area_um2": layer_area_um2[:, LAYER_NAMES.index("ONL")],
            "INL_area_um2": layer_area_um2[:, LAYER_NAMES.index("INL")],
            "IPL_area_um2": layer_area_um2[:, LAYER_NAMES.index("IPL")],
            "GCL_area_um2": layer_area_um2[:, LAYER_NAMES.index("GCL")],
            "n_tx_in_col": tx_per_col,
        })

        # Build the per-column, per-gene, per-layer transcript density table
        if len(comp_tx_names):
            df_tx = pd.DataFrame({
                "col_id_local": comp_tx_col + next_col,
                "gene": comp_tx_names,
                "layer_code": comp_tx_layer_codes,
            })
            # Count transcripts of each gene per column and layer
            counts = (df_tx.groupby(["col_id_local", "gene", "layer_code"], observed=True)
                      .size().rename("count").reset_index())
            counts["layer"] = counts["layer_code"].map({v: k for k, v in LAYER_CODE.items()})
            # Attach each column and layer's area to normalize the counts
            layer_area_lookup = pd.DataFrame({
                "col_id_local": np.repeat(col_ids, len(LAYER_NAMES)),
                "layer": np.tile(LAYER_NAMES, n_cols),
                "layer_area_um2": layer_area_um2.ravel(),
            })
            long_df = counts.merge(layer_area_lookup, on=["col_id_local", "layer"], how="left", validate="many_to_one")
            # Keep only rows where the layer has area, then compute densities
            long_df = long_df[long_df["layer_area_um2"] > 0].copy()
            long_df["density_um2"] = long_df["count"] / long_df["layer_area_um2"]
            long_df["count_per_arc_um"] = long_df["count"] / float(col_width_um)
            long_df["sample"] = sample
            long_df["condition"] = sample_to_condition(sample)
            long_dfs.append(long_df)

        # Record the skeleton points and their arc coordinates for this piece

        skel_dfs.append(pd.DataFrame({
            "sample": sample,
            "component_id": ci,
            "idx": np.arange(len(rs)),
            "r_ds8": rs, "c_ds8": cs,
            "s_um": s_um,
        }))
        col_metas.append(col_meta)
        next_col += n_cols
        print(f"    component {ci}: {len(rs)} skel pts, {n_cols} cols, arc {s_um[-1]:.0f} um")

    # Combine the per-piece tables into one set of tables for the sample
    col_meta_all = pd.concat(col_metas, ignore_index=True)

    if long_dfs:
        long_all = pd.concat(long_dfs, ignore_index=True)
    else:
        long_all = pd.DataFrame()

    skel_all = pd.concat(skel_dfs, ignore_index=True)
    return col_meta_all, long_all, skel_all


def render_qc_figure(col_meta, out_path):
    """Render a per-sample QC panel of column tissue coverage and layer areas"""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    set_paper_style()

    samples = ALL_SAMPLES
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(
        4, 5, figure=fig, wspace=0.30, hspace=0.55,
        top=0.94, bottom=0.06, left=0.06, right=0.985,
    )

    for i, sample in enumerate(samples):
        sub = col_meta[col_meta["sample"] == sample].sort_values("col_id_local")
        has_tissue = sub["tissue_area_um2"].to_numpy() > 0
        # Top row: mark each column green where it holds tissue and red where empty
        ax_top = fig.add_subplot(gs[(i // 5) * 2, i % 5])
        x = np.arange(len(sub))
        ax_top.fill_between(x, 0, 1, where=has_tissue,
                            color="#1b9e3b", alpha=0.6, step="mid", linewidth=0)
        ax_top.fill_between(x, 0, 1, where=~has_tissue,
                            color="#aa3322", alpha=0.6, step="mid", linewidth=0)
        ax_top.set_ylim(0, 1)
        ax_top.set_yticks([])
        ax_top.set_title(sample, fontsize=8.5,
                         color="black" if sample.startswith("WT") else "#c0392b")
        ax_top.set_xlabel("")
        n_tissue = int(has_tissue.sum())
        n_total = len(sub)
        n_comps = int(sub["component_id"].nunique())
        ax_top.text(0.99, 0.95, f"{n_tissue}/{n_total} ({n_comps}c)",
                    transform=ax_top.transAxes, ha="right", va="top", fontsize=7.5)
        # Bottom row: plot each layer's area across the tissue-bearing columns
        ax_bot = fig.add_subplot(gs[(i // 5) * 2 + 1, i % 5])
        keep_sub = sub[has_tissue].copy()
        if not len(keep_sub):
            ax_bot.set_xticks([])
            ax_bot.set_yticks([])
            continue
        x = np.arange(len(keep_sub))
        ax_bot.plot(x, keep_sub["ONL_area_um2"].to_numpy(), color="#5f6fbd", lw=1.0, label="ONL")
        ax_bot.plot(x, keep_sub["INL_area_um2"].to_numpy(), color="#e08e3a", lw=1.0, label="INL")
        ax_bot.plot(x, keep_sub["IPL_area_um2"].to_numpy(), color="#7da041", lw=1.0, label="IPL")
        ax_bot.plot(x, keep_sub["GCL_area_um2"].to_numpy(), color="#a4569a", lw=1.0, label="GCL")
        ax_bot.set_xlim(0, len(keep_sub) - 1)
        ax_bot.set_xlabel("column index", fontsize=8)
        ax_bot.set_ylabel("area (um2)", fontsize=8)
        ax_bot.tick_params(axis="both", labelsize=7)
        if i == 0:
            ax_bot.legend(fontsize=6.5, ncols=2, frameon=False,
                          loc="upper right", handlelength=1.1, columnspacing=0.7)

    fig.suptitle("Full thickness columns (tissue = green; empty = red)", fontsize=11, y=0.985)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(col_width_um, shift_frac, run_tag=None):
    """Build arc columns for all samples and write the combined column tables"""
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    print(f"build_columns: width={col_width_um}  shift_frac={shift_frac}")
    # Skip any samples named in the FTM_HELD_OUT environment variable
    held_out = set(os.environ.get("FTM_HELD_OUT", "").split(",")) - {""}
    samples = [s for s in ALL_SAMPLES if s not in held_out]
    if held_out:
        print(f"  held out: {sorted(held_out)}")
    col_metas, longs, skels = [], [], []
    # Offset each sample's local column IDs so col_id is unique across all samples
    next_global = 0
    for sample in samples:
        col_meta, long_df, skel_df = build_one_sample(
            sample, col_width_um, shift_frac,
        )
        col_meta["col_id"] = col_meta["col_id_local"] + next_global
        if len(long_df):
            long_df["col_id"] = long_df["col_id_local"] + next_global
        next_global += len(col_meta)
        col_metas.append(col_meta)
        longs.append(long_df)
        skels.append(skel_df)
    # Combine the per-sample tables into one table each
    col_meta_all = pd.concat(col_metas, ignore_index=True)
    long_all = pd.concat(longs, ignore_index=True)
    skel_all = pd.concat(skels, ignore_index=True)

    # Tag the output filenames when this is a parameter-sweep run
    suffix = "" if run_tag is None else f"_{run_tag}"
    long_path = OUT_TAB / f"per_column_gene_layer_long{suffix}.parquet"
    meta_path = OUT_TAB / f"columns_meta{suffix}.parquet"
    skel_path = OUT_TAB / f"skeletons{suffix}.parquet"

    col_meta_all.to_parquet(meta_path, index=False)
    long_all.to_parquet(long_path, index=False)
    skel_all.to_parquet(skel_path, index=False)
    print(f"\nwrote {meta_path}  ({len(col_meta_all):,} columns)")
    n_tissue = int((col_meta_all["tissue_area_um2"] > 0).sum())
    print(f"      with tissue: {n_tissue:,}")
    print(f"      {long_path.name}  rows={len(long_all):,}")
    print(f"      {skel_path.name}  rows={len(skel_all):,}")

    # Render the QC figure only for the canonical default run
    if run_tag is None:
        fig_path = OUT_FIG / "01_column_geometry_qc.png"
        render_qc_figure(col_meta_all, fig_path)
        print(f"      {fig_path}")


def parse_args():
    """Parse the command-line width, shift, tag, and rebuild options"""
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=float, default=DEFAULT_COL_WIDTH_UM)
    p.add_argument("--shift", type=float, default=DEFAULT_SHIFT_FRAC)
    p.add_argument("--tag", default=None)
    p.add_argument("--base-density", type=Path)
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()
    if args.base_density and args.rebuild:
        p.error("choose at most one of --base-density FILE or --rebuild")
    if args.base_density and (args.tag or args.width != DEFAULT_COL_WIDTH_UM or args.shift != DEFAULT_SHIFT_FRAC):
        p.error("width, shift and tag apply only to --rebuild")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.base_density:
        # Reuse an existing density table instead of rebuilding columns
        density = pd.read_parquet(args.base_density)
    else:
        # Build the column tables, then exit early for non-default sweep runs
        main(args.width, args.shift, args.tag)
        if args.tag is not None or args.width != DEFAULT_COL_WIDTH_UM or args.shift != DEFAULT_SHIFT_FRAC:
            raise SystemExit(0)
        density = pd.read_parquet(OUT_TAB / "per_column_gene_layer_long.parquet")
    # Rename to the release schema and select the published density columns
    density = density.rename(columns={"sample": "sample_id", "count": "transcript_count"})
    final = density[["sample_id", "condition", "col_id_local", "col_id", "gene", "layer",
                     "transcript_count", "layer_area_um2", "density_um2", "count_per_arc_um"]]
    data_raw = C.ROOT / "_shared" / "data_raw"
    data_raw.mkdir(parents=True, exist_ok=True)
    final_path = data_raw / "gene_density_by_layer.parquet"
    final.to_parquet(final_path, index=False)
    print(f"Wrote {len(final):,} density rows to {final_path}")
