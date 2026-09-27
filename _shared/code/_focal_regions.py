# Detect rod injury response focal regions along the retinal arc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _superior_inferior_axis as sih  # Provides the sigma 120 cone-opsin S/I field
from _paths import PX_DS_UM

# Define the data paths
_ROOT = Path(__file__).resolve().parents[2]   # Points at analysis/retina_paper_final
FT = _ROOT / "_shared/data_raw/full_thickness_tables"
POOL = _ROOT / "figure_4/data_raw_manifest/all_rods.parquet"
SCORE = _ROOT / "figure_4/data_raw_manifest/rod_injury_response_score.parquet"
PER_CELL = _ROOT / "_shared/data_processed/per_cell_gene_counts.parquet"
DENS = _ROOT / "_shared/data_raw/gene_density_by_layer.parquet"

COL_WIDTH_UM = 25.0
SMOOTH_SIGMA_UM = 37.5
PROMINENCE_MIN = 0.10
PROGRAM_GENES = ["Edn2", "Socs3", "Fgf2", "Stat3"]


# Require the four-gene injury response score
def _check_program(program_genes):
    if program_genes is not None and set(program_genes) != set(PROGRAM_GENES):
        raise ValueError("focal regions require the four gene injury response score; "
                         "build reduced scores before calling this module")

# Assign each rod to the nearest 25 µm column within its tissue piece
def _project_nearest(df, skel, cm):
    tree = cKDTree(np.column_stack([skel.r_ds8, skel.c_ds8]))
    _distance, index = tree.query(np.column_stack([df.cy_um / PX_DS_UM, df.cx_um / PX_DS_UM]), k=1)
    s_um = skel.s_um.values[index]
    comp = skel.component_id.values[index]
    col_ids = np.zeros(len(df), dtype=np.int64)
    for c in sorted(cm.component_id.unique()):
        mask = comp == c
        if not mask.any():
            continue
        cm_c = cm[cm.component_id == c].sort_values("s_lo_um").reset_index(drop=True)
        col_in_comp = np.clip(
            np.searchsorted(cm_c.s_lo_um.values, s_um[mask], side="right") - 1,
            0, len(cm_c) - 1)
        col_ids[mask] = cm_c.col_id_local.values[col_in_comp]
    return df.assign(col_id_local=col_ids)


# Compute the mean rod injury response score in each 25 µm column
def _column_profile(sample, score_override=None):
    skel = pd.read_parquet(FT / "skeletons.parquet")
    skel = skel[skel["sample"] == sample]
    cm = pd.read_parquet(FT / "columns_meta.parquet")
    cm = cm[cm["sample"] == sample].sort_values("col_id_local").reset_index(drop=True)

    # Use score_override when a score with one gene omitted is supplied
    pool = pd.read_parquet(POOL, columns=["sample", "label", "cx_um", "cy_um"])
    pool = pool[pool["sample"] == sample].copy()
    if score_override is None:
        score = pd.read_parquet(SCORE, columns=["sample", "label", "injury_response_score"])
        score = score[score["sample"] == sample][["label", "injury_response_score"]]
    else:
        score = score_override[["label", "injury_response_score"]]
    pool = pool.merge(score, on="label", how="inner", validate="one_to_one")

    # Add the smoothed cone-opsin field used for detrending
    si_src = pd.read_parquet(PER_CELL, columns=["cell_id", "sample_id", "cell_type",
                             "cx_um", "cy_um", "body_Opn1mw", "body_Opn1sw"])
    si_src = si_src[(si_src.sample_id == sample)
                    & si_src.cell_type.isin(["rod", "cone"])].copy()
    si_src["si_smooth"] = sih.smoothed_opsin_field(si_src)
    rod_si = si_src[si_src.cell_type == "rod"][["cell_id", "si_smooth"]].rename(columns={"cell_id": "label"})
    pool = pool.merge(rod_si, on="label", how="left", validate="one_to_one")

    # Add the opsin field used to orient the display
    opsin = pd.read_parquet(PER_CELL, columns=["cell_id", "sample_id", "opsin_field"])
    opsin = opsin[opsin.sample_id == sample].rename(columns={"cell_id": "label"})[["label", "opsin_field"]]
    pool = pool.merge(opsin, on="label", how="left", validate="one_to_one")

    projected = _project_nearest(pool, skel, cm)
    agg = projected.groupby("col_id_local").agg(
        n_rods=("injury_response_score", "size"),
        raw_program_score=("injury_response_score", "mean"),
        si_smooth=("si_smooth", "mean"),
        opsin_field=("opsin_field", "mean"))
    valid_ids = set(agg.index)
    cm = cm.assign(valid_tissue=cm["col_id_local"].isin(valid_ids))

    # Keep the full grid so empty columns remain as gaps
    grid = (cm[["col_id_local", "component_id", "s_mid_um", "valid_tissue"]]
            .rename(columns={"s_mid_um": "arc_col_um"}).set_index("col_id_local"))
    table = grid.join(agg).reset_index().sort_values("col_id_local").reset_index(drop=True)
    table["n_rods"] = table["n_rods"].fillna(0).astype(int)
    # Number the contiguous runs of scored columns within each tissue piece
    valid = table["valid_tissue"].to_numpy(bool)
    comp = table["component_id"].to_numpy(int)
    segment_id = np.full(len(valid), -1, dtype=int)
    seg = -1
    prev_valid, prev_comp = False, -999
    for i in range(len(valid)):
        if valid[i] and (not prev_valid or comp[i] != prev_comp):
            seg += 1
        if valid[i]:
            segment_id[i] = seg
        prev_valid = valid[i]
        prev_comp = comp[i]
    table["segment_id"] = segment_id
    return table


# Regress the column score against the cone-opsin field
def _residual(table):
    raw = table["raw_program_score"].to_numpy(float)
    field = table["si_smooth"].to_numpy(float)
    trend = np.full_like(raw, np.nan)
    ok = np.isfinite(raw) & np.isfinite(field)
    if ok.sum() >= 5 and np.std(field[ok]) > 1e-9:
        coefficients = np.polyfit(field[ok], raw[ok], 1)
        trend[ok] = np.polyval(coefficients, field[ok])
    elif ok.any():
        trend[ok] = float(np.mean(raw[ok]))
    return trend, raw - trend


# Smooth within contiguous segments while respecting component boundaries
def smooth_residual(residual, sigma_um, segment_id=None):
    residual = np.asarray(residual, float)
    smoothed = np.full_like(residual, np.nan)
    if segment_id is not None:
        segment_id = np.asarray(segment_id, int)
        for sid in sorted(set(segment_id[segment_id >= 0])):
            idx = np.where(segment_id == sid)[0]
            seg = residual[idx]
            smoothed[idx] = (seg if (sigma_um <= 0 or len(seg) < 2)
                             else gaussian_filter1d(seg, sigma_um / COL_WIDTH_UM, mode="nearest"))
        return smoothed
    # Fall back for callers without segment_id, such as arc_strip run per component
    finite = np.isfinite(residual)
    start = None
    for i in range(len(residual) + 1):
        inside = i < len(residual) and finite[i]
        if inside and start is None:
            start = i
        elif not inside and start is not None:
            segment = residual[start:i]
            smoothed[start:i] = (segment if sigma_um <= 0
                                 else gaussian_filter1d(segment, sigma_um / COL_WIDTH_UM, mode="nearest"))
            start = None
    return smoothed


# Evaluate a segment endpoint as a truncated focal region candidate
def _endpoint_peak(segment, idx, side, accepted, seg_len, prominence_min):
    peak_val = float(segment[idx])
    if side == "left":
        boundary = accepted[0]["peak"] if accepted else seg_len
        if boundary < 2:
            return None
        observed_min = float(np.min(segment[1:boundary]))
    else:
        boundary = accepted[-1]["peak"] if accepted else 0
        search_lo = boundary + 1 if accepted else 0
        if search_lo >= seg_len - 1:
            return None
        observed_min = float(np.min(segment[search_lo:seg_len - 1]))
    if peak_val - observed_min < prominence_min or peak_val < prominence_min:
        return None
    prom = peak_val - observed_min
    half_h = peak_val - prom / 2
    if side == "left":
        lip = 0.0
        rip = float(seg_len - 1)
        for j in range(1, seg_len):
            if segment[j] <= half_h:
                d = segment[j - 1] - segment[j]
                rip = ((j - 1) + (segment[j - 1] - half_h) / d) if d > 0 else float(j)
                break
        return {"peak": 0, "lip": lip, "rip": rip,
                "prominence": prom, "left_open": True, "right_open": False}
    rip = float(seg_len - 1)
    lip = 0.0
    for j in range(seg_len - 2, -1, -1):
        if segment[j] <= half_h:
            d = segment[j + 1] - segment[j]
            lip = (j + (half_h - segment[j]) / d) if d > 0 else float(j)
            break
    return {"peak": seg_len - 1, "lip": lip, "rip": rip,
            "prominence": prom, "left_open": False, "right_open": True}


# Interior peaks use prominence measured from both sides, while an edge peak must exceed both the
# observed minimum and zero and the segment edge becomes its truncated boundary; detection runs
# per segment_id so disconnected tissue pieces are never merged
def _detect(smoothed, prominence_min=PROMINENCE_MIN, is_valid=None, segment_id=None):
    n = len(smoothed)
    if is_valid is None:
        is_valid = np.zeros(n, dtype=bool)
    focal_regions = []

    if segment_id is not None:
        segment_id = np.asarray(segment_id, int)
        ranges = []
        for sid in sorted(set(segment_id[segment_id >= 0])):
            idx = np.where(segment_id == sid)[0]
            ranges.append((int(idx[0]), int(idx[-1]) + 1))
    else:
        finite = np.isfinite(smoothed)
        ranges = []
        run_start = None
        for i in range(n + 1):
            inside = i < n and finite[i]
            if inside and run_start is None:
                run_start = i
            elif not inside and run_start is not None:
                ranges.append((run_start, i))
                run_start = None

    for start, end in ranges:
        segment = smoothed[start:end]
        seg_len = len(segment)
        if seg_len < 1:
            continue
        left_is_edge = (start == 0 or not is_valid[start - 1])
        right_is_edge = (end >= n or not is_valid[end])

        peaks, _ = find_peaks(segment)
        accepted = []
        if len(peaks):
            prominences, left_bases, right_bases = peak_prominences(segment, peaks)
            _widths, _heights, left_ips, right_ips = peak_widths(segment, peaks, rel_height=0.5)
            for k, pk in enumerate(peaks):
                at_left_edge = (left_bases[k] == 0) and left_is_edge
                at_right_edge = (right_bases[k] == seg_len - 1) and right_is_edge
                lo_open = k == 0 and at_left_edge
                ro_open = k == len(peaks) - 1 and at_right_edge
                peak_val = float(segment[pk])

                if lo_open or ro_open:
                    if lo_open and not ro_open:
                        observed_min = float(segment[right_bases[k]])
                    elif ro_open and not lo_open:
                        observed_min = float(segment[left_bases[k]])
                    else:
                        observed_min = min(float(segment[left_bases[k]]),
                                           float(segment[right_bases[k]]))
                    if peak_val - observed_min < prominence_min:
                        continue
                    if peak_val < prominence_min:
                        continue
                    lip = 0.0 if lo_open else float(left_ips[k])
                    rip = float(seg_len - 1) if ro_open else float(right_ips[k])
                    prom = peak_val - observed_min      # For an edge peak, measure height above the interior valley only
                else:
                    if prominences[k] < prominence_min:
                        continue
                    lip = float(left_ips[k])
                    rip = float(right_ips[k])
                    prom = float(prominences[k])

                accepted.append({"peak": int(pk), "lip": lip, "rip": rip,
                                 "prominence": float(prom),
                                 "left_open": lo_open, "right_open": ro_open})

        # Evaluate the first and last columns under the truncated-peak rule
        if left_is_edge and seg_len >= 2 and segment[0] > segment[1]:
            if not accepted or accepted[0]["peak"] > 0:
                ep = _endpoint_peak(segment, 0, "left", accepted, seg_len, prominence_min)
                if ep is not None:
                    accepted.insert(0, ep)
        if right_is_edge and seg_len >= 2 and segment[-1] > segment[-2]:
            if not accepted or accepted[-1]["peak"] < seg_len - 1:
                ep = _endpoint_peak(segment, seg_len - 1, "right", accepted, seg_len, prominence_min)
                if ep is not None:
                    accepted.append(ep)

        # Split overlapping regions at their shared minimum
        for a, b in zip(accepted, accepted[1:]):
            valley = a["peak"] + int(np.argmin(segment[a["peak"]:b["peak"] + 1]))
            a["rip"] = min(a["rip"], float(valley))
            b["lip"] = max(b["lip"], float(valley))
        for a in accepted:
            left_ip_g, right_ip_g = start + a["lip"], start + a["rip"]
            core_idx = start + a["peak"]
            lo = min(core_idx, max(start, int(np.ceil(left_ip_g))))
            hi = max(core_idx, min(end - 1, int(np.floor(right_ip_g))))
            focal_regions.append({
                "core": core_idx, "lo": lo, "hi": hi,
                "left_ip": left_ip_g, "right_ip": right_ip_g,
                "prominence": a["prominence"],
                "width_um": (a["rip"] - a["lip"]) * COL_WIDTH_UM,
                "truncated": bool(a["left_open"] or a["right_open"]),
                "left_open": bool(a["left_open"]),
                "right_open": bool(a["right_open"])})
    return focal_regions


# Return one row per column with its focal region membership
def focal_region_columns(sample, program_genes=None, sigma_um=None, score_override=None, prominence_min=None):
    _check_program(program_genes)
    sigma_um = SMOOTH_SIGMA_UM if sigma_um is None else sigma_um
    prominence_min = PROMINENCE_MIN if prominence_min is None else prominence_min
    table = _column_profile(sample, score_override)
    trend, residual = _residual(table)
    seg_ids = table["segment_id"].to_numpy(int)
    smoothed = smooth_residual(residual, sigma_um, segment_id=seg_ids)
    n = len(smoothed)

    focal_region_id = np.full(n, -1, dtype=int)
    core = np.zeros(n, dtype=bool)
    edge = np.zeros(n, dtype=bool)
    left_trunc = np.zeros(n, dtype=bool)
    right_trunc = np.zeros(n, dtype=bool)
    is_valid = table["valid_tissue"].to_numpy(bool)
    for region, spot in enumerate(_detect(smoothed, prominence_min, is_valid, segment_id=seg_ids)):
        focal_region_id[spot["lo"]:spot["hi"] + 1] = region
        core[spot["core"]] = True
        if spot["truncated"]:
            edge[spot["lo"]:spot["hi"] + 1] = True
        if spot.get("left_open"):
            left_trunc[spot["lo"]:spot["hi"] + 1] = True
        if spot.get("right_open"):
            right_trunc[spot["lo"]:spot["hi"] + 1] = True

    return table.assign(sample=sample, si_trend=trend, residual=residual,
                        smoothed_residual=smoothed, focal_region_id=focal_region_id,
                        in_focal_region=focal_region_id >= 0,
                        out_of_region=np.isfinite(smoothed) & (focal_region_id < 0),
                        focal_region_core=core, edge_truncated=edge,
                        left_truncated=left_trunc, right_truncated=right_trunc)


# Locate each focal region center on the oriented arc
def core_arc_positions(sample, program_genes=None, sigma_um=None, drop_edge_truncated=False):
    cols = focal_region_columns(sample, program_genes, sigma_um)
    cores = cols[cols["focal_region_core"]]
    if drop_edge_truncated:
        cores = cores[~cores["edge_truncated"]]
    positions = cores["arc_col_um"].to_numpy(float)
    valid = cols[cols["valid_tissue"]].sort_values("arc_col_um")
    opsin = valid["opsin_field"].to_numpy(float)
    if _opsin_orientation(opsin):
        arc = valid["arc_col_um"].to_numpy(float)
        positions = arc[0] + arc[-1] - positions
    return positions


# Assign each rod its column and focal region membership
def rod_membership(sample):
    skel = pd.read_parquet(FT / "skeletons.parquet")
    skel = skel[skel["sample"] == sample]
    cm = pd.read_parquet(FT / "columns_meta.parquet")
    cm = cm[cm["sample"] == sample].sort_values("col_id_local").reset_index(drop=True)
    pool = pd.read_parquet(POOL, columns=["sample", "label", "cx_um", "cy_um"])
    pool = pool[pool["sample"] == sample].copy()
    pool = _project_nearest(pool, skel, cm)
    cols = focal_region_columns(sample)[["col_id_local", "valid_tissue", "segment_id",
                                    "in_focal_region", "out_of_region", "focal_region_id"]]
    return pool[["label", "col_id_local"]].merge(cols, on="col_id_local", how="left", validate="many_to_one")


# Select flanking reference columns, where each side extends up to one region width and stops at a
# gap or another region; complete regions use equal widths on both sides while truncated regions
# use only the observed side
def local_reference(sample, score_override=None, require_rods=True):
    cols = focal_region_columns(sample, score_override=score_override)
    colid = cols["col_id_local"].to_numpy(int)
    seg = cols["segment_id"].to_numpy(int)
    valid = cols["valid_tissue"].to_numpy(bool)
    in_region = cols["in_focal_region"].to_numpy(bool)
    region_id = cols["focal_region_id"].to_numpy(int)
    has_rods = cols["n_rods"].to_numpy(int) > 0
    lt = cols["left_truncated"].to_numpy(bool)
    rt = cols["right_truncated"].to_numpy(bool)
    n = len(cols)
    rows = []

    def flank(start_idx, step, seg_h, window):
        found, p = [], start_idx
        for _ in range(window):
            if p < 0 or p >= n or seg[p] != seg_h or not valid[p] or in_region[p]:
                break
            if not require_rods or has_rods[p]:
                found.append(p)
            p += step
        return found

    for h in sorted(set(region_id[region_id >= 0])):
        in_idx = np.where(region_id == h)[0]
        seg_h = int(seg[in_idx[0]])
        n_focal = len(in_idx)
        is_lt = bool(lt[in_idx[0]])
        is_rt = bool(rt[in_idx[0]])

        if is_lt or is_rt:
            if is_lt and not is_rt:
                reference = flank(in_idx.max() + 1, 1, seg_h, n_focal)
            elif is_rt and not is_lt:
                reference = flank(in_idx.min() - 1, -1, seg_h, n_focal)
            else:
                reference = []
        else:
            left = flank(in_idx.min() - 1, -1, seg_h, n_focal)
            right = flank(in_idx.max() + 1, 1, seg_h, n_focal)
            m = min(len(left), len(right))
            reference = left[:m] + right[:m]
        has_reference = len(reference) > 0
        for i in in_idx:
            rows.append({"sample": sample, "focal_region_id": int(h), "col_id_local": int(colid[i]),
                         "role": "in", "has_reference": has_reference})
        for i in reference:
            rows.append({"sample": sample, "focal_region_id": int(h), "col_id_local": int(colid[i]),
                         "role": "ref", "has_reference": has_reference})
    return pd.DataFrame(rows, columns=["sample", "focal_region_id", "col_id_local", "role", "has_reference"])


# Summarize each focal region
def per_focal_region_table(sample, sigma_um=None, prominence_min=None):
    prominence_min = PROMINENCE_MIN if prominence_min is None else prominence_min
    cols = focal_region_columns(sample, sigma_um=sigma_um, prominence_min=prominence_min)
    arc = cols["arc_col_um"].to_numpy(float)
    si = cols["si_smooth"].to_numpy(float)
    raw = cols["raw_program_score"].to_numpy(float)
    smoothed = cols["smoothed_residual"].to_numpy(float)

    def arc_at(ip):
        low = int(np.floor(ip))
        high = min(low + 1, len(arc) - 1)
        return float(arc[low] + (ip - low) * (arc[high] - arc[low]))

    rows = []
    is_valid = cols["valid_tissue"].to_numpy(bool)
    seg_ids = cols["segment_id"].to_numpy(int)
    for focal_region_id, spot in enumerate(_detect(smoothed, prominence_min, is_valid, segment_id=seg_ids)):
        rows.append({"sample": sample, "focal_region_id": focal_region_id,
                     "core_arc_um": float(arc[spot["core"]]),
                     "si_field_at_core": float(si[spot["core"]]),
                     "left_boundary_um": arc_at(spot["left_ip"]),
                     "right_boundary_um": arc_at(spot["right_ip"]),
                     "width_um": spot["width_um"],
                     "prominence": spot["prominence"],
                     "truncated": spot["truncated"],
                     "left_open": spot["left_open"],
                     "right_open": spot["right_open"],
                     "mean_raw_injury_score": float(np.nanmean(raw[spot["lo"]:spot["hi"] + 1]))})
    columns = ["sample", "focal_region_id", "core_arc_um", "si_field_at_core", "left_boundary_um",
               "right_boundary_um", "width_um", "prominence", "truncated", "left_open", "right_open",
               "mean_raw_injury_score"]
    return pd.DataFrame(rows, columns=columns)


# Build the column profile used by the arc strip
def strip_profile(sample, sigma_um=None):
    cols = focal_region_columns(sample, sigma_um=sigma_um)
    return cols[["col_id_local", "component_id", "arc_col_um", "opsin_field", "raw_program_score",
                 "smoothed_residual", "focal_region_core", "in_focal_region"]].copy()


# Orient the arc strip from superior to inferior
def _opsin_orientation(opsin):
    """Return whether the arc order should be reversed for display"""
    ok = np.isfinite(opsin)
    if ok.sum() < 2:
        return False
    n = max(1, int(ok.sum()) // 10)
    return float(np.nanmean(opsin[ok][:n])) > float(np.nanmean(opsin[ok][-n:]))


VISUAL_GAP_UM = 50.0  # Width of blank display space inserted between tissue pieces

def arc_strip(sample, sigma_um=None):
    prof = strip_profile(sample, sigma_um=sigma_um)
    comp_edges, comp_means = [], []
    for ci in sorted(prof["component_id"].unique()):
        sub = prof[prof["component_id"] == ci].sort_values("arc_col_um")
        arc = sub["arc_col_um"].to_numpy(float)
        opsin = sub["opsin_field"].to_numpy(float)
        means = smooth_residual(sub["raw_program_score"].to_numpy(float),
                                SMOOTH_SIGMA_UM if sigma_um is None else sigma_um)
        scored = np.where(np.isfinite(means))[0]
        if not len(scored):
            continue
        lo, hi = int(scored[0]), int(scored[-1])
        arc, means, opsin = arc[lo:hi + 1], means[lo:hi + 1], opsin[lo:hi + 1]
        if _opsin_orientation(opsin):
            means = means[::-1]
            arc = arc[0] + arc[-1] - arc[::-1]
        edges = np.concatenate([[arc[0] - COL_WIDTH_UM / 2],
                                0.5 * (arc[:-1] + arc[1:]), [arc[-1] + COL_WIDTH_UM / 2]])
        comp_edges.append(edges)
        comp_means.append(means)
    if not comp_edges:
        return np.array([0.0, 1.0]), np.array([np.nan])
    if len(comp_edges) == 1:
        return comp_edges[0], comp_means[0]
    # Insert a blank display gap between adjacent tissue pieces
    combined_edges = list(comp_edges[0])
    combined_means = list(comp_means[0])
    for edges, means in zip(comp_edges[1:], comp_means[1:]):
        offset = combined_edges[-1]
        combined_edges.append(offset + VISUAL_GAP_UM)
        combined_means.append(np.nan)
        shifted = edges - edges[0] + offset + VISUAL_GAP_UM
        combined_edges.extend(shifted[1:].tolist())
        combined_means.extend(means.tolist())
    return np.array(combined_edges), np.array(combined_means)
