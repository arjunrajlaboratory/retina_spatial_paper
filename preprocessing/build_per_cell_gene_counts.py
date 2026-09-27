# Build the master per-cell table by joining cell typing with segmentation body counts
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _preprocessing_config as C
import _superior_inferior_axis as sih

CELL_TYPING_DIR = C.TYPED_DIR
SKELETONS = C.ARC_TABLE_DIR / "skeletons.parquet"

## Define the metadata column order for the output table
META_COLS = [
    "cell_id", "sample_id", "genotype", "age", "biological_replicate", "batch",
    "cx_um", "cy_um", "cx_px", "cy_px", "nucleus_area_um2", "cell_area_um2",
    "cell_type", "argmax_layer", "dist_to_ONL_um",
    "opsin_field", "arc_um",
    "body_total", "body_density_per_um2", "body_source",
]


def build_sample(sample, roi, meta_row, genes):
    """Assemble the master per-cell table for one sample"""
    print(f"  {sample} (roi {roi}) ...", flush=True)

    ## Load the cell typing results
    tv = pd.read_parquet(CELL_TYPING_DIR / f"{sample}_typed.parquet")
    n0 = len(tv)
    if tv["label"].duplicated().any():
        raise ValueError(f"{sample}: duplicate labels in typed table")
    ## Merge body transcript counts and both area measurements
    cells = pd.read_parquet(C.SEGMENTATION_DIR / sample / f"cells_{sample}.parquet")
    body_cols = [f"body_{g}" for g in genes]
    missing = [c for c in body_cols if c not in cells.columns]
    if missing:
        raise ValueError(f"{sample}: missing body columns {missing[:5]}...")
    if "cell_area_um2" not in cells.columns:
        raise ValueError(f"{sample}: cells table missing cell_area_um2 (cell body mask area)")
    cells_keep = cells[["label", "body_source", "area_um2", "cell_area_um2"] + body_cols].copy()
    cells_keep = cells_keep.rename(columns={"area_um2": "nucleus_area_um2_cells"})

    # Join body counts onto the typed cells by segmentation label
    merged = tv.merge(cells_keep, on="label", how="left", validate="one_to_one")
    if merged["body_source"].isna().any():
        n_miss = int(merged["body_source"].isna().sum())
        raise ValueError(f"{sample}: {n_miss}/{n0} typed cells have no body-count match")
    ## Check nuclear area agreement between typed table and segmentation export
    da = (merged["area_um2"] - merged["nucleus_area_um2_cells"]).abs()
    if float(da.max()) > 1e-3:
        print(f"    WARN {sample}: max nucleus area mismatch typed vs cells = {float(da.max()):.4g} um^2")
    ## Give the nuclear and cell body areas distinct names
    merged = merged.rename(columns={"area_um2": "nucleus_area_um2"})
    merged = merged.drop(columns=["nucleus_area_um2_cells"])

    ## Assign each cell its arc position from the nearest skeleton point
    skeleton = pd.read_parquet(SKELETONS)
    skeleton = skeleton[skeleton["sample"] == sample].sort_values("s_um")
    if skeleton.empty:
        raise ValueError(f"{sample}: no skeleton rows in {SKELETONS}")
    # Build a spatial index over the arc skeleton on the ds8 grid
    tree = cKDTree(skeleton[["r_ds8", "c_ds8"]].to_numpy())
    # Map each cell centroid onto the ds8 grid and find its nearest arc point
    query = np.column_stack([
        merged["cy_um"].to_numpy() / C.PX_DS_UM,
        merged["cx_um"].to_numpy() / C.PX_DS_UM,
    ])
    _distance, nearest = tree.query(query, k=1)
    merged["arc_um"] = skeleton["s_um"].to_numpy()[nearest]
    ## Add the smoothed cone opsin field at each cell
    merged["opsin_field"] = sih.smoothed_opsin_field(pd.DataFrame({
        "sample_id": sample,
        "cell_type": merged["final_label"].to_numpy(),
        "cx_um": merged["cx_um"].to_numpy(),
        "cy_um": merged["cy_um"].to_numpy(),
        "body_Opn1mw": merged["body_Opn1mw"].to_numpy(),
        "body_Opn1sw": merged["body_Opn1sw"].to_numpy(),
    }))
    print(f"    opsin_field = smoothed opsin field ({merged['opsin_field'].min():.2f} to {merged['opsin_field'].max():.2f})")

    ## Compute the total transcript count per cell and its density over cell body area
    body_mat = merged[body_cols].to_numpy()
    merged["body_total"] = body_mat.sum(axis=1)
    merged["body_density_per_um2"] = merged["body_total"] / merged["cell_area_um2"]

    ## Rename columns to the output schema and attach sample metadata
    merged = merged.rename(columns={
        "label": "cell_id",
        "sample": "sample_id",
        "final_label": "cell_type",
        "close_call": "typing_close_call",
    })
    merged["sample_id"] = sample
    merged["genotype"] = meta_row["genotype"]
    merged["age"] = meta_row["age"]
    merged["biological_replicate"] = meta_row["biological_replicate"]
    if "batch" not in merged.columns:
        merged["batch"] = pd.NA
    # Verify every expected metadata column is present before selecting them
    for col in META_COLS:
        if col not in merged.columns:
            raise KeyError(f"{sample}: expected metadata column '{col}' absent")

    # Select the metadata columns and body counts, checking no rows were lost
    out = merged[META_COLS + body_cols].copy()
    if len(out) != n0:
        raise ValueError(f"{sample}: row count changed {n0} -> {len(out)}")
    return out


def main():
    """Build every sample's per-cell table and write the combined master table"""
    samples = C.load_samples()
    gc = C.gene_channels()
    genes = list(gc["gene"])
    print(f"Panel: {len(genes)} genes | {len(samples)} samples")

    # Build one per-cell table per sample
    final_cells = []
    summary = []
    for _, row in samples.iterrows():
        s = str(row["sample_id"])
        df = build_sample(s, int(row["roi_number"]), row, genes)
        final_cells.append(df)
        summary.append({
            "sample_id": s, "roi_number": int(row["roi_number"]),
            "n_cells": len(df),
        })

    sdf = pd.DataFrame(summary)
    print("\n" + sdf.to_string(index=False))
    print(f"\nTOTAL: {int(sdf['n_cells'].sum()):,} cells")

    # Concatenate all samples and write the master per-cell table
    final = pd.concat(final_cells, ignore_index=True)
    data_processed = C.ROOT / "_shared" / "data_processed"
    data_processed.mkdir(parents=True, exist_ok=True)
    final_path = data_processed / "per_cell_gene_counts.parquet"
    final.to_parquet(final_path, index=False)
    print(f"Wrote {len(final):,} cells to {final_path}")


if __name__ == "__main__":
    main()
