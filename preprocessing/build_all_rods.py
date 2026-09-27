# Build the typed rod table used by Figure 4
## Load packages
import sys
from pathlib import Path
import pandas as pd

## Resolve the input per-cell table and the output rod table
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _preprocessing_config as C
ROOT = C.ROOT
per_cell_path = ROOT / "_shared" / "data_processed" / "per_cell_gene_counts.parquet"
out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figure_4" / "data_raw_manifest" / "all_rods.parquet"

## Keep rod identity, position, area, and transcript counts
identity_cols = ["cell_id", "sample_id", "cx_um", "cy_um", "cx_px", "cy_px", "nucleus_area_um2", "cell_area_um2", "argmax_layer"]
cells = pd.read_parquet(per_cell_path)
# Select the per-gene cell-body count columns
body_cols = [c for c in cells.columns if c.startswith("body_")]
# Keep only rods and retain the identity and count columns
rods = cells[cells["cell_type"] == "rod"][identity_cols + body_cols].copy()

## Add sample labels used by the Figure 4 scripts
rods["label"] = rods["cell_id"]
rods["sample"] = rods["sample_id"]
# Split the sample id into its replicate and its condition
rods["rep"] = rods["sample_id"].str.extract(r"_(rep\d+)$")[0]
rods["condition"] = rods["sample_id"].str.replace(r"_rep\d+$", "", regex=True)

## Write the rod table
out_path.parent.mkdir(parents=True, exist_ok=True)
rods.to_parquet(out_path, index=False)
print(f"wrote {out_path}  rows={len(rods)}  cols={rods.shape[1]}")
