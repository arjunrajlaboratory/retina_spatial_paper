# Export transcript coordinates used in the figure maps
## Load packages
import sys
from pathlib import Path
import pandas as pd

## Load the shared sample to ROI map and the external data locations
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _preprocessing_config as C
from _paths import SAMPLE_TO_ROI, findpath_transcripts

## Resolve the output base directory and the full sample list
ROOT = C.ROOT
out_base = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_shared" / "data_raw"
ALL_SAMPLES = list(SAMPLE_TO_ROI)

## Define the genes and samples for each export
subsets = {
    "marker_tx": (["Arr3", "Gad1", "Glul", "Nrl", "Opn1mw", "Opn1sw", "Prkca", "Rbpms"],
                  ["WT_P21_rep2"], "{sample}_tx.parquet"),
    "opsin_tx": (["Opn1mw", "Opn1sw"], ["WT_P21_rep2"], "{sample}_opsin_tx.parquet"),
    "calibration_tx": (["Aldh1a1", "Opn1mw", "Opn1sw"],
                       ["LCA5_P21_rep2", "LCA5_P30_rep2", "LCA5_P64_rep1", "WT_P21_rep2", "WT_P64_rep1"],
                       "{sample}_tx.parquet"),
}

## Export the gene name and spatial coordinates for each subset
for subset, (genes, samples, pattern) in subsets.items():
    out_dir = out_base / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        roi = SAMPLE_TO_ROI[sample]
        # Read the raw transcript table and keep only the requested genes
        raw = pd.read_csv(findpath_transcripts(sample))
        subset_df = raw[raw["name"].isin(genes)][["name", "x", "y"]]
        out_path = out_dir / pattern.format(sample=sample)
        subset_df.to_parquet(out_path, index=False)
        print(f"wrote {out_path}  rows={len(subset_df)}")

## Pool the Muller gliosis transcripts across all samples for Fig4b_2
mg_genes = ["Gfap", "Serpina3n"]
mg_out = ROOT / "figure_4" / "data_raw_manifest" / "muller_gliosis_transcripts.parquet"
mg_out.parent.mkdir(parents=True, exist_ok=True)
frames = []
for sample in ALL_SAMPLES:
    # Keep the gliosis genes and tag each row with its sample
    raw = pd.read_csv(findpath_transcripts(sample))
    sub = raw[raw["name"].isin(mg_genes)][["name", "x", "y"]].copy()
    sub.insert(0, "sample", sample)
    frames.append(sub)
# Concatenate the per-sample frames and write the pooled table
pd.concat(frames, ignore_index=True).to_parquet(mg_out, index=False)
print(f"wrote {mg_out}  rows={sum(len(f) for f in frames)}")
