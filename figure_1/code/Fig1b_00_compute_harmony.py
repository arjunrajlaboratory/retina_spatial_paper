# Fig 1b compute the PCA Harmony UMAP embedding

## Load packages
from pathlib import Path
import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scanpy.external as sce

## Load files
HERE = Path(__file__).resolve().parent
FIG = HERE.parent
PER_CELL = Path(os.environ.get("RETINA_PER_CELL_FILE", FIG.parent / "_shared/data_processed/per_cell_gene_counts.parquet"))
OUT_DIR = Path(os.environ.get("RETINA_PREPROCESSING_OUTPUT", FIG.parent / "_shared/data_processed"))
OUT_PATH = OUT_DIR / "umap_coordinates.parquet"

CONDITIONS = ["WT_P21", "WT_P64", "LCA5_P21", "LCA5_P30", "LCA5_P64"]
COND_REPS = {
    "WT_P21":   ["WT_P21_rep1", "WT_P21_rep2"],
    "WT_P64":   ["WT_P64_rep1", "WT_P64_rep2"],
    "LCA5_P21": ["LCA5_P21_rep1", "LCA5_P21_rep2"],
    "LCA5_P30": ["LCA5_P30_rep1", "LCA5_P30_rep2"],
    "LCA5_P64": ["LCA5_P64_rep1", "LCA5_P64_rep2"],
}
NON_GENE_BODY = {"body_total", "body_density_per_um2", "body_source"}

sc.settings.verbosity = 1
sc.settings.n_jobs = 8

## Keep typed cells
df = pd.read_parquet(PER_CELL)
df = df[df["cell_type"] != "unassigned"].copy()

## Add columns used by the figure scripts
df["sample"] = df["sample_id"]
df["condition"] = df["genotype"].astype(str) + "_" + df["age"].astype(str)
df["rep"] = "rep" + df["biological_replicate"].astype(int).astype(str)
df["final_label"] = df["cell_type"]

## Order by condition and sample
ordered = []
for cond in CONDITIONS:
    for sample in COND_REPS[cond]:
        s = df[df["sample"] == sample]
        ordered.append(s)
        print(f"  {sample}: {len(s):>6d} cells")
df = pd.concat(ordered, ignore_index=True)
print(f"total: {len(df)} cells")

## Build the AnnData expression matrix
body_cols = [c for c in df.columns
             if c.startswith("body_") and c not in NON_GENE_BODY]
var_names = [c.replace("body_", "") for c in body_cols]
print(f"n genes: {len(var_names)}")
X = df[body_cols].to_numpy(np.float32)
obs = df.drop(columns=body_cols).copy()
obs.index = obs.index.astype(str)
a = ad.AnnData(X=X, obs=obs)
a.var_names = var_names

## Run normalize, log1p, scale, PCA, UMAP, Harmony
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("batch composition (n cells per batch):")
print(a.obs["batch"].astype("category").value_counts().to_string())
a.obs["batch"] = a.obs["batch"].astype("category")

# Keep the raw counts before normalization
a.layers["counts"] = X.copy()
sc.pp.normalize_total(a, target_sum=1e4)
sc.pp.log1p(a)
sc.pp.scale(a)

sc.tl.pca(a, n_comps=30, random_state=0)
sc.pp.neighbors(a, n_neighbors=15, n_pcs=30, random_state=0,
                key_added="raw")
sc.tl.umap(a, random_state=0, neighbors_key="raw")
a.obsm["X_umap_raw"] = a.obsm["X_umap"].copy()

sce.pp.harmony_integrate(a, key="batch", basis="X_pca",
                         adjusted_basis="X_pca_harmony",
                         max_iter_harmony=20, random_state=0)
sc.pp.neighbors(a, n_neighbors=15, n_pcs=30, use_rep="X_pca_harmony",
                random_state=0, key_added="harmony")
sc.tl.umap(a, random_state=0, neighbors_key="harmony")
a.obsm["X_umap_harmony"] = a.obsm["X_umap"].copy()

## Write the UMAP coordinates table
keep_obs = ["cell_id", "sample", "batch", "rep", "condition", "final_label",
            "cell_type", "argmax_layer", "cx_px", "cy_px", "cx_um", "cy_um",
            "nucleus_area_um2", "cell_area_um2", "dist_to_ONL_um"]
keep_obs = [c for c in keep_obs if c in a.obs.columns]
out = a.obs[keep_obs].copy()
out["umap1"] = a.obsm["X_umap_raw"][:, 0]
out["umap2"] = a.obsm["X_umap_raw"][:, 1]
out["umap1_harmony"] = a.obsm["X_umap_harmony"][:, 0]
out["umap2_harmony"] = a.obsm["X_umap_harmony"][:, 1]
out["pc1"] = a.obsm["X_pca"][:, 0]
out["pc2"] = a.obsm["X_pca"][:, 1]
out["pc1_harmony"] = a.obsm["X_pca_harmony"][:, 0]
out["pc2_harmony"] = a.obsm["X_pca_harmony"][:, 1]
out.to_parquet(OUT_PATH, index=False)
print(f"WROTE {OUT_PATH}  ({len(out)} rows)")
