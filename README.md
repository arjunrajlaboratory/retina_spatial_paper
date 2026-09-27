# Single-cell spatial transcriptomics of inherited retinal degeneration reveals broad and focal organization of disease responses

This repository contains the analysis and plotting code for Figs. 1–4 and Supplementary Figs. S1–S5 for this manuscript.

## Repository organization

| Directory | Contents |
| --- | --- |
| `figure_1/code/` through `figure_4/code/` | Main figure scripts |
| `figure_1_supp/code/`, `figure_2_supp/code/`, `figure_3_supp/code/` | Supplementary Figs. S1–S3 scripts|
| `figure_4_supp_1/code/`, `figure_4_supp_2/code/` | Supplementary Figs. S4-S5 scripts |
| `_shared/code/` | Code shared by multiple scripts |
| `preprocessing/` | Code for preprocessing data|

## Instructions to reproduce analyses and plots

### 1. Setup

Create a Python environment, then install the required dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Download data

Raw and preprocessed data is deposited on Zenodo* at the following link: https://doi.org/10.5281/zenodo.22767514.

Download and unzip the Zenodo data deposit, then set `DATA_DIR`:

```bash
export DATA_DIR=/path/to/retina_spatial_paper_data
```

*See the Zenodo README for more detailed information on the contents of the deposit.

### 3. Run data preprocessing scripts 

From `preprocessing/`, run the scripts in this order:

```bash
python build_cell_typing.py
python build_arc_columns.py
python build_per_cell_gene_counts.py
python build_all_rods.py
python export_transcript_subsets.py
python build_ds8_cache.py
python ../figure_1/code/Fig1b_00_compute_harmony.py
```

**Notes:** 
The cell segmentation masks and the trained U-Net model and layer predictions are already provided in the Zenodo deposit for convenience. To regenerate them: 
1. Run `build_segmentation.py` for regenerating the cell segmentation
2. Run the scripts in `preprocessing/unet/` for regenerating the U-Net model (`predict_unet_layers.py` applies the existing trained model) 

### 4. Run analysis and figure plotting scripts

Run each script from its own `code/` directory:

```bash
cd figure_1/code
python Fig1a.py
```

**Notes on the order to run scripts:** 
1. Run Fig. 3 scripts before Fig. 4 and Supplementary Fig. 3, as some scripts require Fig. 3 outputs.
2. Within Fig. 4, run `Fig4a.py` first, as it produces the rod injury response score used by the subsequent scripts. 
