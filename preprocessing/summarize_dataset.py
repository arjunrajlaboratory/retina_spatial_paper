# Compute headline dataset metrics from deposited raw and segmented data

## Load packages
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as arrow_csv
import pyarrow.parquet as pq

## Load the shared config and define the reporter genes and non-gene columns to exclude
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _preprocessing_config as C
from _paths import findpath_transcripts

transcript_block_size_bytes = 16 * 1024 * 1024
non_gene_body_columns = {"body_total", "body_density_per_um2", "body_source"}
excluded_reporter_genes = set(C.DROP_GENES)
excluded_reporter_body_columns = {
    f"body_{gene}" for gene in excluded_reporter_genes
}


## Define summary functions
# Count all transcript spots and the reporter spots to be excluded
def count_decoded_transcripts(path):
    if not path.exists():
        raise FileNotFoundError(f"missing raw transcript table: {path}")
    # Stream the transcript table in blocks, reading only the gene name column
    reader = arrow_csv.open_csv(
        path,
        read_options=arrow_csv.ReadOptions(block_size=transcript_block_size_bytes),
        convert_options=arrow_csv.ConvertOptions(include_columns=["name"]),
    )
    reporter_names = pa.array(sorted(excluded_reporter_genes))
    all_spots = 0
    excluded_reporter_spots = 0
    # Tally the total and reporter spot counts across every block
    for batch in reader:
        names = batch.column(batch.schema.get_field_index("name"))
        reporter_mask = pc.fill_null(
            pc.is_in(names, value_set=reporter_names),
            False,
        )
        all_spots += batch.num_rows
        excluded_reporter_spots += int(pc.sum(reporter_mask).as_py())
    return all_spots, excluded_reporter_spots


# List the body_<gene> count columns, dropping non-gene and reporter columns
def body_gene_columns(path):
    if not path.exists():
        raise FileNotFoundError(f"missing segmented cell table: {path}")
    columns = pq.ParquetFile(path).schema.names
    genes = [
        column for column in columns
        if column.startswith("body_")
        and column not in non_gene_body_columns
        and column not in excluded_reporter_body_columns
    ]
    if not genes:
        raise ValueError(f"no body_<gene> columns found in {path}")
    return genes


# Compute the per-retina transcript, cell, and gene detection metrics
def summarize_retina(sample, expected_gene_columns=None):
    transcript_path = findpath_transcripts(sample)
    cell_path = C.SEGMENTATION_DIR / sample / f"cells_{sample}.parquet"
    gene_columns = body_gene_columns(cell_path)
    # Require every retina to share the same gene panel
    if expected_gene_columns is not None and gene_columns != expected_gene_columns:
        raise ValueError(f"{sample}: body-gene panel differs from the first retina")

    # Load the body-gene counts and validate them
    cells = pd.read_parquet(cell_path, columns=gene_columns)
    body_counts = cells[gene_columns].to_numpy(dtype=float)
    if len(cells) == 0:
        raise ValueError(f"{sample}: segmented cell table contains no cells")
    if not np.isfinite(body_counts).all():
        raise ValueError(f"{sample}: body-gene counts contain missing or infinite values")
    if (body_counts < 0).any():
        raise ValueError(f"{sample}: body-gene counts contain negative values")

    # Derive transcripts and genes detected per cell from the count matrix
    all_spots, excluded_reporter_spots = count_decoded_transcripts(transcript_path)
    decoded_transcripts = all_spots - excluded_reporter_spots
    transcripts_per_cell = body_counts.sum(axis=1)
    genes_detected = (body_counts > 0).sum(axis=1)
    return {
        "sample_id": sample,
        "decoded_spots_all": all_spots,
        "excluded_reporter_spots": excluded_reporter_spots,
        "decoded_transcripts_excluding_reporters": decoded_transcripts,
        "n_segmented_cells": len(cells),
        "n_panel_genes": len(gene_columns),
        "sum_cell_body_transcripts": float(transcripts_per_cell.sum()),
        "mean_cell_body_transcripts_per_cell": float(transcripts_per_cell.mean()),
        "median_cell_body_transcripts_per_cell": float(np.median(transcripts_per_cell)),
        "sum_genes_detected": int(genes_detected.sum()),
        "mean_genes_detected_per_cell": float(genes_detected.mean()),
        "median_genes_detected_per_cell": float(np.median(genes_detected)),
    }, gene_columns


# Combine the per-retina rows into the headline dataset metrics table
def aggregate_metrics(per_retina):
    n_retinas = len(per_retina)
    total_all_spots = int(per_retina["decoded_spots_all"].sum())
    total_reporter_spots = int(per_retina["excluded_reporter_spots"].sum())
    total_transcripts = int(
        per_retina["decoded_transcripts_excluding_reporters"].sum()
    )
    mean_transcripts = float(
        per_retina["decoded_transcripts_excluding_reporters"].mean()
    )
    total_cells = int(per_retina["n_segmented_cells"].sum())
    total_cell_body_transcripts = float(
        per_retina["sum_cell_body_transcripts"].sum()
    )
    pooled_transcript_mean = total_cell_body_transcripts / total_cells
    equal_retina_transcript_mean = float(
        per_retina["mean_cell_body_transcripts_per_cell"].mean()
    )
    pooled_gene_mean = float(
        per_retina["sum_genes_detected"].sum() / total_cells
    )
    equal_retina_gene_mean = float(
        per_retina["mean_genes_detected_per_cell"].mean()
    )
    excluded_names = ", ".join(sorted(excluded_reporter_genes))
    return pd.DataFrame([
        {
            "metric": "n_retinas",
            "value": n_retinas,
            "definition": "Number of deposited retinal sections",
        },
        {
            "metric": "total_decoded_spots_all",
            "value": total_all_spots,
            "definition": "Total rows across all raw TranscriptList.csv files before reporter exclusion",
        },
        {
            "metric": "total_excluded_reporter_spots",
            "value": total_reporter_spots,
            "definition": f"Raw spots named {excluded_names}",
        },
        {
            "metric": "total_decoded_transcripts_excluding_reporters",
            "value": total_transcripts,
            "definition": f"Raw TranscriptList.csv rows excluding {excluded_names}",
        },
        {
            "metric": "mean_decoded_transcripts_per_retina_excluding_reporters",
            "value": mean_transcripts,
            "definition": "Arithmetic mean of reporter-excluded raw transcript counts across retinas",
        },
        {
            "metric": "rounded_mean_decoded_transcripts_per_retina_excluding_reporters",
            "value": round(mean_transcripts),
            "definition": "Reporter-excluded mean per retina rounded to the nearest whole transcript",
        },
        {
            "metric": "n_panel_genes",
            "value": int(per_retina["n_panel_genes"].iloc[0]),
            "definition": "Number of reporter-excluded body_<gene> count columns in every cell table",
        },
        {
            "metric": "total_segmented_cells",
            "value": total_cells,
            "definition": "Total rows across the supplied segmented cell tables",
        },
        {
            "metric": "total_cell_body_transcripts",
            "value": total_cell_body_transcripts,
            "definition": "Sum of reporter-excluded body_<gene> counts across all segmented cells",
        },
        {
            "metric": "pooled_mean_cell_body_transcripts_per_cell",
            "value": pooled_transcript_mean,
            "definition": "Mean reporter-excluded cell-body transcripts with cells pooled across retinas",
        },
        {
            "metric": "equal_retina_mean_cell_body_transcripts_per_cell",
            "value": equal_retina_transcript_mean,
            "definition": "Arithmetic mean of per-retina mean reporter-excluded cell-body transcripts per cell",
        },
        {
            "metric": "pooled_mean_genes_detected_per_cell",
            "value": pooled_gene_mean,
            "definition": "Mean reporter-excluded panel genes with at least one cell-body transcript across all cells",
        },
        {
            "metric": "equal_retina_mean_genes_detected_per_cell",
            "value": equal_retina_gene_mean,
            "definition": "Arithmetic mean of per-retina mean reporter-excluded genes detected per cell",
        },
    ])


## Parse the command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute headline dataset metrics from raw and segmented data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=C.OUTPUT,
        help="Directory for dataset_summary_per_retina.csv and dataset_summary_metrics.csv",
    )
    return parser.parse_args()


## Compute the per-retina and aggregate summaries and save them
def main():
    args = parse_args()
    rows = []
    expected_gene_columns = None
    # Summarize each retina, pinning the gene panel to the first retina
    for sample in C.ALL_SAMPLES:
        row, gene_columns = summarize_retina(sample, expected_gene_columns)
        if expected_gene_columns is None:
            expected_gene_columns = gene_columns
        rows.append(row)
        print(
            f"{sample}: "
            f"{row['decoded_transcripts_excluding_reporters']:,} decoded transcripts | "
            f"{row['mean_cell_body_transcripts_per_cell']:.2f} mean cell-body "
            "transcripts per cell | "
            f"{row['mean_genes_detected_per_cell']:.2f} mean genes detected per cell"
        )

    # Build both summary tables and write them to the output directory
    per_retina = pd.DataFrame(rows)
    metrics = aggregate_metrics(per_retina)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_retina_path = args.output_dir / "dataset_summary_per_retina.csv"
    metrics_path = args.output_dir / "dataset_summary_metrics.csv"
    per_retina.to_csv(per_retina_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    metric_values = metrics.set_index("metric")["value"]
    print(
        "\nMean decoded transcripts per retina excluding reporters: "
        f"{metric_values['mean_decoded_transcripts_per_retina_excluding_reporters']:,.1f} "
        "(rounded: "
        f"{int(metric_values['rounded_mean_decoded_transcripts_per_retina_excluding_reporters']):,})"
    )
    print(
        "Mean cell-body transcripts per cell excluding reporters: "
        f"{metric_values['pooled_mean_cell_body_transcripts_per_cell']:.2f} pooled | "
        f"{metric_values['equal_retina_mean_cell_body_transcripts_per_cell']:.2f} equal-retina"
    )
    print(
        "Mean genes detected per cell excluding reporters: "
        f"{metric_values['pooled_mean_genes_detected_per_cell']:.2f} pooled | "
        f"{metric_values['equal_retina_mean_genes_detected_per_cell']:.2f} equal-retina"
    )
    print(f"Wrote {per_retina_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
