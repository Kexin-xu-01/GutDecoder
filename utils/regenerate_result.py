"""
HEST Evaluation Results Cleanup and Regeneration Script
=======================================================

This script cleans and regenerates evaluation result files produced by the
HEST benchmark pipeline.

Raw HEST outputs (results_kfold.json) may contain NaN values when Pearson
correlations cannot be computed for certain genes. These NaNs break JSON
serialization and downstream aggregation, making summary files unusable.

The purpose of this script is to:
  1) Sanitize per-model evaluation outputs
  2) Recompute summary statistics in a NaN-safe way
  3) Regenerate per-dataset and cross-dataset ranking files

All outputs produced by this script are fully JSON-valid and compatible with
the original HEST evaluation methodology.
"""

# ---------------------------------------------------------------------
# Output Files
# ---------------------------------------------------------------------
# This script produces the following cleaned and regenerated files:
#
#   - results_kfold.json   : cleaned per-model evaluation results
#   - enc_results.json     : per-dataset encoder rankings and statistics
#   - dataset_results.json : cross-dataset summary with rankings
#   - dataset_results.csv  : flat table of all datasets and models

# ---------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------
# - This script is intended to be run AFTER HEST model evaluation.
# - It is safe to re-run; outputs will be overwritten deterministically.
# - Results are consistent with the original HEST benchmark definitions:
#   https://github.com/mahmoodlab/HEST

# ---------------------------------------------------------------------
# Example entry point
# ---------------------------------------------------------------------
# This block defines how to run the script on a specific HEST evaluation
# run directory. The run directory is expected to contain one subfolder
# per dataset, each with per-model evaluation outputs.
#
# Example:
#   ST_pred_results/
#     └── run_25-12-16-17-30-07/
#         ├── VisiumR1/
#         ├── XeniumPR1/
#         └── ...
#
# For each dataset directory, this script will:
#   1) Clean per-model results_kfold.json files
#   2) Regenerate enc_results.json
#
# After processing all datasets, it will:
#   3) Regenerate dataset_results.json and dataset_results.csv
#      summarizing performance across all datasets
# ---------------------------------------------------------------------

# how to run
# regenerate_all_results(
#     "/project/simmons_hts/kxu/hest/eval/ST_pred_results/run_25-12-16-17-30-07"
# )


import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

def safe_mean(values):
    arr = np.array(values, dtype=float)
    if np.all(np.isnan(arr)):
        return None
    return float(np.nanmean(arr))

def safe_std(values):
    arr = np.array(values, dtype=float)
    if np.all(np.isnan(arr)):
        return None
    return float(np.nanstd(arr))

def fix_results_kfold(results_path):
    """Fix NaNs inside a single results_kfold.json file."""
    with open(results_path, "r") as f:
        data = json.load(f)

    # fix mean_per_split
    if "mean_per_split" in data:
        data["mean_per_split"] = [
            None if (v is None or (isinstance(v, float) and np.isnan(v))) else v
            for v in data["mean_per_split"]
        ]

    # fix pearson_corrs
    if "pearson_corrs" in data:
        for gene in data["pearson_corrs"]:
            corrs = gene.get("pearson_corrs", [])
            gene["mean"] = safe_mean(corrs)
            gene["std"] = safe_std(corrs)

            gene["pearson_corrs"] = [
                None if (v is None or (isinstance(v, float) and np.isnan(v))) else v
                for v in corrs
            ]

    # overwrite with cleaned data
    with open(results_path, "w") as f:
        json.dump(data, f, indent=4)

    return data



def is_dataset_dir(p: Path) -> bool:
    """
    Return True when `p` should be treated as a dataset directory.

    Centralised rule: skip directories listed in SKIP_DIRS and non-directories.
    Implemented once and used wherever we need to decide whether to treat a
    folder as a dataset.
    """
    SKIP_DIRS = {"plots"}
    return p.is_dir() and (p.name not in SKIP_DIRS)


def regenerate_enc_results(dataset_dir):
    """Aggregate all model results into enc_results.json for a dataset, sorted by mean Pearson."""

    # single-place check via helper
    if not is_dataset_dir(dataset_dir):
        print(f"[skip] not a dataset dir: {dataset_dir}")
        return None
    
    enc_results = {"results": []}
    dataset = dataset_dir.name

    for model_dir in dataset_dir.iterdir():
        results_path = model_dir / "results_kfold.json"
        if not results_path.exists():
            continue

        fixed_data = fix_results_kfold(results_path)

        gene_means = [
            g["mean"] for g in fixed_data.get("pearson_corrs", [])
            if g["mean"] is not None
        ]
        pearson_mean = safe_mean(gene_means)
        pearson_std = safe_std(gene_means)

        enc_results["results"].append({
            "encoder_name": model_dir.name,
            "pearson_mean": pearson_mean,
            "pearson_std": pearson_std
        })

    # sort encoders by pearson_mean descending
    enc_results["results"].sort(
        key=lambda x: (x["pearson_mean"] is not None, x["pearson_mean"]),
        reverse=True
    )

    # save enc_results.json
    out_path = dataset_dir / "enc_results.json"
    with open(out_path, "w") as f:
        json.dump(enc_results, f, indent=4)

    print(f"[+] Wrote {out_path}")
    return dataset, enc_results

    

def regenerate_dataset_results(root_dir):
    """Summarize all datasets into dataset_results.json and a wide-format dataset_results.csv.

    Columns in the CSV are ordered so the encoder with the highest average pearson_mean
    is placed first (after the 'dataset' column), then the next best, etc.
    """
    # dataset_results = {"results": []}
    # encoder_scores = {}

    # # Collect per-dataset results
    # dataset_rows = []

    # for dataset_dir in Path(root_dir).iterdir():
    #     if not dataset_dir.is_dir():
    #         continue
    #     enc_results_path = dataset_dir / "enc_results.json"
    #     if not enc_results_path.exists():
    #         continue

    root_dir = Path(root_dir)
    dataset_results = {"results": []}
    encoder_scores = {}
    dataset_rows = []

    for dataset_dir in root_dir.iterdir():
        # use centralised helper to decide whether this is a dataset
        if not is_dataset_dir(dataset_dir):
            continue

        enc_results_path = dataset_dir / "enc_results.json"
        if not enc_results_path.exists():
            # try to generate it if missing (will also respect is_dataset_dir)
            print(f"[warn] enc_results.json missing for {dataset_dir.name}; attempting to generate")
            regenerate_enc_results(dataset_dir)
            if not enc_results_path.exists():
                print(f"[skip] still missing enc_results.json for {dataset_dir.name}; skipping")
                continue

        with open(enc_results_path, "r") as f:
            enc_results = json.load(f)

        # store per-dataset JSON results
        dataset_results["results"].append({
            "dataset_name": dataset_dir.name,
            "results": enc_results["results"]
        })

        # accumulate for wide CSV row
        row = {"dataset": dataset_dir.name}
        for r in enc_results["results"]:
            encoder = r["encoder_name"]
            mean = r["pearson_mean"]
            std = r["pearson_std"]
            row[f"{encoder}_mean"] = mean
            row[f"{encoder}_std"] = std

            if mean is not None:
                encoder_scores.setdefault(encoder, []).append(mean)
        dataset_rows.append(row)

    # compute averages across datasets
    dataset_results["average"] = {
        enc: safe_mean(vals) for enc, vals in encoder_scores.items()
    }

    # create an ordered list of encoders by descending average pearson mean
    # encoders with None average will be placed at the end
    encoder_order = sorted(
        dataset_results["average"].items(),
        key=lambda kv: (kv[1] is not None, kv[1]),
        reverse=True
    )
    ordered_encoders = [enc for enc, _ in encoder_order]

    # sort encoders inside each dataset (leave for compatibility)
    for ds in dataset_results["results"]:
        ds["results"].sort(
            key=lambda x: (x["pearson_mean"] is not None, x["pearson_mean"]),
            reverse=True
        )

    # save JSON
    json_path = Path(root_dir) / "dataset_results.json"
    with open(json_path, "w") as f:
        json.dump(dataset_results, f, indent=4)
    print(f"[+] Wrote {json_path}")

    # save wide-format CSV
    df = pd.DataFrame(dataset_rows)

    # Ensure all encoder_mean/_std columns exist in df (fill missing with NaN)
    all_cols_needed = []
    for enc in ordered_encoders:
        all_cols_needed.extend([f"{enc}_mean", f"{enc}_std"])
    for col in all_cols_needed:
        if col not in df.columns:
            df[col] = np.nan

    # Final column order: dataset, then encoders in ordered_encoders (mean then std)
    cols = ["dataset"] + [c for enc in ordered_encoders for c in (f"{enc}_mean", f"{enc}_std")]

    # If there are any other unexpected columns (e.g. encoders not in averages), append them
    extra_cols = [c for c in df.columns if c not in cols and c != "dataset"]
    cols = cols + sorted(extra_cols)

    # Reindex columns safely (if df doesn't contain some columns they were added above)
    df = df.reindex(columns=cols)

    csv_path = Path(root_dir) / "dataset_results.csv"
    df.to_csv(csv_path, index=False, float_format="%.17f")
    print(f"[+] Wrote {csv_path}")


def regenerate_all_results(run_dir):
    """
    One-line regeneration of all HEST evaluation summaries for a run.

    This function:
      1) Iterates over all dataset directories in `run_dir`
      2) Cleans per-model results_kfold.json files
      3) Regenerates enc_results.json for each dataset
      4) Regenerates dataset_results.json and dataset_results.csv

    Args:
        run_dir (str or Path): Path to a single HEST run directory, e.g.
            /project/simmons_hts/kxu/hest/eval/ST_pred_results/run_25-12-16-17-30-07
    """
    run_dir = Path(run_dir)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    # Per-dataset regeneration
    for dataset_dir in run_dir.iterdir():
        if dataset_dir.is_dir():
            regenerate_enc_results(dataset_dir)

    # Cross-dataset aggregation
    regenerate_dataset_results(run_dir)




