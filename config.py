"""
Centralised path configuration for gutdecoder.

Override any default by setting the corresponding environment variable before
importing gutdecoder, e.g.:

    export GUTDECODER_RESULTS_ROOT=/my/project/results
"""
import os

# --- Prediction / benchmark results ---
RESULTS_ROOT = os.environ.get(
    "GUTDECODER_RESULTS_ROOT",
    "/project/gutdecoder/kxu/hest/eval/ST_pred_results",
)

# --- Benchmark data (splits, adata, patches) ---
DATA_ROOT = os.environ.get(
    "GUTDECODER_DATA_ROOT",
    "/project/gutdecoder/kxu/hest/eval/data",
)

# --- Summary plot output directory ---
SUMMARY_PLOT_DIR = os.environ.get(
    "GUTDECODER_SUMMARY_PLOT_DIR",
    "/project/gutdecoder/kxu/hest/eval/summary_plots",
)

# --- Curated gene list Excel file ---
CURATED_XLSX = os.environ.get(
    "GUTDECODER_CURATED_XLSX",
    "/project/gutdecoder/kxu/hest/curated_gene_list.xlsx",
)

# --- Sample metadata CSVs ---
HEST_METADATA_CSV = os.environ.get(
    "GUTDECODER_HEST_METADATA_CSV",
    "/project/gutdecoder/kxu/hest/metadata/hest_directory.csv",
)
BROAD_METADATA_CSV = os.environ.get(
    "GUTDECODER_BROAD_METADATA_CSV",
    "/project/gutdecoder/kxu/hest/metadata/broad_directory.csv",
)
