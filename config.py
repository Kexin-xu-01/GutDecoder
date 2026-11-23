"""
Centralised path configuration for gutdecoder.

Every path defaults to a location under HEST_ROOT. Override HEST_ROOT to move them all,
or any single path with its own variable, before importing gutdecoder, e.g.:

    export GUTDECODER_HEST_ROOT=/my/project/hest
    export GUTDECODER_RESULTS_ROOT=/my/project/results
"""
import os


def _path(env_var: str, default: str) -> str:
    return os.environ.get(env_var, default)


HEST_ROOT = _path("GUTDECODER_HEST_ROOT", "/project/gutdecoder/kxu/hest")

# --- Prediction / benchmark results ---
RESULTS_ROOT = _path("GUTDECODER_RESULTS_ROOT", f"{HEST_ROOT}/eval/ST_pred_results")
RUNS_YAML = _path("GUTDECODER_RUNS_YAML", f"{HEST_ROOT}/metadata/runs.yaml")

# --- Benchmark data (splits, adata, patches) and patch embeddings ---
DATA_ROOT = _path("GUTDECODER_DATA_ROOT", f"{HEST_ROOT}/eval/data")
EMBEDDING_ROOT = _path("GUTDECODER_EMBEDDING_ROOT", f"{HEST_ROOT}/eval/ST_data_emb")

# --- Processed Xenium samples ---
XENIUM_DATA_ROOT = _path("GUTDECODER_XENIUM_DATA_ROOT", f"{HEST_ROOT}/xenium_data")

# --- Summary plot output directory ---
SUMMARY_PLOT_DIR = _path("GUTDECODER_SUMMARY_PLOT_DIR", f"{HEST_ROOT}/eval/summary_plots")

# --- Metadata sheets ---
CURATED_XLSX = _path("GUTDECODER_CURATED_XLSX", f"{HEST_ROOT}/metadata/curated_gene_list.xlsx")
XENIUM_DIRECTORY_XLSX = _path("GUTDECODER_XENIUM_DIRECTORY_XLSX", f"{HEST_ROOT}/metadata/xenium_directory.xlsx")
MODEL_META_XLSX = _path(
    "GUTDECODER_MODEL_META_XLSX",
    f"{HEST_ROOT}/metadata/foundation_model/pathology_foundation_models_plotting.xlsx",
)
HEST_METADATA_CSV = _path("GUTDECODER_HEST_METADATA_CSV", f"{HEST_ROOT}/metadata/hest_directory.csv")
BROAD_METADATA_CSV = _path("GUTDECODER_BROAD_METADATA_CSV", f"{HEST_ROOT}/metadata/broad_directory.csv")
