#!/usr/bin/env bash
# Run TRIDENT feature extraction for one or more encoders.
# TRIDENT's GigaPath encoder asserts timm==0.9.16, but the cluster has 1.0.x.
# The run() function monkey-patches timm.__version__ at startup via python -,
# so the assertion is satisfied without modifying any installed package.

unset LD_PRELOAD
export NUMEXPR_MAX_THREADS=128
export NUMEXPR_NUM_THREADS=64

module purge
module load python-cbrg/202510
module load cuda/12.9

source /project/gutdecoder/kxu/hest/env.sh

export PYTHONWARNINGS="ignore::FutureWarning"

set -euo pipefail

# ---------------------------------------------------------------------------
# User-configurable variables (can also be set as environment variables)
# ---------------------------------------------------------------------------
WSI_DIR=${WSI_DIR:-"/project/gutdecoder/kxu/xenium/he/xenium/tif_masked"}
JOB_DIR=${JOB_DIR:-"/project/gutdecoder/kxu/xenium/he/xenium/trident_processed/hest"}
GPU=${GPU:-0}

# Preferred: explicit combos you want to run (encoder:patch:mag)
# Example: COMBINATIONS=("prism:224:20" "titan:512:20")
COMBINATIONS=()

# Alternatively: pick encoder names from the DEFAULTS below (comma/space separated)
# Example: ENCODERS=("prism" "chief")
ENCODERS=("gigapath" "titan" "madeleine" "feather" "prism" "chief")

# --- defaults (encoder -> patch size, magnification) ---
declare -A PATCH=( [threads]=512 [titan]=512 [prism]=224 [chief]=256 [gigapath]=256 [madeleine]=256 [feather]=512 )
declare -A MAG=(   [threads]=20  [titan]=20  [prism]=20  [chief]=10  [gigapath]=20  [madeleine]=10  [feather]=20  )

# ---------------------------------------------------------------------------
_TRIDENT_SCRIPT=/ceph/home/k/kxu/.local/lib/python3.11/site-packages/trident/run_batch_of_slides.py

run() {
  # python - reads the heredoc as the script; remaining args go into sys.argv[1:]
  # which argparse picks up normally.  The monkey-patch must happen before trident
  # imports timm, so it lives at the very top of the inline script.
  python - \
    --task feat \
    --wsi_dir "$WSI_DIR" \
    --job_dir "$JOB_DIR" \
    --slide_encoder "$1" \
    --patch_size "$2" \
    --mag "$3" \
    --min_tissue_proportion 0.15 \
    --gpu "$GPU" <<PYEOF
import timm, runpy, sys
timm.__version__ = '0.9.16'  # bypass GigaPath assertion; timm 1.0.x API is compatible
runpy.run_path("$_TRIDENT_SCRIPT", run_name='__main__')
PYEOF
}

# ---------------------------------------------------------------------------

if [[ -z "$WSI_DIR" || -z "$JOB_DIR" ]]; then
  echo "ERROR: set WSI_DIR and JOB_DIR. Example:"
  echo "  WSI_DIR=/path JOB_DIR=/path GPU=0 ./run_encoders.sh"
  exit 1
fi
mkdir -p "$JOB_DIR"

if [[ ${#COMBINATIONS[@]} -gt 0 ]]; then
  for c in "${COMBINATIONS[@]}"; do
    IFS=':' read -r enc patch mag <<<"$c"
    [[ -z ${enc:-} || -z ${patch:-} || -z ${mag:-} ]] && { echo "Invalid combo format: $c (expected encoder:patch:mag)"; exit 1; }
    run "$enc" "$patch" "$mag"
  done
else
  for enc in "${ENCODERS[@]}"; do
    [[ -z ${PATCH[$enc]:-} || -z ${MAG[$enc]:-} ]] && { echo "Unknown encoder: $enc"; exit 1; }
    run "$enc" "${PATCH[$enc]}" "${MAG[$enc]}"
  done
fi

echo "Done."

find . -maxdepth 1 -type f -name "*.tar.gz" -delete
