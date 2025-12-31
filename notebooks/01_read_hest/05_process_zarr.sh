#!/bin/bash
#SBATCH --job-name=xenium
#SBATCH --nodes=1
#SBATCH --output=18_process_zarr.out     # Output file
#SBATCH --time=2:00:00  # 1 hour
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --mem=50G
#SBATCH --partition=gpu

module load python-cbrg

python /project/simmons_hts/kxu/JupyterFolder/hest/18_process_zarr.py

# ---- Cleanup step ----
# Delete any .tar.gz files in current working directory
find . -maxdepth 1 -type f -name "*.tar.gz" -delete