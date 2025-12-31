#!/bin/bash
#SBATCH --job-name=xenium
#SBATCH --nodes=1
#SBATCH --output=11_read_xenium_5k_cell.out     # Output file
#SBATCH --time=1:00:00  # 1 hour
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=30G
#SBATCH --gpus=1
#SBATCH --partition=gpu-ada

module load python-cbrg/202510

for ID in XeniumPR1S1ROI2; do
    echo "Processing $ID..."
    python /project/simmons_hts/kxu/JupyterFolder/hest/01_read_hest/11_read_xenium_5k_cell.py "$ID"
done


# ---- Cleanup step ----
# Delete any .tar.gz files in current working directory
find . -maxdepth 1 -type f -name "*.tar.gz" -delete