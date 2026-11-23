#!/bin/bash
#SBATCH --job-name=XeniumPR2S1ROI4
#SBATCH --nodes=1
#SBATCH --output=XeniumPR2S1ROI4.out     # Output file
#SBATCH --time=2:00:00  # 1 hour
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --partition=short

module load python-cbrg/202510

for ID in XeniumPR1S1ROI3 XeniumPR1S2ROI4 XeniumPR1S2ROI1 XeniumPR1S2ROI3 XeniumPR2S1ROI5; do
    echo "Processing $ID..."
    python /project/simmons_hts/kxu/JupyterFolder/hest/01_read_hest/11_read_xenium_5k_cell.py "$ID"
done


# ---- Cleanup step ----
# Delete any .tar.gz files in current working directory
find . -maxdepth 1 -type f -name "*.tar.gz" -delete