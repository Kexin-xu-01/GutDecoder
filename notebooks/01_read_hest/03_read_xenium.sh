#!/bin/bash
#SBATCH --job-name=xenium
#SBATCH --nodes=1
#SBATCH --output=03_read_xenium.out     # Output file
#SBATCH --time=3:00:00  # 1 hour
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=50G
#SBATCH --partition=short

module load python-cbrg/202510

for ID in XeniumPR5S1ROI6 XeniumPR5S1ROI7 XeniumPR5S1ROI8 XeniumPR5S1ROI9 XeniumPR5S1ROI10 XeniumPR5S2ROI1 XeniumPR5S2ROI2 XeniumPR5S2ROI3 XeniumPR5S2ROI4 XeniumPR5S2ROI5 XeniumPR5S2ROI6 XeniumPR5S2ROI7 XeniumPR5S2ROI8 XeniumPR5S2ROI9 XeniumPR5S2ROI10 XeniumPR5S2ROI11; do
    echo "Processing $ID..."
    python /project/simmons_hts/kxu/JupyterFolder/hest/01_read_hest/03_read_xenium.py "$ID"
done


# ---- Cleanup step ----
# Delete any .tar.gz files in current working directory
find . -maxdepth 1 -type f -name "*.tar.gz" -delete