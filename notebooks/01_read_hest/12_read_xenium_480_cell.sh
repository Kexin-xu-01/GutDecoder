#!/bin/bash
#SBATCH --job-name=xenium
#SBATCH --nodes=1
#SBATCH --output=12_read_xenium_480_cell.out     # Output file
#SBATCH --time=1:00:00  # 1 hour
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --partition=short

module load python-cbrg/202510

for ID in $(cat /project/simmons_hts/kxu/JupyterFolder/hest/01_read_hest/sample_id/XeniumR2356.txt); do
    echo "Processing $ID..."
    python /project/simmons_hts/kxu/JupyterFolder/hest/01_read_hest/12_read_xenium_480_cell.py "$ID"
done


# ---- Cleanup step ----
# Delete any .tar.gz files in current working directory
find . -maxdepth 1 -type f -name "*.tar.gz" -delete

