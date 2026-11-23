#!/bin/bash
#SBATCH --job-name=xenium
#SBATCH --nodes=1
#SBATCH --output=03_read_xenium.out     # Output file
#SBATCH --time=2:00:00  # 1 hour
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=50G
#SBATCH --partition=short

module load python-cbrg

python /project/simmons_hts/kxu/JupyterFolder/hest/01_read_hest/03_read_xenium.py 

# ---- Cleanup step ----
# Delete any .tar.gz files in current working directory
find . -maxdepth 1 -type f -name "*.tar.gz" -delete