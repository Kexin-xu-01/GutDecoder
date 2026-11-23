#!/bin/bash
#SBATCH --job-name=visium
#SBATCH --nodes=1
#SBATCH --output=10_read_visium.out     # Output file
#SBATCH --time=2:00:00  # 1 hour
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gpus=1
#SBATCH --mem=20G
#SBATCH --partition=gpu-turing

module load python-cbrg/202510

for ID in VisiumR1S1ROI1 VisiumR1S1ROI2 VisiumR1S1ROI3 VisiumR1S1ROI4 VisiumR2S1ROI1 VisiumR2S1ROI2 VisiumR2S1ROI3 VisiumR2S1ROI4 VisiumR2S2ROI1 VisiumR2S2ROI2 VisiumR2S2ROI3 VisiumR2S2ROI4 VisiumR3S1ROI1 VisiumR3S1ROI2 VisiumR3S1ROI3 VisiumR3S1ROI4 VisiumR4S1ROI1 VisiumR4S1ROI2 VisiumR4S1ROI3 VisiumR4S1ROI4 VisiumR5S1ROI1 VisiumR5S1ROI2 VisiumR5S1ROI3 VisiumR5S1ROI4 VisiumR5S2ROI1 VisiumR5S2ROI2 VisiumR5S2ROI3 VisiumR6S1ROI1 VisiumR6S1ROI2 VisiumR6S1ROI3 VisiumR6S1ROI4 VisiumR6S2ROI1 VisiumR6S2ROI2 VisiumR6S2ROI3 VisiumR6S2ROI4; do
    echo "Processing $ID..."
    python /project/simmons_hts/kxu/JupyterFolder/hest/01_read_hest/10_read_visium.py "$ID"
done


# ---- Cleanup step ----
# Delete any .tar.gz files in current working directory
find . -maxdepth 1 -type f -name "*.tar.gz" -delete