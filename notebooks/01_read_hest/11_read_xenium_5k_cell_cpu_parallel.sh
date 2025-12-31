#!/bin/bash
#SBATCH --output=11_read_xenium_%A_%a.out   # %A = jobID, %a = array index
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=short
#SBATCH --array=1-32%8    # 1-32 indexes, at most 8 running concurrently

module load python-cbrg/202510

# pick SampleID from file by array index
SAMPLE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /project/simmons_hts/kxu/JupyterFolder/hest/01_read_hest/sample_id/XeniumPR1-3.txt)
echo "[$SLURM_JOB_ID:$SLURM_ARRAY_TASK_ID] Processing $SAMPLE"

python /project/simmons_hts/kxu/JupyterFolder/hest/01_read_hest/11_read_xenium_5k_cell.py "$SAMPLE"

# cleanup in task's workdir if you need it
find . -maxdepth 1 -type f -name "*.tar.gz" -delete