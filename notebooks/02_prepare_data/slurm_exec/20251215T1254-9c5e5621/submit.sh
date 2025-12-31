#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=9c5e5621
#SBATCH --output=/ceph/project/simmons_hts/kxu/JupyterFolder/hest/02_prepare_data/slurm_exec/20251215T1254-9c5e5621/slurm-%j.out
#SBATCH --error=/ceph/project/simmons_hts/kxu/JupyterFolder/hest/02_prepare_data/slurm_exec/20251215T1254-9c5e5621/slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=5G
#SBATCH --gpus=1


exec /package/python-base/3.11.11/bin/python slurm_exec/20251215T1254-9c5e5621/driver.py
