#!/bin/bash
#SBATCH --export=ALL
#SBATCH --chdir=/ceph/project/simmons_hts/kxu/JupyterFolder/hest/02_prepare_data/slurm_exec/20251202T1809-e178167f
#SBATCH --job-name=e178167f
#SBATCH --output=/ceph/project/simmons_hts/kxu/JupyterFolder/hest/02_prepare_data/slurm_exec/20251202T1809-e178167f/slurm-%j.out
#SBATCH --error=/ceph/project/simmons_hts/kxu/JupyterFolder/hest/02_prepare_data/slurm_exec/20251202T1809-e178167f/slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=30
#SBATCH --mem=5G
#SBATCH --gpus=1


exec /package/python-base/3.11.11/bin/python driver.py
