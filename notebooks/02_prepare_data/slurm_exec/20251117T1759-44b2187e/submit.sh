#!/bin/bash
#SBATCH --export=ALL
#SBATCH --chdir=/ceph/project/simmons_hts/kxu/JupyterFolder/hest/02_prepare_data/slurm_exec/20251117T1759-44b2187e
#SBATCH --job-name=44b2187e
#SBATCH --output=/ceph/project/simmons_hts/kxu/JupyterFolder/hest/02_prepare_data/slurm_exec/20251117T1759-44b2187e/slurm-%j.out
#SBATCH --error=/ceph/project/simmons_hts/kxu/JupyterFolder/hest/02_prepare_data/slurm_exec/20251117T1759-44b2187e/slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=5G
#SBATCH --gpus=1


exec /package/python-base/3.11.11/bin/python driver.py
