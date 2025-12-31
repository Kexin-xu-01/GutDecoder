#!/bin/bash
#SBATCH --export=ALL
#SBATCH --chdir=/ceph/project/simmons_hts/kxu/JupyterFolder/hest/02_prepare_data/slurm_exec/20251110T1411-ef0a076a
#SBATCH --job-name=ef0a076a
#SBATCH --output=/ceph/project/simmons_hts/kxu/JupyterFolder/hest/02_prepare_data/slurm_exec/20251110T1411-ef0a076a/slurm-%j.out
#SBATCH --error=/ceph/project/simmons_hts/kxu/JupyterFolder/hest/02_prepare_data/slurm_exec/20251110T1411-ef0a076a/slurm-%j.err
#SBATCH --partition=short
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=20


exec /package/python-base/3.11.11/bin/python driver.py
