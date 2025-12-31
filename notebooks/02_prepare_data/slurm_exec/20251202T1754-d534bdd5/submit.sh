#!/bin/bash
#SBATCH --export=ALL
#SBATCH --chdir=/ceph/project/simmons_hts/kxu/JupyterFolder/hest/02_prepare_data/slurm_exec/20251202T1754-d534bdd5
#SBATCH --job-name=d534bdd5
#SBATCH --output=/ceph/project/simmons_hts/kxu/JupyterFolder/hest/02_prepare_data/slurm_exec/20251202T1754-d534bdd5/slurm-%j.out
#SBATCH --error=/ceph/project/simmons_hts/kxu/JupyterFolder/hest/02_prepare_data/slurm_exec/20251202T1754-d534bdd5/slurm-%j.err
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=40


exec /package/python-base/3.11.11/bin/python driver.py
