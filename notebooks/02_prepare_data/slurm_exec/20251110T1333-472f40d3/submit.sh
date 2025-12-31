#!/bin/bash
#SBATCH --export=ALL
#SBATCH --chdir=/ceph/project/simmons_hts/kxu/JupyterFolder/hest/02_prepare_data/slurm_exec/20251110T1333-472f40d3
#SBATCH --job-name=472f40d3
#SBATCH --output=/ceph/project/simmons_hts/kxu/JupyterFolder/hest/02_prepare_data/slurm_exec/20251110T1333-472f40d3/slurm-%j.out
#SBATCH --error=/ceph/project/simmons_hts/kxu/JupyterFolder/hest/02_prepare_data/slurm_exec/20251110T1333-472f40d3/slurm-%j.err
#SBATCH --partition=short
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=20


exec /package/python-base/3.11.11/bin/python driver.py
