#!/bin/bash
#SBATCH --export=ALL
#SBATCH --chdir=/ceph/project/simmons_hts/kxu/JupyterFolder/hest/02_prepare_data/slurm_exec/20251110T1216-ce417852
#SBATCH --job-name=ce417852
#SBATCH --output=/ceph/project/simmons_hts/kxu/JupyterFolder/hest/02_prepare_data/slurm_exec/20251110T1216-ce417852/slurm-%j.out
#SBATCH --error=/ceph/project/simmons_hts/kxu/JupyterFolder/hest/02_prepare_data/slurm_exec/20251110T1216-ce417852/slurm-%j.err
#SBATCH --partition=gpu-ada
#SBATCH --time=00:20:00
#SBATCH --gpus=1


exec /package/python-base/3.11.11/bin/python driver.py
