#!/bin/bash

#SBATCH --job-name=default
#SBATCH --output=%j-%x.%u.out
#SBATCH --account=m3691

#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8

#SBATCH --signal=SIGUSR1@90

export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=INFO

export CONDA_NAME=pycls
export CONDA_PYTHON=/global/homes/z/zangwei/.conda/envs/$CONDA_NAME

conda activate $CONDA_PYTHON

srun python run.py trainer.gpus=2 $@

# interactive
# salloc -t 10 -C gpu -N 1 --gpus-per-node 2 -A m3691
# salloc -t 10 -C gpu -N 2 --gpus-per-node 2 -A m3691
