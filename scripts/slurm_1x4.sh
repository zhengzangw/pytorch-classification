#!/bin/bash

#SBATCH --job-name=default
#SBATCH --output=%j-%x.%u.out
#SBATCH --account=m3691

#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=12

#SBATCH --signal=SIGUSR1@90

export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=INFO

export CONDA_NAME=pycls
export CONDA_PYTHON=/global/homes/z/zangwei/.conda/envs/$CONDA_NAME

conda activate $CONDA_PYTHON

srun python run.py trainer.gpus=4 $@

# +trainer.weights_save_path=/absolute/path
