#!/bin/bash

#SBATCH --job-name=default
#SBATCH --output=%j-%x.%u.out
#SBATCH --account=m3691

#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --time=02:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12

#SBATCH --signal=SIGUSR1@90

export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth3
export NCCL_IB_DISABLE=1

export CONDA_NAME=pycls
export CONDA_PYTHON=/global/homes/z/zangwei/.conda/envs/$CONDA_NAME

conda activate $CONDA_PYTHON

srun python run.py trainer.gpus=8 +trainer.num_nodes=8 $@
