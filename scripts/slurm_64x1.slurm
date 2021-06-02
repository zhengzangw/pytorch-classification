#!/bin/bash -l

#SBATCH --job-name=default
#SBATCH --output=%j-%x.%u.out
#SBATCH --account=s1054

#SBATCH --constraint=gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12

#SBATCH --signal=SIGUSR1@90

module load daint-gpu

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_HCA=ipogif0

. /users/lyongbin/miniconda3/bin/activate pycls
cd /scratch/snx3000/lyongbin/zangwei/classification

srun python run.py trainer.gpus=1 +trainer.num_nodes=64 $@
