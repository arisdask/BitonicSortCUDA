#!/bin/bash
#SBATCH --job-name=sort_27_0
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30
#SBATCH --output=logs/cuda_sort_27_0_%j.out
#SBATCH --error=logs/cuda_sort_27_0_%j.err

nvidia-smi

# load the required modules
module load gcc/13.2.0-iqpfkya cuda/12.4.0-zk32gam

# compile and run bitonic_sort_cuda
make clean
make

./bitonic_sort_cuda 27 0
