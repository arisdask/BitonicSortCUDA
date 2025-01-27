#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <q: 2^q numbers to sort> <v: version>"
    exit 1
fi

q=$1
v=$2

# Check if both arguments are non-negative integers
if ! [[ "$q" =~ ^[0-9]+$ ]] || ! [[ "$v" =~ ^[0-9]+$ ]]; then
    echo "Error: Both arguments must be non-negative integers."
    exit 1
fi

# Define the path of the SLURM script
SBATCH_SCRIPT="sbatch-hpc-bitonic-cuda.sh"

# Check if the SLURM script exists
if [ ! -f "$SBATCH_SCRIPT" ]; then
    echo "Error: $SBATCH_SCRIPT not found."
    exit 1
fi

# Modify the SLURM script to update its q & v values
sed -i \
    -e "s|#SBATCH --output=logs/cuda_sort_[0-9]\+_[0-9]\+_%j.out|#SBATCH --output=logs/cuda_sort_${q}_${v}_%j.out|" \
    -e "s|#SBATCH --error=logs/cuda_sort_[0-9]\+_[0-9]\+_%j.err|#SBATCH --error=logs/cuda_sort_${q}_${v}_%j.err|" \
    -e "s|#SBATCH --job-name=sort_[0-9]\+_[0-9]\+|#SBATCH --job-name=sort_${q}_${v}|" \
    -e "s|./bitonic_sort_cuda [0-9]\+ [0-9]\+|./bitonic_sort_cuda ${q} ${v}|" \
    "$SBATCH_SCRIPT"

# Submit the modified script using sbatch
sbatch "$SBATCH_SCRIPT"

# Print success message
echo "SLURM job submitted with q=$q and v=$v."
