#!/bin/bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/bitonic_sort_%j.out
#SBATCH --error=logs/bitonic_sort_%j.err

# Load required modules
module load gcc openmpi

# Set environment variables
export UCX_WARN_UNUSED_ENV_VARS=n

# In case of `UCX  ERROR connect(fd=.., dest_addr=..) failed: Connection timed out` 
# uncomment *one* of the following exports (recommended to use tcp) to ensure "safer" transportation of data:
# export UCX_TLS=tcp               # Use TCP for UCX transport
# export UCX_RNDV_TIMEOUT=5000     # Time in milliseconds

# Run the program using srun
# Usage: $0 <q: 2^q numbers/process> <p: 2^p processes>
# When we change the number of processes (2^p) we should set the nodes/ntasks-per-node
srun ./bin/bitonic_mpi 20 3
