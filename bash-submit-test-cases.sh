#!/bin/bash

# Check if at least one argument is provided
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <i> [j]"
    exit 1
fi

# Assign arguments to variables
i=$1
j=${2:-$1} # If j is not provided, set j to i

# Clean, build, and run the program:
make clean
module load gcc openmpi
make

# Handle the case where i = -1 (run all test cases)
if [ "$i" -eq -1 ]; then
    echo "Running all test cases..."
    for script in tests/hpc_bitonic_mpi_testcase*.sh; do
        if [ -f "$script" ]; then
            echo "Submitting $script..."
            sbatch "$script"
        else
            echo "No test case scripts found."
            exit 1
        fi
    done
    exit 0
fi

# Ensure i and j are valid integers
if ! [[ "$i" =~ ^[0-9]+$ ]] || ! [[ "$j" =~ ^[0-9]+$ ]]; then
    echo "Error: i and j must be integers."
    exit 1
fi

# Ensure i <= j
if [ "$i" -gt "$j" ]; then
    echo "Error: i must be less than or equal to j."
    exit 1
fi

# Run the specified range of test cases
echo "Running test cases from $i to $j..."
for num in $(seq "$i" "$j"); do
    script="tests/hpc_bitonic_mpi_testcase${num}.sh"
    if [ -f "$script" ]; then
        echo "Submitting $script..."
        sbatch "$script"
    else
        echo "Warning: $script not found, skipping..."
    fi
done

squeue -u "$USER"