#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 <q_1> [q_2] [-v <v>] "
    echo "  <q_1>: Starting value of q (must be > 0)."
    echo "  [q_2]: Ending value of q (optional, must be >= q_1)."
    echo "  [-v <v>]: Specific version (v) to use (optional, must be 0, 1, or 2)."
    echo "If -v is not provided, it will run for v=0, v=1, and v=2."
    exit 1
}

# Default values for arguments
q_1=""
q_2=""
v_list=(0 1 2) # Default versions

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--version|-ver|-vers)
            if [[ $# -lt 2 ]]; then
                echo "Error: Missing value for -v."
                usage
            fi
            v_list=($2)
            if ! [[ "${v_list[0]}" =~ ^[0-2]$ ]]; then
                echo "Error: Invalid version number. Allowed values are 0, 1, or 2."
                exit 1
            fi
            shift 2
            ;;
        *)
            if [[ -z "$q_1" ]]; then
                q_1=$1
            elif [[ -z "$q_2" ]]; then
                q_2=$1
            else
                echo "Error: Too many positional arguments."
                usage
            fi
            shift
            ;;
    esac
done

# Validate q_1
if [[ -z "$q_1" ]] || ! [[ "$q_1" =~ ^[0-9]+$ ]] || [[ "$q_1" -le 0 ]]; then
    echo "Error: q_1 must be a positive integer."
    exit 1
fi

# Set q_2 to q_1 if not provided
if [[ -z "$q_2" ]]; then
    q_2=$q_1
fi

# Validate q_2
if ! [[ "$q_2" =~ ^[0-9]+$ ]] || [[ "$q_2" -lt "$q_1" ]]; then
    echo "Error: q_2 must be an integer greater than or equal to q_1."
    exit 1
fi

# Run the bash script for the specified range of q and versions
echo "Running bash-submit-bitonic-cuda.sh for q=$q_1 to q=$q_2 with versions=${v_list[*]}..."
for q in $(seq "$q_1" "$q_2"); do
    for v in "${v_list[@]}"; do
        echo "Running for q=$q, v=$v..."
        bash bash-submit-bitonic-cuda.sh "$q" "$v"
    done
done

# Display the SLURM queue
squeue -u "$USER"
