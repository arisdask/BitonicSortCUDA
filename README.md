# BitonicSortCUDA

## Introduction

### Problem Overview

The **Bitonic Sort** algorithm is a parallel comparison-based sorting algorithm that is particularly well-suited for GPU execution. The algorithm constructs a ***bitonic sequence*** (a sequence that first increases and then decreases, or vice versa) and recursively sorts it into a fully ordered sequence. The algorithm operates in $O(\log^2 N)$ steps for an array of size $N = 2^k$, where pairs of elements are compared and swapped.

#### Key Challenges
1. **High Kernel Launch Overhead**:
   - In the baseline implementation (**V0**), each step of the algorithm requires a separate kernel launch, resulting in $O(\log^2 N)$ kernel launches. This introduces significant overhead due to frequent global synchronization and kernel setup/teardown.

2. **Inefficient Global Memory Access**:
   - All data exchanges in **V0** and **V1** occur in global memory, which is slow compared to shared memory. Frequent global memory accesses lead to high latency and poor memory bandwidth utilization.

#### Optimizations
To address these challenges, three versions of the Bitonic Sort algorithm were implemented:

1. **V0 (Baseline Implementation)**:
   - A direct translation of the Bitonic Sort algorithm to CUDA.
   - Each step is executed in a separate kernel, leading to high kernel launch overhead and inefficient global memory access.

2. **V1 (Kernel Optimization)**:
   - Optimizes **V0** by fusing multiple steps into a single kernel, reducing the number of total kernel launches.
   - Uses global memory for all data exchanges but improves thread utilization and reduces *global synchronization* overhead.

3. **V2 (Shared Memory Optimization)**:
   - Introduces shared memory to further optimize the algorithm.
   - Reduces global memory accesses by performing intermediate computations in shared memory, significantly improving performance.
   - Achieves the best performance among the three versions by leveraging the GPU's memory hierarchy.


***Note:*** For further information about the algorithms we implemented in CUDA, refer to the [report](https://github.com/arisdask/BitonicSortCUDA/blob/main/report/bitonic_cuda_arisdask.pdf).


## Prerequisites for Running the Code

To successfully run the Bitonic Sort CUDA implementations on your computer, ensure that you have the following software and hardware requirements:

### 1. **Hardware Requirements**
   - **NVIDIA GPU**: The code is designed to run on NVIDIA GPUs with CUDA support. Ensure your system has a compatible GPU.
   - **CUDA-Capable GPU**: Check if your GPU supports CUDA by visiting the [NVIDIA CUDA GPUs page](https://developer.nvidia.com/cuda-gpus).
   - **Sufficient Memory**: Ensure your GPU has enough memory to handle the input array size. For large arrays, a GPU with at least 2GB of memory is recommended.

### 2. **Software Requirements**
   - **NVIDIA CUDA Toolkit**: Install the CUDA Toolkit compatible with your GPU and operating system. Download it from the [NVIDIA CUDA Toolkit website](https://developer.nvidia.com/cuda-toolkit).
   - **CUDA Libraries**: The CUDA Toolkit includes all necessary libraries (e.g., `cuda_runtime.h`).
   - _**Tip:** Remember to add the `nvcc` compiler to the PATH, based the location you have downloaded it,_ e.g.:
   ```bash
   export PATH=/usr/local/cuda-12.8/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
   source ~/.bashrc
   ```

### 3. **Project Setup**

#### 1. **Clone the Repository**
   Clone the repository to your local machine using the following command:
   ```bash
   git clone https://github.com/arisdask/BitonicSortCUDA.git
   cd BitonicSortCUDA
   ```

#### 2. **Build the Project**
   The project uses a `Makefile` to compile the code. Run the following commands to clean and build the project:
   ```bash
   make clean
   make
   ```
   This will compile all the source files (both C++ and CUDA) and generate an executable named `bitonic_sort_cuda`.

#### 3. **Run the Executable**
   After building the project, you can run the executable with the following command:
   ```bash
   ./bitonic_sort_cuda <q> <v>
   ```
   - Replace `<q>` with the exponent for the array size. The array size will be $2^q$. For example, if $q = 10$, the array size will be $1024$.
   - Replace `<v>` with the version of the Bitonic Sort algorithm you want to run:
     - `0` for **V0 (Baseline Implementation)**.
     - `1` for **V1 (Kernel Optimization)**.
     - `2` for **V2 (Shared Memory Optimization)**.

   **Example**:
   ```bash
   ./bitonic_sort_cuda 10 2
   ```
   This will run the **V2 implementation** on an array of size $1024$.

   ***Note:*** We run the code not on a local device but in the [Aristotelis HPC System](https://hpc.it.auth.gr/). Therefore, in the following section, we will also show the way and scripts to run the code in order to test it on this system (or a similar one).


## Cloning the Repository to HPC System
This repository can be integrated with the Aristotelis HPC system in the following ways:  

1. **Direct Cloning on the HPC System:**  
   After connecting to the Aristotelis HPC system, clone the repository directly into your home directory using the command:
   ```bash
   git clone https://github.com/arisdask/BitonicSortCUDA.git
   ```
2. **Local Cloning and Uploading:**  
   Alternatively, you can clone the repository to your local machine:  
   ```bash
   git clone https://github.com/arisdask/BitonicSortCUDA.git
   ```  
   Then, upload the repository to the Aristotelis HPC system using a secure file transfer tool such as `scp` or `rsync`:  
   ```bash
   scp -r BitonicSortMPI [username]@aristotle.it.auth.gr:/path/to/home/
   ```
Replace `username` and `/path/to/home/` with your actual HPC credentials and desired directory.


## Submitting the Code Using `bash-submit-test-cases.sh`

#### **Purpose**
The `bash-submit-test-cases.sh` script is designed to automate the submission of SLURM jobs for a range of test cases based on user-provided input parameters. Specifically, it executes the `bash-submit-bitonic-cuda.sh` script with combinations of parameters `q` (the power of 2 for the number of elements to sort) and `v` (the version). It ensures input validation and offers flexible usage options for running multiple test cases efficiently.

#### **Script Functionality**
The script performs the following steps:

1. **Argument Parsing and Validation**:
   - Accepts positional arguments and flags:
     - `<q_1>`: The starting value of `q`. (Required)
     - `[q_2]`: The ending value of `q`. (Optional, defaults to `q_1`.)
     - `-v <v>`: Specifies a version `v` (0, 1, or 2). (Optional, defaults to running for all valid versions: 0, 1, and 2.)
   - Validates that:
     - `q_1` and `q_2` are positive integers (`> 0`).
     - `q_2` is greater than or equal to `q_1` if provided.
     - `v` is one of the allowed values (0, 1, or 2) when specified.

2. **Default Behavior**:
   - If no `-v` flag is given, it runs for all three versions: 0, 1, and 2.
   - If only `q_1` is provided, it runs for that specific `q_1` (across all versions or the specified version).

3. **Iterative Execution**:
   - Iterates over the range `[q_1, q_2]` and executes `bash-submit-bitonic-cuda.sh` for each value of `q` and for all specified versions `v`.

4. **SLURM Job Submission**:
   - Submits the SLURM jobs by invoking `bash-submit-bitonic-cuda.sh` with the appropriate parameters.
   - Displays a confirmation message for each job submitted.

5. **SLURM Queue Monitoring**:
   - At the end, the script runs `squeue -u $USER` to display the current SLURM queue for the user, showing the status of submitted jobs.


#### **How to Use the Script**

1. **Basic Syntax**:
   ```bash
   ./bash-submit-test-cases.sh <q_1> [q_2] [-v <v>]
   ```
   - `<q_1>`: Required. The starting value of `q` (must be a positive integer).
   - `[q_2]`: Optional. The ending value of `q` (defaults to `q_1`).
   - `-v <v>`: Optional. Specifies the version to run (must be 0, 1, or 2). If omitted, it runs for all versions.

   ***Note:*** In case we need to run the code for $q > 27$, then we probably need to change the total time allocated for the job in the SLURM script `sbatch-hpc-bitonic-cuda.sh` from **2 minutes to 5 minutes** to ensure that the code executes before the time limit.

2. **Examples**:

   - **Run for a single `q` and all versions**:
     ```bash
     ./bash-submit-test-cases.sh 5
     ```
     Submits jobs for `q=5` and `v=0`, `v=1`, `v=2`.

   - **Run for a range of `q` and all versions**:
     ```bash
     ./bash-submit-test-cases.sh 5 8
     ```
     Submits jobs for `q=5, 6, 7, 8` and `v=0`, `v=1`, `v=2`.

   - **Run for a single `q` and a specific version**:
     ```bash
     ./bash-submit-test-cases.sh 5 -v 1
     ```
     Submits jobs for `q=5` and `v=1`.

   - **Run for a range of `q` and a specific version**:
     ```bash
     ./bash-submit-test-cases.sh 5 8 -v 2
     ```
     Submits jobs for `q=5, 6, 7, 8` and `v=2`.

   - **Handle invalid input**:
     - If `q_1` is `<= 0` or `v` is not one of `0`, `1`, or `2`, the script prints an error message and exits.

3. **Error Handling**:
   - If required arguments are missing, the script provides a usage guide.
   - If invalid values for `q` or `v` are provided, it displays an appropriate error message.


#### **Important Notes**
- Ensure that `bash-submit-bitonic-cuda.sh` is present in the same directory or accessible in the system's PATH.
- This script is designed for SLURM-based high-performance computing (HPC) environments.
- The user must have permission to submit SLURM jobs and access the necessary modules or software dependencies.
