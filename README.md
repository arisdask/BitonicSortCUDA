# BitonicSortCUDA
Bitonic Sort with CUDA implementation for fast, large-scale processing.


```
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```

---

### Documentation for `bash-submit-test-cases.sh`

#### **Purpose**
The `bash-submit-test-cases.sh` script is designed to automate the submission of SLURM jobs for a range of test cases based on user-provided input parameters. Specifically, it executes the `bash-submit-bitonic-cuda.sh` script with combinations of parameters `q` (the power of 2 for the number of elements to sort) and `v` (the version). It ensures input validation and offers flexible usage options for running multiple test cases efficiently.

---

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

---

#### **How to Use the Script**

1. **Basic Syntax**:
   ```bash
   ./bash-submit-test-cases.sh <q_1> [q_2] [-v <v>]
   ```
   - `<q_1>`: Required. The starting value of `q` (must be a positive integer).
   - `[q_2]`: Optional. The ending value of `q` (defaults to `q_1`).
   - `-v <v>`: Optional. Specifies the version to run (must be 0, 1, or 2). If omitted, it runs for all versions.

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

---

#### **Benefits**
- **Automation**: Automatically generates and submits multiple SLURM jobs for specified parameters.
- **Flexibility**: Allows running for a single value of `q`, a range of values, or specific versions.
- **Error Validation**: Ensures invalid inputs are caught and prevents incorrect job submissions.
- **Monitoring**: Displays the current SLURM queue for the user after job submissions.

---

#### **Important Notes**
- Ensure that `bash-submit-bitonic-cuda.sh` is present in the same directory or accessible in the system's PATH.
- This script is designed for SLURM-based high-performance computing (HPC) environments.
- The user must have permission to submit SLURM jobs and access the necessary modules or software dependencies.