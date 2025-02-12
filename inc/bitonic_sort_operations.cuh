#ifndef BITONIC_SORT_OPERATIONS_CUH
#define BITONIC_SORT_OPERATIONS_CUH

// #include </usr/local/cuda/include/cuda_runtime.h>
#include <cuda_runtime.h>

/**
 * GET_ARR_ID macro.
 *
 * Computes the index based on the thread id (`tid`) and the given `step`.
 *
 * Algorithmic behavior:
 *   - chunk_size = (1 << step)
 *   - jump_size  = (1 << (step + 1))
 *   - chunk_index = tid / chunk_size
 *   - position    = tid % chunk_size
 *   - If chunk_index == 0: returns position,
 *     else returns jump_size * chunk_index + position.
 *
 * Explanation:
 *   (tid >> step) gives tid/chunk_size,
 *   ((tid >> step) << step) equals tid - (tid % chunk_size),
 *   so the macro computes: tid + (tid - (tid % chunk_size)) = 2*tid - (tid % chunk_size).
 *   For tid < chunk_size, (tid % chunk_size) == tid, so it returns tid.
 *
 * @param tid   Thread id.
 * @param step  Exponent such that chunk_size = (1 << step).
 * @return      The computed array index.
 */
#define GET_ARR_ID(tid, step) ((tid) + (((tid) >> (step)) << (step)))


/**
 * Performs the comparison and swap operation for a single step in the bitonic sorting process.
 * 
 * @param data        A pointer to the array containing the data being sorted.
 * @param local_idx   The index of the current thread's local data element in the shared or global memory array.
 * @param global_idx  The index of the current thread's global position in the array.
 * @param partner     The index of the partner element with which the current element is compared and potentially swapped.
 * @param stage       The current sorting stage, which determines the sorting direction (ascending or descending).
 */
__device__ __forceinline__ void exchange(int* data, int local_idx, int global_idx, int partner, int stage) {
    if ( (global_idx & (1 << (stage + 1))) == 0 ) {
        // Swap if elements are out of order in ascending order
        if (data[local_idx] > data[partner]) {
            int temp        = data[local_idx];
            data[local_idx] = data[partner];
            data[partner]   = temp;
        }
    } else {
        // Swap if elements are out of order in descending order
        if (data[local_idx] < data[partner]) {
            int temp        = data[local_idx];
            data[local_idx] = data[partner];
            data[partner]   = temp;
        }
    }
}

/**
 * Loads two consecutive elements from global memory into shared memory.
 * If an index is out of bounds (>= length), the corresponding shared memory
 * slot is assigned INT_MAX as a sentinel value.
 *
 * @param global_idx    The index in the global array to read from.
 * @param length        The total number of elements in the global array.
 * @param shared_array  Pointer to shared memory where data will be stored.
 * @param local_idx     The index in the shared memory array.
 * @param global_array  Pointer to the global memory array.
 */
__device__ __forceinline__ void load_to_shared(const int    global_idx,
                                              const int     length,
                                              int*          shared_array,
                                              const int     local_idx,
                                              const int*    global_array) {
    // Load valid elements; otherwise, assign INT_MAX for out-of-bounds values.
    shared_array[local_idx]     = (global_idx     < length) ? global_array[global_idx]     : INT_MAX;
    shared_array[local_idx + 1] = ((global_idx+1) < length) ? global_array[global_idx + 1] : INT_MAX;
}

/**
 * Writes up to two consecutive elements from shared memory back to global memory.
 * Only writes values if the global index is within bounds.
 *
 * @param global_idx    The index in the global array to write to.
 * @param length        The total number of elements in the global array.
 * @param global_array  Pointer to the global memory array.
 * @param local_idx     The index in the shared memory array.
 * @param shared_array  Pointer to shared memory where sorted data is stored.
 */
__device__ __forceinline__ void write_to_global(const int   global_idx,
                                               const int    length,
                                               int*         global_array,
                                               const int    local_idx,
                                               const int*   shared_array) {
    if (global_idx < length)        { global_array[global_idx]     = shared_array[local_idx]; }
    if ((global_idx + 1) < length)  { global_array[global_idx + 1] = shared_array[local_idx + 1]; }
}

/**
 * Checks for CUDA errors and prints the error message if any.
 * If an error occurs, it frees the given device memory and returns true.
 *
 * @param d_data  Pointer to the device memory to be freed in case of an error.
 * @return        `true` if an error occurred, `false` otherwise.
 * 
 * @note To enable debug mode and error checking, define the `DEBUG` preprocessor macro before compiling the code.
 */
inline bool check_cuda_error(int* d_data) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return true;
    }

    return false;
}

#endif // BITONIC_SORT_OPERATIONS_CUH
