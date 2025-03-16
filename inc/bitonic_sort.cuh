#ifndef BITONIC_SORT_CUH
#define BITONIC_SORT_CUH

// #include <cstdlib>
#include "utils.hpp"
#include "bitonic_sort_operations.cuh"
// #include </usr/local/cuda/include/cuda_runtime.h>
#include <cuda_runtime.h>

/**
 * Number of threads per block for GPU execution. Must be a power of 2 for Bitonic Sort.
 * @note Adjust value based on target GPU specifications for optimal performance
 */
#define THREADS_PER_BLOCK 1024


//////// !!!  Uncomment the next line to enable debug mode in the source code  !!! ////////
// #define DEBUG


////////// !!!  Uncomment the next line to enable internal time measurement  !!! //////////
#define TIME_MEASURE


///////////////////////////////////   Bitonic Sort V0   ///////////////////////////////////
/**
 * CUDA kernel for the comparison and exchange operations in Bitonic Sort.
 *
 * @param data    A pointer to the array data in device memory.
 * @param length  The length of the array to be sorted.
 * @param stage   The current stage of the Bitonic Sort.
 * @param step    The current step within the stage.
 *
 * @return void
 *
 * @note   Each thread identifies its corresponding element and its partner,
 *         performs a comparison, and exchanges values if necessary. The kernel
 *         ensures threads operate only within valid array boundaries.
 */
__global__ void bitonic_kernel_v0(int* data, size_t length, int stage, int step);

/**
 * Performs bitonic sort on an `IntArray` using the V0 implementation.
 *
 * @param array The `IntArray` to be sorted.
 *
 * @return void
 *
 * @note   This version launches a CUDA kernel where each thread handles
 *         a single comparison and exchange. It is simple but involves
 *         many function calls and global synchronizations.
 */
void bitonic_sort_v0(IntArray& array);


///////////////////////////////////   Bitonic Sort V1   ///////////////////////////////////
/**
 * @brief Performs the first stages of bitonic sort where blocks of threads operate 
 *        independently without requiring global synchronizations.
 * 
 * @param data      Pointer to the array of integers in global memory.
 * @param length    Total number of elements in the array.
 * @param step_max  the maximum step that can be handled by a single thread block.
 */
__global__ void bitonic_kernel_v1_first_stages(int* data, size_t length, int step_max);

/**
 * @brief Performs the lower steps of a given stage in bitonic sort 
 *        without requiring global synchronizations.
 * 
 * @param data      Pointer to the array of integers in global memory.
 * @param length    Total number of elements in the array.
 * @param stage     Current sorting stage being processed.
 * @param step_max  The maximum step that can be handled by a single thread block.
 */
__global__ void bitonic_kernel_v1_lower_steps(int* data, size_t length, int stage, int step_max);

/**
 * Performs bitonic sort on an `IntArray` using the V1 implementation.
 *
 * @param array The `IntArray` to be sorted.
 *
 * @return void
 *
 * @note   This version incorporates the `k` inner loops directly into the CUDA V1 kernels 
 *         (when possible), reducing the number of global synchronizations and function calls.
 *         It is more efficient than V0.
 */
void bitonic_sort_v1(IntArray& array);


///////////////////////////////////   Bitonic Sort V2   ///////////////////////////////////
/**
 * @brief Performs the first stages of bitonic sort using shared memory for faster processing.
 * 
 * @param data      Pointer to the array of integers in global memory.
 * @param length    Total number of elements in the array.
 * @param step_max  The maximum step that can be handled by a single thread block.
 */
__global__ void bitonic_kernel_v2_first_stages(int* data, size_t length, int step_max);

/**
 * @brief Performs the lower steps of a given stage in bitonic sort 
 *        using shared memory for faster processing.
 * 
 * @param data      Pointer to the array of integers in global memory.
 * @param length    Total number of elements in the array.
 * @param stage     Current sorting stage being processed.
 * @param step_max  The maximum step that can be handled by a single thread block.
 */
__global__ void bitonic_kernel_v2_lower_steps(int* data, size_t length, int stage, int step_max);

/**
 * Performs bitonic sort on an `IntArray` using the V2 implementation.
 *
 * @param array The `IntArray` to be sorted.
 *
 * @return void
 *
 * @note   This version optimizes the V1 implementation by using local memory
 *         instead of global memory in the CUDA kernels, significantly improving
 *         performance for larger arrays.
 */
void bitonic_sort_v2(IntArray& array);


#endif // BITONIC_SORT_CUH
