#ifndef BITONIC_SORT_CUH
#define BITONIC_SORT_CUH

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "utils.hpp"
// #include "bitonic_sort_operations.cuh"
#include </usr/local/cuda/include/cuda_runtime.h>
// #include <cuda_runtime.h>

// Adjust as needed for optimal performance based on the gpu's specs
#define THREADS_PER_BLOCK 1024

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


/**
 * Performs bitonic sort on an `IntArray` using the V1 implementation.
 *
 * @param array The `IntArray` to be sorted.
 *
 * @return void
 *
 * @note   This version incorporates the `k` inner loop directly into the CUDA kernel,
 *         reducing the number of global synchronizations and function calls.
 *         It is more efficient than V0.
 */
void bitonic_sort_v1(IntArray& array);


/**
 * Performs bitonic sort on an `IntArray` using the V2 implementation.
 *
 * @param array The `IntArray` to be sorted.
 *
 * @return void
 *
 * @note   This version optimizes the V1 implementation by using local memory
 *         instead of global memory in the CUDA kernel, significantly improving
 *         performance for larger arrays.
 */
void bitonic_sort_v2(IntArray& array);


#endif // BITONIC_SORT_CUH
