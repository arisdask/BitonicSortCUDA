#ifndef BITONIC_SORT_OPERATIONS_CUH
#define BITONIC_SORT_OPERATIONS_CUH

#include </usr/local/cuda/include/cuda_runtime.h>
// #include <cuda_runtime.h>

/**
 * Swaps two elements in the array.
 * 
 * @param data     Pointer to the array.
 * @param idx      Index of the first element.
 * @param partner  Index of the second element.
 */
__device__ __inline__ void swap(int* data, int idx, int partner) {
    int temp      = data[idx];
    data[idx]     = data[partner];
    data[partner] = temp;
}

/**
 * Determines if the comparison should be in ascending order.
 * 
 * @param idx    Thread index.
 * @param stage  Current stage of the Bitonic Sort.
 * 
 * @return `true` if ascending, `false` if descending.
 */
__device__ __inline__ bool is_ascending(int idx, int stage) {
    return ( idx & (1 << (stage + 1)) ) == 0;
}

/**
 * Computes the partner index for a given thread.
 * 
 * @param idx   Thread index.
 * @param step  Current substage step of the Bitonic Sort.
 * 
 * @return The partner index for comparison.
 */
__device__ __inline__ int find_partner(int idx, int step) {
    return idx ^ (1 << step);    // Compute partner index using XOR (Hamming distance)
}

/**
 * Performs a comparison and exchange operation between two elements.
 * 
 * @param data       Pointer to the array.
 * @param idx        Index of the "local" element.
 * @param partner    Index of the partner element.
 * @param ascending  Whether to perform the exchange in ascending order.
 */
__device__ __inline__ void exchange(int* data, int idx, int partner, bool ascending) {
    if (ascending) {
        // Swap if elements are out of order in ascending order
        if (data[idx] > data[partner]) {
            swap(data, idx, partner);
        }
    } else {
        // Swap if elements are out of order in descending order
        if (data[idx] < data[partner]) {
            swap(data, idx, partner);
        }
    }
}

#endif // BITONIC_SORT_OPERATIONS_CUH
