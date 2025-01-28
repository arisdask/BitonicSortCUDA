#include "../inc/bitonic_sort.cuh"

// Load data from global memory into shared memory
#define LOAD_TO_SHARED(global_idx, length, shared_array, local_idx, global_array)   \
    if ((global_idx) < (length)) {                                                  \
        (shared_array)[local_idx] = (global_array)[global_idx];                     \
    } else {                                                                        \
        (shared_array)[local_idx] = INT_MAX; /* Sentinel value for out-of-bounds */ \
    }

// Write back sorted data to global memory
#define WRITE_TO_GLOBAL(global_idx, length, global_array, local_idx, shared_array)  \
    if ((global_idx) < (length)) {                                                  \
        (global_array)[global_idx] = (shared_array)[local_idx];                     \
    }


__global__ void bitonic_kernel_v2_first_stages(int* data, int length, int step_min) {
    // Define shared memory for local block storage
    __shared__ int shared_data[THREADS_PER_BLOCK];

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx  = threadIdx.x;

    LOAD_TO_SHARED(global_idx, length, shared_data, local_idx, data);
    __syncthreads();  // Synchronize to ensure all threads have loaded data

    // Perform bitonic sorting for the initial stages (0 to step_min-1)
    for (int stage = 0; stage < step_min; stage++) {
        for (int step = stage; step >= 0; step--) {
            // Compute partner index within the block
            int partner_idx = local_idx ^ (1 << step);

            if (local_idx < partner_idx && partner_idx < blockDim.x) {
                BITONIC_COMPARE_AND_SWAP(local_idx, global_idx, partner_idx, stage, shared_data);
            }
            __syncthreads();  // Synchronize threads after each step
        }
    }

    WRITE_TO_GLOBAL(global_idx, length, data, local_idx, shared_data);
}

__global__ void bitonic_kernel_v2_lower_steps(int* data, int length, int stage, int step_min) {
    // Define shared memory for local block storage
    __shared__ int shared_data[THREADS_PER_BLOCK];

    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx  = threadIdx.x;

    LOAD_TO_SHARED(global_idx, length, shared_data, local_idx, data);
    __syncthreads();  // Synchronize to ensure all threads have loaded data

    // Perform bitonic sorting for the lower steps (step_min-1 to 0)
    for (int step = step_min - 1; step >= 0; step--) {
        // Compute partner index within the block
        int partner_idx = local_idx ^ (1 << step);

        if (local_idx < partner_idx && partner_idx < blockDim.x) {
            BITONIC_COMPARE_AND_SWAP(local_idx, global_idx, partner_idx, stage, shared_data);
        }
        __syncthreads();  // Synchronize threads after each step
    }

    WRITE_TO_GLOBAL(global_idx, length, data, local_idx, shared_data);
}

void bitonic_sort_v2(IntArray& array) {
    int* d_data;
    size_t size = array.length * sizeof(int);

    // Allocate memory on the device
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, array.data, size, cudaMemcpyHostToDevice);

    int num_blocks = (array.length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int stages     = __builtin_ctz(array.length); // log2(array.length)
    
    // This is the minimum step **that exceeds** the current thread block boundary
    int step_min   = __builtin_ctz(THREADS_PER_BLOCK);

    // Launch the v2 kernel for the *first stages*, in which stage = 0:1:(step_min-1)
    bitonic_kernel_v2_first_stages<<<num_blocks, THREADS_PER_BLOCK>>>(d_data, array.length, step_min);

    // Launch the Bitonic Sort for higher stages
    // All the previous stages, from 0 to step_min-1, were handled by a single kernel call
    for (int stage = step_min; stage < stages; stage++) {
        for (int step = stage; step >= step_min; step--) {
            // Launch the kernel v0 for higher steps ( >= step_min )
            bitonic_kernel_v0<<<num_blocks, THREADS_PER_BLOCK>>>(d_data, array.length, stage, step);

            #ifdef DEBUG
            // Optional kernel error-checking
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA error: %s\n", cudaGetErrorString(err));
                cudaFree(d_data);
                return;
            }
            #endif

            cudaDeviceSynchronize();
        }

        // Launch the v2 kernel for all *lower steps*, in which step = (step_min-1):-1:0
        bitonic_kernel_v2_lower_steps<<<num_blocks, THREADS_PER_BLOCK>>>(d_data, array.length, stage, step_min);

        #ifdef DEBUG
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            cudaFree(d_data);
            return;
        }
        #endif

        cudaDeviceSynchronize();
    }

    // Copy the sorted data back to the host
    cudaMemcpy(array.data, d_data, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);
}
