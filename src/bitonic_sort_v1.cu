#include "../inc/bitonic_sort.cuh"

__global__ void bitonic_kernel_v1_first_stages(int* data, int length, int step_min) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure threads operate within valid range
    if (idx >= length) return;

    for (int stage = 0; stage < step_min; stage++) {
        for (int step = stage; step >= 0; step--) {
            // Find the partner index
            int partner = idx ^ (1 << step);

            // Ensure valid partner index
            if (idx < partner && partner < length) {
                BITONIC_COMPARE_AND_SWAP(idx, idx, partner, stage, data)
            }
            __syncthreads();  // Synchronize threads for each step
        }
    }
}

__global__ void bitonic_kernel_v1_lower_steps(int* data, int length, int stage, int step_min) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure threads operate within valid range
    if (idx >= length) return;

    for (int step = step_min - 1; step >= 0; step--) {
        // Find the partner index
        int partner = idx ^ (1 << step);

        // Ensure valid partner index
        if (idx < partner && partner < length) {
            BITONIC_COMPARE_AND_SWAP(idx, idx, partner, stage, data)
        }

        __syncthreads();  // Synchronize threads for each step
    }
}

void bitonic_sort_v1(IntArray& array) {
    int* d_data;
    size_t size = array.length * sizeof(int);

    // Allocate memory on the device
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, array.data, size, cudaMemcpyHostToDevice);

    int num_blocks = (array.length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int stages     = __builtin_ctz(array.length); // log2(array.length)
    
    // This is the minimum step **that exceeds** the current thread block boundary
    int step_min   = __builtin_ctz(THREADS_PER_BLOCK);

    // Launch the v1 kernel for the *first stages*, in which stage = 0:1:(step_min-1)
    bitonic_kernel_v1_first_stages<<<num_blocks, THREADS_PER_BLOCK>>>(d_data, array.length, step_min);

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

        // Launch the v1 kernel for all *lower steps*, in which step = (step_min-1):-1:0
        bitonic_kernel_v1_lower_steps<<<num_blocks, THREADS_PER_BLOCK>>>(d_data, array.length, stage, step_min);

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
