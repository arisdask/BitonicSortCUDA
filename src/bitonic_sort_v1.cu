#include "../inc/bitonic_sort.cuh"

__global__ void bitonic_kernel_v1_first_stages(int* data, size_t length, int step_max) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure threads operate within valid range
    if (tid >= (length >> 1)) return;

    for (int stage = 0; stage <= step_max; stage++) {
        for (int step = stage; step >= 0; step--) {
            // Find the index that this thread will handle
            size_t idx = GET_ARR_ID(tid, (size_t)step);

            // Find the partner index that this thread will handle
            size_t partner = idx ^ (1 << step);

            // Ensure valid partner index
            if (idx < partner && partner < length) {
                exchange(data, idx, idx, partner, stage);
            }
            __syncthreads();  // Synchronize threads for each step
        }
    }
}

__global__ void bitonic_kernel_v1_lower_steps(int* data, size_t length, int stage, int step_max) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure threads operate within valid range
    if (tid >= (length >> 1)) return;

    for (int step = step_max; step >= 0; step--) {
        // Find the index that this thread will handle
        size_t idx = GET_ARR_ID(tid, (size_t)step);

        // Find the partner index
        size_t partner = idx ^ (1 << step);

        // Ensure valid partner index
        if (idx < partner && partner < length) {
            exchange(data, idx, idx, partner, stage);
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

    #ifdef TIME_MEASURE
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    #endif

    size_t num_blocks = ((array.length >> 1) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int    stages     = __builtin_ctz(array.length); // log2(array.length)
    
    // This is the maximum step that can be handled by a single thread block
    int step_max   = __builtin_ctz(THREADS_PER_BLOCK);

    // Launch the v1 kernel for the *first stages*, in which stage = 0:1:step_max
    bitonic_kernel_v1_first_stages<<<num_blocks, THREADS_PER_BLOCK>>>(d_data, array.length, step_max);
    cudaDeviceSynchronize();

    // Launch the Bitonic Sort for higher stages
    // All the previous stages, from 0 to step_max, were handled by a single kernel call
    for (int stage = step_max+1; stage < stages; stage++) {
        for (int step = stage; step > step_max; step--) {
            // Launch the kernel v0 for higher steps ( > step_max )
            bitonic_kernel_v0<<<num_blocks, THREADS_PER_BLOCK>>>(d_data, array.length, stage, step);

            #ifdef DEBUG
            if (check_cuda_error(d_data)) return;
            #endif

            cudaDeviceSynchronize();
        }

        // Launch the v1 kernel for all *lower steps*, in which step = step_max:-1:0
        bitonic_kernel_v1_lower_steps<<<num_blocks, THREADS_PER_BLOCK>>>(d_data, array.length, stage, step_max);

        #ifdef DEBUG
        if (check_cuda_error(d_data)) return;
        #endif

        cudaDeviceSynchronize();
    }

    #ifdef TIME_MEASURE
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[v1 internal time] Execution Time: %f msec, Normalized Execution Time: %f msec per element\n", 
            milliseconds, milliseconds / array.length);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    #endif

    // Copy the sorted data back to the host
    cudaMemcpy(array.data, d_data, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);
}
