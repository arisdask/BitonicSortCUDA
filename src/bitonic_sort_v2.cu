#include "../inc/bitonic_sort.cuh"

__global__ void bitonic_kernel_v2_first_stages(int* data, size_t length, int step_max) {
    // Define shared memory for local block storage
    // Each thread block can handle `2*THREADS_PER_BLOCK` elements (the idx & the partner)
    __shared__ int shared_data[THREADS_PER_BLOCK << 1];

    size_t global_tid  = blockIdx.x * blockDim.x + threadIdx.x;
    size_t local_tid   = threadIdx.x;
    size_t global_idx  = global_tid << 1;  // GET_ARR_ID(global_tid, 0);
    size_t local_idx   = local_tid << 1;   // GET_ARR_ID(local_tid, 0);

    load_to_shared(global_idx, length, shared_data, local_idx, data);
    __syncthreads();  // Synchronize to ensure all threads have loaded data

    // Perform bitonic sorting for the initial stages (0 to step_max)
    for (int stage = 0; stage <= step_max; stage++) {
        for (int step = stage; step >= 0; step--) {
            size_t idx = GET_ARR_ID(local_tid, (size_t)step);

            // Compute partner index within the block
            size_t partner = idx ^ (1 << step);

            if (idx < partner && partner < (THREADS_PER_BLOCK << 1)) {
                global_idx = GET_ARR_ID(global_tid, (size_t)step);  // Global position of element
                exchange(shared_data, idx, global_idx, partner, stage);
            }
            __syncthreads();  // Synchronize threads after each step
        }
    }

    global_idx  = global_tid << 1;  // GET_ARR_ID(global_tid, 0);
    write_to_global(global_idx, length, data, local_idx, shared_data);
}

__global__ void bitonic_kernel_v2_lower_steps(int* data, size_t length, int stage, int step_max) {
    // Define shared memory for local block storage
    // Each thread block can handle `2*THREADS_PER_BLOCK` elements (the idx & the partner)
    __shared__ int shared_data[THREADS_PER_BLOCK << 1];

    size_t global_tid  = blockIdx.x * blockDim.x + threadIdx.x;
    size_t local_tid   = threadIdx.x;
    size_t global_idx  = global_tid << 1;  // GET_ARR_ID(global_tid, 0);
    size_t local_idx   = local_tid << 1;   // GET_ARR_ID(local_tid, 0);

    load_to_shared(global_idx, length, shared_data, local_idx, data);
    __syncthreads();  // Synchronize to ensure all threads have loaded data

    // Perform bitonic sorting for the lower steps (step_max to 0)
    for (int step = step_max; step >= 0; step--) {
        size_t idx = GET_ARR_ID(local_tid, (size_t)step);

        // Compute partner index within the block
        size_t partner = idx ^ (1 << step);

        if (idx < partner && partner < (THREADS_PER_BLOCK << 1)) {
            global_idx = GET_ARR_ID(global_tid, (size_t)step);
            exchange(shared_data, idx, global_idx, partner, stage);
        }
        __syncthreads();  // Synchronize threads after each step
    }

    global_idx  = global_tid << 1;  // GET_ARR_ID(global_tid, 0);
    write_to_global(global_idx, length, data, local_idx, shared_data);
}

void bitonic_sort_v2(IntArray& array) {
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

    // Launch the v2 kernel for the *first stages*, in which stage = 0:1:step_max
    bitonic_kernel_v2_first_stages<<<num_blocks, THREADS_PER_BLOCK>>>(d_data, array.length, step_max);
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

        // Launch the v2 kernel for all *lower steps*, in which step = step_max:-1:0
        bitonic_kernel_v2_lower_steps<<<num_blocks, THREADS_PER_BLOCK>>>(d_data, array.length, stage, step_max);

        #ifdef DEBUG
        if (check_cuda_error(d_data)) return;
        #endif

        cudaDeviceSynchronize();
    }

    #ifdef TIME_MEASURE
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[v2 internal time] Execution Time: %f msec, Normalized Execution Time: %f msec per element\n", 
            milliseconds, milliseconds / array.length);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    #endif

    // Copy the sorted data back to the host
    cudaMemcpy(array.data, d_data, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);
}
