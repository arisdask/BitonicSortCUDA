#include "../inc/bitonic_sort.cuh"

__global__ void bitonic_kernel_v0(int* data, int length, int stage, int step) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure threads operate within valid range
    if (tid >= (length >> 1)) return;

    // Find the index that this thread will handle
    int idx = GET_ARR_ID(tid, step);

    // Find the partner index that this thread will handle
    int partner = idx ^ (1 << step);

    // Ensure valid partner index
    if (idx >= partner || partner >= length) return;

    exchange(data, idx, idx, partner, stage);
}

void bitonic_sort_v0(IntArray& array) {
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

    int num_blocks = ((array.length >> 1) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int stages     = __builtin_ctz(array.length); // __builtin_ctz gets log2(length)

    // Launch the Bitonic Sort
    for (int stage = 0; stage < stages; stage++) {
        for (int step = stage; step >= 0; step--) {
            bitonic_kernel_v0<<<num_blocks, THREADS_PER_BLOCK>>>(d_data, array.length, stage, step);

            #ifdef DEBUG
            if (check_cuda_error(d_data)) return;
            #endif
            
            cudaDeviceSynchronize();
        }
    }

    #ifdef TIME_MEASURE
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[v0 internal time] Execution Time: %f msec, Normalized Execution Time: %f msec per element\n", 
            milliseconds, milliseconds / array.length);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    #endif

    // Copy the sorted data back to the host
    cudaMemcpy(array.data, d_data, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);
}
