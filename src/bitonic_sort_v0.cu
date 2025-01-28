#include "../inc/bitonic_sort.cuh"

__global__ void bitonic_kernel_v0(int* data, int length, int stage, int step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure threads operate within valid range
    if (idx >= length) return;

    // Find the partner index
    int partner = idx ^ (1 << step);

    // Ensure valid partner index
    if (idx >= partner || partner >= length) return;

    // // Determine if the exchange should be in ascending order
    // bool ascending = (idx & (1 << (stage + 1))) == 0;

    BITONIC_COMPARE_AND_SWAP(idx, idx, partner, stage, data)
}

void bitonic_sort_v0(IntArray& array) {
    int* d_data;
    size_t size = array.length * sizeof(int);

    // Allocate memory on the device
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, array.data, size, cudaMemcpyHostToDevice);

    int num_blocks = (array.length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int stages     = __builtin_ctz(array.length); // __builtin_ctz gets log2(length)

    // Launch the Bitonic Sort
    for (int stage = 0; stage < stages; stage++) {
        for (int step = stage; step >= 0; step--) {
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
    }

    // Copy the sorted data back to the host
    cudaMemcpy(array.data, d_data, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);
}
