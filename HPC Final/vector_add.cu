%%writefile vector_add.cu
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <stdbool.h>

// CPU vector addition
void vectorAddCPU(int* a, int* b, int* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

// GPU vector addition kernel
__global__ void vectorAddGPU(int* a, int* b, int* c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

// Verify correctness
bool verifyVectorResults(int* c_cuda, int* c_cpu, int size) {
    for (int i = 0; i < size; i++) {
        if (c_cuda[i] != c_cpu[i]) {
            printf("Mismatch at index %d: GPU = %d, CPU = %d\n", i, c_cuda[i], c_cpu[i]);
            return false;
        }
    }
    return true;
}

int main() {
//size defines how many elements are in each vector a, b, and c
//Memory Allocation , Loops and Execution
    const int size = 1000000;
    const size_t bytes = size * sizeof(int);

    // Host memory allocation
    int *a = (int*)malloc(bytes);
    int *b = (int*)malloc(bytes);
    int *c_cpu = (int*)malloc(bytes);
    int *c_gpu = (int*)malloc(bytes);
    
    //Checks for allocation failure
    if (!a || !b || !c_cpu || !c_gpu) {
        fprintf(stderr, "Host malloc failed\n");
        return EXIT_FAILURE;
    }

    // Initialize input arrays
    for (int i = 0; i < size; i++) {
        a[i] = rand() % 1000;
        b[i] = rand() % 1000;
    }

    // Allocate gpu device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy to device
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    // GPU execution
    clock_t start_gpu = clock();
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    vectorAddGPU<<<gridSize, blockSize>>>(d_a, d_b, d_c, size);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    
    cudaDeviceSynchronize();
    
    //Waits for the GPU to finish.
    cudaMemcpy(c_gpu, d_c, bytes, cudaMemcpyDeviceToHost);
    clock_t end_gpu = clock();
    double gpu_time = double(end_gpu - start_gpu) / CLOCKS_PER_SEC;
    printf("Time Taken GPU : %f s\n", gpu_time);

    // CPU execution
    clock_t start_cpu = clock();
    vectorAddCPU(a, b, c_cpu, size);
    clock_t end_cpu = clock();
    double cpu_time = double(end_cpu - start_cpu) / CLOCKS_PER_SEC;
    printf("Time Taken CPU : %f s\n", cpu_time);

    // Verify result
    bool match = verifyVectorResults(c_gpu, c_cpu, size);
    printf("Output Match   : %s\n", match ? "True" : "False");
    printf("Speedup Factor : %f×\n", cpu_time / gpu_time);

    // Print first 5 results (optional)
    printf("Sample Output (first 5 elements):\n");
    for (int i = 0; i < 5; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c_gpu[i]);
    }

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(a); free(b); free(c_cpu); free(c_gpu);
    return 0;
}

//!nvcc -arch=sm_75 vector_add.cu -o vector_add
//!./vector_add
