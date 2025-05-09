
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <stdbool.h>

void matrix_mult(int* a, int* b, int* c, int rowsA, int colsA, int colsB) {
    for (int row = 0; row < rowsA; row++) {
        for (int col = 0; col < colsB; col++) {
            int sum = 0;
            for (int i = 0; i < colsA; i++) {
                sum += a[row * colsA + i] * b[i * colsB + col];
            }
            c[row * colsB + col] = sum;
        }
    }
}

__global__ void matrixMul(int* a, int* b, int* c, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (row < rowsA && col < colsB) {
        for (int i = 0; i < colsA; i++) {
            sum += a[row * colsA + i] * b[i * colsB + col];
        }
        c[row * colsB + col] = sum;
    }
}


// Compare results
bool verifyMatrixResults(int* c_cuda, int* c_normal, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        if (c_cuda[i] != c_normal[i]) {
            return false;
        }
    }
    return true;
}
int main() {
    int rowsA = 1000;
    int colsA = 1000;
    int rowsB = 1000;
    int colsB = 1000;

    int *a, *b, *c_cuda, *c_normal;
    int *dev_a, *dev_b, *dev_c;

    a = (int*)malloc(rowsA * colsA * sizeof(int));
    b = (int*)malloc(rowsB * colsB * sizeof(int));
    c_cuda = (int*)malloc(rowsA * colsB * sizeof(int));
    c_normal = (int*)malloc(rowsA * colsB * sizeof(int));

    for (int i = 0; i < rowsA * colsA; i++) {
        a[i] = rand() % 100;
    }
    for (int i = 0; i < rowsB * colsB; i++) {
        b[i] = rand() % 100;
    }

    cudaMalloc((void**)&dev_a, rowsA * colsA * sizeof(int));
    cudaMalloc((void**)&dev_b, rowsB * colsB * sizeof(int));
    cudaMalloc((void**)&dev_c, rowsA * colsB * sizeof(int));

    cudaMemcpy(dev_a, a, rowsA * colsA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, rowsB * colsB * sizeof(int), cudaMemcpyHostToDevice);

    clock_t start_cuda = clock();

    dim3 blockSize(16, 16);
    dim3 gridSize((colsB + blockSize.x - 1) / blockSize.x, (rowsA + blockSize.y - 1) / blockSize.y);
    matrixMul<<<gridSize, blockSize>>>(dev_a, dev_b, dev_c, rowsA, colsA, colsB);
    cudaMemcpy(c_cuda, dev_c, rowsA * colsB * sizeof(int), cudaMemcpyDeviceToHost);

    clock_t end_cuda = clock();
    double cuda_time = (double)(end_cuda - start_cuda) / CLOCKS_PER_SEC;
    printf("Time Taken GPU : %f", cuda_time);

    clock_t start_normal = clock();
    matrix_mult(a, b, c_normal, rowsA, colsA, colsB);
    clock_t end_normal = clock();
    double normal_time = (double)(end_normal - start_normal) / CLOCKS_PER_SEC;
    printf("\nTime Taken CPU : %f ", normal_time);

    bool match = verifyMatrixResults(c_cuda, c_normal, rowsA, colsB);
    printf("\nOutput Match: %s", match ? "True" : "False");

    double speedup = normal_time / cuda_time;
    printf("\nSpeedup Factor: %f\n", speedup);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c_cuda);
    free(c_normal);

    return 0;
}