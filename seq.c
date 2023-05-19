%%writefile matrixmulbasic.cu

#include <stdio.h>

#include <stdio.h>

#define M 1000
#define N 800
#define BLOCK_SIZE 32

__global__ void matrixMultiply(float* A, float* B, float* C, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;
    if(row < m && col < n) {
        for(int i = 0; i < n; i++) {
            sum += A[row*n+i] * B[i*n+col];
        }
        C[row*n+col] = sum;
    }
}

int main() {
    float A[M*N], B[N*M], C[M*M];
    float *d_A, *d_B, *d_C;

    // Allocation of device memory
    cudaMalloc((void**)&d_A, sizeof(float)*M*N);
    cudaMalloc((void**)&d_B, sizeof(float)*N*M);
    cudaMalloc((void**)&d_C, sizeof(float)*M*M);

    // Copy matrices to device memory
    cudaMemcpy(d_A, A, sizeof(float)*M*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float)*N*M, cudaMemcpyHostToDevice);

    // Define CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record event on the default stream
    cudaEventRecord(start, 0);

    // Kernel invocation
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);
    matrixMultiply<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N);

    // Record event on the default stream
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Compute and print the elapsed time in millisec
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time: %f ms\n", elapsedTime);
    printf("AAAAAAAAAAAAAAAAAAA");

    // Copy the results back to the host
    cudaMemcpy(C, d_C, sizeof(float)*M*M, cudaMemcpyDeviceToHost);

    // Clean-up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
