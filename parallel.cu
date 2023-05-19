#===CUDA===
%%cu

#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define ROWS_A 70
#define COLS_A 80
#define ROWS_B COLS_A
#define COLS_B 70

#define SHMEM_SIZE (1 << 10)

//CUDA Kernel function to multiply the matrices
__global__ void matrixMul(const int *a, const int *b, int *c) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int s_a[SHMEM_SIZE];
  __shared__ int s_b[SHMEM_SIZE];

  int tmp = 0;

  for (int i = 0; i < COLS_A; i += blockDim.x) {
    s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * COLS_A + i + threadIdx.x];
    s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[i * COLS_B + threadIdx.y * COLS_B + col];

    __syncthreads();

    for (int j = 0; j < blockDim.x; j++) {
      tmp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    }

    __syncthreads();
  }

  c[row * COLS_B + col] = tmp;
}

int verify_result(int *a, int *b, int *c) {
  for (int i = 0; i < ROWS_A; i++) {
    for (int j = 0; j < COLS_B; j++) {
      int tmp = 0;
      for (int k = 0; k < COLS_A; k++) {
        tmp += a[i * COLS_A + k] * b[k * COLS_B + j];
      }

      // If the operation did not succeed, return false
      if (tmp != c[i * COLS_B + j]) {
        return 0;
      }
    }
  }

  // If the operation succeeded, return true
  return 1;
}

void print_matrix(int *mat, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      printf("%d ", mat[i * cols + j]);
    }
    printf("\n");
  }
}

int main() {
  size_t bytes_a = ROWS_A * COLS_A * sizeof(int);
  size_t bytes_b = ROWS_B * COLS_B * sizeof(int);
  size_t bytes_c = ROWS_A * COLS_B * sizeof(int);

  int *h_a = (int*)malloc(bytes_a);
  int *h_b = (int*)malloc(bytes_b);
  int *h_c = (int*)malloc(bytes_c);

  for (int i = 0; i < ROWS_A * COLS_A; i++) {
    h_a[i] = rand() % 100;
  }

  for (int i = 0; i < ROWS_B * COLS_B; i++) {
    h_b[i] = rand() % 100;
  }

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes_a);
  cudaMalloc(&d_b, bytes_b);
  cudaMalloc(&d_c, bytes_c);

  cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice);

  int THREADS = 10;
  int BLOCKS_A = ROWS_A / THREADS;
  int BLOCKS_B = COLS_B / THREADS;

  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS_B, BLOCKS_A);

  int totalThreads = THREADS * THREADS * BLOCKS_B * BLOCKS_A;
  printf("Total threads: %d\n", totalThreads);
  clock_t start, end;
  double cpu_time_used;
  
  start = clock();

  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);

  cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost);

  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

  printf("Matrix A:\n");
  print_matrix(h_a, ROWS_A, COLS_A);
  printf("\nMatrix B:\n");
  print_matrix(h_b, ROWS_B, COLS_B);
  printf("\nMatrix C (result):\n");
  print_matrix(h_c, ROWS_A, COLS_B);

  int success = verify_result(h_a, h_b, h_c);

  if (success) {
    printf("COMPLETED SUCCESSFULLY\n");
  } else {
    printf("ERROR WITH COMPUTATION\n");
  }

  printf("Time taken: %f seconds\n", cpu_time_used);

  free(h_a);
  free(h_b);
  free(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}
