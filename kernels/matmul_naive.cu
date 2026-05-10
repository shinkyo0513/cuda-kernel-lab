#include "matmul_naive.h"

#include <cuda_runtime.h>

#include "../common/check_cuda.h"

#ifndef MATMUL_BLOCK_SIZE
#define MATMUL_BLOCK_SIZE 16
#endif

__global__ void matmul_naive_kernel(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N dimension
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M dimension

    if (row < M && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < K; ++k)
        {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

void launch_matmul_naive(
    const float *d_A,
    const float *d_B,
    float *d_C,
    int M,
    int N,
    int K)
{
    dim3 block(MATMUL_BLOCK_SIZE, MATMUL_BLOCK_SIZE);
    dim3 grid(
        (N + MATMUL_BLOCK_SIZE - 1) / MATMUL_BLOCK_SIZE,
        (M + MATMUL_BLOCK_SIZE - 1) / MATMUL_BLOCK_SIZE);

    matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}