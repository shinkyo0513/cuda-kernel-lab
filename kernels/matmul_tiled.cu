#include "matmul_tiled.h"

#include <cuda_runtime.h>

#include "../common/check_cuda.h"

#ifndef MATMUL_TILE_SIZE
#define MATMUL_TILE_SIZE 32
#endif

static_assert(MATMUL_TILE_SIZE * MATMUL_TILE_SIZE <= 1024, "MATMUL_TILE_SIZE is too large");

__global__ void matmul_tiled_kernel(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K)
{
    __shared__ float tile_A[MATMUL_TILE_SIZE][MATMUL_TILE_SIZE];
    __shared__ float tile_B[MATMUL_TILE_SIZE][MATMUL_TILE_SIZE];

    int col = blockIdx.x * MATMUL_TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * MATMUL_TILE_SIZE + threadIdx.y;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE; ++tile)
    {
        int a_col = tile * MATMUL_TILE_SIZE + threadIdx.x;
        int b_row = tile * MATMUL_TILE_SIZE + threadIdx.y;

        tile_A[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        tile_B[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < MATMUL_TILE_SIZE; ++k)
        {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}

void launch_matmul_tiled(
    const float *d_A,
    const float *d_B,
    float *d_C,
    int M,
    int N,
    int K)
{
    dim3 block(MATMUL_TILE_SIZE, MATMUL_TILE_SIZE);
    dim3 grid(
        (N + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE,
        (M + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE);

    matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
}