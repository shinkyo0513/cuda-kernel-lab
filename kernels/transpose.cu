#include "transpose.h"

#include <cuda_runtime.h>

#include "../common/check_cuda.h"

#ifndef TRANSPOSE_TILE_SIZE
#define TRANSPOSE_TILE_SIZE 32
#endif

static_assert(
    TRANSPOSE_TILE_SIZE * TRANSPOSE_TILE_SIZE <= 1024,
    "TRANSPOSE_TILE_SIZE is too large");

__global__ void transpose_naive_kernel(
    const float *in,
    float *out,
    int rows,
    int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols)
    {
        out[col * rows + row] = in[row * cols + col];
    }
}

__global__ void transpose_shared_kernel(
    const float *in,
    float *out,
    int rows,
    int cols)
{
    __shared__ float tile[TRANSPOSE_TILE_SIZE][TRANSPOSE_TILE_SIZE + 1];

    int x_in = blockIdx.x * TRANSPOSE_TILE_SIZE + threadIdx.x;
    int y_in = blockIdx.y * TRANSPOSE_TILE_SIZE + threadIdx.y;

    if (x_in < cols && y_in < rows)
    {
        tile[threadIdx.y][threadIdx.x] = in[y_in * cols + x_in];
    }

    __syncthreads();

    int x_out = blockIdx.y * TRANSPOSE_TILE_SIZE + threadIdx.x;
    int y_out = blockIdx.x * TRANSPOSE_TILE_SIZE + threadIdx.y;

    if (x_out < rows && y_out < cols)
    {
        out[y_out * rows + x_out] = tile[threadIdx.x][threadIdx.y];
    }
}

void launch_transpose_naive(
    const float *d_in,
    float *d_out,
    int rows,
    int cols)
{
    dim3 block(TRANSPOSE_TILE_SIZE, TRANSPOSE_TILE_SIZE);
    dim3 grid(
        (cols + TRANSPOSE_TILE_SIZE - 1) / TRANSPOSE_TILE_SIZE,
        (rows + TRANSPOSE_TILE_SIZE - 1) / TRANSPOSE_TILE_SIZE);

    transpose_naive_kernel<<<grid, block>>>(d_in, d_out, rows, cols);
    CHECK_CUDA(cudaGetLastError());
}

void launch_transpose_shared(
    const float *d_in,
    float *d_out,
    int rows,
    int cols)
{
    dim3 block(TRANSPOSE_TILE_SIZE, TRANSPOSE_TILE_SIZE);
    dim3 grid(
        (cols + TRANSPOSE_TILE_SIZE - 1) / TRANSPOSE_TILE_SIZE,
        (rows + TRANSPOSE_TILE_SIZE - 1) / TRANSPOSE_TILE_SIZE);

    transpose_shared_kernel<<<grid, block>>>(d_in, d_out, rows, cols);
    CHECK_CUDA(cudaGetLastError());
}