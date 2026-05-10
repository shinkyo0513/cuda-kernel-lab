#include "softmax.h"

#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

#include "../common/check_cuda.h"

#ifndef SOFTMAX_BLOCK_SIZE
#define SOFTMAX_BLOCK_SIZE 32
#endif

__global__ void max_row_wise_kernel(
    const float *in,
    float *out,
    int active_cols,
    int stride,
    int nrows)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    int sidx = threadIdx.y * blockDim.x + threadIdx.x;

    extern __shared__ float sdata[];

    sdata[sidx] = (col < active_cols && row < nrows)
                      ? in[row * stride + col]
                      : -INFINITY;

    __syncthreads();

    for (int stride_reduce = blockDim.x / 2; stride_reduce > 0; stride_reduce >>= 1)
    {
        if (threadIdx.x < stride_reduce)
        {
            float a = sdata[sidx];
            float b = sdata[sidx + stride_reduce];
            sdata[sidx] = (a < b) ? b : a;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && row < nrows)
    {
        out[row * stride + blockIdx.x] = sdata[threadIdx.y * blockDim.x];
    }
}

__global__ void subtract_max_and_exp_kernel(
    const float *in,
    const float *row_max,
    float *out,
    int ncols,
    int nrows)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < nrows && col < ncols)
    {
        int gidx = row * ncols + col;
        float max_val = row_max[row * ncols];
        out[gidx] = expf(in[gidx] - max_val);
    }
}

__global__ void sum_row_wise_kernel(
    const float *in,
    float *out,
    int active_cols,
    int stride,
    int nrows)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    int sidx = threadIdx.y * blockDim.x + threadIdx.x;

    extern __shared__ float sdata[];

    sdata[sidx] = (col < active_cols && row < nrows)
                      ? in[row * stride + col]
                      : 0.0f;

    __syncthreads();

    for (int stride_reduce = blockDim.x / 2; stride_reduce > 0; stride_reduce >>= 1)
    {
        if (threadIdx.x < stride_reduce)
        {
            sdata[sidx] += sdata[sidx + stride_reduce];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && row < nrows)
    {
        out[row * stride + blockIdx.x] = sdata[threadIdx.y * blockDim.x];
    }
}

__global__ void normalize_kernel(
    const float *exp_vals,
    const float *row_sum,
    float *out,
    int ncols,
    int nrows)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (row < nrows && col < ncols)
    {
        int gidx = row * ncols + col;
        float sum_val = row_sum[row * ncols];
        out[gidx] = exp_vals[gidx] / sum_val;
    }
}

void launch_softmax(
    const float *d_in,
    float *d_out,
    int nrows,
    int ncols)
{
    const int block_size = SOFTMAX_BLOCK_SIZE;

    dim3 block(block_size, block_size);
    size_t shared_bytes = block_size * block_size * sizeof(float);

    size_t nelements = static_cast<size_t>(nrows) * ncols;
    size_t bytes = nelements * sizeof(float);

    float *d_tmp_a = nullptr;
    float *d_tmp_b = nullptr;

    CHECK_CUDA(cudaMalloc((void **)&d_tmp_a, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_tmp_b, bytes));

    // ------------------------------------------------------------------
    // 1. Row-wise max reduction
    // d_tmp_a initially stores a copy of input.
    // After the reduction loop, d_tmp_a[row * ncols] stores row max.
    // ------------------------------------------------------------------
    CHECK_CUDA(cudaMemcpy(d_tmp_a, d_in, bytes, cudaMemcpyDeviceToDevice));

    int curr_ncols = ncols;
    float *reduce_in = d_tmp_a;
    float *reduce_out = d_tmp_b;

    while (curr_ncols > 1)
    {
        dim3 grid(
            (curr_ncols + block_size - 1) / block_size,
            (nrows + block_size - 1) / block_size);

        max_row_wise_kernel<<<grid, block, shared_bytes>>>(
            reduce_in,
            reduce_out,
            curr_ncols,
            ncols,
            nrows);
        CHECK_CUDA(cudaGetLastError());

        curr_ncols = (curr_ncols + block_size - 1) / block_size;
        std::swap(reduce_in, reduce_out);
    }

    float *d_row_max = reduce_in;

    // ------------------------------------------------------------------
    // 2. exp(x - max)
    // d_out stores exp values.
    // ------------------------------------------------------------------
    {
        dim3 grid(
            (ncols + block_size - 1) / block_size,
            (nrows + block_size - 1) / block_size);

        subtract_max_and_exp_kernel<<<grid, block>>>(
            d_in,
            d_row_max,
            d_out,
            ncols,
            nrows);
        CHECK_CUDA(cudaGetLastError());
    }

    // ------------------------------------------------------------------
    // 3. Row-wise sum reduction over exp values
    // After the reduction loop, reduce_in[row * ncols] stores row exp sum.
    // ------------------------------------------------------------------
    CHECK_CUDA(cudaMemcpy(d_tmp_a, d_out, bytes, cudaMemcpyDeviceToDevice));

    curr_ncols = ncols;
    reduce_in = d_tmp_a;
    reduce_out = d_tmp_b;

    while (curr_ncols > 1)
    {
        dim3 grid(
            (curr_ncols + block_size - 1) / block_size,
            (nrows + block_size - 1) / block_size);

        sum_row_wise_kernel<<<grid, block, shared_bytes>>>(
            reduce_in,
            reduce_out,
            curr_ncols,
            ncols,
            nrows);
        CHECK_CUDA(cudaGetLastError());

        curr_ncols = (curr_ncols + block_size - 1) / block_size;
        std::swap(reduce_in, reduce_out);
    }

    float *d_row_sum = reduce_in;

    // ------------------------------------------------------------------
    // 4. Normalize
    // d_out currently stores exp values, and is overwritten by softmax.
    // ------------------------------------------------------------------
    {
        dim3 grid(
            (ncols + block_size - 1) / block_size,
            (nrows + block_size - 1) / block_size);

        normalize_kernel<<<grid, block>>>(
            d_out,
            d_row_sum,
            d_out,
            ncols,
            nrows);
        CHECK_CUDA(cudaGetLastError());
    }

    CHECK_CUDA(cudaFree(d_tmp_a));
    CHECK_CUDA(cudaFree(d_tmp_b));
}