#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 16;

__global__ void max_row_wise_kernel(const float *in, float *out, int active_cols, int stride, int nrows)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    int sidx = threadIdx.y * blockDim.x + threadIdx.x;

    extern __shared__ float sdata[];
    sdata[sidx] = (col < active_cols && row < nrows) ? in[row * stride + col] : -INFINITY;
    __syncthreads();

    for (int stride_reduce = blockDim.x / 2; stride_reduce > 0; stride_reduce >>= 1)
    {
        if (threadIdx.x < stride_reduce)
        {
            sdata[sidx] = (sdata[sidx] < sdata[sidx + stride_reduce]) ? sdata[sidx + stride_reduce] : sdata[sidx];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && row < nrows)
    {
        out[row * stride + blockIdx.x] = sdata[threadIdx.y * blockDim.x];
    }
}

__global__ void subtract_max_and_exp_kernel(const float *in, const float *max_val, float *out, int ncols, int nrows)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int gidx = row * ncols + col;
    int row_base = row * ncols;

    if (row < nrows && col < ncols)
    {
        out[gidx] = expf(in[gidx] - max_val[row_base]);
    }
}

__global__ void sum_row_wise_kernel(const float *in, float *out, int active_cols, int stride, int nrows)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int sidx = threadIdx.y * blockDim.x + threadIdx.x;

    extern __shared__ float sdata[];
    sdata[sidx] = (col < active_cols && row < nrows) ? in[row * stride + col] : 0.0f;
    __syncthreads();

    for (int stride_reduce = blockDim.x / 2; stride_reduce > 0; stride_reduce >>= 1)
    {
        if (threadIdx.x < stride_reduce)
        {
            sdata[sidx] += sdata[sidx + stride_reduce];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        out[row * stride + blockIdx.x] = sdata[threadIdx.y * blockDim.x];
    }
}

__global__ void normalize_kernel(const float *e, const float *esum, float *out, int ncols, int nrows)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int gidx = row * ncols + col;
    int row_base = row * ncols;

    if (row < nrows && col < ncols)
    {
        out[gidx] = e[gidx] / esum[row_base];
    }
}

int main()
{
    int nrows = 20, ncols = 20;
    int block_size = BLOCK_SIZE;

    int nelements = nrows * ncols;
    size_t bytes = nelements * sizeof(float);

    std::vector<float> h_in(nelements);
    for (int row = 0; row < nrows; ++row)
    {
        for (int col = 0; col < ncols; ++col)
        {
            h_in[row * ncols + col] = static_cast<float>(col);
        }
    }
    std::vector<float> h_out(nelements, 0.0f);

    float *d_in = nullptr;
    float *d_out = nullptr;
    float *d_tem = nullptr;

    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMalloc(&d_tem, bytes);

    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tem, h_in.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(block_size, block_size);
    size_t shared_bytes = block_size * block_size * sizeof(float);

    // -------------------------
    // 1. Row-wise max reduction
    // -------------------------
    int curr_ncols = ncols;

    while (curr_ncols > 1)
    {
        dim3 grid(
            (curr_ncols + block_size - 1) / block_size,
            (nrows + block_size - 1) / block_size);

        max_row_wise_kernel<<<grid, block, shared_bytes>>>(
            d_tem, d_out, curr_ncols, ncols, nrows);
        cudaDeviceSynchronize();

        curr_ncols = (curr_ncols + block_size - 1) / block_size;
        std::swap(d_tem, d_out);
    }
    // Now d_tem[row * ncols] stores row-wise max.

    // -------------------------
    // 2. exp(x - max)
    // -------------------------
    {
        dim3 grid(
            (ncols + block_size - 1) / block_size,
            (nrows + block_size - 1) / block_size);
        subtract_max_and_exp_kernel<<<grid, block>>>(
            d_in, d_tem, d_out, ncols, nrows);
        cudaDeviceSynchronize();
    }
    // d_out stores exp(x - max).
    // Keep one copy of exp values in d_out for final normalization.
    cudaMemcpy(d_in, d_out, bytes, cudaMemcpyDeviceToDevice);

    // -------------------------
    // 3. Row-wise sum reduction
    // -------------------------
    curr_ncols = ncols;

    float *reduce_in = d_in;
    float *reduce_out = d_tem;
    while (curr_ncols > 1)
    {
        dim3 grid(
            (curr_ncols + block_size - 1) / block_size,
            (nrows + block_size - 1) / block_size);

        sum_row_wise_kernel<<<grid, block, shared_bytes>>>(
            reduce_in, reduce_out, curr_ncols, ncols, nrows);
        cudaDeviceSynchronize();

        curr_ncols = (curr_ncols + block_size - 1) / block_size;
        std::swap(reduce_in, reduce_out);
    }
    // Now reduce_in[row * ncols] stores row-wise exp sum.

    // -------------------------
    // 4. Normalize
    // -------------------------
    {
        dim3 grid(
            (ncols + block_size - 1) / block_size,
            (nrows + block_size - 1) / block_size);
        normalize_kernel<<<grid, block>>>(
            d_out, reduce_in, d_out, ncols, nrows);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Matrix A:" << std::endl;
    for (int r = 0; r < nrows; ++r)
    {
        for (int c = 0; c < ncols; ++c)
        {
            std::cout << h_in[r * ncols + c] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nMatrix B:" << std::endl;
    for (int r = 0; r < ncols; ++r)
    {
        for (int c = 0; c < nrows; ++c)
        {
            std::cout << h_out[r * nrows + c] << " ";
        }
        std::cout << std::endl;
    }

    float row_sum = std::accumulate(h_out.begin(), h_out.begin() + ncols, 0.0f);
    std::cout << "Row sum: " << row_sum << std::endl;

    cudaFree(d_in);
    cudaFree(d_tem);
    cudaFree(d_out);

    return 0;
}