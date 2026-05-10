#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <cuda_runtime.h>

#include "../common/check_cuda.h"

#ifndef REDUCTION_BLOCK_SIZE
#define REDUCTION_BLOCK_SIZE 256
#endif

__global__ void sum_reduction_kernel(
    const float *input,
    float *partial_sums,
    int n)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (gid < n) ? input[gid] : 0.0f;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

float cpu_sum(const std::vector<float> &input)
{
    return std::accumulate(input.begin(), input.end(), 0.0f);
}

float gpu_sum(const std::vector<float> &h_input)
{
    int n = static_cast<int>(h_input.size());
    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float *d_in = nullptr;
    float *d_out = nullptr;

    CHECK_CUDA(cudaMalloc((void **)&d_in, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_out, bytes));

    CHECK_CUDA(cudaMemcpy(
        d_in,
        h_input.data(),
        bytes,
        cudaMemcpyHostToDevice));

    int curr_n = n;

    while (curr_n > 1)
    {
        int grid_size = (curr_n + REDUCTION_BLOCK_SIZE - 1) / REDUCTION_BLOCK_SIZE;

        sum_reduction_kernel<<<
            grid_size,
            REDUCTION_BLOCK_SIZE,
            REDUCTION_BLOCK_SIZE * sizeof(float)>>>(d_in, d_out, curr_n);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        std::swap(d_in, d_out);
        curr_n = grid_size;
    }

    float result = 0.0f;

    CHECK_CUDA(cudaMemcpy(
        &result,
        d_in,
        sizeof(float),
        cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return result;
}

int main()
{
    const int n = 1000;

    std::vector<float> h_input(n);

    for (int i = 0; i < n; ++i)
    {
        h_input[i] = 1.0f;
    }

    float cuda_result = gpu_sum(h_input);
    float cpu_result = cpu_sum(h_input);

    std::cout << "CUDA sum: " << cuda_result << std::endl;
    std::cout << "CPU sum:  " << cpu_result << std::endl;

    bool correct = std::abs(cuda_result - cpu_result) < 1e-5f;

    std::cout << "Correctness: "
              << (correct ? "PASS" : "FAIL")
              << std::endl;

    return correct ? 0 : 1;
}