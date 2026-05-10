#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cuda_runtime.h>

#include "../common/check_cuda.h"

#ifndef REDUCTION_BLOCK_SIZE
#define REDUCTION_BLOCK_SIZE 256
#endif

struct ArgMaxPair
{
    float value;
    int index;
};

__device__ __host__ inline bool is_better_argmax_pair(
    const ArgMaxPair &a,
    const ArgMaxPair &b)
{
    // Return true if a is better than b.
    // Tie-break rule: smaller index wins.
    if (a.value != b.value)
    {
        return a.value > b.value;
    }
    return a.index < b.index;
}

__global__ void argmax_reduction_kernel(
    const ArgMaxPair *input,
    ArgMaxPair *partial_max,
    int n)
{
    extern __shared__ ArgMaxPair sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n)
    {
        sdata[tid] = input[gid];
    }
    else
    {
        sdata[tid].value = -INFINITY;
        sdata[tid].index = -1;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            ArgMaxPair left = sdata[tid];
            ArgMaxPair right = sdata[tid + stride];

            if (is_better_argmax_pair(right, left))
            {
                sdata[tid] = right;
            }
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        partial_max[blockIdx.x] = sdata[0];
    }
}

ArgMaxPair cpu_argmax(const std::vector<ArgMaxPair> &input)
{
    auto it = std::max_element(
        input.begin(),
        input.end(),
        [](const ArgMaxPair &a, const ArgMaxPair &b)
        {
            // Return true if a is worse than b.
            if (a.value != b.value)
            {
                return a.value < b.value;
            }
            return a.index > b.index;
        });

    return *it;
}

ArgMaxPair gpu_argmax(const std::vector<ArgMaxPair> &h_input)
{
    int n = static_cast<int>(h_input.size());
    size_t bytes = static_cast<size_t>(n) * sizeof(ArgMaxPair);

    ArgMaxPair *d_in = nullptr;
    ArgMaxPair *d_out = nullptr;

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

        argmax_reduction_kernel<<<
            grid_size,
            REDUCTION_BLOCK_SIZE,
            REDUCTION_BLOCK_SIZE * sizeof(ArgMaxPair)>>>(d_in, d_out, curr_n);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        std::swap(d_in, d_out);
        curr_n = grid_size;
    }

    ArgMaxPair result;
    CHECK_CUDA(cudaMemcpy(
        &result,
        d_in,
        sizeof(ArgMaxPair),
        cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return result;
}

int main()
{
    const int n = 1024;

    std::vector<ArgMaxPair> h_input(n);

    for (int i = 0; i < n; ++i)
    {
        h_input[i].value = static_cast<float>(i);
        h_input[i].index = i;
    }

    // Inject a duplicated maximum to test tie-breaking.
    // Smaller index should win.
    h_input[500].value = 2048.0f;
    h_input[700].value = 2048.0f;

    ArgMaxPair cuda_result = gpu_argmax(h_input);
    ArgMaxPair cpu_result = cpu_argmax(h_input);

    std::cout << "CUDA max: " << cuda_result.value
              << ", index: " << cuda_result.index << std::endl;

    std::cout << "CPU max:  " << cpu_result.value
              << ", index: " << cpu_result.index << std::endl;

    bool correct =
        std::abs(cuda_result.value - cpu_result.value) < 1e-6f &&
        cuda_result.index == cpu_result.index;

    std::cout << "Correctness: "
              << (correct ? "PASS" : "FAIL")
              << std::endl;

    return correct ? 0 : 1;
}