#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 128;

struct ArgMaxPair
{
    float value;
    int index;
};

__global__ void argmax_reduction_kernel(const ArgMaxPair *arr, ArgMaxPair *partial_max, int n)
{
    extern __shared__ ArgMaxPair sdata[];

    int tidx = threadIdx.x;
    int gidx = threadIdx.x + blockDim.x * blockIdx.x;

    // Shared memory copy
    sdata[tidx].value = gidx < n ? arr[gidx].value : -INFINITY;
    sdata[tidx].index = gidx < n ? arr[gidx].index : -1;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tidx < stride)
        {
            auto p1 = sdata[tidx];
            auto p2 = sdata[tidx + stride];
            if (p1.value < p2.value || (p1.value == p2.value && p1.index > p2.index))
            {
                sdata[tidx].value = sdata[tidx + stride].value;
                sdata[tidx].index = sdata[tidx + stride].index;
            }
        }
        __syncthreads();
    }

    if (tidx == 0)
    {
        partial_max[blockIdx.x] = sdata[0];
    }
}

int main()
{
    int n = 1024;
    int bytes = n * sizeof(ArgMaxPair);

    std::vector<ArgMaxPair> h_in(n);
    for (int i = 0; i < n; ++i)
    {
        h_in[i].value = i;
        if (i == 500)
        {
            h_in[i].value = std::pow(2, 10);
        }
        h_in[i].index = i;
    }

    ArgMaxPair *d_in = nullptr;
    ArgMaxPair *d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

    int curr_n = n;
    while (curr_n > 1)
    {
        int grid_size = (curr_n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        argmax_reduction_kernel<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(ArgMaxPair)>>>(
            d_in, d_out, curr_n);
        cudaDeviceSynchronize();

        std::swap(d_in, d_out);
        curr_n = grid_size;
    }

    ArgMaxPair res_cuda;
    cudaMemcpy(&res_cuda, d_in, sizeof(ArgMaxPair), cudaMemcpyDeviceToHost);

    std::cout << "CUDA max: " << res_cuda.value << " Index: " << res_cuda.index << std::endl;

    auto it_res = std::max_element(
        h_in.begin(),
        h_in.end(),
        [](const ArgMaxPair &p1, const ArgMaxPair &p2)
        {
            if (p1.value != p2.value)
            {
                return p1.value < p2.value;
            }
            return p1.index > p2.index;
        });
    int res_cpu = it_res->value;
    int arg_cpu = std::distance(h_in.begin(), it_res);

    std::cout << "CPU max: " << res_cpu << " Index: " << arg_cpu << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}