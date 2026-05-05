#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 16;

__global__ void sum_reduction_kernel(const float *arr, float *partial_sums, int n)
{
    // Shared memory
    extern __shared__ float sdata[];

    int tidx = threadIdx.x;
    int gidx = threadIdx.x + blockDim.x * blockIdx.x;

    sdata[tidx] = (gidx < n) ? arr[gidx] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tidx < stride)
        {
            sdata[tidx] += sdata[tidx + stride];
        }
        __syncthreads();
    }

    if (tidx == 0)
    {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

int main()
{
    int n = 128;
    size_t bytes = n * sizeof(float);

    std::vector<float> h_in(n, 1.0f);

    float *d_in = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

    float *d_out = nullptr;
    cudaMalloc(&d_out, bytes);

    int curr_n = n;
    while (curr_n > 1)
    {
        int grid_size = ((curr_n + BLOCK_SIZE - 1) / BLOCK_SIZE);

        sum_reduction_kernel<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_in, d_out, curr_n);
        cudaDeviceSynchronize();

        std::swap(d_in, d_out);
        curr_n = grid_size;
    }

    float res_cuda;
    cudaMemcpy(&res_cuda, d_in, sizeof(float), cudaMemcpyDeviceToHost);

    float res_cpu = std::accumulate(h_in.begin(), h_in.end(), 0.0f);

    std::cout << "CUDA Sum: " << res_cuda << std::endl;
    std::cout << "CPU Sum:  " << res_cpu << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}