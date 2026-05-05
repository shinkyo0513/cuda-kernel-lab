#include <iostream>
#include <vector>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 16;

__global__ void matrix_transpose_sharedmem_kernel(
    const float *in,
    float *out,
    int nrows,
    int ncols)
{
    __shared__ float sdata[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int in_col = tx + blockDim.x * blockIdx.x;
    int in_row = ty + blockDim.y * blockIdx.y;

    if (in_row < nrows && in_col < ncols)
    {
        sdata[ty][tx] = in[in_row * ncols + in_col];
    }
    else
    {
        sdata[ty][tx] = 0.0f;
    }

    __syncthreads();

    int out_row = blockDim.x * blockIdx.x + ty;
    int out_col = blockDim.y * blockIdx.y + tx;

    if (out_row < ncols && out_col < nrows)
    {
        out[out_row * nrows + out_col] = sdata[tx][ty];
    }
}

int main()
{
    int nrows = 10000, ncols = 10000;
    int nelements = nrows * ncols;
    size_t bytes = nelements * sizeof(float);

    std::vector<float> h_in(nelements);
    std::vector<float> h_out(nelements, 0.0f);

    for (int i = 0; i < nelements; ++i)
    {
        h_in[i] = static_cast<float>(i);
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (ncols + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (nrows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    for (int i = 0; i < 10; ++i)
    {
        matrix_transpose_sharedmem_kernel<<<grid, block>>>(d_in, d_out, nrows, ncols);
    }

    cudaEventRecord(start);

    // Multiple run
    for (int i = 0; i < 100; ++i)
    {
        matrix_transpose_sharedmem_kernel<<<grid, block>>>(d_in, d_out, nrows, ncols);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "CUDA calculation duration: " << ms / 100.0f << " ms" << std::endl;

    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    /*
    std::cout << "Matrix A:" << std::endl;
    for (int r = 0; r < nrows; ++r) {
        for (int c = 0; c < ncols; ++c) {
            std::cout << h_in[r * ncols + c] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nMatrix B:" << std::endl;
    for (int r = 0; r < ncols; ++r) {
        for (int c = 0; c < nrows; ++c) {
            std::cout << h_out[r * nrows + c] << " ";
        }
        std::cout << std::endl;
    }
    */

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}