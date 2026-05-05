#include <iostream>
#include <vector>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 16;

__global__ void matrix_transpose_naive_kernel(const float *A, float *B, int nrows, int ncols)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < ncols && row < nrows)
    {
        int idx_a = row * ncols + col;
        int idx_b = col * nrows + row;
        B[idx_b] = A[idx_a];
    }
}

int main()
{
    int nrows = 10000, ncols = 10000;
    int elements = nrows * ncols;
    size_t bytes = elements * sizeof(float);

    std::vector<float> h_A(elements), h_B(elements);
    for (int i = 0; i < elements; ++i)
    {
        h_A[i] = i;
    }

    float *d_A, *d_B;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size(
        (ncols + block_size.x - 1) / block_size.y,
        (nrows + block_size.y - 1) / block_size.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < 10; ++i)
    {
        matrix_transpose_naive_kernel<<<grid_size, block_size>>>(d_A, d_B, nrows, ncols);
    }

    cudaEventRecord(start);

    for (int i = 0; i < 100; ++i)
    {
        matrix_transpose_naive_kernel<<<grid_size, block_size>>>(d_A, d_B, nrows, ncols);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "CUDA calculation duration: " << ms / 100.0f << "ms" << std::endl;

    /*
    cudaMemcpy(h_B.data(), d_B, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Matrix A:" << std::endl;
    for (int r = 0; r < nrows; ++r) {
        for (int c = 0; c < ncols; ++c) {
            std::cout << h_A[r * ncols + c] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nMatrix B:" << std::endl;
    for (int r = 0; r < ncols; ++r) {
        for (int c = 0; c < nrows; ++c) {
            std::cout << h_B[r * nrows + c] << " ";
        }
        std::cout << std::endl;
    }
    */

    return 0;
}