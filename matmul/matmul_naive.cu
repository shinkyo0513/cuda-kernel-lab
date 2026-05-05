% % writefile naive_matmul.cu
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>

    constexpr int BLOCK_SIZE = 32;

__global__ void matmul_naive_kernel(
    const float *A,
    const float *B,
    float *C,
    int M,
    int K,
    int N)
{
    // C[i, j] = sum_{k=0,1,...,N-1}(A[i, k] * B[k, j])
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if (row < M && col < N)
    {
        for (int k = 0; k < K; k++)
        {
            C[row * N + col] += A[row * K + k] * B[k * N + col];
        }
    }
}

int main()
{
    int M = 800;
    int K = 600;
    int N = 1000;

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            h_A[i * K + j] = 1.0f;
        }
    }

    for (int i = 0; i < K; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            h_B[i * N + j] = 2.0f;
        }
    }

    float *d_A;
    float *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = BLOCK_SIZE;
    dim3 block(block_size, block_size);
    dim3 grid(
        (N + block_size - 1) / block_size,
        (M + block_size - 1) / block_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    for (int i = 0; i < 10; ++i)
    {
        matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    }

    cudaEventRecord(start);

    // Multiple run
    for (int i = 0; i < 100; ++i)
    {
        matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "CUDA calculation duration: " << ms / 100.0f << " ms" << std::endl;

    /*
    cudaMemcpy(h_C.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        std::cout << h_C[i * N + j] << " ";
      }
      std::cout << std::endl;
    }
    */

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}