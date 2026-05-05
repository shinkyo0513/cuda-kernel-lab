#include <iostream>
#include <vector>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 32;

__global__ void matmul_tiled_kernel(
    const float *A,
    const float *B,
    float *C,
    int M,
    int K,
    int N)
{
    __shared__ float sdata_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sdata_b[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int g_row = ty + blockIdx.y * blockDim.y;
    int g_col = tx + blockIdx.x * blockDim.x;

    float sum = 0.0f;

    int num_blocks = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int m = 0; m < num_blocks; ++m)
    {

        int a_col = m * BLOCK_SIZE + tx;
        if (g_row < M && a_col < K)
        {
            sdata_a[ty][tx] = A[g_row * K + a_col];
        }
        else
        {
            sdata_a[ty][tx] = 0.0f;
        }

        int b_row = m * BLOCK_SIZE + ty;
        if (b_row < K && g_col < N)
        {
            sdata_b[ty][tx] = B[b_row * N + g_col];
        }
        else
        {
            sdata_b[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            sum += sdata_a[ty][k] * sdata_b[k][tx];
        }

        __syncthreads();
    }

    if (g_row < M && g_col < N)
    {
        C[g_row * N + g_col] = sum;
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

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    for (int i = 0; i < 10; ++i)
    {
        matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    }

    cudaEventRecord(start);

    // Multiple run
    for (int i = 0; i < 100; ++i)
    {
        matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
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