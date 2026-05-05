#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void matmul_cpu(
    const std::vector<float> &A,
    const std::vector<float> &B,
    std::vector<float> &C,
    int M,
    int N,
    int K
) {
    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for(int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
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
    std::vector<float> h_D(M * N);

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

    matmul_cpu(h_A, h_B, h_D, M, N, K);

    float *d_A;
    float *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,
        d_A, K,
        &beta,
        d_C, N
    );

    cublasDestroy(handle);

    cudaMemcpy(h_C.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    float diff = 0.0f;
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        diff += std::fabs(h_D[i * N + j] - h_C[i * N + j]);
      }
    }
    std::cout << "Abs diff sum: " << diff << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}