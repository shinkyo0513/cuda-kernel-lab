#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "check_cuda.h"
#include "timer.h"

constexpr int TILE_SIZE = 16;

#define CHECK_CUBLAS(call)                                                 \
    do                                                                     \
    {                                                                      \
        cublasStatus_t status = (call);                                    \
        if (status != CUBLAS_STATUS_SUCCESS)                               \
        {                                                                  \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                      << " status=" << status << std::endl;                \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

__global__ void matmul_naive_kernel(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N dimension
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M dimension

    if (row < M && col < N)
    {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_tiled_kernel(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K)
{
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile)
    {
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;

        if (row < M && a_col < K)
        {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        }
        else
        {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < K && col < N)
        {
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        }
        else
        {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}

void matmul_cpu(
    const std::vector<float> &A,
    const std::vector<float> &B,
    std::vector<float> &C,
    int M,
    int N,
    int K)
{
    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

float max_abs_error(const std::vector<float> &ref, const std::vector<float> &out)
{
    float max_err = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i)
    {
        max_err = std::max(max_err, std::abs(ref[i] - out[i]));
    }
    return max_err;
}

double benchmark_cpu(
    const std::vector<float> &A,
    const std::vector<float> &B,
    std::vector<float> &C,
    int M,
    int N,
    int K,
    int repeats)
{
    CpuTimer timer;
    timer.start();
    for (int i = 0; i < repeats; ++i)
    {
        matmul_cpu(A, B, C, M, N, K);
    }
    timer.stop();
    return timer.elapsed_ms() / repeats;
}

float benchmark_cuda_naive(
    const float *d_A,
    const float *d_B,
    float *d_C,
    int M,
    int N,
    int K,
    int repeats)
{
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    for (int i = 0; i < 5; ++i)
    {
        matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    for (int i = 0; i < repeats; ++i)
    {
        matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    timer.stop();

    CHECK_CUDA(cudaGetLastError());
    return timer.elapsed_ms() / repeats;
}

float benchmark_cuda_tiled(
    const float *d_A,
    const float *d_B,
    float *d_C,
    int M,
    int N,
    int K,
    int repeats)
{
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    for (int i = 0; i < 5; ++i)
    {
        matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    for (int i = 0; i < repeats; ++i)
    {
        matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    timer.stop();

    CHECK_CUDA(cudaGetLastError());
    return timer.elapsed_ms() / repeats;
}

float benchmark_cublas(
    cublasHandle_t handle,
    const float *d_A,
    const float *d_B,
    float *d_C,
    int M,
    int N,
    int K,
    int repeats)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Row-major C = A(MxK) * B(KxN)
    // cuBLAS is column-major. This computes equivalent row-major result:
    // C_row = A_row * B_row
    // interpreted as C_col^T = B_col^T * A_col^T
    for (int i = 0; i < 5; ++i)
    {
        CHECK_CUBLAS(cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, N,
            d_A, K,
            &beta,
            d_C, N));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    for (int i = 0; i < repeats; ++i)
    {
        CHECK_CUBLAS(cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, N,
            d_A, K,
            &beta,
            d_C, N));
    }
    timer.stop();

    return timer.elapsed_ms() / repeats;
}

double gflops(int M, int N, int K, double ms)
{
    double ops = 2.0 * static_cast<double>(M) * N * K;
    return ops / (ms * 1e6);
}

void run_case(int M, int N, int K, bool run_cpu)
{
    std::cout << "\n=== Matmul case: M=" << M
              << ", N=" << N
              << ", K=" << K << " ===" << std::endl;

    size_t bytes_A = static_cast<size_t>(M) * K * sizeof(float);
    size_t bytes_B = static_cast<size_t>(K) * N * sizeof(float);
    size_t bytes_C = static_cast<size_t>(M) * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_cpu(M * N, 0.0f);
    std::vector<float> h_C_naive(M * N, 0.0f);
    std::vector<float> h_C_tiled(M * N, 0.0f);
    std::vector<float> h_C_cublas(M * N, 0.0f);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto &x : h_A)
        x = dist(rng);
    for (auto &x : h_B)
        x = dist(rng);

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc((void **)&d_C, bytes_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    int repeats = 20;
    int cpu_repeats = 1;

    double cpu_ms = -1.0;
    if (run_cpu)
    {
        cpu_ms = benchmark_cpu(h_A, h_B, h_C_cpu, M, N, K, cpu_repeats);
        std::cout << "CPU naive:        " << cpu_ms << " ms, "
                  << gflops(M, N, K, cpu_ms) << " GFLOP/s" << std::endl;
    }

    float naive_ms = benchmark_cuda_naive(d_A, d_B, d_C, M, N, K, repeats);
    CHECK_CUDA(cudaMemcpy(h_C_naive.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    float tiled_ms = benchmark_cuda_tiled(d_A, d_B, d_C, M, N, K, repeats);
    CHECK_CUDA(cudaMemcpy(h_C_tiled.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float cublas_ms = benchmark_cublas(handle, d_A, d_B, d_C, M, N, K, repeats);
    CHECK_CUDA(cudaMemcpy(h_C_cublas.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    CHECK_CUBLAS(cublasDestroy(handle));

    std::cout << "CUDA naive:       " << naive_ms << " ms, "
              << gflops(M, N, K, naive_ms) << " GFLOP/s" << std::endl;

    std::cout << "CUDA tiled:       " << tiled_ms << " ms, "
              << gflops(M, N, K, tiled_ms) << " GFLOP/s" << std::endl;

    std::cout << "cuBLAS SGEMM:     " << cublas_ms << " ms, "
              << gflops(M, N, K, cublas_ms) << " GFLOP/s" << std::endl;

    if (run_cpu)
    {
        std::cout << "Max error naive vs CPU:   "
                  << max_abs_error(h_C_cpu, h_C_naive) << std::endl;
        std::cout << "Max error tiled vs CPU:   "
                  << max_abs_error(h_C_cpu, h_C_tiled) << std::endl;
        std::cout << "Max error cuBLAS vs CPU:  "
                  << max_abs_error(h_C_cpu, h_C_cublas) << std::endl;
    }
    else
    {
        std::cout << "Max error tiled vs naive: "
                  << max_abs_error(h_C_naive, h_C_tiled) << std::endl;
        std::cout << "Max error cuBLAS vs naive:"
                  << max_abs_error(h_C_naive, h_C_cublas) << std::endl;
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}

int main()
{
    CHECK_CUDA(cudaSetDevice(0));

    run_case(256, 256, 256, true);
    run_case(512, 512, 512, true);
    run_case(1024, 1024, 1024, false);
    run_case(2048, 2048, 2048, false);

    run_case(1024, 2048, 512, false);
    run_case(4096, 1024, 256, false);

    return 0;
}