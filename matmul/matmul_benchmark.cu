#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/check_cuda.h"
#include "../common/check_cublas.h"
#include "../common/timer.h"

#include "../kernels/matmul_cpu.h"
#include "../kernels/matmul_tiled.h"
#include "../kernels/matmul_cublas.h"

double gflops(int M, int N, int K, double ms)
{
    double ops = 2.0 * static_cast<double>(M) * N * K;
    return ops / (ms * 1e6);
}

double benchmark_cpu(
    const std::vector<float> &h_A,
    const std::vector<float> &h_B,
    std::vector<float> &h_C,
    int M,
    int N,
    int K,
    int repeats)
{
    CpuTimer timer;

    timer.start();
    for (int i = 0; i < repeats; ++i)
    {
        matmul_cpu(h_A, h_B, h_C, M, N, K);
    }
    timer.stop();

    return timer.elapsed_ms() / repeats;
}

float benchmark_tiled(
    const float *d_A,
    const float *d_B,
    float *d_C,
    int M,
    int N,
    int K,
    int repeats)
{
    for (int i = 0; i < 5; ++i)
    {
        launch_matmul_tiled(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();

    for (int i = 0; i < repeats; ++i)
    {
        launch_matmul_tiled(d_A, d_B, d_C, M, N, K);
    }

    timer.stop();
    CHECK_CUDA(cudaDeviceSynchronize());

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
    for (int i = 0; i < 5; ++i)
    {
        launch_matmul_cublas(handle, d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();

    for (int i = 0; i < repeats; ++i)
    {
        launch_matmul_cublas(handle, d_A, d_B, d_C, M, N, K);
    }

    timer.stop();
    CHECK_CUDA(cudaDeviceSynchronize());

    return timer.elapsed_ms() / repeats;
}

void run_case(
    cublasHandle_t handle,
    int M,
    int N,
    int K,
    bool run_cpu)
{
    std::cout << "\n=== Matmul case: M=" << M
              << ", N=" << N
              << ", K=" << K << " ===" << std::endl;

    size_t num_A = static_cast<size_t>(M) * K;
    size_t num_B = static_cast<size_t>(K) * N;
    size_t num_C = static_cast<size_t>(M) * N;

    size_t bytes_A = num_A * sizeof(float);
    size_t bytes_B = num_B * sizeof(float);
    size_t bytes_C = num_C * sizeof(float);

    std::vector<float> h_A(num_A);
    std::vector<float> h_B(num_B);
    std::vector<float> h_C_cpu(num_C, 0.0f);
    std::vector<float> h_C_tiled(num_C, 0.0f);
    std::vector<float> h_C_cublas(num_C, 0.0f);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float &x : h_A)
    {
        x = dist(rng);
    }

    for (float &x : h_B)
    {
        x = dist(rng);
    }

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc((void **)&d_C, bytes_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    int gpu_repeats = 20;
    int cpu_repeats = 1;

    double cpu_ms = -1.0;
    if (run_cpu)
    {
        cpu_ms = benchmark_cpu(h_A, h_B, h_C_cpu, M, N, K, cpu_repeats);

        std::cout << "CPU naive:     "
                  << std::fixed << std::setprecision(4)
                  << cpu_ms << " ms, "
                  << std::setprecision(2)
                  << gflops(M, N, K, cpu_ms)
                  << " GFLOP/s" << std::endl;
    }

    float tiled_ms = benchmark_tiled(d_A, d_B, d_C, M, N, K, gpu_repeats);
    CHECK_CUDA(cudaMemcpy(h_C_tiled.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    float cublas_ms = benchmark_cublas(handle, d_A, d_B, d_C, M, N, K, gpu_repeats);
    CHECK_CUDA(cudaMemcpy(h_C_cublas.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    std::cout << "CUDA tiled:    "
              << std::fixed << std::setprecision(4)
              << tiled_ms << " ms, "
              << std::setprecision(2)
              << gflops(M, N, K, tiled_ms)
              << " GFLOP/s" << std::endl;

    std::cout << "cuBLAS SGEMM:  "
              << std::fixed << std::setprecision(4)
              << cublas_ms << " ms, "
              << std::setprecision(2)
              << gflops(M, N, K, cublas_ms)
              << " GFLOP/s" << std::endl;

    if (run_cpu)
    {
        std::cout << "Max error tiled vs CPU:  "
                  << max_abs_error(h_C_cpu, h_C_tiled) << std::endl;

        std::cout << "Max error cuBLAS vs CPU: "
                  << max_abs_error(h_C_cpu, h_C_cublas) << std::endl;
    }
    else
    {
        std::cout << "Max error cuBLAS vs tiled: "
                  << max_abs_error(h_C_tiled, h_C_cublas) << std::endl;
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}

int main()
{
    CHECK_CUDA(cudaSetDevice(0));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    run_case(handle, 256, 256, 256, true);
    run_case(handle, 512, 512, 512, true);
    run_case(handle, 1024, 1024, 1024, false);
    run_case(handle, 2048, 2048, 2048, false);

    run_case(handle, 1024, 2048, 512, false);
    run_case(handle, 4096, 1024, 256, false);

    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}