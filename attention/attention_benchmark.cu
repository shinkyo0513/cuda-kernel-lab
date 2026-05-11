#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>

#include "../common/check_cuda.h"
#include "../common/timer.h"

#include "attention_cpu.h"
#include "attention_cuda.h"

double attention_gflops(int seq_len, int dim, double ms)
{
    double sd = static_cast<double>(seq_len) * dim;
    double score_ops = 2.0 * static_cast<double>(seq_len) * seq_len * dim;
    double output_ops = 2.0 * static_cast<double>(seq_len) * dim * seq_len;
    double scale_ops = static_cast<double>(seq_len) * seq_len;
    double softmax_ops = 5.0 * static_cast<double>(seq_len) * seq_len;
    double ops = score_ops + output_ops + scale_ops + softmax_ops + sd;
    return ops / (ms * 1e6);
}

float max_abs_error(
    const std::vector<float> &ref,
    const std::vector<float> &out)
{
    float max_err = 0.0f;

    for (size_t i = 0; i < ref.size(); ++i)
    {
        max_err = std::max(max_err, std::abs(ref[i] - out[i]));
    }

    return max_err;
}

double benchmark_cpu(
    const std::vector<float> &h_Q,
    const std::vector<float> &h_K,
    const std::vector<float> &h_V,
    std::vector<float> &h_scores,
    std::vector<float> &h_probs,
    std::vector<float> &h_O,
    int seq_len,
    int dim,
    int repeats
) {
    CpuTimer timer;

    timer.start();
    for (int i = 0; i < repeats; ++i)
    {
        attention_cpu_with_workspace(
            h_Q,
            h_K,
            h_V,
            h_scores,
            h_probs,
            h_O,
            seq_len,
            dim);
    }
    timer.stop();

    return timer.elapsed_ms() / repeats;
}

float benchmark_cuda(
    const float *d_Q,
    const float *d_K,
    const float *d_V,
    float *d_O,
    float *d_KT,
    float *d_scores,
    float *d_probs,
    int seq_len,
    int dim,
    int repeats)
{
    for (int i = 0; i < 5; ++i)
    {
        launch_attention_cuda_with_workspace(
            d_Q,
            d_K,
            d_V,
            d_O,
            d_KT,
            d_scores,
            d_probs,
            seq_len,
            dim);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();

    for (int i = 0; i < repeats; ++i)
    {
        launch_attention_cuda_with_workspace(
            d_Q,
            d_K,
            d_V,
            d_O,
            d_KT,
            d_scores,
            d_probs,
            seq_len,
            dim);
    }

    timer.stop();
    CHECK_CUDA(cudaDeviceSynchronize());

    return timer.elapsed_ms() / repeats;
}

void run_case(
    int seq_len,
    int dim,
    bool run_cpu
) {
    std::cout << "\n=== Attention case: S=" << seq_len
              << ", D=" << dim << " ===" << std::endl;

    size_t num_elements = static_cast<size_t>(seq_len) * dim;

    size_t bytes = num_elements * sizeof(float);

    std::vector<float> h_Q(num_elements);
    std::vector<float> h_K(num_elements);
    std::vector<float> h_V(num_elements);
    std::vector<float> h_O_cpu(num_elements, 0.0f);
    std::vector<float> h_O_cuda(num_elements, 0.0f);
    std::vector<float> h_scores(static_cast<size_t>(seq_len) * seq_len);
    std::vector<float> h_probs(static_cast<size_t>(seq_len) * seq_len);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float &x : h_Q)
    {
        x = dist(rng);
    }

    for (float &x : h_K)
    {
        x = dist(rng);
    }

    for (float &x : h_V)
    {
        x = dist(rng);
    }

    float *d_Q = nullptr;
    float *d_K = nullptr;
    float *d_V = nullptr;
    float *d_O = nullptr;
    float *d_KT = nullptr;
    float *d_scores = nullptr;
    float *d_probs = nullptr;

    size_t bytes_KT = static_cast<size_t>(dim) * seq_len * sizeof(float);
    size_t bytes_scores = static_cast<size_t>(seq_len) * seq_len * sizeof(float);

    CHECK_CUDA(cudaMalloc((void **)&d_Q, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_K, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_V, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_O, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_KT, bytes_KT));
    CHECK_CUDA(cudaMalloc((void **)&d_scores, bytes_scores));
    CHECK_CUDA(cudaMalloc((void **)&d_probs, bytes_scores));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));

    int gpu_repeats = 20;
    int cpu_repeats = 20;

    double cpu_ms = -1.0;
    if (run_cpu)
    {
        cpu_ms = benchmark_cpu(
            h_Q, h_K, h_V, h_scores, h_probs, h_O_cpu,
            seq_len, dim, cpu_repeats
        );

        std::cout << "CPU:     "
                  << std::fixed << std::setprecision(4)
                  << cpu_ms << " ms, "
                  << std::setprecision(2)
                  << attention_gflops(seq_len, dim, cpu_ms)
                  << " GFLOP/s" << std::endl;
    }

    float tiled_ms = benchmark_cuda(
        d_Q, d_K, d_V, d_O, d_KT, d_scores, d_probs,
        seq_len, dim, gpu_repeats
    );
    CHECK_CUDA(cudaMemcpy(h_O_cuda.data(), d_O, bytes, cudaMemcpyDeviceToHost));

    std::cout << "CUDA:    "
              << std::fixed << std::setprecision(4)
              << tiled_ms << " ms, "
              << std::setprecision(2)
              << attention_gflops(seq_len, dim, tiled_ms)
              << " GFLOP/s" << std::endl;

    if (run_cpu)
    {
        float err = max_abs_error(h_O_cpu, h_O_cuda);
        std::cout << "Max error CUDA vs CPU: "
                  << err << std::endl;
        std::cout << "Correctness:           "
                  << (err < 1e-3f ? "PASS" : "FAIL")
                  << std::endl;
    }

    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));
    CHECK_CUDA(cudaFree(d_KT));
    CHECK_CUDA(cudaFree(d_scores));
    CHECK_CUDA(cudaFree(d_probs));
}

int main()
{
    CHECK_CUDA(cudaSetDevice(0));

    run_case(128, 64, true);
    run_case(256, 64, true);
    run_case(512, 64, true);
    run_case(1024, 64, false);
    run_case(1000, 80, true);

    return 0;
}
