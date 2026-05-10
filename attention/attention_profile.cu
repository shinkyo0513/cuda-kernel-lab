#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>

#include "../common/check_cuda.h"

#include "attention_cpu.h"
#include "attention_cuda.h"

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

void fill_inputs(
    std::vector<float> &h_Q,
    std::vector<float> &h_K,
    std::vector<float> &h_V)
{
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
}

void run_attention_cuda_once(
    const std::vector<float> &h_Q,
    const std::vector<float> &h_K,
    const std::vector<float> &h_V,
    std::vector<float> &h_O_cuda,
    int seq_len,
    int dim)
{
    size_t num_elements = static_cast<size_t>(seq_len) * dim;
    size_t bytes = num_elements * sizeof(float);

    float *d_Q = nullptr;
    float *d_K = nullptr;
    float *d_V = nullptr;
    float *d_O = nullptr;

    CHECK_CUDA(cudaMalloc((void **)&d_Q, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_K, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_V, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_O, bytes));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));

    launch_attention_cuda(d_Q, d_K, d_V, d_O, seq_len, dim);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_O_cuda.data(), d_O, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));
}

bool run_correctness_case(int seq_len, int dim)
{
    std::cout << "\n=== Attention correctness: S=" << seq_len
              << ", D=" << dim << " ===" << std::endl;

    size_t num_elements = static_cast<size_t>(seq_len) * dim;

    std::vector<float> h_Q(num_elements);
    std::vector<float> h_K(num_elements);
    std::vector<float> h_V(num_elements);
    std::vector<float> h_O_cpu(num_elements, 0.0f);
    std::vector<float> h_O_cuda(num_elements, 0.0f);

    fill_inputs(h_Q, h_K, h_V);

    attention_cpu(h_Q, h_K, h_V, h_O_cpu, seq_len, dim);
    run_attention_cuda_once(h_Q, h_K, h_V, h_O_cuda, seq_len, dim);

    float err = max_abs_error(h_O_cpu, h_O_cuda);
    bool correct = err < 1e-3f;

    std::cout << "Max error CUDA vs CPU: " << err << std::endl;
    std::cout << "Correctness:           "
              << (correct ? "PASS" : "FAIL")
              << std::endl;

    return correct;
}

void run_profile_case(int seq_len, int dim)
{
    std::cout << "\n=== Attention profile: S=" << seq_len
              << ", D=" << dim << " ===" << std::endl;

    size_t num_elements = static_cast<size_t>(seq_len) * dim;

    std::vector<float> h_Q(num_elements);
    std::vector<float> h_K(num_elements);
    std::vector<float> h_V(num_elements);
    std::vector<float> h_O_cuda(num_elements, 0.0f);

    fill_inputs(h_Q, h_K, h_V);
    run_attention_cuda_once(h_Q, h_K, h_V, h_O_cuda, seq_len, dim);
}

int main()
{
    CHECK_CUDA(cudaSetDevice(0));

    int seq_len = 1024;
    int dim = 64;

    if (std::getenv("ATTENTION_S") != nullptr)
    {
        seq_len = std::atoi(std::getenv("ATTENTION_S"));
    }

    if (std::getenv("ATTENTION_D") != nullptr)
    {
        dim = std::atoi(std::getenv("ATTENTION_D"));
    }

    if (seq_len <= 0 || dim <= 0)
    {
        std::cerr << "ATTENTION_S and ATTENTION_D must be positive"
                  << std::endl;
        return 1;
    }

    const bool check_mode =
        std::getenv("ATTENTION_CHECK") != nullptr;

    if (check_mode)
    {
        return run_correctness_case(seq_len, dim) ? 0 : 1;
    }

    run_profile_case(seq_len, dim);
    return 0;
}
