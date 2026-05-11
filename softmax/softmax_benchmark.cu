#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#include "../common/check_cuda.h"
#include "../common/timer.h"
#include "../kernels/softmax.h"

void softmax_cpu(
    const std::vector<float> &in,
    std::vector<float> &out,
    int nrows,
    int ncols)
{
    for (int row = 0; row < nrows; ++row)
    {
        float row_max = -std::numeric_limits<float>::infinity();

        for (int col = 0; col < ncols; ++col)
        {
            row_max = std::max(row_max, in[row * ncols + col]);
        }

        float row_sum = 0.0f;
        for (int col = 0; col < ncols; ++col)
        {
            float value = std::exp(in[row * ncols + col] - row_max);
            out[row * ncols + col] = value;
            row_sum += value;
        }

        for (int col = 0; col < ncols; ++col)
        {
            out[row * ncols + col] /= row_sum;
        }
    }
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

double softmax_gflops(int nrows, int ncols, double ms)
{
    double elements = static_cast<double>(nrows) * ncols;
    double ops = 5.0 * elements;
    return ops / (ms * 1e6);
}

float benchmark_softmax(
    const float *d_in,
    float *d_out,
    int nrows,
    int ncols,
    int repeats)
{
    for (int i = 0; i < 5; ++i)
    {
        launch_softmax(d_in, d_out, nrows, ncols);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();

    for (int i = 0; i < repeats; ++i)
    {
        launch_softmax(d_in, d_out, nrows, ncols);
    }

    timer.stop();
    CHECK_CUDA(cudaDeviceSynchronize());

    return timer.elapsed_ms() / repeats;
}

float benchmark_softmax_fused(
    const float *d_in,
    float *d_out,
    int nrows,
    int ncols,
    int repeats)
{
    for (int i = 0; i < 5; ++i)
    {
        launch_softmax_fused(d_in, d_out, nrows, ncols);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();

    for (int i = 0; i < repeats; ++i)
    {
        launch_softmax_fused(d_in, d_out, nrows, ncols);
    }

    timer.stop();
    CHECK_CUDA(cudaDeviceSynchronize());

    return timer.elapsed_ms() / repeats;
}

void run_case(int nrows, int ncols)
{
    std::cout << "\n=== Softmax case: rows=" << nrows
              << ", cols=" << ncols << " ===" << std::endl;

    size_t nelements = static_cast<size_t>(nrows) * ncols;
    size_t bytes = nelements * sizeof(float);

    std::vector<float> h_in(nelements);
    std::vector<float> h_ref(nelements, 0.0f);
    std::vector<float> h_out_old(nelements, 0.0f);
    std::vector<float> h_out_fused(nelements, 0.0f);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);

    for (float &x : h_in)
    {
        x = dist(rng);
    }

    softmax_cpu(h_in, h_ref, nrows, ncols);

    float *d_in = nullptr;
    float *d_out = nullptr;

    CHECK_CUDA(cudaMalloc((void **)&d_in, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_out, bytes));

    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    int repeats = 100;

    float old_ms = benchmark_softmax(d_in, d_out, nrows, ncols, repeats);
    CHECK_CUDA(cudaMemcpy(h_out_old.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    float fused_ms = benchmark_softmax_fused(d_in, d_out, nrows, ncols, repeats);
    CHECK_CUDA(cudaMemcpy(h_out_fused.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    float old_err = max_abs_error(h_ref, h_out_old);
    float fused_err = max_abs_error(h_ref, h_out_fused);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Softmax old:   " << old_ms << " ms, "
              << std::setprecision(2)
              << softmax_gflops(nrows, ncols, old_ms)
              << " GFLOP/s" << std::endl;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Softmax fused: " << fused_ms << " ms, "
              << std::setprecision(2)
              << softmax_gflops(nrows, ncols, fused_ms)
              << " GFLOP/s" << std::endl;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Speedup:       " << old_ms / fused_ms << "x" << std::endl;

    std::cout << std::scientific << std::setprecision(3);
    std::cout << "Max error old:   " << old_err << std::endl;
    std::cout << "Max error fused: " << fused_err << std::endl;

    bool correct = old_err < 1e-5f && fused_err < 1e-5f;
    std::cout << "Correctness:     "
              << (correct ? "PASS" : "FAIL")
              << std::endl;

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
}

int main()
{
    CHECK_CUDA(cudaSetDevice(0));

    run_case(1024, 128);
    run_case(1024, 256);
    run_case(1024, 512);
    run_case(1024, 1000);

    return 0;
}
