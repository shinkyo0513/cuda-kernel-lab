#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>

#include "../common/check_cuda.h"
#include "../common/timer.h"
#include "../kernels/transpose.h"

void transpose_cpu(
    const std::vector<float> &in,
    std::vector<float> &out,
    int rows,
    int cols)
{
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            out[c * rows + r] = in[r * cols + c];
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

float benchmark_transpose_naive(
    const float *d_in,
    float *d_out,
    int rows,
    int cols,
    int repeats)
{
    for (int i = 0; i < 5; ++i)
    {
        launch_transpose_naive(d_in, d_out, rows, cols);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();

    for (int i = 0; i < repeats; ++i)
    {
        launch_transpose_naive(d_in, d_out, rows, cols);
    }

    timer.stop();
    CHECK_CUDA(cudaDeviceSynchronize());

    return timer.elapsed_ms() / repeats;
}

float benchmark_transpose_shared(
    const float *d_in,
    float *d_out,
    int rows,
    int cols,
    int repeats)
{
    for (int i = 0; i < 5; ++i)
    {
        launch_transpose_shared(d_in, d_out, rows, cols);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();

    for (int i = 0; i < repeats; ++i)
    {
        launch_transpose_shared(d_in, d_out, rows, cols);
    }

    timer.stop();
    CHECK_CUDA(cudaDeviceSynchronize());

    return timer.elapsed_ms() / repeats;
}

void run_case(int rows, int cols)
{
    std::cout << "\n=== Transpose case: "
              << rows << " x " << cols << " ===" << std::endl;

    size_t num_elements = static_cast<size_t>(rows) * cols;
    size_t bytes = num_elements * sizeof(float);

    std::vector<float> h_in(num_elements);
    std::vector<float> h_ref(num_elements, 0.0f);
    std::vector<float> h_out_naive(num_elements, 0.0f);
    std::vector<float> h_out_shared(num_elements, 0.0f);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float &x : h_in)
    {
        x = dist(rng);
    }

    transpose_cpu(h_in, h_ref, rows, cols);

    float *d_in = nullptr;
    float *d_out = nullptr;

    CHECK_CUDA(cudaMalloc((void **)&d_in, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_out, bytes));

    CHECK_CUDA(cudaMemcpy(
        d_in,
        h_in.data(),
        bytes,
        cudaMemcpyHostToDevice));

    int repeats = 100;

    float naive_ms = benchmark_transpose_naive(
        d_in,
        d_out,
        rows,
        cols,
        repeats);

    CHECK_CUDA(cudaMemcpy(
        h_out_naive.data(),
        d_out,
        bytes,
        cudaMemcpyDeviceToHost));

    float shared_ms = benchmark_transpose_shared(
        d_in,
        d_out,
        rows,
        cols,
        repeats);

    CHECK_CUDA(cudaMemcpy(
        h_out_shared.data(),
        d_out,
        bytes,
        cudaMemcpyDeviceToHost));

    float err_naive = max_abs_error(h_ref, h_out_naive);
    float err_shared = max_abs_error(h_ref, h_out_shared);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Naive transpose:  " << naive_ms << " ms" << std::endl;
    std::cout << "Shared transpose: " << shared_ms << " ms" << std::endl;
    std::cout << "Speedup:          " << naive_ms / shared_ms << "x" << std::endl;

    std::cout << "Max error naive:  " << err_naive << std::endl;
    std::cout << "Max error shared: " << err_shared << std::endl;

    bool correct = err_naive < 1e-6f && err_shared < 1e-6f;
    std::cout << "Correctness:      "
              << (correct ? "PASS" : "FAIL")
              << std::endl;

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
}

int main()
{
    CHECK_CUDA(cudaSetDevice(0));

    run_case(512, 512);
    run_case(1024, 1024);
    run_case(2048, 2048);

    run_case(1024, 2048);
    run_case(2048, 1024);

    run_case(1000, 777);

    return 0;
}