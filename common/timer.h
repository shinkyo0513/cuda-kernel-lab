#pragma once

#include <cuda_runtime.h>
#include <chrono>
#include "check_cuda.h"

class GpuTimer {
public:
    GpuTimer() {
        CHECK_CUDA(cudaEventCreate(&start_));
        CHECK_CUDA(cudaEventCreate(&stop_));
    }

    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() {
        CHECK_CUDA(cudaEventRecord(start_));
    }

    void stop() {
        CHECK_CUDA(cudaEventRecord(stop_));
        CHECK_CUDA(cudaEventSynchronize(stop_));
    }

    float elapsed_ms() const {
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

class CpuTimer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        stop_ = std::chrono::high_resolution_clock::now();
    }

    float elapsed_ms() {
        std::chrono::duration<double, std::milli> elapsed = stop_ - start_;
        return elapsed.count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::high_resolution_clock::time_point stop_;
};