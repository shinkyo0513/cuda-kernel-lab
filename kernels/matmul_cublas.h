#pragma once

void launch_matmul_cublas(
    cublasHandle_t handle,
    const float *d_A,
    const float *d_B,
    float *d_C,
    int M,
    int N,
    int K
);