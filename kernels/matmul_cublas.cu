#include "matmul_cublas.h"

#include <iostream>
#include <cstdlib>
#include "../common/check_cuda.h"

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

void launch_matmul_cublas(
    cublasHandle_t handle,
    const float *d_A,
    const float *d_B,
    float *d_C,
    int M,
    int N,
    int K)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Input matrices are row-major:
    // A_row: M x K
    // B_row: K x N
    // C_row: M x N
    //
    // cuBLAS assumes column-major.
    // Row-major C = A * B can be computed by calling cuBLAS as:
    // C_col_equiv = B_col_equiv * A_col_equiv
    //
    // This writes the correct row-major result into d_C.
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

    CHECK_CUDA(cudaGetLastError());
}