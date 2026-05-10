#include "attention_cuda.h"

#include <cmath>
#include <cuda_runtime.h>

#include "../kernels/matmul_tiled.h"
#include "../kernels/softmax.h"
#include "../kernels/transpose.h"
#include "../common/check_cuda.h"

#ifndef ATTENTION_BLOCK_SIZE
#define ATTENTION_BLOCK_SIZE 256
#endif

__global__ void scale_kernel (
    float* data,
    int n,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        data[idx] *= scale;
    }
}

void launch_scale (
    float* data,
    int n,
    float scale
) {
    int block = ATTENTION_BLOCK_SIZE;
    int grid = (n + block - 1) / block;

    scale_kernel<<<grid, block>>>(data, n, scale);
    CHECK_CUDA(cudaGetLastError());
}

void launch_attention_cuda(
    const float *d_Q,
    const float *d_K,
    const float *d_V,
    float *d_O,
    int seq_len,
    int dim)
{
    // Q:      [S, D]
    // K:      [S, D]
    // V:      [S, D]
    // K_T:    [D, S]
    // scores: [S, S]
    // probs:  [S, S]
    // O:      [S, D]

    const int S = seq_len;
    const int D = dim;

    size_t bytes_KT = static_cast<size_t>(D) * S * sizeof(float);
    size_t bytes_scores = static_cast<size_t>(S) * S * sizeof(float);

    float *d_KT = nullptr;
    float *d_scores = nullptr;
    float *d_probs = nullptr;

    CHECK_CUDA(cudaMalloc((void **)&d_KT, bytes_KT));
    CHECK_CUDA(cudaMalloc((void **)&d_scores, bytes_scores));
    CHECK_CUDA(cudaMalloc((void **)&d_probs, bytes_scores));

    // 1. K_T = transpose(K)
    // K:   [S, D]
    // K_T: [D, S]
    launch_transpose_shared(d_K, d_KT, S, D);

    // 2. scores = Q * K_T
    // Q:      [S, D]
    // K_T:    [D, S]
    // scores: [S, S]
    launch_matmul_tiled(d_Q, d_KT, d_scores, S, S, D);

    // 3. scores *= 1 / sqrt(D)
    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    launch_scale(d_scores, S * S, scale);

    // 4. probs = softmax(scores)
    launch_softmax(d_scores, d_probs, S, S);

    // 5. O = probs * V
    // probs: [S, S]
    // V:     [S, D]
    // O:     [S, D]
    launch_matmul_tiled(d_probs, d_V, d_O, S, D, S);

    CHECK_CUDA(cudaFree(d_KT));
    CHECK_CUDA(cudaFree(d_scores));
    CHECK_CUDA(cudaFree(d_probs));
}
