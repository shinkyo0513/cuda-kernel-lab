#pragma once

#include <cublas_v2.h>

void launch_attention_cuda(
    const float *d_Q,
    const float *d_K,
    const float *d_V,
    float *d_O,
    int seq_len,
    int dim
);

void launch_attention_cuda_with_workspace(
    const float *d_Q,
    const float *d_K,
    const float *d_V,
    float *d_O,
    float *d_KT,
    float *d_scores,
    float *d_probs,
    int seq_len,
    int dim
);

void launch_attention_cublas_fused_softmax(
    cublasHandle_t handle,
    const float *d_Q,
    const float *d_K,
    const float *d_V,
    float *d_O,
    int seq_len,
    int dim
);

void launch_attention_cublas_fused_softmax_with_workspace(
    cublasHandle_t handle,
    const float *d_Q,
    const float *d_K,
    const float *d_V,
    float *d_O,
    float *d_KT,
    float *d_scores,
    float *d_probs,
    int seq_len,
    int dim
);
