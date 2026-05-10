#include "matmul_cpu.h"

#include <cmath>
#include <algorithm>

void matmul_cpu(
    const std::vector<float> &A,
    const std::vector<float> &B,
    std::vector<float> &C,
    int M,
    int N,
    int K)
{
    // A: M x K
    // B: K x N
    // C: M x N
    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
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