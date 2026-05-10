#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <limits>
#include <algorithm>

#ifndef S
#define S 16
#endif

#ifndef D
#define D 16
#endif

/* R = A * B
   A: M x K
   B: K x N
   R: M x N
*/
void matmul(
    const std::vector<float> &A,
    const std::vector<float> &B,
    std::vector<float> &R,
    int M,
    int K,
    int N)
{
    assert(A.size() == static_cast<size_t>(M * K));
    assert(B.size() == static_cast<size_t>(K * N));
    assert(R.size() == static_cast<size_t>(M * N));

    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k)
            {
                sum += A[row * K + k] * B[k * N + col];
            }

            R[row * N + col] = sum;
        }
    }
}

/* R = A * B^T
   A: M x K
   B: N x K
   R: M x N
*/
void matmul_w_transpose(
    const std::vector<float> &A,
    const std::vector<float> &B,
    std::vector<float> &R,
    int M,
    int K,
    int N)
{
    assert(A.size() == static_cast<size_t>(M * K));
    assert(B.size() == static_cast<size_t>(N * K));
    assert(R.size() == static_cast<size_t>(M * N));

    for (int row = 0; row < M; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k)
            {
                sum += A[row * K + k] * B[col * K + k];
            }

            R[row * N + col] = sum;
        }
    }
}

/* Row-wise softmax
   in:  nrows x ncols
   out: nrows x ncols
*/
void softmax_row_wise(
    const std::vector<float> &in,
    std::vector<float> &out,
    int nrows,
    int ncols)
{
    assert(in.size() == static_cast<size_t>(nrows * ncols));
    assert(out.size() == static_cast<size_t>(nrows * ncols));

    std::vector<float> row_max(nrows);
    std::vector<float> row_exp_sum(nrows);
    std::vector<float> exp_vals(nrows * ncols);

    // 1. Row-wise maximum
    for (int row = 0; row < nrows; ++row)
    {
        float max_val = -std::numeric_limits<float>::infinity();

        for (int col = 0; col < ncols; ++col)
        {
            max_val = std::max(max_val, in[row * ncols + col]);
        }

        row_max[row] = max_val;
    }

    // 2. Max-subtraction and exp
    for (int row = 0; row < nrows; ++row)
    {
        for (int col = 0; col < ncols; ++col)
        {
            exp_vals[row * ncols + col] =
                std::exp(in[row * ncols + col] - row_max[row]);
        }
    }

    // 3. Row-wise exp sum
    for (int row = 0; row < nrows; ++row)
    {
        float sum = 0.0f;

        for (int col = 0; col < ncols; ++col)
        {
            sum += exp_vals[row * ncols + col];
        }

        row_exp_sum[row] = sum;
    }

    // 4. Normalize
    for (int row = 0; row < nrows; ++row)
    {
        for (int col = 0; col < ncols; ++col)
        {
            out[row * ncols + col] =
                exp_vals[row * ncols + col] / row_exp_sum[row];
        }
    }
}

/*
   Single-head self-attention:

   Q: S x D
   K: S x D
   V: S x D

   scores = Q * K^T / sqrt(D), shape S x S
   probs  = softmax(scores),   shape S x S
   R      = probs * V,         shape S x D
*/
void attention_cpu_with_workspace(
    const std::vector<float> &Q,
    const std::vector<float> &K,
    const std::vector<float> &V,
    std::vector<float> &scores,
    std::vector<float> &probs,
    std::vector<float> &R,
    int seq_len,
    int dim)
{
    assert(Q.size() == static_cast<size_t>(seq_len * dim));
    assert(K.size() == static_cast<size_t>(seq_len * dim));
    assert(V.size() == static_cast<size_t>(seq_len * dim));

    assert(scores.size() == static_cast<size_t>(seq_len * seq_len));
    assert(probs.size() == static_cast<size_t>(seq_len * seq_len));
    assert(R.size() == static_cast<size_t>(seq_len * dim));

    matmul_w_transpose(Q, K, scores, seq_len, dim, seq_len);

    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));

    for (int i = 0; i < seq_len * seq_len; ++i)
    {
        scores[i] *= scale;
    }

    softmax_row_wise(scores, probs, seq_len, seq_len);

    matmul(probs, V, R, seq_len, seq_len, dim);
}

void attention_cpu(
    const std::vector<float> &Q,
    const std::vector<float> &K,
    const std::vector<float> &V,
    std::vector<float> &O,
    int seq_len,
    int dim)
{
    std::vector<float> scores(seq_len * seq_len);
    std::vector<float> probs(seq_len * seq_len);

    attention_cpu_with_workspace(
        Q,
        K,
        V,
        scores,
        probs,
        O,
        seq_len,
        dim);
}
/*
int main()
{
    std::vector<float> Q(S * D);
    std::vector<float> K(S * D);
    std::vector<float> V(S * D);

    std::vector<float> scores(S * S);
    std::vector<float> probs(S * S);
    std::vector<float> R(S * D);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto &x : Q)
    {
        x = dist(rng);
    }

    for (auto &x : K)
    {
        x = dist(rng);
    }

    for (auto &x : V)
    {
        x = dist(rng);
    }

    attention_cpu(Q, K, V, scores, probs, R, S, D);

    for (int row = 0; row < S; ++row)
    {
        for (int col = 0; col < D; ++col)
        {
            std::cout << R[row * D + col] << " ";
        }
        std::cout << '\n';
    }

    return 0;
}
*/