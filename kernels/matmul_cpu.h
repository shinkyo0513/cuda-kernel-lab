#pragma once

#include <vector>

void matmul_cpu(
    const std::vector<float> &A,
    const std::vector<float> &B,
    std::vector<float> &C,
    int M,
    int N,
    int K);

float max_abs_error(
    const std::vector<float> &ref,
    const std::vector<float> &out);