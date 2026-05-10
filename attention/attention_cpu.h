#pragma once

#include <vector>

void attention_cpu_with_workspace(
    const std::vector<float> &Q,
    const std::vector<float> &K,
    const std::vector<float> &V,
    std::vector<float> &scores,
    std::vector<float> &probs,
    std::vector<float> &O,
    int seq_len,
    int dim
);

void attention_cpu(
    const std::vector<float> &Q,
    const std::vector<float> &K,
    const std::vector<float> &V,
    std::vector<float> &O,
    int seq_len,
    int dim
);