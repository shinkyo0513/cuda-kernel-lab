#pragma once

void launch_transpose_naive(
    const float *d_in,
    float *d_out,
    int rows,
    int cols
);

void launch_transpose_shared(
    const float *d_in,
    float *d_out,
    int rows,
    int cols
);