#pragma once

void launch_softmax(
    const float *d_in,
    float *d_out,
    int nrows,
    int ncols
);

void launch_softmax_fused(
    const float *d_in,
    float *d_out,
    int nrows,
    int ncols
);
