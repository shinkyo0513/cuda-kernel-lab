#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>

#include "../common/check_cuda.h"
#include "../kernels/softmax.h"

int main()
{
    int nrows = 20;
    int ncols = 20;

    int nelements = nrows * ncols;
    size_t bytes = nelements * sizeof(float);

    std::vector<float> h_in(nelements);
    std::vector<float> h_out(nelements, 0.0f);

    for (int row = 0; row < nrows; ++row)
    {
        for (int col = 0; col < ncols; ++col)
        {
            h_in[row * ncols + col] = static_cast<float>(col);
        }
    }

    float *d_in = nullptr;
    float *d_out = nullptr;

    CHECK_CUDA(cudaMalloc((void **)&d_in, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_out, bytes));

    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    launch_softmax(d_in, d_out, nrows, ncols);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    std::cout << "Softmax output:" << std::endl;
    for (int r = 0; r < nrows; ++r)
    {
        for (int c = 0; c < ncols; ++c)
        {
            std::cout << h_out[r * ncols + c] << " ";
        }
        std::cout << std::endl;
    }

    float row_sum = std::accumulate(
        h_out.begin(),
        h_out.begin() + ncols,
        0.0f);

    std::cout << "First row sum: " << row_sum << std::endl;

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}