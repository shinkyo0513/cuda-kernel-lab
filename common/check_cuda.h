#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

#define CHECK_CUDA(call)                                            \
    do                                                              \
    {                                                               \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess)                                     \
        {                                                           \
        std:                                                        \
            cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                 << "code=" << static_cast<int>(err)                \
                 << "\"" << cudaGetErrorString(err) << "\""         \
                 << std::endl;                                      \
            std::exit(EXIT_FAILURE);                                \
        }                                                           \
    } while (0)

#define CUDA_KERNEL()                        \
    do                                       \
    {                                        \
        CHECK_CUDA(cudaGetLastError());      \
        CHECK_CUDA(cudaDeviceSynchronize()); \
    } while (0)