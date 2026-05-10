#pragma once

#include <cublas_v2.h>
#include <cstdlib>
#include <iostream>

#define CHECK_CUBLAS(call)                                                   \
    do                                                                       \
    {                                                                        \
        cublasStatus_t status = (call);                                      \
        if (status != CUBLAS_STATUS_SUCCESS)                                 \
        {                                                                    \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__   \
                      << " status=" << static_cast<int>(status) << std::endl; \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)
