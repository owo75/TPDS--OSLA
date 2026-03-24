#pragma once

#include <cublas_v2.h>     // cuBLAS API
#include <cuda_runtime.h>  // CUDA 运行时 API
#include <cusolverDn.h>    // cuSOLVER API

#include <cstdlib>   // 用于 EXIT_FAILURE
#include <iostream>  // 用于 std::cerr 和 std::endl

// 错误检查宏
#define CHECK_CUDA(call)                                                 \
    {                                                                    \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " : " << cudaGetErrorString(err) << std::endl;  \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    }

#define CHECK_CUSOLVER(call)                                                 \
    {                                                                        \
        cusolverStatus_t status = (call);                                    \
        if (status != CUSOLVER_STATUS_SUCCESS) {                             \
            std::cerr << "CUSOLVER error in " << __FILE__ << ":" << __LINE__ \
                      << " : " << status << std::endl;                       \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    }

#define CHECK_CUBLAS(call)                                                 \
    {                                                                      \
        cublasStatus_t status = (call);                                    \
        if (status != CUBLAS_STATUS_SUCCESS) {                             \
            std::cerr << "CUBLAS error in " << __FILE__ << ":" << __LINE__ \
                      << " : " << status << std::endl;                     \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }
