#ifndef CUDA_UTILS_CHECK_HPP
#define CUDA_UTILS_CHECH_HPP

#include "cublas_v2.h"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>
#include <iostream>
#include <istream>
#include <iterator>
#include <ostream>

#define CHECK_Cusolver(call)                                                   \
  {                                                                            \
    cusolverStatus_t err;                                                      \
    if ((err = (call)) != CUSOLVER_STATUS_SUCCESS) {                           \
      fprintf(stderr, "Got CuSolver error %d at %s:%d\n", err, __FILE__,       \
              __LINE__);                                                       \
      exit(1);                                                                 \
    }                                                                          \
  }

#define CHECK_Runtime(call)                                                    \
  {                                                                            \
    cudaError_t err;                                                           \
    if ((err = (call)) != cudaSuccess) {                                       \
      fprintf(stderr, "Got CuRuntime error %d at %s:%d\n", err, __FILE__,      \
              __LINE__);                                                       \
      exit(1);                                                                 \
    }                                                                          \
  }

#define CHECK_Curand(call)                                                     \
  {                                                                            \
    curandStatus_t err;                                                        \
    if ((err = (call)) != CURAND_STATUS_SUCCESS) {                             \
      fprintf(stderr, "Got CuRand error %d at %s:%d\n", err, __FILE__,         \
              __LINE__);                                                       \
      exit(1);                                                                 \
    }                                                                          \
  }
#define CHECK_Cublas(call)                                                     \
  {                                                                            \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS) {                             \
      fprintf(stderr, "Got CuBlas error %d at %s:%d\n", err, __FILE__,         \
              __LINE__);                                                       \
      exit(1);                                                                 \
    }                                                                          \
  }

#endif