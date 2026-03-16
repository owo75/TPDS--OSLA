#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

#define TSLU_BLOCK_SIZE 256
#define TSLU_BLOCK_DIM_X 32
#define TSLU_BLOCK_DIM_Y 8
#define TSLU_NUM_DATA_ROW 2

// 共享内存定义
template <typename T>
struct shared_memory;
template <>
struct shared_memory<float> {
    __device__ static float *get_pointer() {
        extern __shared__ float shared_mem_float[];
        return shared_mem_float;
    }
};
template <>
struct shared_memory<double> {
    __device__ static double *get_pointer() {
        extern __shared__ double shared_mem_double[];
        return shared_mem_double;
    }
};

// Warp 规约求和
template <typename T>
__device__ T warpAllReduceSum(T val) {
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// LU 分解
template <typename T>
__device__ void local_lu(int n, T *A, int lda, int *pivot) {
    for (int i = 0; i < n; ++i) {
        int max_idx = i;
        for (int j = i + 1; j < n; ++j) {
            if (fabs(A[j * lda + i]) > fabs(A[max_idx * lda + i])) {
                max_idx = j;
            }
        }
        pivot[i] = max_idx;

        if (max_idx != i) {
            for (int k = 0; k < n; ++k) {
                T tmp = A[i * lda + k];
                A[i * lda + k] = A[max_idx * lda + k];
                A[max_idx * lda + k] = tmp;
            }
        }

        if (fabs(A[i * lda + i]) > 1e-10) {
            for (int j = i + 1; j < n; ++j) {
                T factor = A[j * lda + i] / A[i * lda + i];
                A[j * lda + i] = factor;
                for (int k = i + 1; k < n; ++k) {
                    A[j * lda + k] -= factor * A[i * lda + k];
                }
            }
        }
    }
}

// 核函数：TSLU 分解
template <typename T>
__global__ void tslu_kernel(int m, int n, T *A, int lda, int *P, T *U, int ldu) {
    extern __shared__ T shared_mem[];
    T *shared_A = shared_mem;
    int *shared_P = reinterpret_cast<int *>(shared_mem + TSLU_BLOCK_SIZE * n);

    int tidx = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = min(TSLU_BLOCK_SIZE, m - bid * TSLU_BLOCK_SIZE);

    A = A + bid * TSLU_BLOCK_SIZE;
    U = U + bid * n;

    for (int k = 0; k < TSLU_NUM_DATA_ROW; ++k) {
        int row_idx = tidx + k * TSLU_BLOCK_DIM_X;
        if (row_idx < block_size) {
            for (int h = 0; h < TSLU_BLOCK_DIM_Y; ++h) {
                int col_idx = threadIdx.y + h * TSLU_BLOCK_DIM_Y;
                if (col_idx < n) {
                    shared_A[row_idx + col_idx * TSLU_BLOCK_SIZE] = A[row_idx + col_idx * lda];
                }
            }
        }
    }
    __syncthreads();

    local_lu<T>(block_size, shared_A, TSLU_BLOCK_SIZE, shared_P);
    __syncthreads();

    if (tidx == 0) {
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < n; ++j) {
                A[i + j * lda] = shared_A[shared_P[i] + j * TSLU_BLOCK_SIZE];
            }
        }
    }
    __syncthreads();
}

// 递归 TSLU 分解
template <typename T>
void tslu(cublasHandle_t handle, int m, int n, T *A, int lda, int *P, T *L, int ldl, T *U, int ldu) {
    assert(m >= n);
    int blockNum = (m + TSLU_BLOCK_SIZE - 1) / TSLU_BLOCK_SIZE;
    int shared_memory_size = TSLU_BLOCK_SIZE * n * sizeof(T) + TSLU_BLOCK_SIZE * sizeof(int);

    cudaFuncSetAttribute(tslu_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size);
    dim3 blockDim(TSLU_BLOCK_DIM_X, TSLU_BLOCK_DIM_Y);
    tslu_kernel<T><<<blockNum, blockDim, shared_memory_size>>>(m, n, A, lda, P, U, ldu);
    cudaDeviceSynchronize();
}

// 实例化模板函数
template void tslu<float>(cublasHandle_t, int, int, float *, int, int *, float *, int, float *, int);
template void tslu<double>(cublasHandle_t, int, int, double *, int, int *, double *, int, double *, int);