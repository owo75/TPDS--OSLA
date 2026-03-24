#pragma once

#include <cublas_v2.h>     // cuBLAS API
#include <cuda_runtime.h>  // 提供 CUDA 运行时函数，如 cudaMemcpy
#include <curand.h>
#include <cusolverDn.h>  // cuSOLVER API

#include <cstdio>    // 用于 printf
#include <fstream>   // 用于 std::ofstream
#include <iomanip>   // 用于控制输出格式
#include <iostream>  // 用于 std::cout 和 std::endl
#include <string>    // 用于 std::string

// 初始化cuSOLVER和cuBLAS句柄
struct CudaHandles {
    cusolverDnHandle_t solver_handle;
    cublasHandle_t blas_handle;

    CudaHandles() {
        cusolverDnCreate(&solver_handle);
        cublasCreate(&blas_handle);
    }

    ~CudaHandles() {
        cusolverDnDestroy(solver_handle);
        cublasDestroy(blas_handle);
    }
};

// 拷贝上三角
void launchKernel_getU(dim3 gridDim, dim3 blockDim, int m, int n, double* A,
                       int ldA, double* U, int ldU);
__global__ void getU(int m, int n, double* A, int ldA, double* U, int ldU);
// 生成矩阵
void generateNormalMatrix(double* dA, long int m, long int n, double mean,
                          double stddev);
// 检测后向误差
void checkBackwardError(int m, int n, double* A, int lda, double* Q, int ldq,
                        double* R, int ldr);
// 检测正交性
void checkOtho(long int m, long int n, double* Q, int ldq);
// 矩阵乘法
void dgemm(int m, int n, int k, double* dA, int lda, double* dB, int ldb,
           double* dC, int ldc, double alpha, double beta);
// 计算范数
double Dnorm(long int m, long int n, double* dA);
// 生成单位矩阵
__global__ void setEye(double* I, long int n);
// 计算I - Q
__global__ void IminusQ(long m, long n, double* Q, long ldq);

// float版本
// 拷贝上三角
void launchKernel_getU(dim3 gridDim, dim3 blockDim, int m, int n, float* A,
                       int ldA, float* U, int ldU);
__global__ void getU(int m, int n, float* A, int ldA, float* U, int ldU);

// 生成矩阵
void generateNormalMatrix(float* dA, long int m, long int n, float mean,
                          float stddev);

// 检测后向误差
void checkBackwardError(int m, int n, float* A, int lda, float* Q, int ldq,
                        float* R, int ldr);

// 检测正交性
void checkOtho(long int m, long int n, float* Q, int ldq);

// 矩阵乘法
void sgemm(int m, int n, int k, float* dA, int lda, float* dB, int ldb,
           float* dC, int ldc, float alpha, float beta);

// 计算范数
float Snorm(long int m, long int n, float* dA);

// 生成单位矩阵
__global__ void setEye(float* I, long int n);
// 计算I - Q
__global__ void IminusQ(long m, long n, float* Q, long ldq);

void writeMatrixToCsvV2(float* dA, long ldA, long rows, long cols,
                        const std::string& fileName,
                        const std::string& matrixName);
void writeMatrixToCsvV2(double* dA, long ldA, long rows, long cols,
                        const std::string& fileName,
                        const std::string& matrixName);
void writeMatrixToCsvV2(__half* dA, long ldA, long rows, long cols,
                        const std::string& fileName,
                        const std::string& matrixName);
__global__ void h2s(int m, int n, __half* ah, int ldah, float* as, int ldas);
__global__ void s2h(int m, int n, float* as, int ldas, __half* ah, int ldah);

namespace {
cudaEvent_t start, stop;
}
inline void startTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
}

inline float stopTimer() {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

inline void startTimer1(cudaEvent_t* start, cudaEvent_t* stop) {
    cudaEventCreate(start);   // 创建 start 事件
    cudaEventCreate(stop);    // 创建 stop 事件
    cudaEventRecord(*start);  // 记录起始时间
}

// 停止计时器并返回耗时
inline float stopTimer1(cudaEvent_t* start, cudaEvent_t* stop) {
    cudaEventRecord(*stop);       // 记录结束时间
    cudaEventSynchronize(*stop);  // 等待事件完成
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, *start, *stop);  // 计算时间
    cudaEventDestroy(*start);                            // 销毁 start 事件
    cudaEventDestroy(*stop);                             // 销毁 stop 事件
    return milliseconds;                                 // 返回耗时
}

// 矩阵打印
template <typename T>
void printDeviceMatrixV2(T* dA, long ldA, long rows, long cols) {
    T matrix;

    for (long i = 0; i < rows; i++) {
        for (long j = 0; j < cols; j++) {
            cudaMemcpy(&matrix, dA + i + j * ldA, sizeof(T),
                       cudaMemcpyDeviceToHost);
            // printf("%f ", matrix[i * cols + j]);//按行存储优先
            // printf("%10.4f", matrix); // 按列存储优先
            // printf("%12.6f", matrix); // 按列存储优先
            // printf("%.20f ", matrix); // 按列存储优先
            printf("%5.2f ", matrix);  // 按列存储优先
        }
        printf("\n");
    }
}
