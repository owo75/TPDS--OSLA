#pragma once

#include <cuda_runtime.h>  
#include <cusolverDn.h>  
#include <cublas_v2.h>  
#include "myutils.h"


//同时提取L和U矩阵
__global__ void extractLowerUpper(int n, const double* A, int lda, double* L, double* U, int ldu);
//拷贝矩阵最后几行
__global__ void copyLastRowsColumnMajor(double* A, double* A2, int m, int n, int cols);
//拷贝矩阵前几行
__global__ void copyFirstRowsColumnMajor(double* A, double* A2, int m, int n, int cols);
//垂直拼接两个矩阵
__global__ void concatenateVertical(const double* L1, const double* L2, double* Y, int m, int n, int cols);
//
void ReconstructWY(double* d_Q, int ldq, int m, int n, double* d_W, int ldw, double* d_Y, int ldy, CudaHandles& handles);


//float 版本
//同时提取L和U矩阵  
__global__ void extractLowerUpper(int n, const float* A, int lda, float* L, float* U, int ldu);  

//拷贝矩阵最后几行  
__global__ void copyLastRowsColumnMajor(float* A, float* A2, int m, int n, int cols);  

//拷贝矩阵前几行  
__global__ void copyFirstRowsColumnMajor(float* A, float* A2, int m, int n, int cols);  

//垂直拼接两个矩阵  
__global__ void concatenateVertical(const float* L1, const float* L2, float* Y, int m, int n, int cols);  

//  
void ReconstructWY(float* d_Q, int ldq, int m, int n, float* d_W, int ldw, float* d_Y, int ldy, CudaHandles& handles);