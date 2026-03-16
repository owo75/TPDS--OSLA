#pragma once

// #include <cuda.h>
// void launchKernel_ClearMatrix(dim3 gridDim, dim3 blockDim, long m, long n, double *A, long ldA);

//清除矩阵A所有元素为0
template <typename T>
void launchKernel_ClearMatrix(dim3 gridDim, dim3 blockDim, long m, long n, T *A, long ldA);

// void launchKernel_setMetrixTrValue(dim3 gridDim,
//                                    dim3 blockDim,
//                                    long m,
//                                    long n,
//                                    double *A,
//                                    long ldA,
//                                    double v);

//矩阵A元素设为常数v
template <typename T>
void launchKernel_setMetrixTrValue(dim3 gridDim, dim3 blockDim, long m, long n, T *A, long ldA, T v);

// 将矩阵的某个对称部分的下三角部分复制到上三角部分
// void launchKernel_CpyMatrixL2U(dim3 gridDim, dim3 blockDim, long n, double *A, long ldA);
template <typename T>
void launchKernel_CpyMatrixL2U(dim3 gridDim, dim3 blockDim, long n, T *A, long ldA);

// void launchKernel_copyAndClear(dim3 gridDim,
//                                dim3 blockDim,
//                                long m,
//                                long n,
//                                double *srcM,
//                                long lds,
//                                double *dstM,
//                                long ldd);

//从源矩阵 srcM 复制数据到目标矩阵 dstM，然后清除源矩阵 srcM 的内容。
template <typename T>
void launchKernel_copyAndClear(dim3 gridDim,
                               dim3 blockDim,
                               long m,
                               long n,
                               T *srcM,
                               long lds,
                               T *dstM,
                               long ldd);

// void launchKernel_IminusQ(dim3 gridDim, dim3 blockDim, long m, long n, double *Q, long ldq);

//计算I-Q
template <typename T>
void launchKernel_IminusQ(dim3 gridDim, dim3 blockDim, long m, long n, T *Q, long ldq);
//计算A-B
void launchKernel_AminusB(dim3 gridDim,
                          dim3 blockDim,
                          long m,
                          long n,
                          double *A,
                          long ldA,
                          double *B,
                          long ldB);

//计算|A|-|B|
void launchKernel_AbsAminusAbsB(dim3 gridDim,
                                dim3 blockDim,
                                long m,
                                long n,
                                double *A,
                                long ldA,
                                double *B,
                                long ldB);

// void launchKernel_copyMatrix(dim3 gridDim,
//                              dim3 blockDim,
//                              long m,
//                              long n,
//                              double *srcM,
//                              long lds,
//                              double *dstM,
//                              long ldd);

//将矩阵 srcM 复制到 dstM
template <typename T>
void launchKernel_copyMatrix(dim3 gridDim,
                             dim3 blockDim,
                             long m,
                             long n,
                             T *srcM,
                             long lds,
                             T *dstM,
                             long ldd);

// void launchKernel_copyMatrixAToTranpB(dim3 gridDim,
//                                       dim3 blockDim,
//                                       long m,
//                                       long n,
//                                       double *srcM,
//                                       long lds,
//                                       double *dstM,
//                                       long ldd);

//将矩阵 srcM 的转置（即A^T）复制到矩阵 dstM。
template <typename T>
void launchKernel_copyMatrixAToTranpB(dim3 gridDim,
                                      dim3 blockDim,
                                      long m,
                                      long n,
                                      T *srcM,
                                      long lds,
                                      T *dstM,
                                      long ldd);

// void launchKernel_getU(dim3 gridDim,
//                        dim3 blockDim,
//                        int m,
//                        int n,
//                        double *A,
//                        int ldA,
//                        double *U,
//                        int ldU);

//从矩阵 A 中提取上三角部分，存储到矩阵 U 中
template <typename T>
void launchKernel_getU(dim3 gridDim, dim3 blockDim, int m, int n, T *A, int ldA, T *U, int ldU);

// void launchKernel_getLower(dim3 gridDim, dim3 blockDim, long m, long n, double *A, long ldA);

//提取矩阵 A的下三角部分。
template <typename T>
void launchKernel_getLower(dim3 gridDim, dim3 blockDim, long m, long n, T *A, long ldA);

// void launch_kernel_cpyATr2Vector(dim3 gridDim,
//                                  dim3 blockDim,
//                                  long m,
//                                  long n,
//                                  double *A,
//                                  long ldA,
//                                  double *B);

//A^T的某一部分复制到向量B中
template <typename T>
void launch_kernel_cpyATr2Vector(dim3 gridDim, dim3 blockDim, long m, long n, T *A, long ldA, T *B);

//A=A*scaler
void launchKernel_scaleMatrixA(dim3 gridDim,
                               dim3 blockDim,
                               long m,
                               long n,
                               double *A,
                               long ldA,
                               double scaler);

double findVectorAbsMax(double *d_array, int n);