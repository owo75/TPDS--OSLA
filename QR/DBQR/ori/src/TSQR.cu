/** 
 * 
 * 
 * 
 * 现在有一个问题：对于大矩阵（比如8192 1024 1024）不能正确执行，会出现内存上的错误，还没有排查出来
 *排查方向1. cuda-memcheck
 *排查方向2. 打印到文件中查看
 *对ldr的修改，是不是应该用n * blocksize
*/

#include "TSQR.h"


// #define MYDEBUG
#define threadsPerDim 32


// TSQR函数  
void tsqr(double* d_A, int lda, int m, int n, int blockSize,  
          double* d_Q, int ldq, double* d_R, int ldr,  
          CudaHandles& handles) {  
    // 计算分块数  
    
    int numBlocks = (m + blockSize - 1) / blockSize;  

    // 创建cuSOLVER和cuBLAS句柄  
    cusolverDnHandle_t solver_handle = handles.solver_handle;  
    cublasHandle_t blas_handle = handles.blas_handle;  

    // int threadsPerDim = 2; 
    dim3 blockDim(threadsPerDim,threadsPerDim);
    
    // 为QR分解分配空间  
    int maxBlockSize = std::min(blockSize, m);  
    int lwork_geqrf = 0;  
    int lwork_orgqr = 0;  

    // 直接创建最大的辅助空间
    CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(  
        solver_handle, maxBlockSize, n, d_A, lda, &lwork_geqrf));  

    CHECK_CUSOLVER(cusolverDnDorgqr_bufferSize(  
        solver_handle, maxBlockSize, n, n, d_A, lda, nullptr, &lwork_orgqr));  

    int lwork = std::max(lwork_geqrf, lwork_orgqr);  
    double* d_work;  
    CHECK_CUDA(cudaMalloc((void**)&d_work, sizeof(double) * lwork));  

    // 分配 tau 和 info  
    double* d_tau;  
    CHECK_CUDA(cudaMalloc((void**)&d_tau, sizeof(double) * n));  

    int* devInfo;  
    CHECK_CUDA(cudaMalloc((void**)&devInfo, sizeof(int)));  

    // 分配用于存储所有块 R 的设备内存  
    double* d_R_blocks;  
    CHECK_CUDA(cudaMalloc((void**)&d_R_blocks, sizeof(double) * numBlocks * n * n));  

    // 逐块处理  
    for (int i = 0; i < numBlocks; ++i) {  
        int currentBlockSize = std::min(blockSize, m - i * blockSize);  
        double* d_A_block = d_A + i * blockSize; // 假设A按列主序存储  

        // 对当前块执行 QR 分解  
        CHECK_CUSOLVER(cusolverDnDgeqrf(  
            solver_handle, currentBlockSize, n, d_A_block, lda,  
            d_tau, d_work, lwork_geqrf, devInfo));  

        // 提取 R 块并存储,其中R为上三角
        dim3 gridDim((currentBlockSize + threadsPerDim - 1) / threadsPerDim, (n + threadsPerDim - 1) / threadsPerDim);
        getU<<<gridDim,blockDim>>>(currentBlockSize, n, d_A_block, lda, d_R_blocks + i * n, n * numBlocks);
        // CHECK_CUDA(cudaGetLastError());  
        CHECK_CUDA(cudaDeviceSynchronize());  

#ifdef MYDEBUG
        printf("矩阵R%d\n", i);
        printDeviceMatrixV2<double>(d_R_blocks + i * n, n * numBlocks, n, n);
#endif
        // 生成 Q 块并保存在 d_A_block  
        CHECK_CUSOLVER(cusolverDnDorgqr(  
            solver_handle, currentBlockSize, n, n, d_A_block, lda,  
            d_tau, d_work, lwork_orgqr, devInfo));  

#ifdef MYDEBUG
        // 在打印三位小数的情况下测试正交性为e-3
        printf("矩阵Q%d\n", i);
        printDeviceMatrixV2<double>(d_A_block, m, currentBlockSize, n);
#endif

    }  

    // 合并所有 R 块，形成一个大的 R 矩阵，然后对其进行全局 QR 分解  
    double* d_R_combined;  
    CHECK_CUDA(cudaMalloc((void**)&d_R_combined, sizeof(double) * numBlocks * n * n));  
////为什么需要重新转移到d_R_combined中呢
    CHECK_CUDA(cudaMemcpy(  
        d_R_combined,  
        d_R_blocks,  
        sizeof(double) * numBlocks * n * n,  
        cudaMemcpyDeviceToDevice));  

    // 对 d_R_combined 执行全局 QR 分解  
    int lwork_geqrf_global = 0;  
    CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(  
        solver_handle, numBlocks * n, n, d_R_combined, numBlocks * n, &lwork_geqrf_global));  

    double* d_work_global;  
    CHECK_CUDA(cudaMalloc((void**)&d_work_global, sizeof(double) * lwork_geqrf_global));  

    double* d_tau_global;  
    CHECK_CUDA(cudaMalloc((void**)&d_tau_global, sizeof(double) * n));  

    CHECK_CUSOLVER(cusolverDnDgeqrf(  
        solver_handle, numBlocks * n, n, d_R_combined, numBlocks * n,  
        d_tau_global, d_work_global, lwork_geqrf_global, devInfo));  

    // 提取最终的 R 矩阵，存储在 d_R  
    dim3 gridDim2((n + threadsPerDim - 1) / threadsPerDim, (n + threadsPerDim - 1) / threadsPerDim);
    getU<<<gridDim2, blockDim>>>(n, n, d_R_combined,  n * numBlocks, d_R, ldr);
    // CHECK_CUDA(cudaGetLastError());  
    CHECK_CUDA(cudaDeviceSynchronize());  
#ifdef MYDEBUG
    printf("final R\n");
    printDeviceMatrixV2(d_R, ldr, n, n);
#endif

    // 生成全局 Q 矩阵  
    CHECK_CUSOLVER(cusolverDnDorgqr(  
        solver_handle, numBlocks * n, n, n, d_R_combined, numBlocks * n,  
        d_tau_global, d_work_global, lwork_geqrf_global, devInfo));  

    // 将全局 Q 应用于每个局部 Q 块，计算最终的 Q 矩阵  
    for (int i = 0; i < numBlocks; ++i) {  
        int currentBlockSize = std::min(blockSize, m - i * blockSize);  
        double* d_Q_block = d_A + i * blockSize; // 局部 Q 块  
        double* d_Q_global = d_R_combined + i * n; // 对应的全局 Q 矩阵  

        // 计算 d_Q_block = d_Q_block * d_Q_global  
        // 使用 cublasDgemm 进行矩阵乘法  
        double alpha = 1.0;  
        double beta = 0.0;  

        CHECK_CUBLAS(cublasDgemm(  
            blas_handle,  
            CUBLAS_OP_N, CUBLAS_OP_N,  
            currentBlockSize, n, n,  
            &alpha,  
            d_Q_block, lda,  
            d_Q_global, numBlocks * n,  
            &beta,  
            d_Q + i * blockSize, ldq));  
    }  
#ifdef MYDEBUG
    printf("final Q:\n");
    printDeviceMatrixV2(d_Q, ldq, m, n);
#endif
    // CHECK_CUDA(cudaGetLastError()); 
    // 释放设备内存  
    CHECK_CUDA(cudaFree(d_tau));  
    CHECK_CUDA(cudaFree(d_tau_global));  
    CHECK_CUDA(cudaFree(d_work));  
    CHECK_CUDA(cudaFree(d_work_global));  
    CHECK_CUDA(cudaFree(d_R_blocks));  
    CHECK_CUDA(cudaFree(d_R_combined));  
    CHECK_CUDA(cudaFree(devInfo));  
}  




// int main(int argc, char *argv[]) {  
//     // 矩阵大小  

//     // int m = 8192; // 行数  
//     // int n = 256;   // 列数  
//     // int blockSize = n; // 分块大小  
//     int m = atoi(argv[1]);
//     int n = atoi(argv[2]);
//     int blockSize = atoi(argv[3]);

//     double* d_A;  
//     CHECK_CUDA(cudaMalloc((void**)&d_A, sizeof(double) * m * n));  
//     generateNormalMatrix(d_A, m, n, 0.0, 1.0);

//     double* d_A2;
//     CHECK_CUDA(cudaMalloc((void**)&d_A2, sizeof(double) * m * n));  
//     CHECK_CUDA(cudaMemcpy(d_A2, d_A, sizeof(double) * m * n, cudaMemcpyDeviceToDevice));  
    

// #ifdef MYDEBUG
//     printf("生成的矩阵A为:\n");
//     printDeviceMatrixV2<double>(d_A, m, m, n);
// #endif

//     // 分配设备内存用于Q和R  
//     double* d_Q;  
//     double* d_R;  
//     CHECK_CUDA(cudaMalloc((void**)&d_Q, sizeof(double) * m * n));  
//     CHECK_CUDA(cudaMalloc((void**)&d_R, sizeof(double) * n * n));  

//     // 初始化CUDA句柄  
//     CudaHandles handles;  

//     // 执行TSQR  
//     tsqr(d_A, m, n, blockSize, d_Q, d_R, handles);  

//     //正交性和后向误差检测
// // #ifdef MYDEBUG
//     checkOtho(m, n, d_Q, m);
//     checkBackwardError(m, n, d_A2, m, d_Q, m, d_R, n);
// // #endif



//     // 释放设备内存  
//     CHECK_CUDA(cudaFree(d_A));  
//     CHECK_CUDA(cudaFree(d_Q));  
//     CHECK_CUDA(cudaFree(d_R));  

//     return 0;  
// }
