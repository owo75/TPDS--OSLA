#include "TSQR.h"  

#define threadsPerDim 32  

/**
 * 问题：8192 8192 1024 32是可以运行的，但是8192 8192 1024 64不可以
 * 
 * 
 * 
 */



void tsqr(float* d_A, int lda, int m, int n, int blockSize,  
          float* d_Q, int ldq, float* d_R, int ldr,  
          CudaHandles& handles) {  
    // printf("m = %d, n = %d, blockSize = %d, lda = %d\n", m, n, blockSize, lda);
    printf("调用了我的tsqr\n");
    int numBlocks = (m + blockSize - 1) / blockSize;  

    cusolverDnHandle_t solver_handle = handles.solver_handle;  
    cublasHandle_t blas_handle = handles.blas_handle;  

    dim3 blockDim(threadsPerDim,threadsPerDim);  
    
    int maxBlockSize = std::min(blockSize, m);  
    int lwork_geqrf = 0;  
    int lwork_orgqr = 0;  

    // 修改为float版本的cuSOLVER函数  
    CHECK_CUSOLVER(cusolverDnSgeqrf_bufferSize(  
        solver_handle, maxBlockSize, n, d_A, lda, &lwork_geqrf));  

    CHECK_CUSOLVER(cusolverDnSorgqr_bufferSize(  
        solver_handle, maxBlockSize, n, n, d_A, lda, nullptr, &lwork_orgqr));  

    int lwork = std::max(lwork_geqrf, lwork_orgqr);  
    float* d_work;  
    CHECK_CUDA(cudaMalloc((void**)&d_work, sizeof(float) * lwork));  

    float* d_tau;  
    CHECK_CUDA(cudaMalloc((void**)&d_tau, sizeof(float) * n));  

    int* devInfo;  
    CHECK_CUDA(cudaMalloc((void**)&devInfo, sizeof(int)));  

    float* d_R_blocks;  
    CHECK_CUDA(cudaMalloc((void**)&d_R_blocks, sizeof(float) * numBlocks * n * n));  

    for (int i = 0; i < numBlocks; ++i) {  
        int currentBlockSize = std::min(blockSize, m - i * blockSize);  
        float* d_A_block = d_A + i * blockSize;  

        CHECK_CUSOLVER(cusolverDnSgeqrf(  
            solver_handle, currentBlockSize, n, d_A_block, lda,  
            d_tau, d_work, lwork_geqrf, devInfo));  

        dim3 gridDim((currentBlockSize + threadsPerDim - 1) / threadsPerDim,   
                     (n + threadsPerDim - 1) / threadsPerDim);  
        getU<<<gridDim,blockDim>>>(currentBlockSize, n, d_A_block, lda,   
                                  d_R_blocks + i * n, n * numBlocks);  
        CHECK_CUDA(cudaDeviceSynchronize());  

#ifdef MYDEBUG  
        printf("矩阵R%d\n", i);  
        printDeviceMatrixV2<float>(d_R_blocks + i * n, n * numBlocks, n, n);  
#endif  

        CHECK_CUSOLVER(cusolverDnSorgqr(  
            solver_handle, currentBlockSize, n, n, d_A_block, lda,  
            d_tau, d_work, lwork_orgqr, devInfo));  

#ifdef MYDEBUG  
        printf("矩阵Q%d\n", i);  
        printDeviceMatrixV2<float>(d_A_block, m, currentBlockSize, n);  
#endif  
    }  

    float* d_R_combined;  
    CHECK_CUDA(cudaMalloc((void**)&d_R_combined, sizeof(float) * numBlocks * n * n));  
    CHECK_CUDA(cudaMemcpy(d_R_combined, d_R_blocks,   
                         sizeof(float) * numBlocks * n * n,   
                         cudaMemcpyDeviceToDevice));  

    int lwork_geqrf_global = 0;  
    CHECK_CUSOLVER(cusolverDnSgeqrf_bufferSize(  
        solver_handle, numBlocks * n, n, d_R_combined, numBlocks * n,   
        &lwork_geqrf_global));  

    float* d_work_global;  
    CHECK_CUDA(cudaMalloc((void**)&d_work_global, sizeof(float) * lwork_geqrf_global));  

    float* d_tau_global;  
    CHECK_CUDA(cudaMalloc((void**)&d_tau_global, sizeof(float) * n));  

    CHECK_CUSOLVER(cusolverDnSgeqrf(  
        solver_handle, numBlocks * n, n, d_R_combined, numBlocks * n,  
        d_tau_global, d_work_global, lwork_geqrf_global, devInfo));  

    dim3 gridDim2((n + threadsPerDim - 1) / threadsPerDim,   
                  (n + threadsPerDim - 1) / threadsPerDim);  
    getU<<<gridDim2, blockDim>>>(n, n, d_R_combined, n * numBlocks, d_R, ldr);  
    CHECK_CUDA(cudaDeviceSynchronize());  

#ifdef MYDEBUG  
    printf("final R\n");  
    printDeviceMatrixV2(d_R, ldr, n, n);  
#endif  

    CHECK_CUSOLVER(cusolverDnSorgqr(  
        solver_handle, numBlocks * n, n, n, d_R_combined, numBlocks * n,  
        d_tau_global, d_work_global, lwork_geqrf_global, devInfo));  

    for (int i = 0; i < numBlocks; ++i) {  
        int currentBlockSize = std::min(blockSize, m - i * blockSize);  
        float* d_Q_block = d_A + i * blockSize;  
        float* d_Q_global = d_R_combined + i * n;  

        float alpha = 1.0f;  
        float beta = 0.0f;  

        CHECK_CUBLAS(cublasSgemm(  
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

    CHECK_CUDA(cudaFree(d_tau));  
    CHECK_CUDA(cudaFree(d_tau_global));  
    CHECK_CUDA(cudaFree(d_work));  
    CHECK_CUDA(cudaFree(d_work_global));  
    CHECK_CUDA(cudaFree(d_R_blocks));  
    CHECK_CUDA(cudaFree(d_R_combined));  
    CHECK_CUDA(cudaFree(devInfo));  
}