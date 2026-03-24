#include "TSQR.h"  
#include "myutils.h"  
#include "ReconstructWY.h"  

#define threadsPerDim 32  
// #define MYDEBUG2
// #define MYTIME

float memcpyTime = 0;
float luTime = 0;
float trsmTime = 0;

__global__ void extractLowerUpper(int n, const float* A, int lda, float* L, float* U, int ldu) {  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;  

    if (i < n && j < n) {  
        float val = A[j + i * lda];  
        if (i < j) {  
            L[j + i * n] = val;  
            U[j + i * ldu] = 0.0f;  
        } else if (i == j) {  
            L[j + i * n] = 1.0f;  
            U[j + i * ldu] = val;  
        } else {  
            L[j + i * n] = 0.0f;  
            U[j + i * ldu] = val;  
        }  
    }  
}  

__global__ void copyLastRowsColumnMajor(float* A, float* A2, int m, int n, int cols, int lda, int lda2) {  
    int col = blockIdx.x * blockDim.x + threadIdx.x;  

    if (col < cols) {  
        float* A_col = A + col * lda + n;  
        float* A2_col = A2 + col * lda2;  

        for (int row = 0; row < (m - n); ++row) {  
            A2_col[row] = A_col[row];  
        }  
    }  
}  

__global__ void copyFirstRowsColumnMajor(float* A, float* A2, int m, int n, int cols) {  
    int col = blockIdx.x * blockDim.x + threadIdx.x;  

    if (col < cols) {  
        float* A_col = A + col * m;  
        float* A2_col = A2 + col * n;  

        for (int row = 0; row < n; ++row) {  
            A2_col[row] = A_col[row];  
        }  
    }  
}  

__global__ void concatenateVertical(const float* L1, const float* L2, float* Y, int m, int n, int cols, int ldy) {  
    int col = blockIdx.x * blockDim.x + threadIdx.x;  

    if (col < cols) {  
        float* Y_col = Y + col * ldy;  
        const float* L1_col = L1 + col * n;  
        const float* L2_col = L2 + col * (m - n);  

        for(int row = 0; row < n; ++row) {  
            Y_col[row] = L1_col[row];  
        }  

        for(int row = 0; row < m - n; ++row) {  
            Y_col[n + row] = L2_col[row];  
        }  
    }  
}   

// 这个函数貌似是非合并访问的函数
// 将数据从d_A搬运到d_W，考虑d_W的lda
__global__ void copyWithLdw(const float* d_A, int lda, float* d_W, int ldw, int m, int n) {  
    // 计算当前线程负责的行和列  
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 行索引  
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 列索引  

    // 确保线程索引在矩阵范围内  
    if (row < m && col < n) {  
        // 计算源矩阵 d_A 和目标矩阵 d_W 的索引  
        int srcIdx = col * lda + row;       // d_A 的索引（列主序存储）  
        int dstIdx = col * ldw + row;    // d_W 的索引（考虑列步长 ldw）  

        // 复制数据  
        d_W[dstIdx] = d_A[srcIdx];  
    }  
}

void ReconstructWY(float* d_Q, int ldq, int m, int n, float* d_W, int ldw, float* d_Y, int ldy, CudaHandles& handles) {  
    cusolverDnHandle_t solver_handle = handles.solver_handle;  
    cublasHandle_t blas_handle = handles.blas_handle;  
    // printf("当前的m = %d , n = %d , ldq = %d , ldy = %d\n", m, n, ldq, ldy);
    float* d_A;  
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeof(float) * m * n));   
    // CHECK_CUDA(cudaMemcpy(d_A, d_Q, sizeof(float)* m * n, cudaMemcpyDeviceToDevice));  ///这里好像就不对，需要考虑ldq来写入吗
    


    dim3 blockDim(threadsPerDim, threadsPerDim);  
    // 这两个有啥区别呢
    dim3 gridDimMN((m + threadsPerDim - 1) / threadsPerDim, (n + threadsPerDim -1) / threadsPerDim);  
    dim3 gridDimNM((n + threadsPerDim - 1) / threadsPerDim, (m + threadsPerDim -1) / threadsPerDim);  
#ifdef MYTIME
    startTimer();
#endif
    copyWithLdw<<<gridDimNM, blockDim>>>(d_Q, ldq, d_A, m, m, n); 
    CHECK_CUDA(cudaDeviceSynchronize());  

    IminusQ<<<gridDimMN, blockDim>>>(m, n, d_A, m);  
    CHECK_CUDA(cudaDeviceSynchronize());        

    // CHECK_CUDA(cudaMemcpy(d_W, d_A, sizeof(float)* m * n, cudaMemcpyDeviceToDevice));    //////这里需要改
    copyWithLdw<<<gridDimNM, blockDim>>>(d_A, m, d_W, ldw, m, n); 
#ifdef MYTIME
    memcpyTime += stopTimer();
#endif

#ifdef MYDEBUG2
    printf("A:\n");  
    printDeviceMatrixV2(d_A, m, m, n);  
    printf("W:\n");  
    printDeviceMatrixV2(d_W, ldw, m, n);  
#endif  

    
    float* d_A1 = d_A;  
    int m_A2 = m - n;  
    float alpha = 1.0f;  
    int lda = m;  
#ifdef MYTIME
    startTimer();
#endif
    int* devInfo;  
    CHECK_CUDA(cudaMalloc((void**)&devInfo, sizeof(int)));  

    int lwork_getrf = 0;  
    CHECK_CUSOLVER(cusolverDnSgetrf_bufferSize(solver_handle, n, n, d_A1, lda, &lwork_getrf));  

    float* d_work_getrf;  
    CHECK_CUDA(cudaMalloc((void**)&d_work_getrf, sizeof(float) * lwork_getrf));  

    int* d_pivots = NULL;  

    CHECK_CUSOLVER(cusolverDnSgetrf(solver_handle, n, n, d_A1, lda, d_work_getrf, d_pivots, devInfo));  
#ifdef MYTIME
    luTime += stopTimer();
#endif

    float* d_L1;  
    float* d_U;  
    CHECK_CUDA(cudaMalloc((void**)&d_L1, sizeof(float) * n * n));  
    CHECK_CUDA(cudaMalloc((void**)&d_U, sizeof(float) * n * n));  
#ifdef MYTIME
    startTimer();
#endif
    dim3 gridDimNN((n + threadsPerDim - 1) / threadsPerDim, (n + threadsPerDim - 1) / threadsPerDim);  
    extractLowerUpper<<<gridDimNN, blockDim>>>(n, d_A1, lda, d_L1, d_U, n);  
    // CHECK_CUDA(cudaDeviceSynchronize());  
#ifdef MYTIME
    memcpyTime += stopTimer();
#endif
#ifdef MYDEBUG
    printf("L1:\n");  
    printDeviceMatrixV2(d_L1, n, n, n);  
    printf("U:\n");  
    printDeviceMatrixV2(d_U, n, n, n);  
    // printf("LU分解后的A:\n");  
    // printDeviceMatrixV2(d_A, m, m, n);
#endif  

    if(m_A2 != 0) {  
        float* d_A2_t;  
        CHECK_CUDA(cudaMalloc((void**)&d_A2_t, sizeof(float) * m_A2 * n));  

        int threadsPerBlock = 256;  
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;   
#ifdef MYTIME
    startTimer();
#endif
        copyLastRowsColumnMajor<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_A2_t, m, n, n, m, m_A2);  
#ifdef MYTIME
    memcpyTime += stopTimer();
#endif

#ifdef MYDEBUG1  
        printf("LU分解后下面m-n行矩阵A2:\n");  
        printDeviceMatrixV2(d_A2_t, m_A2, m_A2, n);  
#endif  
#ifdef MYTIME
    startTimer();
#endif
        CHECK_CUBLAS(cublasStrsm(blas_handle,  
                                CUBLAS_SIDE_RIGHT,         
                                CUBLAS_FILL_MODE_UPPER,   
                                CUBLAS_OP_N,              
                                CUBLAS_DIAG_NON_UNIT,     
                                m_A2,                    
                                n,                        
                                &alpha,                    
                                d_U,                     
                                n,                         
                                d_A2_t,                   
                                m_A2));                      
#ifdef MYTIME
    trsmTime += stopTimer();
#endif
#ifdef MYDEBUG
        printf("L2:\n");  
        printDeviceMatrixV2(d_A2_t, m_A2, m_A2, n);  
#endif  
#ifdef MYTIME
    startTimer();
#endif
        concatenateVertical<<<blocksPerGrid, threadsPerBlock>>>(d_L1, d_A2_t, d_Y, m, n, n, ldy);  
        // CHECK_CUDA(cudaDeviceSynchronize());  
#ifdef MYTIME
    memcpyTime += stopTimer();
#endif
#ifdef MYDEBUG  
        printf("Y:\n");  
        printDeviceMatrixV2(d_Y, m, m, n);  
#endif  

        CHECK_CUDA(cudaFree(d_A2_t));  
    } else {  
        // CHECK_CUDA(cudaMemcpy(d_Y, d_L1, sizeof(float) * n * n, cudaMemcpyDeviceToDevice));    //// 这里需要改
        copyWithLdw<<<gridDimNN, blockDim>>>(d_L1, n, d_Y, ldy, n, n); 
    }  
#ifdef MYTIME
    startTimer();
#endif
    CHECK_CUBLAS(cublasStrsm(blas_handle,  
                            CUBLAS_SIDE_RIGHT,         
                            CUBLAS_FILL_MODE_LOWER,   
                            CUBLAS_OP_T,              
                            CUBLAS_DIAG_UNIT,     
                            m,                        
                            n,                        
                            &alpha,                    
                            d_L1,                     
                            n,                         
                            d_W,                   
                            ldw));                      
#ifdef MYTIME
    trsmTime += stopTimer();
#endif
#ifdef MYDEBUG2
    printf("d_W:\n");  
    printDeviceMatrixV2(d_W, m, m, n);  
#endif  
#ifdef MYTIME
    printf("memcpy time = %.4f(s)\n", memcpyTime / 1000);
    printf("lu time = %.4f(s)\n", luTime / 1000);
    printf("trsm time = %.4f(s)\n", trsmTime / 1000);
    printf("total time = %.4f(s)\n", (memcpyTime + luTime + trsmTime) / 1000);
#endif

    CHECK_CUDA(cudaFree(d_work_getrf));  
    CHECK_CUDA(cudaFree(devInfo));  
    CHECK_CUDA(cudaFree(d_pivots));  
    CHECK_CUDA(cudaFree(d_L1));  
    CHECK_CUDA(cudaFree(d_U));  
}