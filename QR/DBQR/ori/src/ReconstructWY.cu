#include "TSQR.h"  
#include "myutils.h"
#include "ReconstructWY.h"



/**
 * 1. 需要考虑的一个点是当L2不需要的时候，即m = n时，如何处理？
 *    这是cuda实现和matlab实现一个比较大的区别：在出现极端情况的时候，cuda需要额外处理
 * 
 */




// #define MYDEBUG  
#define threadsPerDim 32  


// 提取下三角和上三角矩阵的内核  
__global__ void extractLowerUpper(int n, const double* A, int lda, double* L, double* U, int ldu) {  
//这里怎么和矩阵里面元素对应起来的

    int i = blockIdx.x * blockDim.x + threadIdx.x; // 列索引  
    int j = blockIdx.y * blockDim.y + threadIdx.y; // 行索引  

    if (i < n && j < n) {  
        double val = A[j + i * lda];  
        if (i < j) {  
            L[j + i * n] = val;  
            U[j + i * ldu] = 0.0;  
        } else if (i == j) {  
            L[j + i * n] = 1.0;  
            U[j + i * ldu] = val;  
        } else {  
            L[j + i * n] = 0.0;  
            U[j + i * ldu] = val;  
        }  
    }  
}  

// 将 A 的最后 (m - n) 行复制到 A2  
// m是A的行数，n是A开始拷贝到饿行数，cols是A和A2的列数
__global__ void copyLastRowsColumnMajor(double* A, double* A2, int m, int n, int cols) {  
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 列索引  

    if (col < cols) {  
        // 计算 A 和 A2 中当前列的起始地址  
        double * A_col = A + col * m + n;       // A 的第 (n) 行开始  
        double* A2_col = A2 + col * (m - n);        // A2 的起始位置  

        // 复制 (m - n) 元素  
        for (int row = 0; row < (m - n); ++row) {  
            A2_col[row] = A_col[row];  
        }  
    }  
}  

// 将 A 的前 n 行复制到 A2  
// m 是 A 的行数，n 是要拷贝的行数，cols 是 A 和 A2 的列数  
__global__ void copyFirstRowsColumnMajor(double* A, double* A2, int m, int n, int cols) {  
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 列索引  

    if (col < cols) {  
        // 计算 A 和 A2 中当前列的起始地址  
        double* A_col = A + col * m;          // A 的起始位置  
        double* A2_col = A2 + col * n;        // A2 的起始位置  

        // 复制前 n 个元素  
        for (int row = 0; row < n; ++row) {  
            A2_col[row] = A_col[row];  
        }  
    }  
}  



//将L1 L2拼接成Y
__global__ void concatenateVertical(const double* L1, const double* L2, double* Y, int m, int n, int cols, int ldy) {  
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 列索引  
    // int m_minus_n =  m - n;

    if (col < cols) {  
        // 指向 Y 的当前列的起始位置  
        double* Y_col = Y + col * ldy;  
        // 指向 L1 和 L2 的当前列的起始位置  
        const double* L1_col = L1 + col * n;  
        const double* L2_col = L2 + col * (m - n);  

        // 复制 L1 的数据到 Y  
        for(int row = 0; row < n; ++row) {  
            Y_col[row] = L1_col[row];  
        }  

        // 复制 L2 的数据到 Y  
        for(int row = 0; row < m - n; ++row) {  
            Y_col[n + row] = L2_col[row];  
        }  
    }  
} 



// ReconstructWY 函数  
void ReconstructWY(double* d_Q, int ldq, int m, int n, double* d_W, int ldw, double* d_Y,int ldy, CudaHandles& handles) {  
    cusolverDnHandle_t solver_handle = handles.solver_handle;  
    cublasHandle_t blas_handle = handles.blas_handle;  

    //保存Q副本
    
    double* d_A;
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeof(double) * m * n)); 
    CHECK_CUDA(cudaMemcpy(d_A, d_Q, sizeof(double)* m * n, cudaMemcpyDeviceToDevice));

    // 1. 计算 A = I - Q  
    dim3 blockDim(threadsPerDim, threadsPerDim);
    dim3 gridDimMN((m + threadsPerDim - 1) / threadsPerDim, (n + threadsPerDim -1) / threadsPerDim);
    IminusQ<<<gridDimMN, blockDim>>>(m, n, d_A, m);
//这里需不需要同步呢
    CHECK_CUDA(cudaDeviceSynchronize());        
    CHECK_CUDA(cudaMemcpy(d_W, d_A, sizeof(double)* m * n, cudaMemcpyDeviceToDevice));
    

#ifdef MYDEBUG
    printf("I - Q:\n");
    printDeviceMatrixV2(d_A, m, m, n);
#endif

    
    // 2. 分割 A 为 A1 和 A2  
    double* d_A1 = d_A;               // A 的前 n 行，大小 n x n  
    // double* d_A2 = d_A + n;           // A 的第 n+1 行开始，大小 (m - n) x n  

    int m_A2 = m - n;  
    double alpha = 1.0;
    // 3. 对 A1 执行 LU 分解（无主元）  
    // 准备参数  
    int lda = m;  
    int* devInfo;  
    CHECK_CUDA(cudaMalloc((void**)&devInfo, sizeof(int)));  

    int lwork_getrf = 0;  
    CHECK_CUSOLVER(cusolverDnDgetrf_bufferSize(solver_handle, n, n, d_A1, lda, &lwork_getrf));  

    double* d_work_getrf;  
    CHECK_CUDA(cudaMalloc((void**)&d_work_getrf, sizeof(double) * lwork_getrf));  

//这里是不是应该设置成空指针，可以先把d_pivots输出看一下
    int* d_pivots = NULL; // 虽然不需要主元，但函数可能仍然需要该参数  
    // CHECK_CUDA(cudaMalloc((void**)&d_pivots, sizeof(int) * n));  

    // 执行 LU 分解  
    CHECK_CUSOLVER(cusolverDnDgetrf(solver_handle, n, n, d_A1, lda, d_work_getrf, d_pivots, devInfo));  

    // 提取 L1 和 U  
    double* d_L1;  
    double* d_U;  
    CHECK_CUDA(cudaMalloc((void**)&d_L1, sizeof(double) * n * n));  
    CHECK_CUDA(cudaMalloc((void**)&d_U, sizeof(double) * n * n));  

    // 内核提取 L 和 U  
    // dim3 blockDim(threadsPerDim, threadsPerDim);  
//这一步是不是多余的，其实并不需要提取，直接计算即可
    dim3 gridDimNN((n + threadsPerDim - 1) / threadsPerDim, (n + threadsPerDim - 1) / threadsPerDim);  
    extractLowerUpper<<<gridDimNN, blockDim>>>(n, d_A1, lda, d_L1, d_U, n);  
    // CHECK_CUDA(cudaGetLastError());  
    CHECK_CUDA(cudaDeviceSynchronize());

#ifdef MYDEBUG
    printf("L1:\n");
    printDeviceMatrixV2(d_L1, n, n, n);
    printf("U:\n");
    printDeviceMatrixV2(d_U, n, n, n);
#endif


    if(m_A2 != 0)
    {
        //这一步的后向误差还不能保证，在保留三位小数的情况下不能到e-3
            // 4. 计算 L2 = A2 * U_inv
            
            double* d_A2_t;  
            CHECK_CUDA(cudaMalloc((void**)&d_A2_t, sizeof(double) * m_A2 * n));
        //如果想要拷贝子矩阵，不应该这么写，这样是顺序写入，但是ldA2和ldAt其实是不同的
            // CHECK_CUDA(cudaMemcpy(d_A2_t, d_A2, sizeof(double) * m_A2 * n, cudaMemcpyDeviceToDevice));
        //这里如果想改成二维的应该怎么写
            int threadsPerBlock = 256;  
            int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;   
            copyLastRowsColumnMajor<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_A2_t, m, n, n);

        #ifdef MYDEBUG
            printf("A2:\n");
            printDeviceMatrixV2(d_A2_t, m_A2, m_A2, n);
        #endif

            //本质上就是求解XA = B
            CHECK_CUBLAS(cublasDtrsm(blas_handle,  
                                    CUBLAS_SIDE_RIGHT,         
                                    CUBLAS_FILL_MODE_UPPER,   
                                    CUBLAS_OP_N,              
                                    CUBLAS_DIAG_NON_UNIT,     
                                    m_A2,                    //A2的m    
                                    n,                        
                                    &alpha,                    
                                    d_U,                     
                                    n,                         
                                    d_A2_t,                   //输出   
                                    m_A2));                      
        #ifdef MYDEBUG
            printf("L2:\n");
            printDeviceMatrixV2(d_A2_t, m_A2, m_A2, n);
        #endif


            // 5. 构建 Y  
            concatenateVertical<<<blocksPerGrid, threadsPerBlock>>>(d_L1, d_A2_t, d_Y, m, n, n, ldy);
            // CHECK_CUDA(cudaGetLastError());  
            CHECK_CUDA(cudaDeviceSynchronize());

        #ifdef MYDEBUG
            printf("Y:\n");
            printDeviceMatrixV2(d_Y, m, m, n);
        #endif

    }
    else{
        //如果满足 m == n，那么需要直接姜L1复制给Y
        CHECK_CUDA(cudaMemcpy(d_Y, d_L1, sizeof(double) * n * n, cudaMemcpyDeviceToDevice));
    }

    // 6. 计算 W*Y1' = A：其中Y1就是L1
    CHECK_CUBLAS(cublasDtrsm(blas_handle,  
                            CUBLAS_SIDE_RIGHT,         
                            CUBLAS_FILL_MODE_LOWER,   
                            CUBLAS_OP_T,              
                            CUBLAS_DIAG_UNIT,     //这里应该是NON还是UNIT
                            m,                        
                            n,                        
                            &alpha,                    
                            d_L1,                     
                            n,                         
                            d_W,                   //输出   
                            ldw));                      
#ifdef MYDEBUG
    printf("d_W:\n");
    printDeviceMatrixV2(d_W, ldw, m, n);
#endif
    

    // 释放临时内存  
    CHECK_CUDA(cudaFree(d_work_getrf));  
    CHECK_CUDA(cudaFree(devInfo));  
    CHECK_CUDA(cudaFree(d_pivots));  
    CHECK_CUDA(cudaFree(d_L1));  
    CHECK_CUDA(cudaFree(d_U));  
    // if(m_A2 != 0){
    //     cudaFree(d_A2_t);  
    // }
    
}  


/*

int main(int argc, char *argv[]) {  
    // 矩阵大小  

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int blockSize = atoi(argv[3]);

    double* d_A;  
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeof(double) * m * n));  

    generateNormalMatrix(d_A, m, n, 0.0, 1.0);

    double* d_A2;
    CHECK_CUDA(cudaMalloc((void**)&d_A2, sizeof(double) * m * n));  
    CHECK_CUDA(cudaMemcpy(d_A2, d_A, sizeof(double) * m * n, cudaMemcpyDeviceToDevice));  
    

#ifdef MYDEBUG
    printf("生成的矩阵A为:\n");
    printDeviceMatrixV2<double>(d_A, m, m, n);
#endif

    // 分配设备内存用于Q和R  
    double* d_Q;  
    double* d_R;  
    CHECK_CUDA(cudaMalloc((void**)&d_Q, sizeof(double) * m * n));  
    CHECK_CUDA(cudaMalloc((void**)&d_R, sizeof(double) * n * n));  



    // 初始化CUDA句柄  
    CudaHandles handles;  

    // 执行TSQR  
    tsqr(d_A, m, n, blockSize, d_Q, d_R, handles);  

    //正交性和后向误差检测
#ifdef MYDEBUG
    //保存原始Q
    double* d_Qori;  
    CHECK_CUDA(cudaMalloc((void**)&d_Qori, sizeof(double) * m * n));  
    CHECK_CUDA(cudaMemcpy(d_Qori, d_Q, sizeof(double)* m * n, cudaMemcpyDeviceToDevice));
    dim3 blockDim(threadsPerDim, threadsPerDim);
    dim3 gridDimMN((m + threadsPerDim - 1) / threadsPerDim, (n + threadsPerDim -1) / threadsPerDim);
    IminusQ<<<gridDimMN, blockDim>>>(m, n, d_Qori, m);

    //测试TSQR结果
    checkOtho(m, n, d_Q, m);
    checkBackwardError(m, n, d_A2, m, d_Q, m, d_R, n);

    // printf("d_Q:\n");
    // printDeviceMatrixV2(d_Q, m, m, n);
#endif

    // 分配设备内存用于 W 和 Y  
    double* d_W;  
    double* d_Y;  
    CHECK_CUDA(cudaMalloc((void**)&d_W, sizeof(double) * m * n));  
    CHECK_CUDA(cudaMalloc((void**)&d_Y, sizeof(double) * m * n));  

    // 执行 ReconstructWY  
    ReconstructWY(d_Q, m, n, d_W, d_Y, handles);  

    // 验证WY' =? I -Q
#ifdef MYDEBUG
    double normQ = Dnorm(m, n, d_Qori);   
    // printf("normQ: %f\n", normQ);
    double* d_Y1;
    CHECK_CUDA(cudaMalloc(&d_Y1, sizeof(double) * n * n));  
    int threadsPerBlock = 256;  
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;   
    copyFirstRowsColumnMajor<<<blocksPerGrid, threadsPerBlock>>>(d_Y, d_Y1, m, n, n);
    // printf("d_Y1:\n");
    // printDeviceMatrixV2(d_Y1, n, n, n);

    // checkBackwardError(m, n, d_Q, m, d_Q, m, d_R, n);
    double done = 1.0;  
    double dnegone = -1.0;  
    CHECK_CUBLAS(cublasDgemm(handles.blas_handle, CUBLAS_OP_N, CUBLAS_OP_T,  
                            m, n, n,  
                            &done, d_W, m,  
                            d_Y1, n,  
                            &dnegone, d_Qori, m));
    double normRes = Dnorm(m, n, d_Qori);  
    // printf("normRes: %f\n", normRes);
    printf("ReconstructWY Backward error: ||WY - (I - Q)||/(||I - Q||) = %.6e\n", normRes / normQ);  

#endif

    // 释放设备内存  
    CHECK_CUDA(cudaFree(d_A));  
    CHECK_CUDA(cudaFree(d_Q));  
    CHECK_CUDA(cudaFree(d_R));  

    return 0;  
}


*/