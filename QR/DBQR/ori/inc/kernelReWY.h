#pragma once

#include <iostream>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iomanip> // 用于控制输出格式 
#include <fstream>  
#include <curand.h>
#include "../utils/inc/myutils.h"
#include "../utils/inc/errorcheck.h"

// #define MYKERREWYDEBUG
// #define KerWYTIME

#define kerWYMAX_N 32
#define kerWY_TRSM_BLOCKDIM 64
//-----------------------------------------------
// kernelIminusQ_inplace
// 并行将 Q -> (I - Q)
//-----------------------------------------------
__global__ void kernelIminusQ_inplace(float* d_Q,
                                      int ldq,
                                      int m, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y;
    if (row < m && col < n) {
        float val = d_Q[col * ldq + row];
        float valI = (row == col) ? 1.0f : 0.0f;
        d_Q[col * ldq + row] = valI - val;
    }
}
//-----------------------------------------------
// kernelLUDecompSharedV1
// 并行进行LU分解
// 矩阵大小固定为32 * 32，采用一个（32，32）的block进行计算
//-----------------------------------------------

__global__ void kernelLUDecompSharedV1(float* d_Q,  int ldq,
                                     float* d_Y,  int ldy,
                                     float* d_U,
                                     int n)
{
    __shared__ float tile[kerWYMAX_N][kerWYMAX_N];
    int row = threadIdx.x;
    int col = threadIdx.y;
    if (row < n && col < n) {
        tile[row][col] = d_Q[col*ldq + row]; // 合并访问
    }
    __syncthreads();

    // 执行简易LU(不含pivot)
    for (int k = 0; k < n - 1; k++) {
        __syncthreads();
        if (row > k && row < n && col == k) {
            tile[row][col] /= tile[k][k];
        }
        __syncthreads();
        if (row > k && row < n && col > k && col < n) {
            tile[row][col] -= tile[row][k] * tile[k][col];
        }
    }
    __syncthreads();

    // 写 L->d_Y，U->d_U
    if (row < n && col < n) {
        float val = tile[row][col];
        if (row > col) {
            // L区域
            d_Y[col*ldy + row] = val;
            d_U[col*n + row]   = 0.f; 
        } else if (row == col) {
            d_Y[col*ldy + row] = 1.f;
            d_U[col*n + row]   = val;
        } else {
            d_Y[col*ldy + row] = 0.f;
            d_U[col*n + row]   = val;
        }
    }
}
// V1和V2的区别主要在于共享内存的存储方式不同,V2的耗时更短
__global__ void kernelLUDecompSharedV2(float* d_Q, int ldq,  
                                    float* d_Y, int ldy,  
                                    float* d_U,  
                                    int n)  
{  
    // printf("进入了kernelLUDecompSharedV2\n");
    __shared__ float tile[kerWYMAX_N][kerWYMAX_N];  
    int row = threadIdx.x;  
    int col = threadIdx.y;  
    
    // 加载数据到共享内存（列主序）  
    if (row < n && col < n) {  
        tile[col][row] = d_Q[col*ldq + row];  
    }  
    __syncthreads();  

    // 执行LU分解  
    for (int k = 0; k < n - 1; k++) {  
        __syncthreads();  
        if (row > k && row < n && col == k) {  
            tile[col][row] /= tile[k][k];  
        }  
        __syncthreads();  
        if (row > k && row < n && col > k && col < n) {  
            tile[col][row] -= tile[k][row] * tile[col][k];  
        }  
    }  
    __syncthreads();  
    // // // 打印调试信息  
        // if (row == 0 && col == 0) {  // 仅使用一个线程负责打印，避免重复打印  
        //     printf("Matrix Q after LU\n");  
        //     for (int i = 0; i < n; i++) {  
        //         for (int j = 0; j < n; j++) {  
        //             printf("%6.3f ", d_Q[j * ldq + i]);  // 按列主序打印  
        //         }  
        //         printf("\n");  
        //     }  
        // }    

    // 写回结果  
    if (row < n && col < n) {  
        float val = tile[col][row];  
        if (row > col) {  
            // L区域  
            d_Y[col*ldy + row] = val;  
            d_U[col*n + row]   = 0.f;  
        } else if (row == col) {  
            d_Y[col*ldy + row] = 1.f;  
            d_U[col*n + row]   = val;  
        } else {  
            d_Y[col*ldy + row] = 0.f;  
            d_U[col*n + row]   = val;  
        }  
    }  
}

//-----------------------------------------------
// kernelLUDecompGlobal: 并行进行LU分解
// 矩阵大小固定为32 * 32，采用一个（32，32）的block进行计算
// 取消了共享内存，全部在dQ上面进行计算
// 问题：误差有点大，但是每一步结果又是正确的
//-----------------------------------------------
__global__ void kernelLUDecompGlobal(float* d_Q, int ldq,  
                                     float* d_Y, int ldy,  
                                     float* d_U,  
                                     int n)  
{  
    int row = threadIdx.x;  // 当前线程的行索引  
    int col = threadIdx.y;  // 当前线程的列索引  

    // 执行 LU 分解（直接对 d_Q 进行操作）  
    for (int k = 0; k < n - 1; k++) {  
        __syncthreads();  // 确保每次步骤前的同步  
        
        // 计算 L 矩阵的第 k 列（除数操作）  
        if (row > k && row < n && col == k) {   
            d_Q[k * ldq + row] /= d_Q[k * ldq + k];  // 列主序索引  
        }  
        __syncthreads();  

        // 对 U 矩阵的第 k 行剩余元素更新  
        if (row > k && row < n && col > k && col < n) {   
            d_Q[col * ldq + row] -= d_Q[k * ldq + row] * d_Q[col * ldq + k];  
        }  
        __syncthreads();  

        // // // 打印调试信息  
        // if (row == 0 && col == 0) {  // 仅使用一个线程负责打印，避免重复打印  
        //     printf("Matrix Q after k=%d\n", k);  
        //     for (int i = 0; i < n; i++) {  
        //         for (int j = 0; j < n; j++) {  
        //             printf("%6.3f ", d_Q[j * ldq + i]);  // 按列主序打印  
        //         }  
        //         printf("\n");  
        //     }  
        // }    
    }  
    // 执行结果写回 d_Y 和 d_U  
    if (row < n && col < n) {  
        float val = d_Q[col * ldq + row];  // 按列主序获取元素  
        
        if (row > col) {  
            d_Y[col * ldy + row] = val;  
            d_U[col * n + row]   = 0.f;  
        }   
        else if (row == col) {  
            d_Y[col * ldy + row] = 1.f;  
            d_U[col * n + row]   = val;  
        }   
        else {   
            d_Y[col * ldy + row] = 0.f;  
            d_U[col * n + row]   = val;  
        }  
    }  
}  


//-----------------------------------------------
// kernelTrsmRightUpperWARP
// 处理 Q 的下部 (m-n)×n 右侧乘 inv(U), 直接写到 Y[n..m-1]行
// 共享内存载入n×n的U
//
// 优化点1：将共享内存改为常量只读内存会不会好点
// 优化点2：U是个上三角，只要搬运一半就行了
// 优化点：stride的设置，现在可能还不是很合理
//-----------------------------------------------
__global__ void kernelTrsmRightUpperWARP(const float* __restrict__ d_U,
                                         int ldu,
                                         float* d_Q, // Q底部
                                         int ldq,
                                         float* d_Y, // 目标: (m×n), 写[m-n..m-1]行。
                                         int ldy,
                                         int m, int n, int offsetRow)
{
    int rowIdxInBlock = threadIdx.x; 
    int globalRow = blockIdx.x * blockDim.x + rowIdxInBlock;
    if (globalRow >= (m - offsetRow)) return;

    int actualRow = globalRow + offsetRow; // 在Q/Y中的绝对行号

    extern __shared__ float sU[];
    int t = threadIdx.x; 
    // 这里比较特殊，因为最后一个块可能没有blockDim.x个线程，所以就会导致搬运出错
    int stride = (m - 32 -  blockIdx.x * blockDim.x) > blockDim.x ? blockDim.x : (m - 32 -  blockIdx.x * blockDim.x); 
    // int stride = blockDim.x;
    for(int i=t; i<n*n; i+=stride){
        sU[i] = __ldg(&d_U[i]);         //dU是列主序，所以sU也就是列主序
    }
    __syncthreads();

    // 读 Q(actualRow, 0..n-1)
    float local[kerWYMAX_N];              ///////////////怎么确定用的是寄存器还是全局内存
    for (int c=0; c<n; c++){
        local[c] = d_Q[c*ldq + actualRow];
    }

    // 右上三角TRSM => local = local * inv(sU)
    // 下面做简单循环展开
    for (int c=0; c<n; c++){
        float diag = sU[c*ldu + c];
        local[c] /= diag;
        // if(diag == 0)
        //     printf("当c = %d时出现了0\n", c);

        #pragma unroll 4
        for (int k=c+1; k<n; k++){
            local[k] -= local[c]*sU[k*ldu + c]; //这里应该用sU[c*ldu + k]还是sU[k*ldu + c]
        }
    }

    // 写回到 Y(actualRow, 0..n-1)
    for (int c=0; c<n; c++){
        d_Y[c*ldy + actualRow] = local[c];
    }
}

//-----------------------------------------------
// kernelTrsmRightLowerT
// W = Q × inv(L^T)
// L^T是上三角 => 逐列回代
// blockDim.x=32/64 => 1个warp处理一行
// 共享内存载入L^T
// 优化点：如果采用只读内存或者提前将L进行转置，是不是可以将两个函数进行统一
//-----------------------------------------------
__global__ void kernelTrsmRightLowerT(const float* __restrict__ d_L,
                                                 int ldL,
                                                 float* d_W,
                                                 int ldW,
                                                 float* d_Q,
                                                 int ldq,
                                                 int m, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;

    // 加载 L^T => n×n
    extern __shared__ float sLT[];
    int t = threadIdx.x;
    // int stride = blockDim.x;   // 需要考虑不满足一个blockDim.x的情况
    int stride = (m - blockIdx.x * blockDim.x) > blockDim.x ? blockDim.x : m - blockIdx.x * blockDim.x; 
    for(int i=t; i<n*n; i+=stride){
        // 虽然是L的转置，但是直接以下三角的形式存入共享内存，在计算时候进行调整
        // 第一次写这里有问题，因为d_L本质上是d_Y，所以写入的时候需要考虑到ldy，
        // 最好改成二维的内存对应起来
        sLT[i] = __ldg(&d_L[(i/32) * ldL + (i%32)]);
    }
    __syncthreads();

    // 读W[row, :]
    float local[kerWYMAX_N];
    for(int c=0; c<n; ++c){
        local[c] = d_Q[c*ldq + row];
    }

    // 上三角回代: local = local * inv(L^T)
    // 这里共享内存中元素位置和上一个函数就不相同
    for(int c=0; c<n; c++){
        float diag = sLT[c*n + c];
        local[c] /= diag;

        #pragma unroll 4
        for(int k=c+1; k<n; k++){
            local[k] -= local[c]*sLT[c*n + k];  // (c,k) --> (k,c)
        }
    }

    // 写回
    for(int c=0; c<n; ++c){
        d_W[c*ldW + row] = local[c];
    }
}

//-----------------------------------------------
// 外部接口: ReconstructWY_Advanced
// - Q <- (I - Q)
// - LU(Q[0..n-1,0..n-1]) => L->Y[0..n-1], U->d_U
// - Q[n..m-1,0..n-1]*inv(U) => Y[n..m-1], fuse写
// - W = Q*inv(L^T)
//-----------------------------------------------
void ReconstructWYKernel(
    float* d_Q,  // in/out: m×n, 被覆盖 => (I-Q)
    int ldq,
    float* d_W,  // out: m×n
    int ldw,
    float* d_Y,  // out: m×n, 前n行为 L, 后(m-n)行为 L2
    int ldy,
    float* d_U,  // out: n×n => U
    int m, int n)
{
    if(n>kerWYMAX_N){
        std::cerr<<"[Error] n>"<<kerWYMAX_N<<" not supported.\n";
        return;
    }
    // std::ofstream file("kernelReWY.csv", std::ios::trunc);  
    // file.close(); 
    // std::string filename = "kernelReWY.csv";
    // std::string matrixname = "matrixname";
    
    //--------------------------------------------------
    // 1) Q <- (I - Q)
    //--------------------------------------------------
    {
        // warp级内核: blockDim=(warpSize,1), 这里可以尝试32或64
#ifdef MYKERREWYDEBUG
        matrixname = "I-Q前";
        writeMatrixToCsvV2(d_Q, ldq, m, n, filename, matrixname);
        printf("I-Q前\n");
        printDeviceMatrixV2(d_Q,ldq,m,n);
#endif
#ifdef kerWYTIME
        startTimer();
#endif
        dim3 block(32,1);
        dim3 grid((m+block.x-1)/block.x, n);
        kernelIminusQ_inplace<<<grid, block>>>(d_Q, ldq, m, n);
        // CHECK_CUDA(cudaDeviceSynchronize());
        // float IminusQTime = stopTimer();

#ifdef MYKERREWYDEBUG
        matrixname = "I-Q后";
        writeMatrixToCsvV2(d_Q, ldq, n, n, filename, matrixname);
        printf("I-Q后\n");
        printDeviceMatrixV2(d_Q,ldq,m,n);
#endif
    }

    //--------------------------------------------------
    // 2) 前 n×n 做LU => L写Y前n行, U写d_U
    //--------------------------------------------------
    {   
        dim3 blockLU(n,n); //这里其实就限制了n只能为32，n > 32时会超过线程块中线程最大数量限制
        dim3 gridLU(1);
        // printf("n = %d\n", n);
#ifdef kerWYTIME
        startTimer();
#endif
        kernelLUDecompSharedV2<<<gridLU, blockLU>>>(d_Q, ldq,
                                                  d_Y, ldy,
                                                  d_U,
                                                  n);
        // CHECK_CUDA(cudaDeviceSynchronize());
        // float LUTime = stopTimer();
        // printf("LU的时间为: %.4f(ms)\n",LUTime);
#ifdef MYKERREWYDEBUG
        matrixname = "A1";
        writeMatrixToCsvV2(d_Q, ldq, n, n, filename, matrixname);
        matrixname = "L";
        writeMatrixToCsvV2(d_Y, ldy, n, n, filename, matrixname);
        matrixname = "U";
        writeMatrixToCsvV2(d_U, n, n, n, filename, matrixname);

        printf("A1\n");  
        printDeviceMatrixV2(d_Q, ldq, n, n);  
        printf("L\n");  
        printDeviceMatrixV2(d_Y, ldy, n, n);  
        printf("U\n");  
        printDeviceMatrixV2(d_U, n, n, n);  
#endif
    }

    //--------------------------------------------------
    // 3) 对 Q 的下(m-n)行右侧乘inv(U)，写入 Y 的 [n..m-1]
    //--------------------------------------------------
    int rowsA2 = m - n;
    if(rowsA2>0){
        // warpSize=32 => blockDim.x=32, 共享内存n*n
        // 这里的blockSize需要尝试一下: 貌似不同的blocksize区别都不大， 不确定是不是计时函数的问题
        int blockSize  = kerWY_TRSM_BLOCKDIM;  
        dim3 gridTRSM((rowsA2+blockSize-1)/blockSize, 1);
        size_t shmBytes = n*n*sizeof(float);
#ifdef kerWYTIME
        startTimer();
#endif
        kernelTrsmRightUpperWARP<<<gridTRSM, blockSize, shmBytes>>>(
            d_U, n,
            d_Q, ldq,
            d_Y, ldy,
            m, n, n  // offsetRow=n => 处理Q底部
        );
        // CHECK_CUDA(cudaDeviceSynchronize());
        // float Y2Time = stopTimer();
        // printf("trsm计算Y2的时间为: %.4f(ms)\n", Y2Time);
    } else {
        // m==n => 已经写好 L
    }
#ifdef MYKERREWYDEBUG
        matrixname = "Y2";
        writeMatrixToCsvV2(d_Y + n, ldy, m - n, n, filename, matrixname);
        matrixname = "A2";
        writeMatrixToCsvV2(d_Q + n, ldq, m - n, n, filename, matrixname);
        matrixname = "U";
        writeMatrixToCsvV2(d_U, n, n, n, filename, matrixname);

        printf("Y2\n");  
        printDeviceMatrixV2(d_Y + n, ldy, m - n, n);  
        printf("A2\n");  
        printDeviceMatrixV2(d_Q + n, ldq, m - n, n);  
        printf("U\n");  
        printDeviceMatrixV2(d_U, n, n, n);  
#endif

    //--------------------------------------------------
    // 4) 计算 W = (I-Q) * inv(L^T)
    //    此时 Q 已经是(I-Q)
    //--------------------------------------------------
    {
        int blockSize = kerWY_TRSM_BLOCKDIM; 
        dim3 gridW((m+blockSize-1)/blockSize,1);
        size_t shmBytes = n*n*sizeof(float);
#ifdef kerWYTIME
        startTimer();
#endif
        kernelTrsmRightLowerT<<<gridW, blockSize, shmBytes>>>(
            d_Y, ldy,
            d_W, ldw,
            d_Q, ldq,
            m, n
        );
        // CHECK_CUDA(cudaDeviceSynchronize());
        // float WTime = stopTimer();
        // printf("trsm计算W的时间为: %.4f(ms)\n", WTime);
#ifdef MYKERREWYDEBUG
        matrixname = "d_A";
        writeMatrixToCsvV2(d_Q, ldq, m, n, filename, matrixname);
        matrixname = "d_Y";
        writeMatrixToCsvV2(d_Y, ldy, n, n, filename, matrixname);
        matrixname = "d_W";
        writeMatrixToCsvV2(d_W, ldw, m, n, filename, matrixname);

        printf("d_A\n");  
        printDeviceMatrixV2(d_Q, ldq, m, n);  
        printf("d_Y\n");  
        printDeviceMatrixV2(d_Y, ldy, n, n);  
        printf("d_W\n");  
        printDeviceMatrixV2(d_W, ldw, m, n);  
#endif
    }
}

