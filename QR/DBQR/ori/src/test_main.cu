// test_main.cu

#include <iostream>
#include <string>
#include <vector>

// 包含我们自己的库和工具
#include "DBQR.h"
#include "utils.h" // 确保这里有您所有辅助函数的声明

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "用法: " << argv[0] << " m n nb b" << std::endl;
        return -1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int nb = atoi(argv[3]);
    int b = atoi(argv[4]);

    std::cout << "测试参数: m=" << m << ", n=" << n << ", nb=" << nb << ", b=" << b << std::endl;

    // --- 1. 分配内存 (和您原来的main函数一样) ---
    float* d_A;
    float* d_A_original;
    float* d_W;
    float* d_Y;
    float* d_Q;
    float* d_R;
    int lda = m;

    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeof(float) * m * n));
    CHECK_CUDA(cudaMalloc((void**)&d_A_original, sizeof(float) * m * n));
    CHECK_CUDA(cudaMalloc(&d_W, sizeof(float) * m * n));
    CHECK_CUDA(cudaMalloc(&d_Y, sizeof(float) * m * n));
    CHECK_CUDA(cudaMalloc(&d_Q, sizeof(float) * m * m));
    CHECK_CUDA(cudaMalloc(&d_R, sizeof(float) * n * n));
    
    // 初始化 cuBLAS
    cublasHandle_t cublas_handle;  
    CHECK_CUBLAS(cublasCreate(&cublas_handle));  

    // --- 2. 生成数据并计时 ---
    generateNormalMatrix(d_A, m, n, 0.0f, 1.0f);
    CHECK_CUDA(cudaMemcpy(d_A_original, d_A, sizeof(float) * m * n, cudaMemcpyDeviceToDevice));
    
    // 在这里计时，因为我们想测试整个库函数的性能
    startTimer();

    // --- 3. 调用我们的新库函数！ ---
    DBQR_IterativeQR(d_A, m, n, nb, b, d_W, d_Y, lda, cublas_handle);
    DBQR_ConstructQ(m, n, d_W, d_Y, d_Q, cublas_handle);
    DBQR_ExtractR(m, n, d_A, d_R);

    // 等待所有GPU操作完成
    CHECK_CUDA(cudaDeviceSynchronize());
    float totalTime = stopTimer();
    printf("DBQR 库函数总耗时 = %.4f (ms)\n", totalTime);

    // --- 4. 验证结果 (和您原来的main函数一样) ---
    std::cout << "\n--- 验证结果 ---" << std::endl;
    checkOtho(m, m, d_Q, m);  
    checkBackwardError(m, n, d_A_original, m, d_Q, m, d_R, n); // 注意这里R的lda是n
    std::cout << "--- 验证结束 ---\n" << std::endl;

    // --- 5. 释放内存 ---
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_A_original));
    CHECK_CUDA(cudaFree(d_W));
    CHECK_CUDA(cudaFree(d_Y));
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_R));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));

    return 0;
}
