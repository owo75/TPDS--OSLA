
#include "inc/myutils.h"

// 矩阵生成
void generateNormalMatrix(double* dA, long int m, long int n, double mean,
                          double stddev) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    int seed = 3000;
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    curandGenerateNormalDouble(gen, dA, long(m * n), mean, stddev);
}

// 矩阵上三角拷贝
__global__ void getU(int m, int n, double* A, int ldA, double* U, int ldU) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < m && j < n) {
        if (i > j)
            U[i + j * ldU] = 0;
        else
            U[i + j * ldU] = A[i + j * ldA];
    }
}

void launchKernel_getU(dim3 gridDim, dim3 blockDim, int m, int n, double* A,
                       int ldA, double* U, int ldU) {
    getU<<<gridDim, blockDim>>>(m, n, A, ldA, U, ldU);
}

// 正交性和后向误差检测
void checkBackwardError(int m, int n, double* A, int lda, double* Q, int ldq,
                        double* R, int ldr) {
    double normA = Dnorm(m, n, A);
    // printf("normA: %f\n", normA);
    double alpha = 1.0;
    double beta = -1.0;

    dgemm(m, n, n, Q, ldq, R, ldr, A, lda, alpha, beta);

    // 计算 ||A - QR|| / ||A||
    double normRes = Dnorm(m, n, A);
    // printf("normRes: %f\n", normRes);
    printf("Backward error: ||A-QR||/(||A||) = %.6e\n", normRes / normA);
}

// 二范数
double Dnorm(long int m, long int n, double* dA) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    double dn;
    int incx = 1;
    cublasDnrm2(handle, m * n, dA, incx, &dn);
    cublasDestroy(handle);
    return dn;
}
// 矩阵乘法
// C = AB + C
void dgemm(int m, int n, int k, double* dA, int lda, double* dB, int ldb,
           double* dC, int ldc, double alpha, double beta) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    double done = alpha;
    double dzero = beta;
    cublasStatus_t status =
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &done, dA, lda,
                    dB, ldb, &dzero, dC, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS Dgemm failed\n");
    }
    cublasDestroy(handle);
}

void checkOtho(long int m, long int n, double* Q, int ldq) {
    double* I;
    cudaMalloc(&I, sizeof(double) * n * n);

    // Define grid and block sizes
    dim3 grid((n + 31) / 32, (n + 31) / 32);
    dim3 block(32, 32);

    // Generate the identity matrix on the device
    setEye<<<grid, block>>>(I, n);
    cudaDeviceSynchronize();

    double dnegone = -1.0;
    double done = 1.0;

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Compute I - Q^T * Q
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, &dnegone, Q,
                 CUDA_R_64F, ldq, Q, CUDA_R_64F, ldq, &done, I, CUDA_R_64F, n,
                 CUDA_R_64F, CUBLAS_GEMM_DEFAULT);
    // Compute the norm of (I - Q^T * Q)
    double normRes = Dnorm(n, n, I);
    printf("orthogonal error: ||I - Q'*Q||/N = %.6e\n", normRes / n);
    cudaFree(I);
    cublasDestroy(handle);
}

__global__ void setEye(double* I, long int n) {
    // 获取当前线程的行和列索引
    long int row = blockIdx.y * blockDim.y + threadIdx.y;  // 行索引
    long int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列索引

    // 确保线程索引在矩阵维度范围内
    if (row < n && col < n) {
        if (row == col) {
            I[row * n + col] = 1.0;  // 对角线元素设为1
        } else {
            I[row * n + col] = 0.0;  // 其他元素设为0
        }
    }
}

__global__ void setEye(double* I, long int m, long int n) {
    // 获取当前线程的行和列索引
    long int row = blockIdx.y * blockDim.y + threadIdx.y;  // 行索引
    long int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列索引

    // 确保线程索引在矩阵维度范围内
    if (row < m && col < n) {
        if (row == col) {
            I[row * n + col] = 1.0;  // 对角线元素设为1
        } else {
            I[row * n + col] = 0.0;  // 其他元素设为0
        }
    }
}

// 计算I - Q
__global__ void IminusQ(long m, long n, double* Q, long ldq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // printf("come %d, %d, %d,\n", __LINE__, i, j);
    // __syncthreads();

    if (i < m && j < n) {
        if (i == j) {
            Q[i + j * ldq] = 1.0 - Q[i + j * ldq];
        } else {
            Q[i + j * ldq] = -Q[i + j * ldq];
        }

        // printf("come %d, %d, %d,\n", __LINE__, i, j);
        // __syncthreads();
    }
}

// float版本的实现

// 矩阵生成
void generateNormalMatrix(float* dA, long int m, long int n, float mean,
                          float stddev) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    int seed = 3000;
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    curandGenerateNormal(gen, dA, long(m * n), mean, stddev);
}

// 矩阵上三角拷贝
__global__ void getU(int m, int n, float* A, int ldA, float* U, int ldU) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < m && j < n) {
        if (i > j)
            U[i + j * ldU] = 0.0f;
        else
            U[i + j * ldU] = A[i + j * ldA];
    }
}

void launchKernel_getU(dim3 gridDim, dim3 blockDim, int m, int n, float* A,
                       int ldA, float* U, int ldU) {
    getU<<<gridDim, blockDim>>>(m, n, A, ldA, U, ldU);
}

// 正交性和后向误差检测
void checkBackwardError(int m, int n, float* A, int lda, float* Q, int ldq,
                        float* R, int ldr) {
    float normA = Snorm(m, n, A);
    float alpha = 1.0f;
    float beta = -1.0f;

    sgemm(m, n, n, Q, ldq, R, ldr, A, lda, alpha, beta);

    float normRes = Snorm(m, n, A);
    printf("Backward error: ||A-QR||/(||A||) = %.6e\n", normRes / normA);
}

// 二范数
float Snorm(long int m, long int n, float* dA) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float sn;
    int incx = 1;
    cublasSnrm2(handle, m * n, dA, incx, &sn);
    cublasDestroy(handle);
    return sn;
}

// 矩阵乘法
void sgemm(int m, int n, int k, float* dA, int lda, float* dB, int ldb,
           float* dC, int ldc, float alpha, float beta) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float done = alpha;
    float dzero = beta;
    cublasStatus_t status =
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &done, dA, lda,
                    dB, ldb, &dzero, dC, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS Sgemm failed\n");
    }
    cublasDestroy(handle);
}

void checkOtho(long int m, long int n, float* Q, int ldq) {
    float* I;
    cudaMalloc(&I, sizeof(float) * n * n);

    dim3 grid((n + 31) / 32, (n + 31) / 32);
    dim3 block(32, 32);

    setEye<<<grid, block>>>(I, n);
    cudaDeviceSynchronize();

    float snegone = -1.0f;
    float sone = 1.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, &snegone, Q,
                 CUDA_R_32F, ldq, Q, CUDA_R_32F, ldq, &sone, I, CUDA_R_32F, n,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

    float normRes = Snorm(n, n, I);
    printf("orthogonal error: ||I - Q'*Q||/N = %.6e\n", normRes / n);
    cudaFree(I);
    cublasDestroy(handle);
}

__global__ void setEye(float* I, long int n) {
    long int row = blockIdx.y * blockDim.y + threadIdx.y;
    long int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        if (row == col) {
            I[row * n + col] = 1.0f;
        } else {
            I[row * n + col] = 0.0f;
        }
    }
}

__global__ void setEye(float* I, long int m, long int n) {
    long int row = blockIdx.y * blockDim.y + threadIdx.y;
    long int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        if (row == col) {
            I[row * n + col] = 1.0f;
        } else {
            I[row * n + col] = 0.0f;
        }
    }
}

// 计算I - Q
__global__ void IminusQ(long m, long n, float* Q, long ldq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n) {
        if (i == j) {
            Q[i + j * ldq] = 1.0f - Q[i + j * ldq];
        } else {
            Q[i + j * ldq] = -Q[i + j * ldq];
        }
    }
}
// 将矩阵输出到文件中，支持文件追加写入
void writeMatrixToCsvV2(float* dA, long ldA, long rows, long cols,
                        const std::string& fileName,
                        const std::string& matrixName) {
    float matrix;

    // 以 "追加写入" 模式打开文件
    std::ofstream file(fileName, std::ios::app);

    if (file.is_open()) {
        // 如果需要，每次追加写入前可标明当前矩阵的名称
        file << matrixName << std::endl;

        // 设置浮点数输出固定为 4 位小数 + 列对齐
        file << std::fixed << std::setprecision(3);

        // 假设每列宽度固定为 10，可以调整此值以适配更宽或更小的输出
        const int columnWidth = 6;

        for (long i = 0; i < rows; i++) {
            for (long j = 0; j < cols; j++) {
                // 从设备内存中复制一个元素到主机
                cudaMemcpy(&matrix, dA + i + j * ldA, sizeof(float),
                           cudaMemcpyDeviceToHost);

                // 输出当前元素，使用宽度和右对齐
                file << std::right << std::setw(columnWidth) << matrix;

                // 如果不是当前行的最后一个元素，添加逗号对齐
                if ((cols - 1) != j) {
                    file << " ";
                }
            }
            // 每一行以换行符结束
            file << std::endl;
        }
        file.close();
        std::cout << "Appended matrix to " << fileName << std::endl;
    } else {
        std::cout << "Failed to open file: " << fileName << std::endl;
    }
}

// 将矩阵输出到文件中，支持文件追加写入
void writeMatrixToCsvV2(double* dA, long ldA, long rows, long cols,
                        const std::string& fileName,
                        const std::string& matrixName) {
    double matrix;

    // 以 "追加写入" 模式打开文件
    std::ofstream file(fileName, std::ios::app);

    if (file.is_open()) {
        // 如果需要，每次追加写入前可标明当前矩阵的名称
        file << matrixName << std::endl;

        // 设置浮点数输出固定为 4 位小数 + 列对齐
        file << std::fixed << std::setprecision(3);

        // 假设每列宽度固定为 10，可以调整此值以适配更宽或更小的输出
        const int columnWidth = 6;

        for (long i = 0; i < rows; i++) {
            for (long j = 0; j < cols; j++) {
                // 从设备内存中复制一个元素到主机
                cudaMemcpy(&matrix, dA + i + j * ldA, sizeof(double),
                           cudaMemcpyDeviceToHost);

                // 输出当前元素，使用宽度和右对齐
                file << std::right << std::setw(columnWidth) << matrix;

                // 如果不是当前行的最后一个元素，添加逗号对齐
                if ((cols - 1) != j) {
                    file << " ";
                }
            }
            // 每一行以换行符结束
            file << std::endl;
        }
        file.close();
        std::cout << "Appended matrix to " << fileName << std::endl;
    } else {
        std::cout << "Failed to open file: " << fileName << std::endl;
    }
}

// 将半精度转换为单精度
__global__ void h2s(int m, int n, __half* ah, int ldah, float* as, int ldas) {
    long int i = threadIdx.x + blockDim.x * blockIdx.x;
    long int j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i < m && j < n) {
        as[i + j * ldas] = __half2float(ah[i + j * ldah]);
    }
}

// 将单精度浮点数转换为半精度
__global__ void s2h(int m, int n, float* as, int ldas, __half* ah, int ldah) {
    long int i = threadIdx.x + blockDim.x * blockIdx.x;
    long int j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i < m && j < n) {
        ah[i + j * ldah] = __float2half(as[i + j * ldas]);
    }
}

// 将矩阵输出到文件中，支持 __half 类型矩阵，文件追加写入
void writeMatrixToCsvV2(__half* dA, long ldA, long rows, long cols,
                        const std::string& fileName,
                        const std::string& matrixName) {
    float matrix;  // 用于存储从半精度转换为单精度后的值

    // 以 "追加写入" 模式打开文件
    std::ofstream file(fileName, std::ios::app);

    if (file.is_open()) {
        // 如果需要，每次追加写入前可标明当前矩阵的名称
        file << matrixName << std::endl;

        // 设置浮点数输出固定为 4 位小数 + 列对齐
        file << std::fixed << std::setprecision(3);

        // 假设每列宽度固定为 10，可以调整此值以适配更宽或更小的输出
        const int columnWidth = 6;

        for (long i = 0; i < rows; i++) {
            for (long j = 0; j < cols; j++) {
                // 分步将 __half 数据从设备内存复制到主机内存，并转换为 float
                __half h_data;
                cudaMemcpy(&h_data, dA + i + j * ldA, sizeof(__half),
                           cudaMemcpyDeviceToHost);
                matrix = __half2float(h_data);  // 将半精度转为单精度浮点数

                // 输出当前元素，使用宽度和右对齐
                file << std::right << std::setw(columnWidth) << matrix;

                // 如果不是当前行的最后一个元素，添加空格用于对齐
                if ((cols - 1) != j) {
                    file << " ";
                }
            }
            // 每一行以换行符结束
            file << std::endl;
        }
        file.close();
        std::cout << "Appended matrix to " << fileName << std::endl;
    } else {
        std::cout << "Failed to open file: " << fileName << std::endl;
    }
}