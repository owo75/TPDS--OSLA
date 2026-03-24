#pragma once
#include <cublas_v2.h>  
#include <cuda_runtime_api.h>

#include <cassert>

#pragma message("Including TallShinnyQR.h")  

#define TSQR_BLOCK_SIZE 256
#define TSQR_BLOCK_DIM_Y 32
#define TSQR_BLOCK_DIM_X 32
#define TSQR_NUM_DATA_ROW 8

template <typename T>
struct shared_memory;
template <>
struct shared_memory<float> {
    __device__ static float *get_pointer() {
        extern __shared__ float shared_mem_float[];
        return shared_mem_float;
    }
};
template <>
struct shared_memory<double> {
    __device__ static double *get_pointer() {
        extern __shared__ double shared_mem_double[];
        return shared_mem_double;
    }
};

#pragma once
template <typename T>
static __inline__ __device__ T warpAllReduceSum(T val) {
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename T>
__global__ void tsqr_kernel(int m, int n, T *A, int lda, T *R, int ldr) {
    // 创建shared memory，让整个block的线程能够进行数据共享
    shared_memory<T> shared;
    T *shared_A = shared.get_pointer();

    int ldsa = TSQR_BLOCK_SIZE;

    const int thread_idx_x = threadIdx.x;
    const int thread_idx_y = threadIdx.y;
    const int block_idx_x = blockIdx.x;

    int block_size = min(TSQR_BLOCK_SIZE, m - block_idx_x * TSQR_BLOCK_SIZE);

    A = A + block_idx_x * TSQR_BLOCK_SIZE;
    R = R + block_idx_x * n;

    // 每个线程处理的数据个数
    int num_data_col = (n + TSQR_BLOCK_DIM_Y - 1) / TSQR_BLOCK_DIM_Y;

    T acc[TSQR_NUM_DATA_ROW];

    // 假定n=N=32，每一个线程拷贝2列
#pragma unroll
    for (int k = 0; k < TSQR_NUM_DATA_ROW; ++k) {
        int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
        if (row_idx < block_size) {
            for (int h = 0; h < num_data_col; ++h) {
                int col_idx = thread_idx_y + h * TSQR_BLOCK_DIM_Y;
                if (col_idx < n) {
                    shared_A[row_idx + col_idx * ldsa] =
                        A[row_idx + col_idx * lda];
                }
            }
        }
    }

    // 需要进行整个block的同步，应该只需要1个lane进行同步就行---需要思考一下
    // __syncwarp();

    T q[TSQR_NUM_DATA_ROW];
    // 进行HouseHolder分解，先计算HouseHolder向量
    // HouseHolder向量的求法如下:1、u=x/norm(x); 2、u(1)= u(1)+sign(u(1));
    // 3、u=u/sqrt(abs(u(1)))
    for (int cols = 0; cols < n; cols++) {
        // 先计算HouseHolder向量
        // HouseHolder向量的求法如下:1、u=x/norm(x); 2、u(1)= u(1)+sign(u(1));
        // 3、u=u/sqrt(abs(u(1)))
        T nu = 0.0;
        if (thread_idx_y == cols % TSQR_BLOCK_DIM_Y) {
            // 0.求normx
            // 是将下面的循环体进行展开，提高效率，所以需要acc[dataNum]
#pragma unroll
            for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                acc[k] = 0.0;
                int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
                // if条件中，前部部分是为了防止最后一个block中线程行越界；后半部分在计算HouseHolder向量是只计算对角线一下的元素
                if (row_idx >= cols && row_idx < block_size) {
                    q[k] = shared_A[row_idx + cols * ldsa];
                    acc[k] = q[k] * q[k];
                }
                nu += acc[k];
            }

            // 需要将1个lane中所有线程求出的norm_squre加到一起,同时进行同步
            T norm_x_squre = warpAllReduceSum(nu);
            T norm_x = sqrt(norm_x_squre);

            // 1、求u=x/norm(x);
            T scale = 1.0 / norm_x;
#pragma unroll
            for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
                if (row_idx >= cols && row_idx < block_size) {
                    q[k] *= scale;
                }
            }

            int thread_idx = cols % TSQR_BLOCK_DIM_X;
            int thread_off = cols / TSQR_BLOCK_DIM_X;
            T u1 = 0;
            if (thread_idx_x == thread_idx) {
                q[thread_off] += (q[thread_off] >= 0) ? 1 : -1;
                u1 = q[thread_off];
                R[cols + cols * ldr] = (u1 >= 0) ? -norm_x : norm_x;
            }
            u1 = __shfl_sync(0xFFFFFFFF, u1, thread_idx);

            // 3、u=u/sqrt(abs(u(1))),计算HouseHolder向量
            scale = 1 / (sqrt(abs(u1)));
#pragma unroll
            for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
                if (row_idx >= cols && row_idx < block_size) {
                    shared_A[row_idx + cols * ldsa] = q[k] * scale;
                }
            }
        }

        __syncthreads();

        // 用HouseHolder向量去更新HouseHolder向量所在列后面的所有列
        // 因为(I-uu')x=x-uu'x，先计算u'x，在计算x-uu'x
        // 每个线程按列需要处理多个列
        for (int h = 0; h < num_data_col; h++) {
            int opCols = thread_idx_y + h * TSQR_BLOCK_DIM_Y;

            // 只更新当前列后面的列
            if (cols < opCols && opCols < n) {
                nu = 0.0;
                // 先计算u'x
#pragma unroll
                for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                    acc[k] = 0.0;
                    int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
                    // if条件中，前部部分是为了防止最后一个block中线程行越界；后半部分在计算HouseHolder向量是只计算对角线一下的元素
                    if (row_idx >= cols && row_idx < block_size) {
                        q[k] = shared_A[row_idx + cols * ldsa];
                        acc[k] = q[k] * shared_A[row_idx + opCols * ldsa];
                    }
                    nu += acc[k];
                }
                T utx = warpAllReduceSum(nu);

                // 计算x-uu'x
#pragma unroll
                for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                    int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
                    // if条件中，前部部分是为了防止最后一个block中线程行越界；后半部分在计算HouseHolder向量是只计算对角线一下的元素
                    if (row_idx >= cols && row_idx < block_size) {
                        shared_A[row_idx + opCols * ldsa] -= utx * q[k];
                    }
                }
            }
        }
    }

    __syncthreads();
    // 此时已经完成HouseHolder更新，在AA中存放着HouseHolder向量和R矩阵的上三角部分,RR中存放在对角线元素

    // 获得R矩阵，将AA的上三角部分拷贝到R中
    // 以R矩阵来进行循环
    int rRowDataNum = (n + (TSQR_BLOCK_DIM_X - 1)) / TSQR_BLOCK_DIM_X;
    for (int h = 0; h < num_data_col; h++) {
        int opCols = thread_idx_y + h * TSQR_BLOCK_DIM_Y;

        if (opCols >= n) continue;

#pragma unroll
        for (int k = 0; k < rRowDataNum; k++) {
            int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
            if (row_idx < opCols && row_idx < n) {
                R[row_idx + opCols * ldr] = shared_A[row_idx + opCols * ldsa];
                shared_A[row_idx + opCols * ldsa] = 0.0;
            }
            if (row_idx > opCols && row_idx < n) {
                R[row_idx + opCols * ldr] = 0.0;
            }
        }
    }

    // 来求Q，使用的方法是Q=(I-uu')Q, 所以对于Q的一列而言q=(I-uu')q，计算q-uu'q
    // q表示是Q矩阵的1列
    for (int h = 0; h < num_data_col; h++) {
        // 1、构造出每个线程需要处理的Q矩阵的一列q的一部分
        int opCols = thread_idx_y + h * TSQR_BLOCK_DIM_Y;

        if (opCols >= n) continue;

#pragma unroll
        for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
            int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
            if (row_idx == opCols) {
                q[k] = 1.0;
            } else {
                q[k] = 0.0;
            }
        }

        __syncwarp();

        for (int cols = n - 1; cols >= 0; cols--) {
            // 这个判断没有问题，很经典，实际上不带这个判断也是正确的。这个判断是利用矩阵特点对矩阵乘法的一种优化
            // 因为Q_k-1=(I-u_k-1*u_k-1')*Q_k-2也是一个左上角是单位矩阵，右下角是一个k-1xk-1的矩阵，其他部分都是0；
            // 而I-uk*uk'也是一个左上角是单位矩阵，右下角是一个kxk的矩阵，其他部分为0；所以两者相乘只影响后面大于等于k的列
            if (opCols >= cols) {
                // 2、计算u'q
                T nu = 0.0;
#pragma unroll
                for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                    acc[k] = 0.0;
                    int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
                    if (row_idx < block_size) {
                        acc[k] = shared_A[row_idx + cols * ldsa] * q[k];
                        nu += acc[k];
                    }
                }

                T utq = warpAllReduceSum(nu);

                // 3.计算q-uu'q
#pragma unroll
                for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                    int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
                    if (row_idx < block_size) {
                        q[k] -= utq * shared_A[row_idx + cols * ldsa];
                    }
                }

                __syncwarp();
            }
        }

        // 4.把计算出来的q拷贝到A中
#pragma unroll
        for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
            int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
            if (row_idx < block_size) {
                A[row_idx + opCols * lda] = q[k];
            }
        }
    }
}

template __global__ void tsqr_kernel<float>(int m, int n, float *A, int lda,
                                            float *R, int ldr);
template __global__ void tsqr_kernel<double>(int m, int n, double *A, int lda,
                                             double *R, int ldr);

template <typename T>
void tsqr_func(cublasHandle_t cublas_handle, cudaDataType_t cuda_data_type,
               cublasComputeType_t cublas_compute_type, int share_memory_size,
               int m, int n, T *A, int lda, T *R, int ldr, T *work,
               int ldwork) {
    // 一个block最大为32x32，一个block中的thread可以使用共享内存进行通信，
    //  所以使用一个block处理一个最大为<TSQR_BLOCK_SIZE,N>的矩阵块，并对它进行QR分解
    dim3 blockDim(TSQR_BLOCK_DIM_X, TSQR_BLOCK_DIM_Y);

    // 1.如果m<=TSQR_BLOCK_SIZE,就直接调用核函数进行QR分解
    if (m <= TSQR_BLOCK_SIZE) {
        // 调用核函数进行QR分解
        // 分解后A矩阵中存放的是Q矩阵，R矩阵中存放的是R矩阵
        tsqr_kernel<T>
            <<<1, blockDim, share_memory_size>>>(m, n, A, lda, R, ldr);
        cudaDeviceSynchronize();
        return;
    }

    // 2.使用按列进行分段的方式进行QR分解
    // 2.1 把瘦高矩阵进行按列分段
    int blockNum = (m + TSQR_BLOCK_SIZE - 1) / TSQR_BLOCK_SIZE;

    // 2.2直接创建这么多个核函数进行QR分解,A中存放Q, work中存放R
    tsqr_kernel<T>
        <<<blockNum, blockDim, share_memory_size>>>(m, n, A, lda, work, ldwork);

    // 2.3再对R进行QR分解,也就是对work进行递归调用此函数
    tsqr_func<T>(cublas_handle, cuda_data_type, cublas_compute_type,
                 share_memory_size, blockNum * n, n, work, ldwork, R, ldr,
                 work + n * ldwork, ldwork);

    // 3.求出最终的Q，存放到A中
    // 注意这里使用了一个batch乘积的方法，是一个非常有趣的思想,需要结合瘦高矩阵的分块矩阵理解，非常有意思
    T tone = 1.0, tzero = 0.0;
    cublasGemmStridedBatchedEx(
        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, TSQR_BLOCK_SIZE, n, n, &tone,
        A, cuda_data_type, lda, TSQR_BLOCK_SIZE, work, cuda_data_type, ldwork,
        n, &tzero, A, cuda_data_type, lda, TSQR_BLOCK_SIZE, m / TSQR_BLOCK_SIZE,
        cublas_compute_type, CUBLAS_GEMM_DEFAULT);

    // 3.2如果m/M还有剩余的话，还需要计算最后一个块的Q进行乘法计算，才能得到最终的Q
    int mm = m % TSQR_BLOCK_SIZE;
    if (0 < mm) {
        cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, mm, n, n, &tone,
                     A + (m - mm), cuda_data_type, lda,
                     work + (m / TSQR_BLOCK_SIZE * n), cuda_data_type, ldwork,
                     &tzero, A + (m - mm), cuda_data_type, lda,
                     cublas_compute_type, CUBLAS_GEMM_DEFAULT);
    }
}
template void tsqr_func<float>(cublasHandle_t cublas_handle,
                               cudaDataType_t cuda_data_type,
                               cublasComputeType_t cublas_compute_type,
                               int share_memory_size, int m, int n, float *A,
                               int lda, float *R, int ldr, float *work,
                               int ldwork);
template void tsqr_func<double>(cublasHandle_t cublas_handle,
                                cudaDataType_t cuda_data_type,
                                cublasComputeType_t cublas_compute_type,
                                int share_memory_size, int m, int n, double *A,
                                int lda, double *R, int ldr, double *work,
                                int ldwork);

// 注意M必须<=256,N必须<=32
// 另外n必须<=N
template <typename T>
void tsqr(cublasHandle_t cublas_handle, int m, int n, T *A, int lda, T *R,
          int ldr, T *work, int ldwork) {
    assert(m >= n);
    assert(m % n == 0);
    assert(((m % TSQR_BLOCK_SIZE) % n) == 0);
    static_assert(TSQR_BLOCK_SIZE % TSQR_BLOCK_DIM_X == 0);
    static_assert(TSQR_BLOCK_DIM_X * TSQR_NUM_DATA_ROW == TSQR_BLOCK_SIZE);

    cudaDataType_t cuda_data_type;
    cublasComputeType_t cublas_compute_type;

    if (std::is_same<T, double>::value) {
        cuda_data_type = CUDA_R_64F;
        cublas_compute_type = CUBLAS_COMPUTE_64F;
    } else if (std::is_same<T, float>::value) {
        cuda_data_type = CUDA_R_32F;
        cublas_compute_type = CUBLAS_COMPUTE_32F;
    } else if (std::is_same<T, half>::value) {
        cuda_data_type = CUDA_R_16F;
        cublas_compute_type = CUBLAS_COMPUTE_16F;
    }

    int share_memory_size = TSQR_BLOCK_SIZE * n * sizeof(T);
    cudaFuncSetAttribute(tsqr_kernel<T>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         share_memory_size);

    tsqr_func(cublas_handle, cuda_data_type, cublas_compute_type,
              share_memory_size, m, n, A, lda, R, ldr, work, ldwork);
}

template void tsqr<float>(cublasHandle_t cublas_handle, int m, int n, float *A,
                          int lda, float *R, int ldr, float *work, int ldwork);
template void tsqr<double>(cublasHandle_t cublas_handle, int m, int n,
                           double *A, int lda, double *R, int ldr, double *work,
                           int ldwork);