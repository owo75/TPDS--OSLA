#include <iostream> 
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <cusolverDn.h>

#ifndef CHECK_Runtime
#define CHECK_Runtime(call)                                                     \
  do {                                                                          \
    cudaError_t _status = (call);                                               \
    if (_status != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %d (%s) at %s:%d\n",                          \
              (int)_status, cudaGetErrorString(_status), __FILE__, __LINE__);  \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)
#endif

#ifndef CHECK_Cublas
#define CHECK_Cublas(call)                                                      \
  do {                                                                          \
    cublasStatus_t _status = (call);                                            \
    if (_status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)_status, __FILE__,    \
              __LINE__);                                                        \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)
#endif

#ifndef CHECK_Curand
#define CHECK_Curand(call)                                                      \
  do {                                                                          \
    curandStatus_t _status = (call);                                            \
    if (_status != CURAND_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "cuRAND error %d at %s:%d\n", (int)_status, __FILE__,    \
              __LINE__);                                                        \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)
#endif

#ifndef CHECK_Cusolver
#define CHECK_Cusolver(call)                                                    \
  do {                                                                          \
    cusolverStatus_t _status = (call);                                          \
    if (_status != CUSOLVER_STATUS_SUCCESS) {                                   \
      fprintf(stderr, "cuSOLVER error %d at %s:%d\n", (int)_status, __FILE__,  \
              __LINE__);                                                        \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)
#endif

int n;
bool use_double = false;
bool debug_mode = false;

int parseArguments(int argc,char *argv[])
{
    if(argc < 2)
    {
        printf("Usage: %s n [float|double] [--debug]\n", argv[0]);
        return -1;
    }
    n = atoi(argv[1]);
    if (n <= 0) {
        printf("n must be positive\n");
        return -1;
    }
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "double") == 0 || strcmp(argv[i], "d") == 0) {
            use_double = true;
        } else if (strcmp(argv[i], "float") == 0 || strcmp(argv[i], "f") == 0) {
            use_double = false;
        } else if (strcmp(argv[i], "--debug") == 0 || strcmp(argv[i], "-d") == 0) {
            debug_mode = true;
        } else {
            printf("Unknown argument: %s\n", argv[i]);
            return -1;
        }
    }
    return 0;
}

template <typename T>
__global__
void addDiagonalShift(int m, T *a, int lda, T shift)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < m) {
		size_t idx = (size_t)i + (size_t)i * (size_t)lda;
		a[idx] += shift;
	}
}

template <typename T>
__device__ __forceinline__ T absVal(T x);

template <>
__device__ __forceinline__ float absVal<float>(float x) {
    return fabsf(x);
}

template <>
__device__ __forceinline__ double absVal<double>(double x) {
    return fabs(x);
}

template <typename T>
__global__
void symmetrizeMatrix(int n, T *a, int lda)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n || i >= j) {
        return;
    }
    size_t idx_ij = (size_t)i + (size_t)j * (size_t)lda;
    size_t idx_ji = (size_t)j + (size_t)i * (size_t)lda;
    T aij = a[idx_ij];
    T aji = a[idx_ji];
    T v = (aij + aji) * (T)0.5;
    a[idx_ij] = v;
    a[idx_ji] = v;
}

template <typename T>
__global__
void setDiagonalValue(int n, T *a, int lda, T diag_value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    size_t idx = (size_t)i + (size_t)i * (size_t)lda;
    a[idx] = diag_value;
}

template <typename T>
struct GpuOps;

template <>
struct GpuOps<float> {
    static curandStatus_t GenerateUniform(curandGenerator_t gen, float *data, size_t n_elem) {
        return curandGenerateUniform(gen, data, n_elem);
    }
    static cublasStatus_t Gemm(cublasHandle_t handle,
                               int n,
                               const float *alpha,
                               const float *B,
                               const float *beta,
                               float *A) {
        return cublasSgemm(handle,
                           CUBLAS_OP_N,
                           CUBLAS_OP_T,
                           n, n, n,
                           alpha,
                           B, n,
                           B, n,
                           beta,
                           A, n);
    }
    static cusolverStatus_t PotrfBufferSize(cusolverDnHandle_t handle, int n, float *A, int *Lwork) {
        return cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, n, A, n, Lwork);
    }
    static cusolverStatus_t Potrf(cusolverDnHandle_t handle,
                                  int n,
                                  float *A,
                                  float *work,
                                  int Lwork,
                                  int *devInfo) {
        return cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_LOWER, n, A, n, work, Lwork, devInfo);
    }
    static cublasStatus_t Gemv(cublasHandle_t handle, int n, const float *A, const float *x, float *y) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        return cublasSgemv(handle, CUBLAS_OP_N, n, n, &alpha, A, n, x, 1, &beta, y, 1);
    }
    static cublasStatus_t Trmv(cublasHandle_t handle, int n, const float *A, float *x, cublasOperation_t trans) {
        return cublasStrmv(handle, CUBLAS_FILL_MODE_LOWER, trans, CUBLAS_DIAG_NON_UNIT, n, A, n, x, 1);
    }
    static cublasStatus_t Axpy(cublasHandle_t handle, int n, const float *alpha, const float *x, float *y) {
        return cublasSaxpy(handle, n, alpha, x, 1, y, 1);
    }
    static cublasStatus_t Nrm2(cublasHandle_t handle, int n, const float *x, float *res) {
        return cublasSnrm2(handle, n, x, 1, res);
    }
};

template <>
struct GpuOps<double> {
    static curandStatus_t GenerateUniform(curandGenerator_t gen, double *data, size_t n_elem) {
        return curandGenerateUniformDouble(gen, data, n_elem);
    }
    static cublasStatus_t Gemm(cublasHandle_t handle,
                               int n,
                               const double *alpha,
                               const double *B,
                               const double *beta,
                               double *A) {
        return cublasDgemm(handle,
                           CUBLAS_OP_N,
                           CUBLAS_OP_T,
                           n, n, n,
                           alpha,
                           B, n,
                           B, n,
                           beta,
                           A, n);
    }
    static cusolverStatus_t PotrfBufferSize(cusolverDnHandle_t handle, int n, double *A, int *Lwork) {
        return cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, n, A, n, Lwork);
    }
    static cusolverStatus_t Potrf(cusolverDnHandle_t handle,
                                  int n,
                                  double *A,
                                  double *work,
                                  int Lwork,
                                  int *devInfo) {
        return cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_LOWER, n, A, n, work, Lwork, devInfo);
    }
    static cublasStatus_t Gemv(cublasHandle_t handle, int n, const double *A, const double *x, double *y) {
        const double alpha = 1.0;
        const double beta = 0.0;
        return cublasDgemv(handle, CUBLAS_OP_N, n, n, &alpha, A, n, x, 1, &beta, y, 1);
    }
    static cublasStatus_t Trmv(cublasHandle_t handle, int n, const double *A, double *x, cublasOperation_t trans) {
        return cublasDtrmv(handle, CUBLAS_FILL_MODE_LOWER, trans, CUBLAS_DIAG_NON_UNIT, n, A, n, x, 1);
    }
    static cublasStatus_t Axpy(cublasHandle_t handle, int n, const double *alpha, const double *x, double *y) {
        return cublasDaxpy(handle, n, alpha, x, 1, y, 1);
    }
    static cublasStatus_t Nrm2(cublasHandle_t handle, int n, const double *x, double *res) {
        return cublasDnrm2(handle, n, x, 1, res);
    }
};

cudaEvent_t begin, end;
void startTimer()
{
      
    CHECK_Runtime(cudaEventCreate(&begin));

    cudaEventRecord(begin);
    cudaEventCreate(&end);
}

float stopTimer()
{
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, begin, end);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return milliseconds;
}

template <typename T>
int runCholesky()
{
    printf("n = %d, precision = %s, debug = %s\n",
           n, sizeof(T) == sizeof(double) ? "double" : "float",
           debug_mode ? "on" : "off");

    T *A = nullptr;
    size_t matrix_bytes = sizeof(T) * (size_t)n * (size_t)n;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CHECK_Runtime(cudaMemGetInfo(&free_bytes, &total_bytes));
    printf("matrix bytes = %.2f GiB, free GPU memory = %.2f GiB\n",
           (double)matrix_bytes / (1024.0 * 1024.0 * 1024.0),
           (double)free_bytes / (1024.0 * 1024.0 * 1024.0));
    CHECK_Runtime(cudaMalloc(&A, sizeof(T) * (size_t)n * (size_t)n));

    // Build SPD matrix in-place:
    // random A -> symmetrize -> set diagonal to n (strictly diagonally dominant).
    curandGenerator_t gen;
    CHECK_Curand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_Curand(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    CHECK_Curand(GpuOps<T>::GenerateUniform(gen, A, (size_t)n * (size_t)n));

    dim3 sym_block(16, 16);
    dim3 sym_grid((n + sym_block.x - 1) / sym_block.x,
                  (n + sym_block.y - 1) / sym_block.y);
    symmetrizeMatrix<<<sym_grid, sym_block>>>(n, A, n);
    CHECK_Runtime(cudaGetLastError());

    dim3 gridDim((n + 255) / 256);
    dim3 blockDim(256);
    setDiagonalValue<<<gridDim, blockDim>>>(n, A, n, (T)n);
    CHECK_Runtime(cudaGetLastError());
    CHECK_Runtime(cudaDeviceSynchronize());

    cublasHandle_t cublas_handle;
    CHECK_Cublas(cublasCreate(&cublas_handle));

    T *A_ref = nullptr;
    if (debug_mode) {
        if (cudaMalloc(&A_ref, matrix_bytes) == cudaSuccess) {
            CHECK_Runtime(cudaMemcpy(A_ref, A, matrix_bytes, cudaMemcpyDeviceToDevice));
        } else {
            printf("debug: skip residual check (not enough memory for A_ref)\n");
            A_ref = nullptr;
        }
    }

    cusolverDnHandle_t handle;
    
    CHECK_Cusolver(cusolverDnCreate(&handle));

    int Lwork;

    CHECK_Cusolver(GpuOps<T>::PotrfBufferSize(handle, n, A, &Lwork));
    size_t work_bytes = sizeof(T) * (size_t)Lwork;
    CHECK_Runtime(cudaMemGetInfo(&free_bytes, &total_bytes));
    printf("potrf workspace bytes = %.2f GiB, free GPU memory(before work alloc) = %.2f GiB\n",
           (double)work_bytes / (1024.0 * 1024.0 * 1024.0),
           (double)free_bytes / (1024.0 * 1024.0 * 1024.0));

    T *work = nullptr;
    CHECK_Runtime(cudaMalloc(&work, sizeof(T) * (size_t)Lwork));
    int *devInfo;
    CHECK_Runtime(cudaMalloc(&devInfo, sizeof(int)));
    startTimer();
    CHECK_Cusolver(GpuOps<T>::Potrf(handle, n, A, work, Lwork, devInfo));
    CHECK_Runtime(cudaDeviceSynchronize());
    float ms = stopTimer();

    int info_h = 0;
    CHECK_Runtime(cudaMemcpy(&info_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_h != 0) {
        printf("potrf failed, devInfo = %d\n", info_h);
    }
    printf("Cholesky factorization takes %f ms: %f TFLOPs\n", ms, 1.0/3.0*n*n*n/ms/1e9);

    if (debug_mode && A_ref != nullptr && info_h == 0) {
        T *x = nullptr;
        T *yA = nullptr;
        T *yLLT = nullptr;
        CHECK_Runtime(cudaMalloc(&x, sizeof(T) * (size_t)n));
        CHECK_Runtime(cudaMalloc(&yA, sizeof(T) * (size_t)n));
        CHECK_Runtime(cudaMalloc(&yLLT, sizeof(T) * (size_t)n));
        CHECK_Curand(GpuOps<T>::GenerateUniform(gen, x, (size_t)n));

        CHECK_Cublas(GpuOps<T>::Gemv(cublas_handle, n, A_ref, x, yA));
        CHECK_Runtime(cudaMemcpy(yLLT, x, sizeof(T) * (size_t)n, cudaMemcpyDeviceToDevice));
        CHECK_Cublas(GpuOps<T>::Trmv(cublas_handle, n, A, yLLT, CUBLAS_OP_T));
        CHECK_Cublas(GpuOps<T>::Trmv(cublas_handle, n, A, yLLT, CUBLAS_OP_N));

        const T minus_one = (T)-1;
        CHECK_Cublas(GpuOps<T>::Axpy(cublas_handle, n, &minus_one, yLLT, yA));
        T norm_r = (T)0;
        T norm_a = (T)0;
        CHECK_Cublas(GpuOps<T>::Nrm2(cublas_handle, n, yA, &norm_r));
        CHECK_Cublas(GpuOps<T>::Nrm2(cublas_handle, n, yLLT, &norm_a));
        double rel = (norm_a > (T)0) ? (double)(norm_r / norm_a) : 0.0;
        printf("debug residual ||A*x-L*L^T*x||/||L*L^T*x|| = %.6e\n", rel);

        cudaFree(x);
        cudaFree(yA);
        cudaFree(yLLT);
    }

    cudaFree(A);
    if (A_ref) cudaFree(A_ref);
    cudaFree(work);
    cudaFree(devInfo);
    CHECK_Curand(curandDestroyGenerator(gen));
    CHECK_Cublas(cublasDestroy(cublas_handle));
    CHECK_Cusolver(cusolverDnDestroy(handle));

    return 0;
}

int main(int argc,char *argv[])
{
    if(parseArguments(argc, argv)==-1)
        return 0;

    if (use_double) {
        return runCholesky<double>();
    }
    return runCholesky<float>();
}
