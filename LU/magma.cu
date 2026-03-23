#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <magma_lapack.h>
#include <magma_v2.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

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

#define CHECK_MAGMA(call)                                                       \
  do {                                                                          \
    magma_int_t _status = (call);                                               \
    if (_status != MAGMA_SUCCESS) {                                             \
      fprintf(stderr, "MAGMA error %d at %s:%d\n", (int)_status, __FILE__,     \
              __LINE__);                                                        \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)

int n = 0;
bool use_double = false;
bool debug_mode = false;
int warmup_runs = 2;
int repeat_runs = 10;

int parseArguments(int argc, char* argv[]) {
  if (argc < 2) {
    std::printf("Usage: %s n [float|double] [--debug] [--warmup k] [--repeat k]\n", argv[0]);
    return -1;
  }
  n = std::atoi(argv[1]);
  if (n <= 0) {
    std::printf("n must be positive\n");
    return -1;
  }
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "double") == 0 || std::strcmp(argv[i], "d") == 0) {
      use_double = true;
    } else if (std::strcmp(argv[i], "float") == 0 || std::strcmp(argv[i], "f") == 0) {
      use_double = false;
    } else if (std::strcmp(argv[i], "--debug") == 0) {
      debug_mode = true;
    } else if (std::strcmp(argv[i], "--warmup") == 0) {
      if (i + 1 >= argc) {
        std::printf("Missing value for --warmup\n");
        return -1;
      }
      warmup_runs = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--repeat") == 0) {
      if (i + 1 >= argc) {
        std::printf("Missing value for --repeat\n");
        return -1;
      }
      repeat_runs = std::atoi(argv[++i]);
    } else {
      std::printf("Unknown argument: %s\n", argv[i]);
      return -1;
    }
  }
  if (warmup_runs < 0 || repeat_runs <= 0) {
    std::printf("--warmup must be >= 0 and --repeat must be > 0\n");
    return -1;
  }
  return 0;
}

template <typename T>
__global__ void applyInversePivotsKernel(int n, const magma_int_t* ipiv, T* y) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int i = n - 1; i >= 0; --i) {
    magma_int_t piv = ipiv[i] - 1;
    if (piv != i) {
      T tmp = y[i];
      y[i] = y[piv];
      y[piv] = tmp;
    }
  }
}

template <typename T>
__global__ void applyForwardPivotsMatrixKernel(int n, const magma_int_t* ipiv, T* A, int lda) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int i = 0; i < n; ++i) {
    magma_int_t piv = ipiv[i] - 1;
    if (piv != i) {
      for (int j = 0; j < n; ++j) {
        size_t idx_i = (size_t)i + (size_t)j * (size_t)lda;
        size_t idx_p = (size_t)piv + (size_t)j * (size_t)lda;
        T tmp = A[idx_i];
        A[idx_i] = A[idx_p];
        A[idx_p] = tmp;
      }
    }
  }
}

template <typename T>
__global__ void extractLUKernel(int n, const T* A, int lda, T* L, int ldl, T* U, int ldu) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || j >= n) {
    return;
  }
  size_t idxA = (size_t)i + (size_t)j * (size_t)lda;
  size_t idxL = (size_t)i + (size_t)j * (size_t)ldl;
  size_t idxU = (size_t)i + (size_t)j * (size_t)ldu;
  T a = A[idxA];
  if (i > j) {
    L[idxL] = a;
    U[idxU] = (T)0;
  } else if (i == j) {
    L[idxL] = (T)1;
    U[idxU] = a;
  } else {
    L[idxL] = (T)0;
    U[idxU] = a;
  }
}

template <typename T>
__global__ void diffKernel(int n, const T* A, int lda, const T* B, int ldb, T* C, int ldc) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || j >= n) {
    return;
  }
  size_t idxA = (size_t)i + (size_t)j * (size_t)lda;
  size_t idxB = (size_t)i + (size_t)j * (size_t)ldb;
  size_t idxC = (size_t)i + (size_t)j * (size_t)ldc;
  C[idxC] = A[idxA] - B[idxB];
}

template <typename T>
__global__ void frobSumKernel(int n, const T* A, int lda, double* out) {
  double sum = 0.0;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  for (int j = blockIdx.y; j < n; j += gridDim.y) {
    for (int i = row; i < n; i += blockDim.x * gridDim.x) {
      size_t idx = (size_t)i + (size_t)j * (size_t)lda;
      double v = (double)A[idx];
      sum += v * v;
    }
  }
  if (sum != 0.0) {
    atomicAdd(out, sum);
  }
}

template <typename T>
double frobNorm(int n, const T* A, int lda) {
  double* dsum = nullptr;
  CHECK_Runtime(cudaMalloc(&dsum, sizeof(double)));
  CHECK_Runtime(cudaMemset(dsum, 0, sizeof(double)));
  int block = 256;
  int grid_x = (n + block - 1) / block;
  if (grid_x > 256) grid_x = 256;
  int grid_y = n < 32 ? n : 32;
  dim3 grid(grid_x, grid_y);
  frobSumKernel<<<grid, block>>>(n, A, lda, dsum);
  CHECK_Runtime(cudaGetLastError());
  CHECK_Runtime(cudaDeviceSynchronize());
  double hsum = 0.0;
  CHECK_Runtime(cudaMemcpy(&hsum, dsum, sizeof(double), cudaMemcpyDeviceToHost));
  cudaFree(dsum);
  return std::sqrt(hsum);
}

template <typename T>
struct Ops;

template <>
struct Ops<float> {
  static curandStatus_t GenerateUniform(curandGenerator_t gen, float* data, size_t n_elem) {
    return curandGenerateUniform(gen, data, n_elem);
  }
  static magma_int_t GetrfGpu(magma_int_t m, magma_int_t n, float* dA, magma_int_t ldda,
                              magma_int_t* ipiv, magma_int_t* info) {
    return magma_sgetrf_gpu(m, n, dA, ldda, ipiv, info);
  }
  static cublasStatus_t Gemv(cublasHandle_t handle, int n, const float* A, int lda,
                             const float* x, float* y) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublasSgemv(handle, CUBLAS_OP_N, n, n, &alpha, A, lda, x, 1, &beta, y, 1);
  }
  static cublasStatus_t Trmv(cublasHandle_t handle, cublasFillMode_t uplo,
                             cublasDiagType_t diag, int n, const float* A, int lda,
                             float* x) {
    return cublasStrmv(handle, uplo, CUBLAS_OP_N, diag, n, A, lda, x, 1);
  }
  static cublasStatus_t Axpy(cublasHandle_t handle, int n, const float* alpha,
                             const float* x, float* y) {
    return cublasSaxpy(handle, n, alpha, x, 1, y, 1);
  }
  static cublasStatus_t Nrm2(cublasHandle_t handle, int n, const float* x, float* r) {
    return cublasSnrm2(handle, n, x, 1, r);
  }
};

template <>
struct Ops<double> {
  static curandStatus_t GenerateUniform(curandGenerator_t gen, double* data, size_t n_elem) {
    return curandGenerateUniformDouble(gen, data, n_elem);
  }
  static magma_int_t GetrfGpu(magma_int_t m, magma_int_t n, double* dA, magma_int_t ldda,
                              magma_int_t* ipiv, magma_int_t* info) {
    return magma_dgetrf_gpu(m, n, dA, ldda, ipiv, info);
  }
  static cublasStatus_t Gemv(cublasHandle_t handle, int n, const double* A, int lda,
                             const double* x, double* y) {
    const double alpha = 1.0;
    const double beta = 0.0;
    return cublasDgemv(handle, CUBLAS_OP_N, n, n, &alpha, A, lda, x, 1, &beta, y, 1);
  }
  static cublasStatus_t Trmv(cublasHandle_t handle, cublasFillMode_t uplo,
                             cublasDiagType_t diag, int n, const double* A, int lda,
                             double* x) {
    return cublasDtrmv(handle, uplo, CUBLAS_OP_N, diag, n, A, lda, x, 1);
  }
  static cublasStatus_t Axpy(cublasHandle_t handle, int n, const double* alpha,
                             const double* x, double* y) {
    return cublasDaxpy(handle, n, alpha, x, 1, y, 1);
  }
  static cublasStatus_t Nrm2(cublasHandle_t handle, int n, const double* x, double* r) {
    return cublasDnrm2(handle, n, x, 1, r);
  }
};

template <typename T>
int runBenchmark() {
  std::printf("n = %d, precision = %s, debug = %s\n",
              n, sizeof(T) == sizeof(double) ? "double" : "float",
              debug_mode ? "on" : "off");
  std::printf("warmup = %d, repeat = %d\n", warmup_runs, repeat_runs);

  magma_int_t ldda = magma_roundup(n, 32);
  size_t matrix_bytes = sizeof(T) * (size_t)ldda * (size_t)n;

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CHECK_Runtime(cudaMemGetInfo(&free_bytes, &total_bytes));
  std::printf("matrix bytes (with ldda=%d) = %.2f GiB, free GPU memory = %.2f GiB\n",
              (int)ldda,
              (double)matrix_bytes / (1024.0 * 1024.0 * 1024.0),
              (double)free_bytes / (1024.0 * 1024.0 * 1024.0));

  T* dA = nullptr;
  CHECK_Runtime(cudaMalloc(&dA, matrix_bytes));

  curandGenerator_t gen;
  CHECK_Curand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CHECK_Curand(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
  T* dA_ref = nullptr;
  if (debug_mode) {
    if (cudaMalloc(&dA_ref, matrix_bytes) != cudaSuccess) {
      std::printf("debug: skip residual check (not enough memory for A_ref)\n");
    }
  }
  if (dA_ref) {
    CHECK_Curand(Ops<T>::GenerateUniform(gen, dA, (size_t)ldda * (size_t)n));
    CHECK_Runtime(cudaDeviceSynchronize());
    CHECK_Runtime(cudaMemcpy(dA_ref, dA, matrix_bytes, cudaMemcpyDeviceToDevice));
  }

  magma_int_t* ipiv = nullptr;
  ipiv = (magma_int_t*)std::malloc(sizeof(magma_int_t) * (size_t)n);
  if (!ipiv) {
    std::fprintf(stderr, "Failed to allocate ipiv\n");
    return 1;
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CHECK_Runtime(cudaEventCreate(&start));
  CHECK_Runtime(cudaEventCreate(&stop));

  magma_int_t info = 0;
  for (int i = 0; i < warmup_runs; ++i) {
    if (dA_ref) {
      CHECK_Runtime(cudaMemcpy(dA, dA_ref, matrix_bytes, cudaMemcpyDeviceToDevice));
    } else {
      CHECK_Curand(Ops<T>::GenerateUniform(gen, dA, (size_t)ldda * (size_t)n));
    }
    CHECK_Runtime(cudaDeviceSynchronize());
    CHECK_MAGMA(Ops<T>::GetrfGpu(n, n, dA, ldda, ipiv, &info));
    CHECK_Runtime(cudaDeviceSynchronize());
    if (info != 0) {
      std::fprintf(stderr, "magma_xgetrf_gpu failed in warmup, info = %d\n", (int)info);
      return 1;
    }
  }

  double total_ms = 0.0;
  double best_ms = 1e30;
  for (int i = 0; i < repeat_runs; ++i) {
    if (dA_ref) {
      CHECK_Runtime(cudaMemcpy(dA, dA_ref, matrix_bytes, cudaMemcpyDeviceToDevice));
    } else {
      CHECK_Curand(Ops<T>::GenerateUniform(gen, dA, (size_t)ldda * (size_t)n));
    }
    CHECK_Runtime(cudaDeviceSynchronize());
    CHECK_Runtime(cudaEventRecord(start));
    CHECK_MAGMA(Ops<T>::GetrfGpu(n, n, dA, ldda, ipiv, &info));
    CHECK_Runtime(cudaEventRecord(stop));
    CHECK_Runtime(cudaEventSynchronize(stop));
    if (info != 0) {
      std::fprintf(stderr, "magma_xgetrf_gpu failed, info = %d\n", (int)info);
      return 1;
    }
    float ms = 0.0f;
    CHECK_Runtime(cudaEventElapsedTime(&ms, start, stop));
    total_ms += (double)ms;
    if ((double)ms < best_ms) best_ms = (double)ms;
  }

  double avg_ms = total_ms / (double)repeat_runs;
  double tflops_avg = (2.0 / 3.0) * (double)n * (double)n * (double)n / (avg_ms * 1e9);
  double tflops_best = (2.0 / 3.0) * (double)n * (double)n * (double)n / (best_ms * 1e9);
  std::printf("MAGMA LU avg %.3f ms (best %.3f ms): avg %.6f TFLOPs (best %.6f TFLOPs)\n",
              avg_ms, best_ms, tflops_avg, tflops_best);

  if (debug_mode && dA_ref != nullptr) {
    std::printf("debug: computing residual (may take a while)...\n");
    std::fflush(stdout);
    cublasHandle_t cublas_handle;
    CHECK_Cublas(cublasCreate(&cublas_handle));

    magma_int_t* dipiv = nullptr;
    T *dPA = nullptr, *dL = nullptr, *dU = nullptr, *dLU = nullptr, *dDiff = nullptr;
    if (cudaMalloc(&dipiv, sizeof(magma_int_t) * (size_t)n) == cudaSuccess &&
        cudaMalloc(&dPA, matrix_bytes) == cudaSuccess &&
        cudaMalloc(&dL, matrix_bytes) == cudaSuccess &&
        cudaMalloc(&dU, matrix_bytes) == cudaSuccess &&
        cudaMalloc(&dLU, matrix_bytes) == cudaSuccess &&
        cudaMalloc(&dDiff, matrix_bytes) == cudaSuccess) {
      CHECK_Runtime(cudaMemcpy(dipiv, ipiv, sizeof(magma_int_t) * (size_t)n,
                               cudaMemcpyHostToDevice));
      CHECK_Runtime(cudaMemcpy(dPA, dA_ref, matrix_bytes, cudaMemcpyDeviceToDevice));

      applyForwardPivotsMatrixKernel<<<1, 1>>>(n, dipiv, dPA, ldda);
      CHECK_Runtime(cudaGetLastError());
      CHECK_Runtime(cudaDeviceSynchronize());

      dim3 block(16, 16);
      dim3 grid((n + block.x - 1) / block.x,
                (n + block.y - 1) / block.y);
      extractLUKernel<<<grid, block>>>(n, dA, ldda, dL, ldda, dU, ldda);
      CHECK_Runtime(cudaGetLastError());

      const T alpha = (T)1;
      const T beta = (T)0;
      if (sizeof(T) == sizeof(float)) {
        CHECK_Cublas(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 n, n, n,
                                 (const float*)&alpha,
                                 (const float*)dL, ldda,
                                 (const float*)dU, ldda,
                                 (const float*)&beta,
                                 (float*)dLU, ldda));
      } else {
        CHECK_Cublas(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 n, n, n,
                                 (const double*)&alpha,
                                 (const double*)dL, ldda,
                                 (const double*)dU, ldda,
                                 (const double*)&beta,
                                 (double*)dLU, ldda));
      }

      diffKernel<<<grid, block>>>(n, dPA, ldda, dLU, ldda, dDiff, ldda);
      CHECK_Runtime(cudaGetLastError());
      CHECK_Runtime(cudaDeviceSynchronize());

      double normA = frobNorm(n, dA_ref, ldda);
      double normR = frobNorm(n, dDiff, ldda);
      double rel = normA > 0.0 ? normR / normA : 0.0;
      std::printf("debug backward error ||P*A-L*U||/||A|| = %.6e\n", rel);
    } else {
      std::printf("debug: skip residual check (not enough memory for LU workspace)\n");
    }

    if (dipiv) cudaFree(dipiv);
    if (dPA) cudaFree(dPA);
    if (dL) cudaFree(dL);
    if (dU) cudaFree(dU);
    if (dLU) cudaFree(dLU);
    if (dDiff) cudaFree(dDiff);
    CHECK_Cublas(cublasDestroy(cublas_handle));
  }

  CHECK_Runtime(cudaEventDestroy(start));
  CHECK_Runtime(cudaEventDestroy(stop));
  if (dA_ref) cudaFree(dA_ref);
  cudaFree(dA);
  std::free(ipiv);
  CHECK_Curand(curandDestroyGenerator(gen));
  return 0;
}

int main(int argc, char* argv[]) {
  if (parseArguments(argc, argv) != 0) {
    return 1;
  }
  CHECK_MAGMA(magma_init());
  int rc = 0;
  if (use_double) {
    rc = runBenchmark<double>();
  } else {
    rc = runBenchmark<float>();
  }
  CHECK_MAGMA(magma_finalize());
  return rc;
}
