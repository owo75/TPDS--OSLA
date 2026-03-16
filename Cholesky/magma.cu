#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <magma_lapack.h>
#include <magma_v2.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

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
bool eye_mode = false;
int warmup_runs = 2;
int repeat_runs = 10;

int parseArguments(int argc, char* argv[]) {
  if (argc < 2) {
    std::printf("Usage: %s n [float|double] [--debug] [--eye] [--warmup k] [--repeat k]\n", argv[0]);
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
    } else if (std::strcmp(argv[i], "--eye") == 0) {
      eye_mode = true;
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
__global__ void symmetrizeMatrix(int n, T* a, int lda) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || j >= n || i >= j) {
    return;
  }
  size_t idx_ij = (size_t)i + (size_t)j * (size_t)lda;
  size_t idx_ji = (size_t)j + (size_t)i * (size_t)lda;
  T v = (a[idx_ij] + a[idx_ji]) * (T)0.5;
  a[idx_ij] = v;
  a[idx_ji] = v;
}

template <typename T>
__global__ void setDiagonalValue(int n, T* a, int lda, T diag_value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  size_t idx = (size_t)i + (size_t)i * (size_t)lda;
  a[idx] = diag_value;
}

template <typename T>
__global__ void addDiagonalValue(int n, T* a, int lda, T diag_value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  size_t idx = (size_t)i + (size_t)i * (size_t)lda;
  a[idx] += diag_value;
}

template <typename T>
__global__ void setIdentityFull(int n, int lda, T* a) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= lda || j >= n) {
    return;
  }
  size_t idx = (size_t)i + (size_t)j * (size_t)lda;
  a[idx] = (i == j && i < n) ? (T)1 : (T)0;
}

template <typename T>
__global__ void zeroUpperTriangle(int n, T* a, int lda) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || j >= n || i >= j) {
    return;
  }
  size_t idx = (size_t)i + (size_t)j * (size_t)lda;
  a[idx] = (T)0;
}

template <typename T>
__global__ void subtractMatrices(int n, const T* a, const T* b, T* c, int lda) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || j >= n) {
    return;
  }
  size_t idx = (size_t)i + (size_t)j * (size_t)lda;
  c[idx] = a[idx] - b[idx];
}

template <typename T>
struct Ops;

template <>
struct Ops<float> {
  static curandStatus_t GenerateUniform(curandGenerator_t gen, float* data, size_t n_elem) {
    return curandGenerateUniform(gen, data, n_elem);
  }
  static void PotrfGpu(magma_uplo_t uplo, magma_int_t n, float* dA, magma_int_t ldda, magma_int_t* info) {
    magma_spotrf_gpu(uplo, n, dA, ldda, info);
  }
  static cublasStatus_t Gemv(cublasHandle_t handle, int n, const float* A, int lda, const float* x, float* y) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublasSgemv(handle, CUBLAS_OP_N, n, n, &alpha, A, lda, x, 1, &beta, y, 1);
  }
  static cublasStatus_t Gemm(cublasHandle_t handle, int n, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }
  static cublasStatus_t Trmv(cublasHandle_t handle, int n, const float* L, int lda, float* x, cublasOperation_t trans) {
    return cublasStrmv(handle, CUBLAS_FILL_MODE_LOWER, trans, CUBLAS_DIAG_NON_UNIT, n, L, lda, x, 1);
  }
  static cublasStatus_t Axpy(cublasHandle_t handle, int n, const float* alpha, const float* x, float* y) {
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
  static void PotrfGpu(magma_uplo_t uplo, magma_int_t n, double* dA, magma_int_t ldda, magma_int_t* info) {
    magma_dpotrf_gpu(uplo, n, dA, ldda, info);
  }
  static cublasStatus_t Gemv(cublasHandle_t handle, int n, const double* A, int lda, const double* x, double* y) {
    const double alpha = 1.0;
    const double beta = 0.0;
    return cublasDgemv(handle, CUBLAS_OP_N, n, n, &alpha, A, lda, x, 1, &beta, y, 1);
  }
  static cublasStatus_t Gemm(cublasHandle_t handle, int n, const double* A, int lda, const double* B, int ldb, double* C, int ldc) {
    const double alpha = 1.0;
    const double beta = 0.0;
    return cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }
  static cublasStatus_t Trmv(cublasHandle_t handle, int n, const double* L, int lda, double* x, cublasOperation_t trans) {
    return cublasDtrmv(handle, CUBLAS_FILL_MODE_LOWER, trans, CUBLAS_DIAG_NON_UNIT, n, L, lda, x, 1);
  }
  static cublasStatus_t Axpy(cublasHandle_t handle, int n, const double* alpha, const double* x, double* y) {
    return cublasDaxpy(handle, n, alpha, x, 1, y, 1);
  }
  static cublasStatus_t Nrm2(cublasHandle_t handle, int n, const double* x, double* r) {
    return cublasDnrm2(handle, n, x, 1, r);
  }
};

template <typename T>
double frobeniusNorm(cublasHandle_t handle, int n, const T* a, int lda) {
  double sum_sq = 0.0;
  for (int j = 0; j < n; ++j) {
    T col_norm = (T)0;
    CHECK_Cublas(Ops<T>::Nrm2(handle, n, a + (size_t)j * (size_t)lda, &col_norm));
    double d = (double)col_norm;
    sum_sq += d * d;
  }
  return std::sqrt(sum_sq);
}

template <typename T>
void buildInputMatrix(T* dA, T* dG, int lda, curandGenerator_t gen, cublasHandle_t cublas_handle) {
  dim3 block2d(16, 16);
  dim3 grid2d((n + block2d.x - 1) / block2d.x,
              (lda + block2d.y - 1) / block2d.y);
  if (eye_mode) {
    setIdentityFull<<<grid2d, block2d>>>(n, lda, dA);
    CHECK_Runtime(cudaGetLastError());
    return;
  }
  CHECK_Curand(Ops<T>::GenerateUniform(gen, dG, (size_t)lda * (size_t)n));
  CHECK_Cublas(Ops<T>::Gemm(cublas_handle, n, dG, lda, dG, lda, dA, lda));
  dim3 block1d(256);
  dim3 grid1d((n + block1d.x - 1) / block1d.x);
  addDiagonalValue<<<grid1d, block1d>>>(n, dA, lda, (T)n);
  CHECK_Runtime(cudaGetLastError());
}

template <typename T>
int runBenchmark() {
  std::printf("n = %d, precision = %s, debug = %s\n",
              n, sizeof(T) == sizeof(double) ? "double" : "float",
              debug_mode ? "on" : "off");
  std::printf("warmup = %d, repeat = %d\n", warmup_runs, repeat_runs);
  std::printf("matrix mode = %s\n", eye_mode ? "identity" : "spd (G*G^T + nI)");

  magma_int_t ldda = magma_roundup(n, 64);
  size_t matrix_bytes = sizeof(T) * (size_t)ldda * (size_t)n;

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CHECK_Runtime(cudaMemGetInfo(&free_bytes, &total_bytes));
  std::printf("matrix bytes (with ldda=%d) = %.2f GiB, free GPU memory = %.2f GiB\n",
              (int)ldda,
              (double)matrix_bytes / (1024.0 * 1024.0 * 1024.0),
              (double)free_bytes / (1024.0 * 1024.0 * 1024.0));

  T* dA = nullptr;
  T* dA_ref = nullptr;
  T* dG = nullptr;
  CHECK_Runtime(cudaMalloc(&dA, matrix_bytes));
  if (!eye_mode) {
    CHECK_Runtime(cudaMalloc(&dG, matrix_bytes));
  }
  if (debug_mode) {
    CHECK_Runtime(cudaMalloc(&dA_ref, matrix_bytes));
  }

  curandGenerator_t gen;
  if (!eye_mode) {
    CHECK_Curand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_Curand(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
  }

  cublasHandle_t cublas_handle = nullptr;
  if (!eye_mode || debug_mode) {
    CHECK_Cublas(cublasCreate(&cublas_handle));
  }

  buildInputMatrix<T>(dA, dG, (int)ldda, gen, cublas_handle);
  CHECK_Runtime(cudaDeviceSynchronize());
  if (debug_mode) {
    CHECK_Runtime(cudaMemcpy(dA_ref, dA, matrix_bytes, cudaMemcpyDeviceToDevice));
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CHECK_Runtime(cudaEventCreate(&start));
  CHECK_Runtime(cudaEventCreate(&stop));

  magma_int_t info = 0;
  for (int i = 0; i < warmup_runs; ++i) {
    if (debug_mode) {
      CHECK_Runtime(cudaMemcpy(dA, dA_ref, matrix_bytes, cudaMemcpyDeviceToDevice));
    } else {
      buildInputMatrix<T>(dA, dG, (int)ldda, gen, cublas_handle);
    }
    Ops<T>::PotrfGpu(MagmaLower, n, dA, ldda, &info);
    CHECK_Runtime(cudaDeviceSynchronize());
    if (info != 0) {
      std::fprintf(stderr, "magma_xpotrf_gpu failed in warmup, info = %d\n", (int)info);
      return 1;
    }
  }

  double total_ms = 0.0;
  double best_ms = 1e30;
  for (int i = 0; i < repeat_runs; ++i) {
    if (debug_mode) {
      CHECK_Runtime(cudaMemcpy(dA, dA_ref, matrix_bytes, cudaMemcpyDeviceToDevice));
    } else {
      buildInputMatrix<T>(dA, dG, (int)ldda, gen, cublas_handle);
    }
    CHECK_Runtime(cudaEventRecord(start));
    Ops<T>::PotrfGpu(MagmaLower, n, dA, ldda, &info);
    CHECK_Runtime(cudaDeviceSynchronize());
    CHECK_Runtime(cudaEventRecord(stop));
    CHECK_Runtime(cudaEventSynchronize(stop));
    if (info != 0) {
      std::fprintf(stderr, "magma_xpotrf_gpu failed, info = %d\n", (int)info);
      return 1;
    }
    float ms = 0.0f;
    CHECK_Runtime(cudaEventElapsedTime(&ms, start, stop));
    total_ms += (double)ms;
    if ((double)ms < best_ms) best_ms = (double)ms;
  }

  double avg_ms = total_ms / (double)repeat_runs;
  double tflops_avg = (1.0 / 3.0) * (double)n * (double)n * (double)n / (avg_ms * 1e9);
  double tflops_best = (1.0 / 3.0) * (double)n * (double)n * (double)n / (best_ms * 1e9);
  std::printf("MAGMA Cholesky avg %.3f ms (best %.3f ms): avg %.6f TFLOPs (best %.6f TFLOPs)\n",
              avg_ms, best_ms, tflops_avg, tflops_best);

  if (debug_mode) {

    T* dLLT = nullptr;
    CHECK_Runtime(cudaMalloc(&dLLT, matrix_bytes));

    dim3 tri_block(16, 16);
    dim3 tri_grid((n + tri_block.x - 1) / tri_block.x,
                  (n + tri_block.y - 1) / tri_block.y);
    zeroUpperTriangle<<<tri_grid, tri_block>>>(n, dA, ldda);
    CHECK_Runtime(cudaGetLastError());

    CHECK_Cublas(Ops<T>::Gemm(cublas_handle, n, dA, ldda, dA, ldda, dLLT, ldda));
    subtractMatrices<<<tri_grid, tri_block>>>(n, dA_ref, dLLT, dLLT, ldda);
    CHECK_Runtime(cudaGetLastError());
    CHECK_Runtime(cudaDeviceSynchronize());

    double normA = frobeniusNorm(cublas_handle, n, dA_ref, ldda);
    double normR = frobeniusNorm(cublas_handle, n, dLLT, ldda);
    double rel = normA > 0.0 ? normR / normA : 0.0;
    std::printf("debug residual ||A-LL^T||_F/||A||_F = %.6e\n", rel);

    cudaFree(dLLT);
  }

  CHECK_Runtime(cudaEventDestroy(start));
  CHECK_Runtime(cudaEventDestroy(stop));
  if (cublas_handle) {
    CHECK_Cublas(cublasDestroy(cublas_handle));
  }
  if (!eye_mode) {
    CHECK_Curand(curandDestroyGenerator(gen));
  }
  if (dA_ref) {
    cudaFree(dA_ref);
  }
  if (dG) {
    cudaFree(dG);
  }
  cudaFree(dA);
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
