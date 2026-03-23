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
bool time_q = false;
int warmup_runs = 2;
int repeat_runs = 5;

int parseArguments(int argc, char* argv[]) {
  if (argc < 2) {
    std::printf("Usage: %s n [float|double] [--debug] [--with-q] [--warmup k] [--repeat k]\n", argv[0]);
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
    } else if (std::strcmp(argv[i], "--with-q") == 0) {
      time_q = true;
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
  static magma_int_t GeqrfGpu(magma_int_t m, magma_int_t n, float* dA, magma_int_t ldda,
                              float* tau, float* dT, magma_int_t* info) {
    return magma_sgeqrf_gpu(m, n, dA, ldda, tau, dT, info);
  }
  static magma_int_t OrgqrGpu(magma_int_t m, magma_int_t n, magma_int_t k, float* dA,
                              magma_int_t ldda, float* tau, float* dT, magma_int_t nb,
                              magma_int_t* info) {
    return magma_sorgqr_gpu(m, n, k, dA, ldda, tau, dT, nb, info);
  }
  static cublasStatus_t Gemm(cublasHandle_t handle, int n, const float* A, int lda,
                             const float* B, int ldb, float* C, int ldc) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    return cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       n, n, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }
};

template <>
struct Ops<double> {
  static curandStatus_t GenerateUniform(curandGenerator_t gen, double* data, size_t n_elem) {
    return curandGenerateUniformDouble(gen, data, n_elem);
  }
  static magma_int_t GeqrfGpu(magma_int_t m, magma_int_t n, double* dA, magma_int_t ldda,
                              double* tau, double* dT, magma_int_t* info) {
    return magma_dgeqrf_gpu(m, n, dA, ldda, tau, dT, info);
  }
  static magma_int_t OrgqrGpu(magma_int_t m, magma_int_t n, magma_int_t k, double* dA,
                              magma_int_t ldda, double* tau, double* dT, magma_int_t nb,
                              magma_int_t* info) {
    return magma_dorgqr_gpu(m, n, k, dA, ldda, tau, dT, nb, info);
  }
  static cublasStatus_t Gemm(cublasHandle_t handle, int n, const double* A, int lda,
                             const double* B, int ldb, double* C, int ldc) {
    const double alpha = 1.0;
    const double beta = 0.0;
    return cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       n, n, n, &alpha, A, lda, B, ldb, &beta, C, ldc);
  }
};

template <typename T>
int runBenchmark() {
  std::printf("n = %d, precision = %s, debug = %s, time_q = %s, warmup = %d, repeat = %d\n",
              n, sizeof(T) == sizeof(double) ? "double" : "float",
              debug_mode ? "on" : "off",
              time_q ? "on" : "off",
              warmup_runs, repeat_runs);

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
  CHECK_Curand(Ops<T>::GenerateUniform(gen, dA, (size_t)ldda * (size_t)n));
  CHECK_Runtime(cudaDeviceSynchronize());

  T* dA_ref = nullptr;
  if (debug_mode) {
    if (cudaMalloc(&dA_ref, matrix_bytes) == cudaSuccess) {
      CHECK_Runtime(cudaMemcpy(dA_ref, dA, matrix_bytes, cudaMemcpyDeviceToDevice));
    } else {
      std::printf("debug: skip residual check (not enough memory for A_ref)\n");
    }
  }
  T* dA0 = nullptr;
  if (warmup_runs > 0 || repeat_runs > 1 || time_q) {
    if (cudaMalloc(&dA0, matrix_bytes) == cudaSuccess) {
      CHECK_Runtime(cudaMemcpy(dA0, dA, matrix_bytes, cudaMemcpyDeviceToDevice));
    } else {
      std::printf("warning: skip A0 buffer (not enough memory); repeats will reuse modified A\n");
    }
  }

  magma_int_t minmn = n;
  T* tau = (T*)std::malloc(sizeof(T) * (size_t)minmn);
  if (!tau) {
    std::fprintf(stderr, "Failed to allocate tau\n");
    return 1;
  }

  magma_int_t nb = (sizeof(T) == sizeof(double)) ? magma_get_dgeqrf_nb(n, n)
                                                 : magma_get_sgeqrf_nb(n, n);
  magma_int_t lddat = magma_roundup(n, 32);
  size_t dT_size = (size_t)(2 * minmn + lddat) * (size_t)nb;
  T* dT = nullptr;
  CHECK_Runtime(cudaMalloc(&dT, sizeof(T) * dT_size));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CHECK_Runtime(cudaEventCreate(&start));
  CHECK_Runtime(cudaEventCreate(&stop));

  magma_int_t info = 0;
  for (int w = 0; w < warmup_runs; ++w) {
    if (dA0) {
      CHECK_Runtime(cudaMemcpy(dA, dA0, matrix_bytes, cudaMemcpyDeviceToDevice));
    }
    CHECK_MAGMA(Ops<T>::GeqrfGpu(n, n, dA, ldda, tau, dT, &info));
    if (info != 0) {
      std::fprintf(stderr, "magma_xgeqrf_gpu failed, info = %d\n", (int)info);
      return 1;
    }
    if (time_q) {
      T* dQ_w = nullptr;
      if (cudaMalloc(&dQ_w, matrix_bytes) == cudaSuccess) {
        CHECK_Runtime(cudaMemcpy(dQ_w, dA, matrix_bytes, cudaMemcpyDeviceToDevice));
        CHECK_MAGMA(Ops<T>::OrgqrGpu(n, n, n, dQ_w, ldda, tau, dT, nb, &info));
        cudaFree(dQ_w);
        if (info != 0) {
          std::fprintf(stderr, "magma_xorgqr_gpu failed, info = %d\n", (int)info);
          return 1;
        }
      }
    }
  }

  double geqrf_ms_sum = 0.0;
  double orgqr_ms_sum = 0.0;
  T* dQ = nullptr;
  if (time_q) {
    if (cudaMalloc(&dQ, matrix_bytes) != cudaSuccess) {
      std::printf("time_q: skip ORGQR (not enough memory for dQ)\n");
      time_q = false;
    }
  }
  for (int r = 0; r < repeat_runs; ++r) {
    if (dA0) {
      CHECK_Runtime(cudaMemcpy(dA, dA0, matrix_bytes, cudaMemcpyDeviceToDevice));
    }
    CHECK_Runtime(cudaEventRecord(start));
    CHECK_MAGMA(Ops<T>::GeqrfGpu(n, n, dA, ldda, tau, dT, &info));
    CHECK_Runtime(cudaEventRecord(stop));
    CHECK_Runtime(cudaEventSynchronize(stop));
    if (info != 0) {
      std::fprintf(stderr, "magma_xgeqrf_gpu failed, info = %d\n", (int)info);
      return 1;
    }
    float ms = 0.0f;
    CHECK_Runtime(cudaEventElapsedTime(&ms, start, stop));
    geqrf_ms_sum += ms;

    if (time_q) {
      CHECK_Runtime(cudaMemcpy(dQ, dA, matrix_bytes, cudaMemcpyDeviceToDevice));
      CHECK_Runtime(cudaEventRecord(start));
      CHECK_MAGMA(Ops<T>::OrgqrGpu(n, n, n, dQ, ldda, tau, dT, nb, &info));
      CHECK_Runtime(cudaEventRecord(stop));
      CHECK_Runtime(cudaEventSynchronize(stop));
      if (info != 0) {
        std::fprintf(stderr, "magma_xorgqr_gpu failed, info = %d\n", (int)info);
        return 1;
      }
      float q_ms = 0.0f;
      CHECK_Runtime(cudaEventElapsedTime(&q_ms, start, stop));
      orgqr_ms_sum += q_ms;
    }
  }

  double geqrf_ms = geqrf_ms_sum / (double)repeat_runs;
  double geqrf_tflops = (4.0 / 3.0) * (double)n * (double)n * (double)n / (geqrf_ms * 1e9);
  std::printf("MAGMA GEQRF takes %.3f ms: %.6f TFLOPs\n", geqrf_ms, geqrf_tflops);

  if (time_q) {
    double orgqr_ms = orgqr_ms_sum / (double)repeat_runs;
    double orgqr_tflops = (4.0 / 3.0) * (double)n * (double)n * (double)n / (orgqr_ms * 1e9);
    std::printf("MAGMA ORGQR takes %.3f ms: %.6f TFLOPs\n", orgqr_ms, orgqr_tflops);
    double total_ms = geqrf_ms + orgqr_ms;
    double total_flops = 4.0 * (double)n * (double)n * ((double)n - (1.0 / 3.0) * (double)n);
    double total_tflops = total_flops / (total_ms * 1e9);
    std::printf("MAGMA GEQRF+ORGQR total %.3f ms: %.6f TFLOPs\n", total_ms, total_tflops);
  }

  if (debug_mode && dA_ref != nullptr) {
    cublasHandle_t cublas_handle;
    CHECK_Cublas(cublasCreate(&cublas_handle));

    T* dQ = nullptr;
    T* dR = nullptr;
    T* dQR = nullptr;
    T* dDiff = nullptr;
    if (cudaMalloc(&dQ, matrix_bytes) == cudaSuccess &&
        cudaMalloc(&dR, matrix_bytes) == cudaSuccess &&
        cudaMalloc(&dQR, matrix_bytes) == cudaSuccess &&
        cudaMalloc(&dDiff, matrix_bytes) == cudaSuccess) {
      CHECK_Runtime(cudaMemcpy(dQ, dA, matrix_bytes, cudaMemcpyDeviceToDevice));

      CHECK_MAGMA(Ops<T>::OrgqrGpu(n, n, n, dQ, ldda, tau, dT, nb, &info));
      if (info != 0) {
        std::fprintf(stderr, "magma_xorgqr_gpu failed, info = %d\n", (int)info);
      } else {
        const T alpha = (T)1;
        const T beta = (T)0;
        if (sizeof(T) == sizeof(float)) {
          CHECK_Cublas(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                   n, n, n,
                                   (const float*)&alpha,
                                   (const float*)dQ, ldda,
                                   (const float*)dA_ref, ldda,
                                   (const float*)&beta,
                                   (float*)dR, ldda));
          CHECK_Cublas(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   n, n, n,
                                   (const float*)&alpha,
                                   (const float*)dQ, ldda,
                                   (const float*)dR, ldda,
                                   (const float*)&beta,
                                   (float*)dQR, ldda));
        } else {
          CHECK_Cublas(cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                   n, n, n,
                                   (const double*)&alpha,
                                   (const double*)dQ, ldda,
                                   (const double*)dA_ref, ldda,
                                   (const double*)&beta,
                                   (double*)dR, ldda));
          CHECK_Cublas(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   n, n, n,
                                   (const double*)&alpha,
                                   (const double*)dQ, ldda,
                                   (const double*)dR, ldda,
                                   (const double*)&beta,
                                   (double*)dQR, ldda));
        }
        dim3 block(16, 16);
        dim3 grid((n + block.x - 1) / block.x,
                  (n + block.y - 1) / block.y);
        diffKernel<<<grid, block>>>(n, dA_ref, ldda, dQR, ldda, dDiff, ldda);
        CHECK_Runtime(cudaGetLastError());
        CHECK_Runtime(cudaDeviceSynchronize());

        double normA = frobNorm(n, dA_ref, ldda);
        double normR = frobNorm(n, dDiff, ldda);
        double rel = normA > 0.0 ? normR / normA : 0.0;
        std::printf("debug backward error ||A-Q*R||/||A|| = %.6e\n", rel);
      }
    } else {
      std::printf("debug: skip residual check (not enough memory for QR workspace)\n");
    }

    if (dQ) cudaFree(dQ);
    if (dR) cudaFree(dR);
    if (dQR) cudaFree(dQR);
    if (dDiff) cudaFree(dDiff);
    CHECK_Cublas(cublasDestroy(cublas_handle));
  }

  CHECK_Runtime(cudaEventDestroy(start));
  CHECK_Runtime(cudaEventDestroy(stop));
  if (dA_ref) cudaFree(dA_ref);
  if (dA0) cudaFree(dA0);
  if (dQ) cudaFree(dQ);
  cudaFree(dA);
  cudaFree(dT);
  std::free(tau);
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
