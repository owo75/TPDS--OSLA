#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>
#include "../GEMMul8/GEMMul8/include/gemmul8.hpp"

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

int n = 0;
int k = 0;
int nb = 0;
bool use_double = false;
bool debug_mode = false;
bool eye_mode = false;
bool use_gemmul8 = false;
int warmup_runs = 1;
int bench_runs = 5;
unsigned gemmul8_moduli = 6;
bool gemmul8_fastmode = true;

int parseArguments(int argc, char* argv[]) {
  if (argc < 4) {
    std::printf("Usage: %s n k nb [float|double] [--debug] [--eye] [--gemmul8] [--gemmul8-moduli N] [--gemmul8-safe|--gemmul8-fast] [--warmup N] [--runs N]\n", argv[0]);
    return -1;
  }
  n = std::atoi(argv[1]);
  k = std::atoi(argv[2]);
  nb = std::atoi(argv[3]);
  if (n <= 0 || k <= 0 || nb <= 0) {
    std::printf("n, k, nb must be positive\n");
    return -1;
  }

  for (int i = 4; i < argc; ++i) {
    if (std::strcmp(argv[i], "double") == 0 || std::strcmp(argv[i], "d") == 0) {
      use_double = true;
    } else if (std::strcmp(argv[i], "float") == 0 || std::strcmp(argv[i], "f") == 0) {
      use_double = false;
    } else if (std::strcmp(argv[i], "--debug") == 0) {
      debug_mode = true;
    } else if (std::strcmp(argv[i], "--eye") == 0) {
      eye_mode = true;
    } else if (std::strcmp(argv[i], "--gemmul8") == 0) {
      use_gemmul8 = true;
    } else if (std::strcmp(argv[i], "--gemmul8-moduli") == 0) {
      if (i + 1 >= argc) {
        std::printf("--gemmul8-moduli requires a value\n");
        return -1;
      }
      gemmul8_moduli = (unsigned)std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--gemmul8-safe") == 0) {
      gemmul8_fastmode = false;
    } else if (std::strcmp(argv[i], "--gemmul8-fast") == 0) {
      gemmul8_fastmode = true;
    } else if (std::strcmp(argv[i], "--warmup") == 0) {
      if (i + 1 >= argc) {
        std::printf("--warmup requires a value\n");
        return -1;
      }
      warmup_runs = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--runs") == 0) {
      if (i + 1 >= argc) {
        std::printf("--runs requires a value\n");
        return -1;
      }
      bench_runs = std::atoi(argv[++i]);
    } else {
      std::printf("Unknown argument: %s\n", argv[i]);
      return -1;
    }
  }

  if (warmup_runs < 0 || bench_runs <= 0) {
    std::printf("warmup must be >= 0 and runs must be > 0\n");
    return -1;
  }
  if (gemmul8_moduli < 3) {
    std::printf("gemmul8-moduli must be >= 3\n");
    return -1;
  }
  return 0;
}

size_t estimateGemmul8WorkspaceBytes() {
  size_t ws_max = 0;
  auto update_ws = [&](int m, int ncol, int kdim) {
    if (m <= 0 || ncol <= 0 || kdim <= 0) return;
    size_t ws = gemmul8::workSize((size_t)m, (size_t)ncol, (size_t)kdim, gemmul8_moduli);
    if (ws > ws_max) ws_max = ws;
  };

  for (int j = 0; j < n; j += k) {
    int j_end = (j + k < n) ? (j + k) : n;
    for (int i = j; i < j_end - 1 && i < n; i += nb) {
      int ib = nb;
      if (i + ib > n) ib = n - i;
      if (ib <= 0) break;
      int below = n - i - ib;
      if (below <= 0) break;
      if (j + k - 1 - i - ib <= 0) break;

      int kdim = i + ib - j;
      int rem = n - i - 2 * ib;
      if (rem > 0 && kdim > 0) update_ws(rem, ib, kdim);
    }

    int tail1 = n - j - k;
    if (tail1 <= 0) break;
    int kblk = (k < tail1) ? k : tail1;
    int kdim2 = j + k;
    int tail2 = n - j - k - kblk;
    if (tail2 > 0 && kblk > 0 && kdim2 > 0) update_ws(tail2, kblk, kdim2);
  }
  return ws_max;
}

template <typename T>
__global__ void setEye(int m, int ncols, T* a, int lda) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i < m && j < ncols) {
    size_t idx = (size_t)i + (size_t)j * (size_t)lda;
    a[idx] = (i == j) ? (T)1 : (T)0;
  }
}

template <typename T>
__global__ void symmetrizeMatrix(int m, T* a, int lda) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i < m && j < m && i < j) {
    size_t idx_ij = (size_t)i + (size_t)j * (size_t)lda;
    size_t idx_ji = (size_t)j + (size_t)i * (size_t)lda;
    T v = (a[idx_ij] + a[idx_ji]) * (T)0.5;
    a[idx_ij] = v;
    a[idx_ji] = v;
  }
}

template <typename T>
__global__ void setDiagonalValue(int m, T* a, int lda, T diag_val) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < m) {
    size_t idx = (size_t)i + (size_t)i * (size_t)lda;
    a[idx] = diag_val;
  }
}

template <typename T>
__global__ void zeroUpperTri(int m, T* a, int lda) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i < m && j < m && i < j) {
    size_t idx = (size_t)i + (size_t)j * (size_t)lda;
    a[idx] = (T)0;
  }
}

template <typename T>
struct Ops;

template <>
struct Ops<float> {
  static curandStatus_t GenerateUniform(curandGenerator_t gen, float* data, size_t n_elem) {
    return curandGenerateUniform(gen, data, n_elem);
  }
  static cublasStatus_t Trsm(cublasHandle_t h, int m, int ncol, const float* alpha, const float* A, int lda, float* B, int ldb) {
    return cublasStrsm(h, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                       CUBLAS_DIAG_NON_UNIT, m, ncol, alpha, A, lda, B, ldb);
  }
  static cublasStatus_t Syrk(cublasHandle_t h, int nrow, int kdim, const float* alpha, const float* A, int lda, const float* beta, float* C, int ldc) {
    return cublasSsyrk(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, nrow, kdim, alpha, A, lda, beta, C, ldc);
  }
  static cublasStatus_t GemmNT(cublasHandle_t h, int m, int ncol, int kdim, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) {
    return cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, m, ncol, kdim, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  static cusolverStatus_t PotrfBufferSize(cusolverDnHandle_t h, int m, float* A, int lda, int* lwork) {
    return cusolverDnSpotrf_bufferSize(h, CUBLAS_FILL_MODE_LOWER, m, A, lda, lwork);
  }
  static cusolverStatus_t Potrf(cusolverDnHandle_t h, int m, float* A, int lda, float* work, int lwork, int* info) {
    return cusolverDnSpotrf(h, CUBLAS_FILL_MODE_LOWER, m, A, lda, work, lwork, info);
  }
  static cublasStatus_t Gemv(cublasHandle_t h, int m, const float* A, int lda, const float* x, float* y) {
    const float one = 1.0f, zero = 0.0f;
    return cublasSgemv(h, CUBLAS_OP_N, m, m, &one, A, lda, x, 1, &zero, y, 1);
  }
  static cublasStatus_t Trmv(cublasHandle_t h, int m, const float* A, int lda, float* x, cublasOperation_t trans) {
    return cublasStrmv(h, CUBLAS_FILL_MODE_LOWER, trans, CUBLAS_DIAG_NON_UNIT, m, A, lda, x, 1);
  }
  static cublasStatus_t Axpy(cublasHandle_t h, int m, const float* alpha, const float* x, float* y) {
    return cublasSaxpy(h, m, alpha, x, 1, y, 1);
  }
  static cublasStatus_t Nrm2(cublasHandle_t h, int m, const float* x, float* out) {
    return cublasSnrm2(h, m, x, 1, out);
  }
  static cublasStatus_t GemmNN(cublasHandle_t h, int m, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) {
    return cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, m, m, m, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  static cublasStatus_t GeamDiff(cublasHandle_t h, int m, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    const float one = 1.0f;
    const float neg_one = -1.0f;
    return cublasSgeam(h, CUBLAS_OP_N, CUBLAS_OP_N, m, m, &one, A, lda, &neg_one, B, ldb, C, ldc);
  }
};

template <>
struct Ops<double> {
  static curandStatus_t GenerateUniform(curandGenerator_t gen, double* data, size_t n_elem) {
    return curandGenerateUniformDouble(gen, data, n_elem);
  }
  static cublasStatus_t Trsm(cublasHandle_t h, int m, int ncol, const double* alpha, const double* A, int lda, double* B, int ldb) {
    return cublasDtrsm(h, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                       CUBLAS_DIAG_NON_UNIT, m, ncol, alpha, A, lda, B, ldb);
  }
  static cublasStatus_t Syrk(cublasHandle_t h, int nrow, int kdim, const double* alpha, const double* A, int lda, const double* beta, double* C, int ldc) {
    return cublasDsyrk(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, nrow, kdim, alpha, A, lda, beta, C, ldc);
  }
  static cublasStatus_t GemmNT(cublasHandle_t h, int m, int ncol, int kdim, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) {
    return cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, m, ncol, kdim, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  static cusolverStatus_t PotrfBufferSize(cusolverDnHandle_t h, int m, double* A, int lda, int* lwork) {
    return cusolverDnDpotrf_bufferSize(h, CUBLAS_FILL_MODE_LOWER, m, A, lda, lwork);
  }
  static cusolverStatus_t Potrf(cusolverDnHandle_t h, int m, double* A, int lda, double* work, int lwork, int* info) {
    return cusolverDnDpotrf(h, CUBLAS_FILL_MODE_LOWER, m, A, lda, work, lwork, info);
  }
  static cublasStatus_t Gemv(cublasHandle_t h, int m, const double* A, int lda, const double* x, double* y) {
    const double one = 1.0, zero = 0.0;
    return cublasDgemv(h, CUBLAS_OP_N, m, m, &one, A, lda, x, 1, &zero, y, 1);
  }
  static cublasStatus_t Trmv(cublasHandle_t h, int m, const double* A, int lda, double* x, cublasOperation_t trans) {
    return cublasDtrmv(h, CUBLAS_FILL_MODE_LOWER, trans, CUBLAS_DIAG_NON_UNIT, m, A, lda, x, 1);
  }
  static cublasStatus_t Axpy(cublasHandle_t h, int m, const double* alpha, const double* x, double* y) {
    return cublasDaxpy(h, m, alpha, x, 1, y, 1);
  }
  static cublasStatus_t Nrm2(cublasHandle_t h, int m, const double* x, double* out) {
    return cublasDnrm2(h, m, x, 1, out);
  }
  static cublasStatus_t GemmNN(cublasHandle_t h, int m, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) {
    return cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, m, m, m, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  static cublasStatus_t GeamDiff(cublasHandle_t h, int m, const double* A, int lda, const double* B, int ldb, double* C, int ldc) {
    const double one = 1.0;
    const double neg_one = -1.0;
    return cublasDgeam(h, CUBLAS_OP_N, CUBLAS_OP_N, m, m, &one, A, lda, &neg_one, B, ldb, C, ldc);
  }
};

cudaEvent_t begin_event, end_event;
void startTimer() {
  CHECK_Runtime(cudaEventCreate(&begin_event));
  CHECK_Runtime(cudaEventCreate(&end_event));
  CHECK_Runtime(cudaEventRecord(begin_event));
}

float stopTimer() {
  CHECK_Runtime(cudaEventRecord(end_event));
  CHECK_Runtime(cudaEventSynchronize(end_event));
  float ms = 0.0f;
  CHECK_Runtime(cudaEventElapsedTime(&ms, begin_event, end_event));
  CHECK_Runtime(cudaEventDestroy(begin_event));
  CHECK_Runtime(cudaEventDestroy(end_event));
  return ms;
}

template <typename T>
void buildInputMatrix(T* A, int lda, curandGenerator_t gen) {
  dim3 block2d(16, 16);
  dim3 grid2d((n + block2d.x - 1) / block2d.x, (n + block2d.y - 1) / block2d.y);
  if (eye_mode) {
    setEye<<<grid2d, block2d>>>(n, n, A, lda);
    CHECK_Runtime(cudaGetLastError());
    return;
  }
  CHECK_Curand(Ops<T>::GenerateUniform(gen, A, (size_t)lda * (size_t)n));
  symmetrizeMatrix<<<grid2d, block2d>>>(n, A, lda);
  CHECK_Runtime(cudaGetLastError());
  dim3 block1d(256);
  dim3 grid1d((n + block1d.x - 1) / block1d.x);
  setDiagonalValue<<<grid1d, block1d>>>(n, A, lda, (T)n);
  CHECK_Runtime(cudaGetLastError());
}

template <typename T>
void runDebugResidual(cublasHandle_t cublas_handle, curandGenerator_t gen, const T* A_orig, const T* A_fact, int lda) {
  T* L = nullptr;
  T* A_rec = nullptr;
  T* A_diff = nullptr;
  CHECK_Runtime(cudaMalloc(&L, sizeof(T) * (size_t)n * (size_t)n));
  CHECK_Runtime(cudaMalloc(&A_rec, sizeof(T) * (size_t)n * (size_t)n));
  CHECK_Runtime(cudaMalloc(&A_diff, sizeof(T) * (size_t)n * (size_t)n));
  CHECK_Runtime(cudaMemcpy(L, A_fact, sizeof(T) * (size_t)n * (size_t)n, cudaMemcpyDeviceToDevice));

  dim3 block2d(16, 16);
  dim3 grid2d((n + block2d.x - 1) / block2d.x, (n + block2d.y - 1) / block2d.y);
  zeroUpperTri<<<grid2d, block2d>>>(n, L, lda);
  CHECK_Runtime(cudaGetLastError());

  const T one = (T)1;
  const T zero = (T)0;
  CHECK_Cublas(Ops<T>::GemmNN(cublas_handle, n, &one, L, lda, L, lda, &zero, A_rec, lda));
  CHECK_Cublas(Ops<T>::GeamDiff(cublas_handle, n, A_orig, lda, A_rec, lda, A_diff, lda));

  T diff_norm = (T)0;
  T orig_norm = (T)0;
  int vec_n = n * n;
  CHECK_Cublas(Ops<T>::Nrm2(cublas_handle, vec_n, A_diff, &diff_norm));
  CHECK_Cublas(Ops<T>::Nrm2(cublas_handle, vec_n, A_orig, &orig_norm));
  double rel = (orig_norm > (T)0) ? (double)(diff_norm / orig_norm) : 0.0;
  std::printf("debug backward error ||A-L*L^T||_F/||A||_F = %.6e\n", rel);

  CHECK_Runtime(cudaFree(L));
  CHECK_Runtime(cudaFree(A_rec));
  CHECK_Runtime(cudaFree(A_diff));
}

template <typename T>
int run() {
  std::printf("n=%d k=%d nb=%d precision=%s eye=%s debug=%s gemmul8=%s moduli=%u fast=%s warmup=%d runs=%d\n",
              n, k, nb, (sizeof(T) == sizeof(double) ? "double" : "float"),
              eye_mode ? "on" : "off", debug_mode ? "on" : "off",
              use_gemmul8 ? "on" : "off",
              gemmul8_moduli, gemmul8_fastmode ? "on" : "off",
              warmup_runs, bench_runs);

  cublasHandle_t cublas_handle;
  cusolverDnHandle_t cusolver_handle;
  curandGenerator_t gen;
  CHECK_Cublas(cublasCreate(&cublas_handle));
  CHECK_Cusolver(cusolverDnCreate(&cusolver_handle));
  CHECK_Curand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CHECK_Curand(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

  T* A = nullptr;
  CHECK_Runtime(cudaMalloc(&A, sizeof(T) * (size_t)n * (size_t)n));

  int Lwork = 0;
  int potrf_n = (n <= 1024) ? n : ((nb < n) ? nb : n);
  CHECK_Cusolver(Ops<T>::PotrfBufferSize(cusolver_handle, potrf_n, A, n, &Lwork));

  T* work = nullptr;
  int* devInfo = nullptr;
  CHECK_Runtime(cudaMalloc(&work, sizeof(T) * (size_t)Lwork));
  CHECK_Runtime(cudaMalloc(&devInfo, sizeof(int)));
  void* gemmul8_work = nullptr;
  if (use_gemmul8) {
    size_t ws = estimateGemmul8WorkspaceBytes();
    size_t free_mem = 0, total_mem = 0;
    CHECK_Runtime(cudaMemGetInfo(&free_mem, &total_mem));
    std::printf("gemmul8 workspace required = %.2f GiB, free GPU memory before alloc = %.2f GiB\n",
                (double)ws / (1024.0 * 1024.0 * 1024.0),
                (double)free_mem / (1024.0 * 1024.0 * 1024.0));
    if (ws > 0) CHECK_Runtime(cudaMalloc(&gemmul8_work, ws));
  }

  auto run_once = [&](bool do_debug,
                      float* total_ms_out,
                      float* potrf_ms_out,
                      float* trsm_ms_out,
                      float* syrk_ms_out,
                      float* gemm_ms_out) -> int {
    CHECK_Curand(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    buildInputMatrix<T>(A, n, gen);
    CHECK_Runtime(cudaDeviceSynchronize());

    T* run_A_orig = nullptr;
    if (do_debug) {
      CHECK_Runtime(cudaMalloc(&run_A_orig, sizeof(T) * (size_t)n * (size_t)n));
      CHECK_Runtime(cudaMemcpy(run_A_orig, A, sizeof(T) * (size_t)n * (size_t)n, cudaMemcpyDeviceToDevice));
    }

    float potrf_ms = 0.0f;
    float trsm_ms = 0.0f;
    float syrk_ms = 0.0f;
    float gemm_ms = 0.0f;

    if (n <= 1024) {
      startTimer();
      CHECK_Cusolver(Ops<T>::Potrf(cusolver_handle, n, A, n, work, Lwork, devInfo));
      potrf_ms = stopTimer();
      int info_h = 0;
      CHECK_Runtime(cudaMemcpy(&info_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
      if (info_h != 0) {
        std::printf("potrf failed, info=%d\n", info_h);
        if (run_A_orig) CHECK_Runtime(cudaFree(run_A_orig));
        return 1;
      }
    } else {
      const T one = (T)1;
      const T neg_one = (T)-1;
      for (int j = 0; j < n; j += k) {
        int j_end = (j + k < n) ? (j + k) : n;
        for (int i = j; i < j_end - 1 && i < n; i += nb) {
          int ib = nb;
          if (i + ib > n) ib = n - i;
          if (ib <= 0) break;

          startTimer();
          CHECK_Cusolver(Ops<T>::Potrf(cusolver_handle, ib, A + i + (size_t)i * n, n, work, Lwork, devInfo));
          potrf_ms += stopTimer();

          int info_h = 0;
          CHECK_Runtime(cudaMemcpy(&info_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
          if (info_h != 0) {
            std::printf("potrf failed at i=%d info=%d\n", i, info_h);
            if (run_A_orig) CHECK_Runtime(cudaFree(run_A_orig));
            return 1;
          }

          int below = n - i - ib;
          if (below <= 0) break;

          startTimer();
          CHECK_Cublas(Ops<T>::Trsm(cublas_handle, below, ib, &one,
                                    A + i + (size_t)i * n, n,
                                    A + (i + ib) + (size_t)i * n, n));
          trsm_ms += stopTimer();

          if (j + k - 1 - i - ib <= 0) {
            break;
          }

          int kdim = i + ib - j;
          if (kdim > 0) {
            startTimer();
            CHECK_Cublas(Ops<T>::Syrk(cublas_handle, ib, kdim, &neg_one,
                                      A + (i + ib) + (size_t)j * n, n, &one,
                                      A + (i + ib) + (size_t)(i + ib) * n, n));
            syrk_ms += stopTimer();
          }

          int rem = n - i - 2 * ib;
          if (rem > 0 && kdim > 0) {
            startTimer();
            if (use_gemmul8) {
              gemmul8::gemm<T>(cublas_handle,
                               CUBLAS_OP_N,
                               CUBLAS_OP_T,
                               (size_t)rem,
                               (size_t)ib,
                               (size_t)kdim,
                               &neg_one,
                               A + (i + 2 * ib) + (size_t)j * n,
                               (size_t)n,
                               A + (i + ib) + (size_t)j * n,
                               (size_t)n,
                               &one,
                               A + (i + 2 * ib) + (size_t)(i + ib) * n,
                               (size_t)n,
                               gemmul8_moduli,
                               gemmul8_fastmode,
                               gemmul8_work);
            } else {
              CHECK_Cublas(Ops<T>::GemmNT(cublas_handle, rem, ib, kdim, &neg_one,
                                          A + (i + 2 * ib) + (size_t)j * n, n,
                                          A + (i + ib) + (size_t)j * n, n,
                                          &one,
                                          A + (i + 2 * ib) + (size_t)(i + ib) * n, n));
            }
            gemm_ms += stopTimer();
          }
        }

        // j-level trailing update (matches original double-blocking flow):
        // A(j+k:n, j+k:j+2k) and A(j+2k:n, j+2k:n)
        int tail1 = n - j - k;
        if (tail1 <= 0) {
          break;
        }
        int kblk = k;
        if (kblk > tail1) kblk = tail1;
        int kdim2 = j + k;
        if (kdim2 > 0 && kblk > 0) {
          startTimer();
          CHECK_Cublas(Ops<T>::Syrk(cublas_handle, kblk, kdim2, &neg_one,
                                    A + j + k, n, &one,
                                    A + j + k + (size_t)(j + k) * n, n));
          syrk_ms += stopTimer();

          int tail2 = n - j - k - kblk;
          if (tail2 > 0) {
            startTimer();
            if (use_gemmul8) {
              gemmul8::gemm<T>(cublas_handle,
                               CUBLAS_OP_N,
                               CUBLAS_OP_T,
                               (size_t)tail2,
                               (size_t)kblk,
                               (size_t)kdim2,
                               &neg_one,
                               A + j + k + kblk,
                               (size_t)n,
                               A + j + k,
                               (size_t)n,
                               &one,
                               A + (j + k + kblk) + (size_t)(j + k) * n,
                               (size_t)n,
                               gemmul8_moduli,
                               gemmul8_fastmode,
                               gemmul8_work);
            } else {
              CHECK_Cublas(Ops<T>::GemmNT(cublas_handle, tail2, kblk, kdim2, &neg_one,
                                          A + j + k + kblk, n,
                                          A + j + k, n,
                                          &one,
                                          A + (j + k + kblk) + (size_t)(j + k) * n, n));
            }
            gemm_ms += stopTimer();
          }
        }
      }
    }

    if (do_debug && run_A_orig != nullptr) {
      runDebugResidual<T>(cublas_handle, gen, run_A_orig, A, n);
      CHECK_Runtime(cudaFree(run_A_orig));
    }

    if (total_ms_out) *total_ms_out = potrf_ms + trsm_ms + syrk_ms + gemm_ms;
    if (potrf_ms_out) *potrf_ms_out = potrf_ms;
    if (trsm_ms_out) *trsm_ms_out = trsm_ms;
    if (syrk_ms_out) *syrk_ms_out = syrk_ms;
    if (gemm_ms_out) *gemm_ms_out = gemm_ms;
    return 0;
  };

  for (int i = 0; i < warmup_runs; ++i) {
    if (run_once(false, nullptr, nullptr, nullptr, nullptr, nullptr) != 0) return 1;
  }

  double sum_total_ms = 0.0;
  double sum_potrf_ms = 0.0;
  double sum_trsm_ms = 0.0;
  double sum_syrk_ms = 0.0;
  double sum_gemm_ms = 0.0;
  float min_ms = 1e30f;
  float max_ms = 0.0f;

  for (int i = 0; i < bench_runs; ++i) {
    float total_ms = 0.0f;
    float potrf_ms = 0.0f;
    float trsm_ms = 0.0f;
    float syrk_ms = 0.0f;
    float gemm_ms = 0.0f;
    bool do_debug = debug_mode && (i == bench_runs - 1);
    if (run_once(do_debug, &total_ms, &potrf_ms, &trsm_ms, &syrk_ms, &gemm_ms) != 0) return 1;

    sum_total_ms += total_ms;
    sum_potrf_ms += potrf_ms;
    sum_trsm_ms += trsm_ms;
    sum_syrk_ms += syrk_ms;
    sum_gemm_ms += gemm_ms;
    if (total_ms < min_ms) min_ms = total_ms;
    if (total_ms > max_ms) max_ms = total_ms;
  }

  double avg_total_ms = sum_total_ms / (double)bench_runs;
  double avg_potrf_ms = sum_potrf_ms / (double)bench_runs;
  double avg_trsm_ms = sum_trsm_ms / (double)bench_runs;
  double avg_syrk_ms = sum_syrk_ms / (double)bench_runs;
  double avg_gemm_ms = sum_gemm_ms / (double)bench_runs;

  std::printf("benchmark summary: avg=%.3f ms, min=%.3f ms, max=%.3f ms, runs=%d\n",
              avg_total_ms, min_ms, max_ms, bench_runs);
  std::printf("Left-looking Cholesky avg: %.3f ms: %.6f TFLOPs\n",
              avg_total_ms, (1.0 / 3.0) * (double)n * n * n / (avg_total_ms * 1e9));
  if (avg_potrf_ms > 0) std::printf("potrf avg: %.3f ms\n", avg_potrf_ms);
  if (avg_trsm_ms > 0) std::printf("trsm  avg: %.3f ms\n", avg_trsm_ms);
  if (avg_syrk_ms > 0) std::printf("syrk  avg: %.3f ms\n", avg_syrk_ms);
  if (avg_gemm_ms > 0) std::printf("gemm  avg: %.3f ms\n", avg_gemm_ms);

  CHECK_Runtime(cudaFree(A));
  CHECK_Runtime(cudaFree(work));
  CHECK_Runtime(cudaFree(devInfo));
  if (gemmul8_work) CHECK_Runtime(cudaFree(gemmul8_work));
  CHECK_Curand(curandDestroyGenerator(gen));
  CHECK_Cublas(cublasDestroy(cublas_handle));
  CHECK_Cusolver(cusolverDnDestroy(cusolver_handle));
  return 0;
}

int main(int argc, char* argv[]) {
  if (parseArguments(argc, argv) != 0) return 1;
  if (use_double) return run<double>();
  return run<float>();
}
