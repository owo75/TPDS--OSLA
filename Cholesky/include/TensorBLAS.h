#include <iostream> 
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_fp16.h>
#include <fstream>
#include <vector>
#include <cusolverDn.h>
#include <string>
#include <cumpsgemm/cumpsgemm.hpp>

__global__
void s2h(long int m, long int n, float *as, long int ldas, __half *ah, long int ldah);

__global__
void s2hTranspose(long int m, long int n, float *as, __half *ah);

__global__
void row2col(long int m, long int n, float *as, float *ad);

__global__
void clearTri(char uplo, long int m, long int n, float *a, long int lda);

__global__
void clearTriDouble(char uplo, long int m, long int n, double *a, long int lda);

__global__
void setEyePlus( long int m, long int n, float *a, long int lda);

__global__
void setEye( long int m, long int n, float *a, long int lda);

__global__
void setInitialValue( long int m, long int n, float *a, long int lda, float val);

__global__
void setInitialValueDouble( long int m, long int n, double *a, long int lda, double val);

__global__
void matrixCpy(long int m, long int n, float *a, long int lda, float *b, long int ldb);

__global__
void matrixCpyDouble(long int m, long int n, double *a, long int lda, double *b, long int ldb);

__global__ 
void copy_lower_to_upper(long int n, float *A, long int lda);
__global__ 
void copy_lower_to_upperDouble(long int n, double *A, long int lda);
__global__
void transpose(int m, int n, float* dA, float *tmpA);

void beginTimer();

float endTimer();

void startTimer();

float stopTimer();

void generateNormalMatrix(float *dA,long int m,long int n);

void generateNormalMatrixDouble(double *dA,long int m,long int n);

void generateUniformMatrix(float *dA,long int m, long int n);

void generateUniformMatrixDouble(double *dA,long int m,long int n);

float snorm(long int m, long int n, float *d_A, long int lda);
double snormDouble(long int m, long int n, double *d_A, long int lda);
size_t free_mem();


// float snorm(long int m, long int n, float *d_A);

int64_t* find_mat_size_syrk(int64_t n, int *length);

int64_t* find_mat_size_trsm(int64_t n, int *length);


template<typename T>
void printMatrixDeviceBlock(char *filename,int m, int n, T* dA, int lda)
{
    FILE *f = fopen(filename, "w");
	if (f == NULL) {
		printf("fault!\n");
		return;
	}
    //printf("Perform printmatrixdevice\n");
    float *ha;
    ha = (float*)malloc(sizeof(float));

    for(int i = 0;i<m;i++)
    {
        for(int j = 0;j<n;j++)
        {
            cudaMemcpy(&ha[0], &dA[i+j*lda], sizeof(float), cudaMemcpyDeviceToHost);
            fprintf(f, "%lf", ha[0]);
            if (j == n - 1) fprintf(f, "\n");
			else fprintf(f, ",");
        }
    }
    fclose(f);
	//cudaMemcpy(ha, dA, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    //printMatrixFloat(filename, m, n, ha, lda);
    free(ha);
}
template<typename T>
void printMatrixDeviceBlockDouble(char *filename,int m, int n, T* dA, int lda)
{
    FILE *f = fopen(filename, "w");
	if (f == NULL) {
		printf("fault!\n");
		return;
	}
    //printf("Perform printmatrixdevice\n");
    double *ha;
    ha = (double*)malloc(sizeof(double));

    for(int i = 0;i<m;i++)
    {
        for(int j = 0;j<n;j++)
        {
            cudaMemcpy(&ha[0], &dA[i+j*lda], sizeof(double), cudaMemcpyDeviceToHost);
            fprintf(f, "%lf", ha[0]);
            if (j == n - 1) fprintf(f, "\n");
			else fprintf(f, ",");
        }
    }
    fclose(f);
	//cudaMemcpy(ha, dA, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    //printMatrixFloat(filename, m, n, ha, lda);
    free(ha);
}
extern "C" __attribute__((visibility("default"))) void print_env();

extern "C" __attribute__((visibility("default"))) int tc_syrk_wrapper(long int n, long int k, float* A, float* C, long int nb);

void tc_syrk(cublasHandle_t handle, long int n,long int k,  float alpha, float* A, long int lda, float beta, float* C, long int ldc, __half* Ah, long int nb);

void tc_cumpsgemm_syrk(cumpsgemm::handle_t cumpsgemm_handle, long int n,long int k,  float alpha, float* A, long int lda, float beta, float* C, long int ldc, long int nb);

void tc_ozimmu_syrk(cublasHandle_t handle, long int n, long int k,  double alpha, double* A, long int lda, double beta, double* C, long int ldc, long int nb);

extern "C" __attribute__((visibility("default"))) int tc_trsm_wrapper(long int m, long int n, float* A, float* B, long int nb);

void tc_trsm(cublasHandle_t handle, long int m, long int n, float* A, long int lda, float* B, long int ldb, __half* hwork, long int nb);

void tc_cumpsgemm_trsm(cublasHandle_t handle, cumpsgemm::handle_t cumpsgemm_handle, long int m, long int n, float* A, long int lda, float* B, long int ldb,long int nb);

void tc_ozimmu_trsm(cublasHandle_t handle, long int m, long int n, double* A, long int lda, double* B, long int ldb, long int nb);

void tc_trmm(cublasHandle_t handle, long int m, long int n, float alpha, float* A, long int lda, float* B, long int ldb, float* C, long int ldc, __half* hwork, long int nb);

void tc_cumpsgemm_trmm(cumpsgemm::handle_t cumpsgemm_handle, long int m, long int n, float alpha, float* A, long int lda, float* B, long int ldb, float* C, long int ldc, long int nb);

void tc_ozimmu_trmm(cublasHandle_t handle, long int m, long int n, double alpha, double* A, long int lda, double* B, long int ldb, double* C, long int ldc, long int nb);

void tc_syr2k(cublasHandle_t handle, long int n, long int k, float alpha, float* A, long int lda, float* B, long int ldb, float beta, float* C, long int ldc, __half* Ah, long int nb);

void tc_cumpsgemm_syr2k(cumpsgemm::handle_t cumpsgemm_handle, long int n, long int k, float alpha, float* A, long int lda, float* B, long int ldb, float beta, float* C, long int ldc, long int nb);

void tc_ozimmu_syr2k(cublasHandle_t handle, long int n, long int k,  double alpha, double* A, long int lda, double* B, long int ldb, double beta, double* C, long int ldc, long int nb);

void tc_symm(cublasHandle_t handle, long int m, long int n,  float alpha, float* A, long int lda, float* B, int ldb, float beta, float* C, long int ldc, __half* Ah);

void tc_cumpsgemm_symm(cumpsgemm::handle_t cumpsgemm_handle, long int m, long int n,  float alpha, float* A, long int lda, float* B, int ldb, float beta, float* C, long int ldc);

const float sone = 1.0;
const float snegone = -1.0;
const float szero = 0.0;

template <unsigned Size>
__device__ inline void cp(void* const dst, const void* const src) {
	static_assert(Size == 4 || Size == 8 || Size == 16, "Size must be one of 4, 8 and 16");
	if (Size == 4) {
		*(reinterpret_cast<uint32_t*>(dst)) = *(reinterpret_cast<const uint32_t*>(src));
	} else if (Size == 8) {
		*(reinterpret_cast<uint64_t*>(dst)) = *(reinterpret_cast<const uint64_t*>(src));
	} else {
		*(reinterpret_cast<ulong2*>(dst)) = *(reinterpret_cast<const ulong2*>(src));
	}
}

template <class IdxT, unsigned block_size, unsigned smem_len = block_size * 8>
__global__
void s2h_swpipe(const IdxT m, const IdxT n, const float * const as, int ldas, __half *ah, int ldah)
{
	__shared__ float smem_f32[smem_len];
	__shared__ half smem_f16[smem_len];

	const auto in = blockIdx.x;

	for (unsigned i = 0; i < m; i += smem_len) {
		if (i + smem_len <= m) {
			// Load FP32 elements
			if (reinterpret_cast<long>(ah) % 16 == 0 && ldah % 4 == 0) {
				for (unsigned j = 0; j < smem_len; j += block_size * 4) {
					const auto smem_i = j + threadIdx.x * 4;
					if (smem_len < block_size * 4 && smem_i >= smem_len) break;
					const auto im = i + smem_i;
					cp<16>(&smem_f32[smem_i], &as[im + ldas * in]);
				}
				__syncthreads();
			} else if (reinterpret_cast<long>(ah) % 8 == 0 && ldah % 2 == 0) {
				for (unsigned j = 0; j < smem_len; j += block_size * 2) {
					const auto smem_i = j + threadIdx.x * 2;
					if (smem_len < block_size * 2 && smem_i >= smem_len) break;
					const auto im = i + smem_i;
					cp<8>(&smem_f32[smem_i], &as[im + ldas * in]);
				}
				__syncthreads();
			} else {
				for (unsigned j = 0; j < smem_len; j += block_size) {
					const auto smem_i = j + threadIdx.x;
					const auto im = i + smem_i;
					cp<4>(&smem_f32[smem_i], &as[im + ldas * in]);
				}
			}
			// Convert to FP16
			for (unsigned j = 0; j < smem_len; j += block_size) {
				const auto smem_i = j + threadIdx.x;
				smem_f16[smem_i] = __float2half(smem_f32[smem_i]);
			}
			// Store FP16s
			if (reinterpret_cast<long>(ah) % 16 == 0 && ldah % 8 == 0) {
				__syncthreads();
				for (unsigned j = 0; j < smem_len; j += block_size * 8) {
					const auto smem_i = j + threadIdx.x * 8;
					if (smem_len < block_size * 8 && smem_i >= smem_len) break;
					const auto im = i + smem_i;
					cp<16>(&ah[im + ldah * in], &smem_f16[smem_i]);
				}
			} else if (reinterpret_cast<long>(ah) % 8 == 0 && ldah % 4 == 0) {
				__syncthreads();
				for (unsigned j = 0; j < smem_len; j += block_size * 4) {
					const auto smem_i = j + threadIdx.x * 4;
					if (smem_len < block_size * 4 && smem_i >= smem_len) break;
					const auto im = i + smem_i;
					cp<8>(&ah[im + ldah * in], &smem_f16[smem_i]);
				}
			} else if (reinterpret_cast<long>(ah) % 4 == 0 && ldah % 2 == 0) {
				__syncthreads();
				for (unsigned j = 0; j < smem_len; j += block_size * 2) {
					const auto smem_i = j + threadIdx.x * 2;
					if (smem_len < block_size * 2 && smem_i >= smem_len) break;
					const auto im = i + smem_i;
					cp<4>(&ah[im + ldah * in], &smem_f16[smem_i]);
				}
			} else {
				for (unsigned j = 0; j < smem_len; j += block_size) {
					const auto smem_i = j + threadIdx.x;
					const auto im = i + smem_i;
					ah[im + ldah * in] = smem_f16[smem_i];
				}
			}
		} else {
			// Load FP32 elements
			unsigned j = 0;
			for (; j < smem_len; j += block_size) {
				const auto smem_i = j + threadIdx.x;
				const auto im = i + smem_i;
				if (im < m) {
					smem_f32[smem_i] = as[im + ldas * in];
				} else {
					break;
				}
			}
			const unsigned max_j = j;

			// Convert to FP16
			for (unsigned j = 0; j < max_j; j += block_size) {
				const auto smem_i = j + threadIdx.x;
				smem_f16[smem_i] = __float2half(smem_f32[smem_i]);
			}
			// Store FP16s
			for (unsigned j = 0; j < max_j; j += block_size) {
				const auto smem_i = j + threadIdx.x;
				const auto im = i + smem_i;
				ah[im + ldah * in] = smem_f16[smem_i];
			}
		}
	}
}