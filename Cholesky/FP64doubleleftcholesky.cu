#include <iostream> 
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <random>
#include <chrono>
#include <cusolverDn.h>
long int n, k, nb;
int parseArguments(int argc,char *argv[])
{
    if(argc < 4)
    {
        printf("Needs n as inputs\n");
        return -1;
    }
    n = atoi(argv[1]);
    k = atoi(argv[2]);
    nb = atoi(argv[3]);
    return 0;
}

__global__
void clearTri(char uplo, long int m, long int n, double *a, long int lda)
{
	long int i = threadIdx.x + blockDim.x * blockIdx.x;
	long int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i<m && j<n) {
		if (uplo == 'l') {
			if (i>j) {
				a[i+j*lda] = 0;
			}
        } 
        else
        {
            if (i<j)
                a[i+j*lda] = 0;
		}
	}
}

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

__global__
void setEye( long int m, long int n, double *a, long int lda)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
		if (i == j) 
			a[i+j*lda] = 1;
		else
			a[i+j*lda] = 0;
	}
}


cudaEvent_t begin, end;
void startTimer()
{
    cudaEventCreate(&begin);
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



float computeFrobeniusNorm(long int n, double *dA)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    double dn;
    int incx = 1;
    cublasDnrm2(handle, n * n, dA, incx, &dn);
    cublasDestroy(handle);
    return dn;
}
void generateNormalMatrix(double*dA, long int m, long int n) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    int seed = rand() % 3000;
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    size_t numElements = static_cast<size_t>(m) * n; // 避免整数溢出
    curandGenerateNormalDouble(gen, dA, numElements, 0, 1); // 使用size_t参数
    curandDestroyGenerator(gen);
}
int main(int argc,char *argv[])
{
    if(parseArguments(argc, argv)==-1)
        return 0;
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    cusolverDnHandle_t cusolverHandle;
    cusolverDnCreate(&cusolverHandle);
    // --- 1. 生成随机矩阵 B ---
    double *d_B;
    cudaMalloc(&d_B, sizeof(double)*n*n);
    generateNormalMatrix(d_B, n, n);

    // --- 2. 生成 SIGMA 对角向量 ---
    double *d_sigma_diag;
    cudaMalloc(&d_sigma_diag, sizeof(double)*n);
    printf("Generating diagonal for SIGMA (size %ld)...\n", n);
    double *h_sigma_diag = (double*)malloc(sizeof(double)*n);
    srand((unsigned int)time(NULL));
    for(long int i = 0; i < n; ++i) {
        h_sigma_diag[i] = (double)(rand()) / RAND_MAX * 5.0 + 1e-6; // 确保为正数
    }
    cudaMemcpy(d_sigma_diag, h_sigma_diag, sizeof(double)*n, cudaMemcpyHostToDevice);
    free(h_sigma_diag); // 主机内存可以立即释放

    // --- 3. 计算 A = B * SIGMA * B^T ---
    
    // 为中间结果和最终结果分配内存
    double *d_temp; // 用于存储 B * SIGMA
    cudaMalloc(&d_temp, sizeof(double)*n*n);
    double *d_A;   // 最终的对称正定矩阵 A
    cudaMalloc(&d_A, sizeof(double)*n*n);

    // 步骤 3.1: 计算 Temp = B * SIGMA
    cublasDdgmm(cublasHandle, CUBLAS_SIDE_RIGHT, n, n, d_B, n,d_sigma_diag, 1, d_temp, n); 
                             
    // 步骤 3.2: 计算 A = Temp * B^T
    const double alpha = 1.0;
    const double beta = 0.0;
    cublasDgemm(cublasHandle, 
                CUBLAS_OP_N,      // Temp 不转置
                CUBLAS_OP_T,      // B 需要转置
                n, n, n,          
                &alpha,
                d_temp, n,        // 第一个输入矩阵: d_temp
                d_B, n,           // 第二个输入矩阵: d_B (修正!)
                &beta,
                d_A, n);          // 输出矩阵: d_A
    
    // // 备份矩阵 A 
    printf("\nMatrix A has been generated. Backing it up...\n");
    double *d_A_backup;
    cudaMalloc(&d_A_backup, sizeof(double)*n*n);
    cudaMemcpy(d_A_backup, d_A, sizeof(double)*n*n, cudaMemcpyDeviceToDevice);
    
    float orig_norm = computeFrobeniusNorm(n, d_A);


    int Lwork;
    cusolverDnDpotrf_bufferSize(cusolverHandle,
                                CUBLAS_FILL_MODE_LOWER,
                                nb,
                                d_A,
                                n,
                                &Lwork);
    
        double *work;
        cudaMalloc((void**)&work, sizeof(double) * Lwork);
        int *devInfo;
        cudaMalloc(&devInfo, sizeof(int));
        float potrf_time=0.0f, trsm_time=0.0f, syrk_time=0.0f, gemm_time=0.0f, syrk2_time=0.0f, gemm2_time=0.0f;
        float potrf_flops, trsm_flops, syrk_flops, gemm_flops, syrk2_flops, gemm2_flops;
        startTimer();
        // if (n <= 8192) {
        //     cusolverDnDpotrf(cusolverHandle,
        //                     CUBLAS_FILL_MODE_LOWER,
        //                     n,
        //                     A,
        //                     n,
        //                     work,
        //                     Lwork,
        //                     devInfo);



        // } else {
    
    for (int j=0;j<n;j+=k){
    for (int i = j; i < j + k -1 && i < n; i += nb) 
    {
        float ms;
        double snegone = -1.0;
        double sone = 1.0;
        startTimer();
        cusolverDnDpotrf(cusolverHandle,
                         CUBLAS_FILL_MODE_LOWER,
                         nb,
                         d_A + i + i * n,
                         n,
                         work,
                         Lwork,
                         devInfo );
        potrf_time += stopTimer();
        potrf_flops += 1.0/3.0*nb*nb*nb;
       
        if (n - i -  nb <= 0) {
        break; 
        }
        
        
        startTimer();
        cublasDtrsm(cublasHandle,
                    CUBLAS_SIDE_RIGHT,
                    CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_T,
                    CUBLAS_DIAG_NON_UNIT,
                    n - i - nb,
                    nb,
                    &sone,
                    d_A + i + i * n,
                    n,
                    d_A + (i + nb) + i * n,
                    n);
        trsm_time += stopTimer();
        trsm_flops += (n-i-nb)*nb*nb;
        if (j + k -1 - i -  nb <= 0) {
            break; 
            }
        startTimer();
       
        cublasDsyrk(cublasHandle,
                    CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N,
                    nb,
                    i+nb-j,
                    &snegone,
                    d_A + (i + nb)+ j*n,
                    n,
                    &sone,
                    d_A + i + nb + (i + nb) * n,
                    n);
        syrk_time += stopTimer();
        syrk_flops += nb*nb*(i+nb-j);
        if (n - i - 2 * nb <= 0) {
            continue; 
        }
       
        startTimer();
        cublasGemmEx(
                    cublasHandle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    n - i - 2 * nb,
                    nb,
                    i + nb - j,
                    &snegone,
                    d_A + (i + 2 * nb) + j * n,
                    CUDA_R_64F,
                    n,
                    d_A + (i + nb) + j * n,
                    CUDA_R_64F,
                    n,
                    &sone,
                    d_A + i + 2 * nb + (i + nb) * n,
                    CUDA_R_64F,
                    n,
                    CUDA_R_64F,  // computeType
                    CUBLAS_GEMM_DEFAULT
                );
        gemm_time += stopTimer();
        gemm_flops += 2.0*(n - i - 2 * nb)*nb*(i+nb-j);

       
        }
        if (n-k-j<=0) {
            break; 
            }
        double negone = -1.0;
        double one = 1.0;
        startTimer();
        cublasDsyrk(cublasHandle,
                    CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N,
                    k,
                    k+j,
                    &negone,
                    d_A + j + k,
                    n,
                    &one,
                    d_A + j + k + (j + k) * n,
                    n);
        syrk2_time += stopTimer();
        syrk2_flops += k*k*(k+j);
        if (n - j - 2 * k <= 0) {
            continue; 
        }
        startTimer();
        cublasGemmEx(
                    cublasHandle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    n - j - 2 * k,                    // m
                    k,                                // n
                    j + k,                            // k
                    &negone,                          // alpha (double*)
                    d_A + j + 2 * k,                    // A
                    CUDA_R_64F,                       // A type
                    n,                                // lda
                    d_A + j + k,                        // B
                    CUDA_R_64F,                       // B type
                    n,                                // ldb
                    &one,                             // beta (double*)
                    d_A + (j + 2 * k) + (j + k) * n,    // C
                    CUDA_R_64F,                       // C type
                    n,                                // ldc
                    CUDA_R_64F,                       // computeType
                    CUBLAS_GEMM_DEFAULT               // algorithm
                );
        gemm2_time += stopTimer();
        gemm2_flops += 2.0*(n - j - 2 * k)*k*(j+k);

       
    }


// }


    double *L;
    cudaMalloc(&L, sizeof(double) * n * n);
    cudaMemcpy(L, d_A, sizeof(double) * n * n, cudaMemcpyDeviceToDevice);
    dim3 gridDim((n+31)/32,(n+31)/32);
    dim3 blockDim(32,32);
    clearTri<<<gridDim, blockDim>>>('u', n, n, L, n);
 //printMatrixDeviceBlock<float>("1LL.csv", n, n,  L , n);  
double *d_A_reconstructed;
cudaMalloc(&d_A_reconstructed, sizeof(double) * n * n);
// 1. 重构 A = L*L^T
const double one = 1.0;
const double zero = 0.0;
cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, 
            &one, L, n, L, n, &zero, d_A_reconstructed, n);

// 2. 计算残差: d_A_backup = 1.0*d_A_backup - 1.0*d_A_reconstructed
const double neg_one = -1.0;
cublasDgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n,
            &one, d_A_backup, n,
            &neg_one, d_A_reconstructed, n,
            d_A_backup, n); // 结果写回 d_A_backup

cudaFree(d_A_reconstructed); // 释放临时内存


float diff_norm = computeFrobeniusNorm(n, d_A_backup);
printf("diff_norm: %e\n", diff_norm);
printf("orig_norm: %e\n", orig_norm);
float backward_error = diff_norm / orig_norm;
printf("Backward error: %e\n", backward_error);
float ms = potrf_time+trsm_time+gemm_time+syrk_time+gemm2_time+syrk2_time;
printf("Left-looking Cholesky factorization takes %f ms: %f TFLOPs\n", potrf_time+trsm_time+gemm_time+syrk_time+gemm2_time+syrk2_time, 1.0/3.0*n*n*n/ms/1e9);
printf("potrf takes %f ms, TFLOPs is %f\n", potrf_time, potrf_flops/potrf_time/1e9);
printf("trsm takes %f ms, TFLOPs is %f\n", trsm_time, trsm_flops/trsm_time/1e9);
printf("syrk takes %f ms, TFLOPs is %f\n", syrk_time, syrk_flops/syrk_time/1e9);
printf("gemm takes %f ms, TFLOPs is %f\n", gemm_time, gemm_flops/gemm_time/1e9);
printf("syrk2 takes %f ms, TFLOPs is %f\n", syrk2_time, syrk2_flops/syrk2_time/1e9);
printf("gemm2 takes %f ms, TFLOPs is %f\n", gemm2_time, gemm2_flops/gemm2_time/1e9);

    
  
    cudaFree(d_A);
    cudaFree(work);
    cudaFree(devInfo);
    // cudaFree(A_reconstructed);  
    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverHandle);

    return 0;
} 