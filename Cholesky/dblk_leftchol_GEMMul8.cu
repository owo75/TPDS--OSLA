#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <cusolverDn.h>
#include <curand.h>
#include "../GEMMul8/GEMMul8/include/gemmul8.hpp"
long int n, k, nb;
bool eye_mode = false;
int parseArguments(int argc,char *argv[])
{
    if(argc < 4)
    {
        printf("Usage: %s n k nb [--eye]\n", argv[0]);
        return -1;
    }
    n = atoi(argv[1]);
    k = atoi(argv[2]);
    nb = atoi(argv[3]);
    for (int i = 4; i < argc; ++i) {
        if (strcmp(argv[i], "--eye") == 0) {
            eye_mode = true;
        } else {
            printf("Unknown argument: %s\n", argv[i]);
            return -1;
        }
    }
    return 0;
}

__global__
void clearTri(char uplo, long int m, long int n, float *a, long int lda)
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
void setEye( long int m, long int n, float *a, long int lda)
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

__global__
void addEye( long int m, long int n, float *a, long int lda)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i < m && j < n) {
		if (i == j) 
			a[i+j*lda] += n/10.0;
		// else
		// 	a[i+j*lda] = 0;
	}
}

__global__
void symmetrizeMatrix(int m, float* a, int lda)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < m && j < m && i < j) {
        size_t idx_ij = (size_t)i + (size_t)j * (size_t)lda;
        size_t idx_ji = (size_t)j + (size_t)i * (size_t)lda;
        float v = 0.5f * (a[idx_ij] + a[idx_ji]);
        a[idx_ij] = v;
        a[idx_ji] = v;
    }
}

__global__
void setDiagonalValue(int m, float* a, int lda, float diag_val)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < m) {
        size_t idx = (size_t)i + (size_t)i * (size_t)lda;
        a[idx] = diag_val;
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

void generateNormalMatrix(float *dA,int m,int n)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    int seed = rand()%3000;
	curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, dA, m*n,0,1);
}
__global__ void symmetricFromProduct(float* A, int n) {
    // 计算线程负责的矩阵元素索引 (i, j)
    int i = threadIdx.x + blockDim.x * blockIdx.x; // 行索引
    int j = threadIdx.y + blockDim.y * blockIdx.y; // 列索引

    // 确保线程在有效范围内
    if (i < n && j < n) {
        float value = 0.0f;
        for (int k = 0; k < n; ++k) {
            // 使用列主序的存储规则访问矩阵元素
            value += A[i + k * n] + A[j + k * n];
        }
        A[i + j * n] = value; // 对称矩阵存储在列主序中
    }
}

float computeFrobeniusNorm(long int n, float *dA)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    float dn;
    int incx = 1;
    cublasSnrm2(handle, n * n, dA, incx, &dn);
    cublasDestroy(handle);
    return dn;
}
// 预热函数
void warmUpGPU() {
    float *dummy;
    cudaMalloc(&dummy, sizeof(float));
    cudaFree(dummy);
}
int main(int argc,char *argv[])
{
    if(parseArguments(argc, argv)==-1)
        return 0;
    
    printf("n = %d, eye=%s\n", n, eye_mode ? "on" : "off");

    // 分配设备矩阵
    float *A;
    cudaMalloc(&A, sizeof(float) * n * n);
    dim3 gridDim((n+31)/32,(n+31)/32);
    dim3 blockDim(32,32);
    if (eye_mode) {
        setEye<<<gridDim, blockDim>>>(n, n, A, n);
    } else {
        generateNormalMatrix(A, n, n);
        symmetrizeMatrix<<<gridDim, blockDim>>>(n, A, n);
        dim3 block1d(256);
        dim3 grid1d((n + block1d.x - 1) / block1d.x);
        setDiagonalValue<<<grid1d, block1d>>>(n, A, n, (float)n);
    }
    cudaDeviceSynchronize();

    // // // 接下来的操作保留不变
    // float *L;
    // cudaMalloc(&L, sizeof(float) * n * n);
    // // float *A;
    // // cudaMalloc(&A, sizeof(float)*n*n);
    // float *A_LL;
    // cudaMalloc(&A_LL, sizeof(float) * n * n);
    //  cublasHandle_t cublasHandle;
    //  cublasCreate(&cublasHandle);
    // dim3 gridDim((n+31)/32,(n+31)/32);
    // dim3 blockDim(32,32);
    //  generateNormalMatrix(A,n, n);
    
    //  float ssone = 1.0;
    //  float sszero = 0.0;
    //  cublasSgemmEx( cublasHandle,
    //                 CUBLAS_OP_N,
    //                 CUBLAS_OP_T,
    //                 n,
    //                 n,
    //                 n,
    //                 &ssone,
    //                 A,
    //                 CUDA_R_32F,
    //                 n,
    //                 A,   
    //                 CUDA_R_32F,
    //                 n,
    //                 &sszero,
    //                 A_LL, 
    //                 CUDA_R_32F,
    //                 n);
    // cudaMemcpy(A , A_LL, sizeof(float)*n*n, cudaMemcpyDeviceToDevice); 
    //printMatrixDeviceBlock<float>("1A0.csv", n, n,  A , n);
    // symmetricFromProduct<<<gridDim, blockDim>>>(A, n);

    // cudaDeviceSynchronize();
    // setEye<<<gridDim, blockDim>>>(n ,n ,A, n);
    //printMatrixDeviceBlock<float>("1.csv", n, n, A, n);
    float *A_orig;
    cudaMalloc(&A_orig, sizeof(float) * n * n);
    cudaMemcpy(A_orig, A, sizeof(float) * n * n, cudaMemcpyDeviceToDevice);
    float orig_norm = computeFrobeniusNorm(n, A_orig);
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    cusolverDnHandle_t cusolverHandle;
    cusolverDnCreate(&cusolverHandle);
   // warmUpGPU();
    int Lwork;
    cusolverDnSpotrf_bufferSize(cusolverHandle,
                                CUBLAS_FILL_MODE_LOWER,
                                nb,
                                A,
                                n,
                                &Lwork);
    
        float *work;
        cudaMalloc((void**)&work, sizeof(float) * Lwork);
        int *devInfo;
        cudaMalloc(&devInfo, sizeof(int));
        unsigned int num_moduli_for_gemmul8 = 6;
        bool use_fast_mode_for_gemmul8 = true;
        size_t worksize_gemmul8 = gemmul8::workSize(
                                                    static_cast<unsigned long>(n),      
                                                    static_cast<unsigned long>(n),      
                                                    static_cast<unsigned long>(n),      
                                                    num_moduli_for_gemmul8              
                                                   );
        void *workspace_gemmul8;
        cudaMalloc(&workspace_gemmul8, worksize_gemmul8);
        cudaDeviceSynchronize();
        //startTimer();
        float potrf_time=0.0f, trsm_time=0.0f, syrk_time=0.0f, gemm_time=0.0f, syrk2_time=0.0f, gemm2_time=0.0f;
        float potrf_flops=0.0f, trsm_flops=0.0f, syrk_flops=0.0f, gemm_flops=0.0f, syrk2_flops=0.0f, gemm2_flops=0.0f;
        double gemmul8_total_ns_scaling = 0.0;
        double gemmul8_total_ns_int8gemm = 0.0;
        double gemmul8_total_ns_conv = 0.0;
        double gemmul8_total_ns_invscaling = 0.0;
        if (n <= 1024) {
    
// 直接调用 cuSOLVER Cholesky 分解
startTimer();
float potrf_time=0.0f;
cusolverDnSpotrf(cusolverHandle,
                 CUBLAS_FILL_MODE_LOWER,
                 n,
                 A,
                 n,
                 work,
                 Lwork,
                 devInfo);
float potrf_time1 = stopTimer();

printf("cusolverDnSpotrf execution time: %f ms: %f TFLOPs\n", potrf_time1, 1.0/3.0*n*n*n/potrf_time1/1e9);


} else {
   
    
    // startTimer();
    for (int j=0;j<n;j+=k){

   
    for (int i = j; i < j + k -1 && i < n; i += nb) 
    {
        float ms;
        float snegone = -1.0;
        float sone = 1.0;
        startTimer();
        cusolverDnSpotrf(cusolverHandle,
                         CUBLAS_FILL_MODE_LOWER,
                         nb,
                         A + i + i * n,
                         n,
                         work,
                         Lwork,
                         devInfo );
        potrf_time += stopTimer();
        potrf_flops += 1.0/3.0*nb*nb*nb;
        //  char filename1[50];
        // snprintf(filename1, sizeof(filename1), "1A1_%d.csv", i);


        // printMatrixDeviceBlock<float>(filename1, n , n, A , n);
 
        // 检查 devInfo 是否为零，表示成功
        // int info;
        // cudaMemcpy(&info, devInfo, sizeof(int)*1, cudaMemcpyDeviceToHost);
        // printf("info = %d\n", info);
        // printMatrixDeviceBlock<float>("POTRF.csv", n, n,  A , n); 
       
        if (n - i -  nb <= 0) {
        break; 
        }
        
        
        startTimer();
        cublasStrsm(cublasHandle,
                    CUBLAS_SIDE_RIGHT,
                    CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_T,
                    CUBLAS_DIAG_NON_UNIT,
                    n - i - nb,
                    nb,
                    &sone,
                    A + i + i * n,
                    n,
                    A + (i + nb) + i * n,
                    n);
        // char filename2[50];
        // snprintf(filename2, sizeof(filename2), "1A2_%d.csv", i);
        // printMatrixDeviceBlock<float>(filename2, n , n, A , n);

        trsm_time += stopTimer();
        trsm_flops += (n-i-nb)*nb*nb;
        
        // printMatrixDeviceBlock<float>("L21.csv", n, n,  A , n);         
        if (j + k -1 - i -  nb <= 0) {
            break; 
            }
        startTimer();
       
        cublasSsyrk(cublasHandle,
                    CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N,
                    nb,
                    i+nb-j,
                    &snegone,
                    A + (i + nb)+ j*n,
                    n,
                    &sone,
                    A + i + nb + (i + nb) * n,
                    n);
        // char filename3[50];
        // snprintf(filename3, sizeof(filename3), "1A3_%d.csv", i);
        // printMatrixDeviceBlock<float>(filename3, n , n, A , n); 
       
        syrk_time += stopTimer();
        syrk_flops += nb*nb*(i+nb-j);

        // printMatrixDeviceBlock<float>("A22up.csv", n, n,  A , n); 
        // break; 
        //startTimer();
        if (n - i - 2 * nb <= 0) {
            continue; 
        }
         std::vector<double> gemmul8_internal_times1_ns =  gemmul8::gemm<float>(
                                                                                cublasHandle,
                                                                                CUBLAS_OP_N,
                                                                                CUBLAS_OP_T,
                                                                                static_cast<unsigned long>(n - i - 2 * nb),    
                                                                                static_cast<unsigned long>(nb),                                
                                                                                static_cast<unsigned long>(i + nb - j),              
                                                                                &snegone,                                          
                                                                                A + (i + 2 * nb) + j * n, 
                                                                                static_cast<unsigned long>(n),                                    
                                                                                A + (i + nb) + j * n, 
                                                                                static_cast<unsigned long>(n),                                      
                                                                                &sone,                                              
                                                                                A + i + 2 * nb + (i + nb) * n, 
                                                                                static_cast<unsigned long>(n),                                      
                                                                                num_moduli_for_gemmul8,
                                                                                use_fast_mode_for_gemmul8,
                                                                                workspace_gemmul8
                                                                               );
        double current_gemm_time_ns = 0.0;
                    for(double t_ns : gemmul8_internal_times1_ns) {
                        current_gemm_time_ns += t_ns;
                    }
                    gemm_time += static_cast<float>(current_gemm_time_ns / 1e6); // 纳秒转毫秒

                    gemmul8_total_ns_scaling += gemmul8_internal_times1_ns[0];
                    gemmul8_total_ns_int8gemm += gemmul8_internal_times1_ns[1];
                    gemmul8_total_ns_conv += gemmul8_internal_times1_ns[2];
                    gemmul8_total_ns_invscaling += gemmul8_internal_times1_ns[3];

                    gemm_flops += 2.0 * static_cast<float>(n - i - 2 * nb) * (nb) * (i + nb - j);
      
        // cublasSgemmEx( cublasHandle,
        //                CUBLAS_OP_N,
        //                CUBLAS_OP_T,
        //                n - i - 2 * nb,
        //                nb,
        //                i+nb-j,
        //                &snegone,
        //                A + ( i + 2 * nb ) + j*n,
        //                CUDA_R_32F,
        //                n,
        //                A + ( i + nb )+j*n,   
        //                CUDA_R_32F,
        //                n,
        //                &sone,
        //                A + i + 2 * nb + (i + nb) * n, 
        //                CUDA_R_32F,
        //                n);
        // char filename4[50];
        // snprintf(filename4, sizeof(filename4), "1A4_%d.csv", i);
        // printMatrixDeviceBlock<float>(filename4, n , n, A , n); 
        // gemm_time += stopTimer();
        // gemm_flops += 2.0*(n - i - 2 * nb)*nb*(i+nb-j);
        }
        if (n-k-j<=0) {
            break; 
            }
        float negone = -1.0;
        float one = 1.0;
        startTimer();
        cublasSsyrk(cublasHandle,
                    CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N,
                    k,
                    k+j,
                    &negone,
                    A + j + k,
                    n,
                    &one,
                    A + j + k + (j + k) * n,
                    n);
        // char filename6[50];
        // snprintf(filename6, sizeof(filename6), "1A5_%d.csv", j);
        // printMatrixDeviceBlock<float>(filename6, n , n, A , n); 
        syrk2_time += stopTimer();
        syrk2_flops += k*k*(k+j);
        //startTimer();
        if (n - j - 2 * k <= 0) {
            continue; 
        }
         cudaDeviceSynchronize();
         std::vector<double> gemmul8_internal_times2_ns =  gemmul8::gemm<float>(
                    cublasHandle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    static_cast<unsigned long>(n - j - 2 * k),    // m
                     static_cast<unsigned long>(k),                                 // n
                     static_cast<unsigned long>(j + k),                         // k
                    &negone,                                            // alpha (double*)
                    A + j + 2 * k,             // A (double*)
                    static_cast<unsigned long>(n),                                          // lda
                    A + j + k,                  // B (double*)
                   static_cast<unsigned long>(n),                                          // ldb
                    &one,                                               // beta (double*)
                    A + (j + 2 * k) + (j + k) * n, // C (double*)
                   static_cast<unsigned long>(n),                                          // ldc
                    num_moduli_for_gemmul8,
                    use_fast_mode_for_gemmul8,
                    workspace_gemmul8
                );
        //          char filename6[50];
        // snprintf(filename6, sizeof(filename6), "A6_%d.csv", j);
        // printMatrixDeviceBlock<double>(filename6, n, n, A, n); 
       
      double current_gemm2_time_ns = 0.0;
                for(double t_ns : gemmul8_internal_times2_ns) {
                    current_gemm2_time_ns += t_ns;
                }
                gemm2_time += static_cast<float>(current_gemm2_time_ns / 1e6); // 纳秒转毫秒

                gemmul8_total_ns_scaling += gemmul8_internal_times2_ns[0];
                gemmul8_total_ns_int8gemm += gemmul8_internal_times2_ns[1];
                gemmul8_total_ns_conv += gemmul8_internal_times2_ns[2];
                gemmul8_total_ns_invscaling += gemmul8_internal_times2_ns[3];
                gemm2_flops += 2.0 * static_cast<float>(n - j - 2 * k) * (k) * (j + k);
        // cublasSgemmEx( cublasHandle,
        //                CUBLAS_OP_N,
        //                CUBLAS_OP_T,
        //                n - j - 2 * k,
        //                k,
        //                j+k,
        //                &negone,
        //                A + j + 2 * k ,
        //                CUDA_R_32F,
        //                n,
        //                A + j + k ,   
        //                CUDA_R_32F,
        //                n,
        //                &one,
        //                A + (j + 2 * k) + (j + k) * n, 
        //                CUDA_R_32F,
        //                n);
        // // char filename7[50];
        // // snprintf(filename7, sizeof(filename7), "1A6_%d.csv", j);
        // // printMatrixDeviceBlock<float>(filename7, n , n, A , n); 
        // gemm2_time += stopTimer();
        // gemm2_flops += 2.0*(n - j - 2 * k)*k*(j+k);
//         cudaError_t err = cudaGetLastError();
// if (err != cudaSuccess) {
//     printf("CUDA error: %s\n", cudaGetErrorString(err));
//     return 1;
// }
  //printMatrixDeviceBlock<float>("A22D.csv", n, n,  A , n);  
    }
    // printMatrixDeviceBlock<float>("1A10.csv", n, n,  A , n);  
}
float *L;
float *A_residual;
cudaMalloc(&L, sizeof(float) * n * n);
cudaMalloc(&A_residual, sizeof(float) * n * n);
cudaMemcpy(L, A, sizeof(float) * n * n, cudaMemcpyDeviceToDevice);
cudaMemcpy(A_residual, A_orig, sizeof(float) * n * n, cudaMemcpyDeviceToDevice);

clearTri<<<gridDim, blockDim>>>('u', n, n, L, n);
cudaDeviceSynchronize();
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA kernel launch error for clearTri: %s\n", cudaGetErrorString(err));
    return 1;
}

float alpha = 1.0f;
float beta = -1.0f;
cublasStatus_t status = cublasSsyrk(
    cublasHandle,
    CUBLAS_FILL_MODE_LOWER,
    CUBLAS_OP_N,
    n,
    n,
    &alpha,
    L,
    n,
    &beta,
    A_residual,
    n
);
if (status != CUBLAS_STATUS_SUCCESS) {
    printf("cuBLAS Ssyrk failed with status %d\n", status);
    return 1;
}

float diff_norm = computeFrobeniusNorm(n, A_residual);
float backward_error = diff_norm / orig_norm;
printf("debug backward error ||A-L*L^T||_F/||A||_F = %e\n", backward_error);
    // float ms = stopTimer();

    // printf("Cholesky factorization takes %f ms: %f TFLOPs\n", ms, 1.0/3.0*n*n*n/ms/1e9);
    float ms = potrf_time+trsm_time+gemm_time+syrk_time+gemm2_time+syrk2_time;
    printf("Left-looking Cholesky factorization takes %f ms: %f TFLOPs\n", potrf_time+trsm_time+gemm_time+syrk_time+gemm2_time+syrk2_time, 1.0/3.0*n*n*n/ms/1e9);
    printf("potrf takes %f ms, TFLOPs is %f\n", potrf_time, potrf_flops/potrf_time/1e9);
    printf("trsm takes %f ms, TFLOPs is %f\n", trsm_time, trsm_flops/trsm_time/1e9);
    printf("syrk takes %f ms, TFLOPs is %f\n", syrk_time, syrk_flops/syrk_time/1e9);
    printf("gemm takes %f ms, TFLOPs is %f\n", gemm_time, gemm_flops/gemm_time/1e9);
    printf("syrk2 takes %f ms, TFLOPs is %f\n", syrk2_time, syrk2_flops/syrk2_time/1e9);
    printf("gemm2 takes %f ms, TFLOPs is %f\n", gemm2_time, gemm2_flops/gemm2_time/1e9);
    
  
    cudaFree(A);
    cudaFree(A_orig);
    cudaFree(L);
    cudaFree(A_residual);
    cudaFree(work);
    cudaFree(devInfo);
    // cudaFree(A_reconstructed);  
    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverHandle);

    return 0;
} 
