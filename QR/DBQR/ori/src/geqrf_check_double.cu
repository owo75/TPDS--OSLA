// #include "TSQR.h"
#include "TallShinnyQR.h"
#include "kernelReWY-double.h"


/**
 * V7第一版本
 * 1. 在V6的基础上合并了两个矩阵更新
 * 2. 尝试重新加上转置并验证正确性
 */


#define threadsPerDim 16
// #define MYDEBUG1
// #define MYDEBUG3
// #define MYDEBUG4

// 启动TIMEFUNC会计算具体函数时间，否则计算double blocking时间
#define TIMEFUNC

static float tsqrTime1 = 0;  
static float GemmTime1 = 0;  
static float reconstructWYTime1 = 0;  
static float TailMatrixTime1 = 0;  
static float TailMatrixTime2 = 0;
static float WYTime1 = 0;  
static float fullWYTime1 = 0;  
static float getUTime = 0;  
static float transWTime = 0;  
int count = 0;
// 
__global__ void copySubmatrixToFull(double* A, int lda, double* A2, int lda2, int m_A2, int n_A2, int posi_m, int posi_n) {  
    int col = blockIdx.x * blockDim.x + threadIdx.x;  

    if (col < n_A2) {  
        // 计算A2当前列的起始地址  
        double* A2_col = A2 + col * lda2;          
        // 计算A中目标位置当前列的起始地址  
        double* A_col = A + (posi_n + col) * lda + posi_m;              

        for (int row = 0; row < m_A2; ++row) {  
            A_col[row] = A2_col[row];  
        }  
    }  
}  

//---------------------------------------------
// 逻辑1：首先判断是不是后块，如果是的话，
//      进行当前块的更新；执行tsqr+reWY（可能递归）
//      最后执行WY的拼接
//      如果是前块，直接执行tsqr+reWY（可能递归）
// 逻辑2：判断i == b，如果不等，那么一定会进行
//      更新和拼接。貌似也很合理
// 当前使用逻辑2：逻辑2是不是不用去判断前块和后块
// 问题1：currn和currm怎么去变化
// 问题2：传入的已经是对应的地址了，不需要再加上整个的currm和currn了
// 问题3：currm和currn是不是都不需要: 好像是的
//---------------------------------------------
void TSQRSegment(double *d_A, double *d_W, 
                 double *d_Y, double *d_R,  
                 double *work1, double *work2,  
                 int ldwork1,int lda, 
                 int cols, int rows,        
                 int b, cublasHandle_t cublas_handle
                )  
{  
    // std::ofstream file("matrix_output6.csv", std::ios::trunc);  
    // file.close();  
    
    // CudaHandles handles
    // CudaHandles handles2;
    if (cols == b) {  
        // --------------------------------  
        // 处理大小为 b 的子块  
        // --------------------------------  
        // 1. 做 TSQR
        //    需要保证当前传入的dA就是计算矩阵的左上角
#ifdef MYDEBUG4
    std::string fileName = "matrix_output6.csv";  
    std::string MatrixName = "input part A";  
    writeMatrixToCsvV2(d_A, lda, rows, cols, fileName, MatrixName);
#endif
#ifdef  TIMEFUNC
            startTimer();
#endif
        tsqr<double>(cublas_handle, rows, cols, d_A, lda, d_R, cols, work1, ldwork1);  
        // cudaDeviceSynchronize();  
#ifdef  TIMEFUNC
            tsqrTime1 += stopTimer();
#endif
#ifdef MYDEBUG4
    MatrixName = "Q after tsqr";  
    writeMatrixToCsvV2(d_A, lda, rows, cols, fileName, MatrixName);
    MatrixName = "R after tsqr";
    writeMatrixToCsvV2(d_R, cols, cols, cols, fileName, MatrixName);
#endif
#ifdef  TIMEFUNC
            startTimer();
#endif
        // 2. ReconstructWY (求出 W、Y)  
        ReconstructWYKernel(d_A, lda, d_W, lda, d_Y, lda, work2, rows, cols);  
        cudaDeviceSynchronize();  
#ifdef  TIMEFUNC
            reconstructWYTime1 += stopTimer();
#endif
#ifdef MYDEBUG4
    MatrixName = "W after ReconstructWY";  
    writeMatrixToCsvV2(d_W, lda, rows, cols, fileName, MatrixName);
    MatrixName = "Y after ReconstructWY";
    writeMatrixToCsvV2(d_Y, lda, rows, cols, fileName, MatrixName);
#endif
#ifdef  TIMEFUNC
            startTimer();
#endif
        // 3. 将 R 写回 A 的前 b×b 块(上三角)  
        dim3 blockDim(threadsPerDim, threadsPerDim);
        dim3 gridDim((cols + threadsPerDim -1) / threadsPerDim, (cols + threadsPerDim -1) / threadsPerDim);
        getU<<<gridDim, blockDim>>>(cols, cols, d_R, cols, d_A, lda);
        cudaDeviceSynchronize();  
#ifdef  TIMEFUNC
            getUTime += stopTimer();
#endif
    } else {  
        // --------------------------------  
        // 处理大小 i > b 的块：拆成前后两部分  
        // --------------------------------  
        int half_i = cols / 2;  
        double alpha = 1.0;
        double beta = 0.0;
        double minusAlpha = -1.0;
        // 1. 递归调用：处理前 half_i 列  
        TSQRSegment(d_A, d_W, d_Y, d_R,  
                    work1, work2,
                    ldwork1, lda,
                    half_i, rows,  
                    b, cublas_handle
                    );  

        // 2. 更新后面half_i 列  
        //    更新就是利用当前块的i进行更新
        // 维度 W,Y：rows * half_i   A: rows * half_i
        // 起始坐标 W,Y：currm, currn   A: currm, currn + half_i    
        // 先计算 work1 = W' * A
#ifdef  TIMEFUNC
                startTimer();
#endif
        CHECK_CUBLAS(cublasDgemm(  
                    cublas_handle,  
                    CUBLAS_OP_T, // W'  
                    CUBLAS_OP_N, // A
                    half_i,        
                    half_i,        
                    rows,          
                    &alpha,  
                    d_W, // W的起始地址  
                    lda,         // lda  
                    d_A + half_i * lda,     // A1的起始地址  
                    lda,         // lda  
                    &beta,  
                    work1,     // temp的起始地址  
                    half_i          
                ));  
        // 在计算A - Y*temp
        CHECK_CUBLAS(cublasDgemm(  
                    cublas_handle,  
                    CUBLAS_OP_N, // Y  
                    CUBLAS_OP_N, // temp  
                    rows,           
                    half_i,        
                    half_i,       
                    &minusAlpha,  
                    d_Y, // Y的起始地址  
                    lda,         // lda  
                    work1,       // temp的起始地址  
                    half_i,         
                    &alpha,  
                    d_A + half_i * lda, 
                    lda          // lda  
                ));  
        cudaDeviceSynchronize();  
#ifdef  TIMEFUNC
            GemmTime1 += stopTimer();
#endif
        // // 还需要修改currm，currn，rows, cols
        // currm = currm + half_i;
        // currn = currn + half_i;


        // 3. 递归调用：处理后 half_i 列  
        TSQRSegment(d_A + half_i + half_i * lda, d_W + half_i + half_i * lda, d_Y + half_i + half_i * lda, d_R,  
                    work1, work2,
                    ldwork1, lda,
                    half_i, rows - half_i,  
                    b, cublas_handle
                    );  

        // 4. 前、后两部分的 W、Y 拼接  
        // W = [W1, W - W1 * Y1' * W]  使用完整的WY进行计算
        // 维度 W1: rows, half_i, W: rows, half_i
        // 起始坐标 W1: currm, currn, W: currm, currn + half_i 
#ifdef  TIMEFUNC
                startTimer();
#endif
        CHECK_CUBLAS(cublasDgemm(  
                    cublas_handle,  
                    CUBLAS_OP_T, // Y1'
                    CUBLAS_OP_N, // W  
                    half_i,        
                    half_i,        
                    rows,          
                    &alpha,  
                    d_Y,         // Y1的起始地址  
                    lda,         // lda  
                    d_W + half_i * lda, // W的起始地址  
                    lda,         // lda  
                    &beta,  
                    work1,     // temp的起始地址  
                    half_i            
                ));  
                
        CHECK_CUBLAS(cublasDgemm(  
                    cublas_handle,  //
                    CUBLAS_OP_N, // W1  
                    CUBLAS_OP_N, // temp  
                    rows,           
                    half_i,        
                    half_i,       
                    &minusAlpha,  
                    d_W, // W1起始地址  
                    lda,         // lda  
                    work1,     // temp的起始地址  
                    half_i,         
                    &alpha,  
                    d_W + half_i * lda,     // W的起始地址  
                    lda          // lda  
                ));  
        cudaDeviceSynchronize();  
#ifdef  TIMEFUNC
            WYTime1 += stopTimer();
#endif
    }  

}  


void IterativeQR(double* d_A, int m, int n, 
                 int nb, int b, 
                 double* d_W, double* d_Y, int lda,     // dW，dY，dA的ld都是lda
                 double* d_Q, double* d_R, double* work1,  //d_Q, d_R, work不需要lda，可以任意修改矩阵形状，只要矩阵大小不超过m*n
                 double* work2, double* transW, int ldwork1, int ldwork2,
                 cublasHandle_t cublas_handle) {
    int currm = 0;
    int currn = 0;
    bool flag1 = true; // 用于判断 n 中的第一个 nb
    int j;
    double* d_W_current;   
    double* d_Y_current; 
    double* d_A_current;   
    std::ofstream file("matrix_output7right.csv", std::ios::trunc);  
    file.close();  
    std::string fileName = "matrix_output7right.csv";  
    std::string MatrixName = "A";

    //将这里的while循环改成for循环，有利于标记当前是第几个块
    for(j = 0; j * nb < n; j++)
    {
        int nt = nb;
        int i = b;
        int prei = b;     //保存上一个i值
        bool flag = true; // 用于判断在 nb 内的第一个 b
        double alpha = 1.0;  
        double beta = 0.0;  
        double minusAlpha = -1.0;  
        //d_W_current不能和currn一起用

        while (i <= nt / 2) {
            int rows = m - currm;
            int cols = i;
            // int colsnum = i;  // 这个参数是保存当前循环开始的i，方便利用currm定位WY的起始地址，而不用分类讨论
            // TSQR + ReconstructWY
            TSQRSegment(d_A + currm + currn * lda, d_W + currm + currn * lda,
                        d_Y + currm + currn * lda, d_R,
                        work1, work2, 
                        ldwork1, lda,
                        cols, rows, 
                        b, cublas_handle);  


            // 流同步函数
            // CHECK_CUDA(cudaDeviceSynchronize());  


            d_W_current = d_W + currm + currn * lda;   
            d_Y_current = d_Y + currm + currn * lda; 
            d_A_current = d_A + currm + currn * lda;   

            if (!flag) {
                // colsnum = i;
                prei = i;
                i *= 2;
            }
            // A = (I - W * Y')' * A
            // i表示第一个更新块的列数；WY和A是不一样的存储地址
            if (i != nt) {
#ifdef  TIMEFUNC
                startTimer();
#endif
            // MatrixName = "W";
            // writeMatrixToCsvV2(d_W_current, lda, m - currm, prei, fileName, MatrixName);
            // MatrixName = "A";
            // writeMatrixToCsvV2( d_A_current + prei * lda, lda, m - currm, nb - i, fileName, MatrixName);
               // 更新完整的块
                // 维度 W，Y：(m - currm) * prei A:(m - currm) * nb - i
                // 起始坐标W,Y：currm, currn   A: currm, currn + prei
                CHECK_CUBLAS(cublasDgemm(  
                    cublas_handle,  
                    CUBLAS_OP_T, // W'  
                    CUBLAS_OP_N, // A
                    prei,        
                    nb - i,        
                    m - currm,          
                    &alpha,  
                    d_W_current, // W的起始地址  
                    lda,         // lda  
                    d_A_current + prei * lda,     // A1的起始地址  
                    lda,         // lda  
                    &beta,  
                    work1,     // temp的起始地址  
                    prei          
                ));  
            // MatrixName = "work1";
            // writeMatrixToCsvV2(work1, prei, prei, nb - i, fileName, MatrixName);
            // MatrixName = "workhalf";
            // writeMatrixToCsvV2(work1, prei, prei, nb - i, fileName, MatrixName);
            // MatrixName = "Yhalf";
            // writeMatrixToCsvV2(d_Y_current, lda, m - currm, prei, fileName, MatrixName);
                // 在计算A - Y*temp
                CHECK_CUBLAS(cublasDgemm(  
                    cublas_handle,  
                    CUBLAS_OP_N, // Y  
                    CUBLAS_OP_N, // temp  
                    m - currm,           
                    nb - i,        
                    prei,       
                    &minusAlpha,  
                    d_Y_current, // Y的起始地址  
                    lda,         // lda  
                    work1,       // temp的起始地址  
                    prei,         
                    &alpha,  
                    d_A_current + prei * lda, 
                    lda          // lda  
                ));  
                                MatrixName = "更新的A";
                // writeMatrixToCsvV2(d_A_current + prei * lda, m, m - currm, nb - i, fileName, MatrixName);
                // cudaDeviceSynchronize();  
#ifdef  TIMEFUNC
            GemmTime1 += stopTimer();
#endif


            }
// #ifdef MYDEBUG5
//                 // printf("更新尾随矩阵\n");
//                 // printDeviceMatrixV2(d_A, m, m, n);
//                 MatrixName = "更新的A";
//                 writeMatrixToCsvV2(d_A_current + prei * lda, m, m - currm, nb - i, fileName, MatrixName);
// #endif
            // 更新W矩阵
            if(!flag){
                // W = [W1, W - W1 * Y1' * W]  Y1' * W部分可以用下面非0，前面的W和W1需要用整列
                // 维度 W1: m - currm, currn - j * nb, W: m - currm, prei
                // 起始坐标 W1: currm, j * nb, W: currm, currn 
#ifdef  TIMEFUNC
                startTimer();
#endif
                CHECK_CUBLAS(cublasDgemm(  
                    cublas_handle,  // 2
                    CUBLAS_OP_T, // Y1'
                    CUBLAS_OP_N, // W  
                    currn - j * nb,        
                    prei,        
                    m - currm,          
                    &alpha,  
                    d_Y + currm + j * nb * lda, // Y1的起始地址  
                    lda,         // lda  
                    d_W_current, // W的起始地址  
                    lda,         // lda  
                    &beta,  
                    work1,     // temp的起始地址  
                    currn - j * nb            
                ));  
                // WheW1起始坐标需要改变一下，变成完整的计算
                CHECK_CUBLAS(cublasDgemm(  
                    cublas_handle,  //
                    CUBLAS_OP_N, // W1  
                    CUBLAS_OP_N, // temp  
                    m - j * nb,           
                    prei,        
                    currm - j * nb,       
                    &minusAlpha,  
                    d_W + j * nb + j * nb * lda, // W1起始地址  
                    lda,         // lda  
                    work1,     // temp的起始地址  
                    currn - j * nb,         
                    &alpha,  
                    d_W + j * nb + currn * lda,     // W的起始地址  
                    lda          // lda  
                ));  
                cudaDeviceSynchronize();  
#ifdef  TIMEFUNC
            WYTime1 += stopTimer();
#endif
#ifdef MYDEBUG1
            printf("更新后的W矩阵\n");
            printf("d_W\n");
            printDeviceMatrixV2(d_W,lda,m,n);
#endif 
            }
            else{
                flag = false;
            }

            currm += prei;         
            currn += prei;
        }

        if (currn != n) {
            // 更新整个尾随矩阵
            // A(currm - nb : m, currn : n) = (I - W * Y')' * A( : m, currn : n)
            // 维度 A: (m - j * nb) * (n - (j + 1) * nb)   W,Y: m - j * nb， nb
            // 起始 A: currm - nb, currn           W,Y: currm - nb, currn - nb
            // float* d_A_current = d_A + j * nb + (j + 1) * nb * lda;  

            // 调整currn和currm的值
            // currm = currm - nb;
            // currn = currn - nb;
            d_W_current = d_W + currm - nb + (currn - nb) * lda;   
            d_Y_current = d_Y + currm - nb + (currn - nb) * lda;    
            d_A_current = d_A + currm - nb + currn * lda;   
#ifdef MYDEBUG
            // printf("更新大的尾随矩阵\n");
            // printDeviceMatrixV2(d_A, m, m, n);
            std::string MatrixName = "初始的A矩阵";
            writeMatrixToCsvV2(d_A_current, m, m - j * nb, n - (j + 1) * nb, fileName, MatrixName);
            MatrixName = "更新前的W矩阵";
            writeMatrixToCsvV2(d_W, m, m, nb, fileName, MatrixName);
            MatrixName = "更新前的Y矩阵";
            writeMatrixToCsvV2(d_Y, m, m, nb, fileName, MatrixName);
#endif
            
#ifdef  TIMEFUNC
                startTimer();
#endif
            CHECK_CUBLAS(cublasDgeam(  
                cublas_handle,  
                CUBLAS_OP_T, // 对矩阵 W 进行转置  
                CUBLAS_OP_N, 
                nb,          
                m - j * nb,  
                &alpha,      // alpha = 1  
                d_W_current, // 矩阵 W 的起始地址  
                m,         // 矩阵 W 的主维度  
                &beta,       // beta = 0  
                nullptr,     // 矩阵 B 不参与运算（为空）  
                m,         // 占位  
                transW,    // 转置后的矩阵 W 存储到 d_transW  
                nb            // 转置后结果矩阵的主维度  
            )); 



#ifdef  TIMEFUNC
                transWTime += stopTimer();
#endif



#ifdef  TIMEFUNC
                startTimer();
#endif
            
            // CHECK_CUBLAS(cublasSgemm(  
            //     cublas_handle,  
            //     CUBLAS_OP_T, // W'  
            //     CUBLAS_OP_N, // A_sub  
            //     nb,        
            //     n - (j + 1) * nb,        
            //     m - j * nb,          
            //     &alpha,  
            //     d_W_current, // W的起始地址  
            //     lda,         // lda  
            //     d_A_current,     // A_current的起始地址  
            //     lda,         // lda  
            //     &beta,  
            //     work2,     // temp的起始地址  
            //     nb            
            // ));  
            CHECK_CUBLAS(cublasDgemm(  
                cublas_handle,  
                CUBLAS_OP_N, // W'  
                CUBLAS_OP_N, // A_sub  
                nb,        
                n - (j + 1) * nb,        
                m - j * nb,          
                &alpha,  
                transW, // W的起始地址  
                nb,         // lda  
                d_A_current,     // A_current的起始地址  
                lda,         // lda  
                &beta,  
                work2,     // temp的起始地址  
                nb            
            ));  
#ifdef  TIMEFUNC
            float temptime1 = stopTimer();
            long long t = (long long)(m - j * nb) * (n - (j+1) * nb) * (2LL * nb); 
            printf("第1部分时间为%.4f tflops = %.4f\n", temptime1, t / temptime1 / 1e9);
            TailMatrixTime1 += temptime1;
#endif
#ifdef  TIMEFUNC
                startTimer();
#endif
            // cudaDeviceSynchronize();  
            // count++;
            // printf("count = %d\n", count);  
            // 在计算A - Y*temp
            CHECK_CUBLAS(cublasDgemm(  
                cublas_handle,  
                CUBLAS_OP_N, // Y  
                CUBLAS_OP_N, // temp  
                m - j * nb,           
                n - (j + 1) * nb,        
                nb,       
                &minusAlpha,  
                d_Y_current, // Y的起始地址  
                lda,         // lda  
                work2,     // temp的起始地址  
                nb,         
                &alpha,  
                d_A_current,     // A_sub的起始地址  
                lda          // lda  
            ));  
            cudaDeviceSynchronize();  
            
#ifdef  TIMEFUNC
            float temptime2 = stopTimer();
            t = (long long)(m - j * nb) * (n - (j+1) * nb) * (2LL * nb); 
            printf("第2部分时间为%.4f tflops = %.4f\n", temptime2, t / temptime2 / 1e9);
            // printf("count = %d\n", count++);
            TailMatrixTime2 += temptime2;
#endif
            // 调整currn和currm的值
            // currm = currm + nb;
            // currn = currn + nb;
#ifdef MYDEBUG
            // printf("更新大的尾随矩阵\n");
            // printDeviceMatrixV2(d_A, m, m, n);
            MatrixName = "更新尾随矩阵";
            writeMatrixToCsvV2(d_A_current, m, m - j * nb, n - (j + 1) * nb, fileName, MatrixName);
#endif
        }

        //这部分的计算只是为了检测正确性
        if (!flag1) {
            // 拼接W = [W1, W2 - W1 * Y1' * W2]
            // 直接在d_W上进行计算即可
            // 维数： W1  m , (j * nb)   W2  m - j * nb , nb
            // 起始： W1  0 0           W2  j * nb, j * nb 
            // float* d_W1_current = d_W + currm - nb;

            double* d_W2_current = d_W + j * nb + j * nb * lda;
            double* d_Y1_current = d_Y + j * nb;
#ifdef  TIMEFUNC
                startTimer();
#endif
            CHECK_CUBLAS(cublasDgemm(  
                cublas_handle,  
                CUBLAS_OP_T, // Y1'  
                CUBLAS_OP_N, // W1
                j * nb,        
                nb,        
                m - j * nb,          
                &alpha,  
                d_Y1_current,// Y1的起始地址  
                lda,         // lda  
                d_W2_current,// W1的起始地址  
                lda,         // lda  
                &beta,  
                work2,     // temp的起始地址  
                j * nb            
            ));  
            
            CHECK_CUBLAS(cublasDgemm(  
                cublas_handle,  
                CUBLAS_OP_N, // W1 
                CUBLAS_OP_N, // temp  
                m,           
                nb,        
                j * nb,       
                &minusAlpha,  
                d_W, // Y的起始地址  
                lda,          // lda  
                work2,      // temp的起始地址  
                j * nb,         
                &alpha,  
                d_W + j * nb * lda, // A_sub的起始地址  
                lda           // lda  
            ));  
            // CHECK_CUDA(cudaFree(d_temp2));
            cudaDeviceSynchronize();  
#ifdef  TIMEFUNC
            fullWYTime1 += stopTimer();
#endif
#ifdef MYDEBUG
            printf("完整拼接的W矩阵\n");
            printDeviceMatrixV2(d_W, m, m, n);
            printf("完整拼接的Y矩阵\n");
            printDeviceMatrixV2(d_Y, m, m, n);
#endif

        } else {
            flag1 = false;

        }
    }
#ifdef MYDEBUG
    printf("最终的矩阵A\n");
    printDeviceMatrixV2(d_A, m, m, n);
#endif

}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " m n nb b" << std::endl;
        return -1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int nb = atoi(argv[3]);
    int b = atoi(argv[4]);

    // CudaHandles handles;
    std::ofstream file("matrix_output6.csv", std::ios::trunc);  
    file.close();  
    std::string fileName = "matrix_output6.csv";  
    std::string MatrixName;
    double* d_A;
    double* d_A_original;
    double* d_A_cusolver;
    double* d_W;
    double* d_Y;
    double* d_Q;
    double* d_R;
    double* work1;
    double* work2;
    double* work3;
    const int ldwork1 = m + 108 * nb / 2;      // blocksize最大为nb/2
    const int ldwork2 = m;
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeof(double) * m * n));
    // 生成随机矩阵 A
    startTimer();
    generateNormalMatrix(d_A, m, n, 0.0f, 1.0f);
    float genetime = stopTimer();
#ifdef MYDEBUG
    printf("原始的d_A\n");
    printDeviceMatrixV2(d_A,m,m,n);
#endif
#ifdef MYDEBUG4
    MatrixName = "orgin A";  
    writeMatrixToCsvV2(d_A, m, m, n, "matrixA.csv", MatrixName);
#endif
    CHECK_CUDA(cudaMalloc((void**)&d_A_original, sizeof(double) * m * n));
    CHECK_CUDA(cudaMalloc((void**)&d_A_cusolver, sizeof(double) * m * n));
    CHECK_CUDA(cudaMalloc(&d_W, sizeof(double) * m * n));
    CHECK_CUDA(cudaMalloc(&d_Y, sizeof(double) * m * n));
    CHECK_CUDA(cudaMalloc(&d_Q, sizeof(double) * m * m));
    CHECK_CUDA(cudaMalloc(&d_R, sizeof(double) * n * n));
    CHECK_CUDA(cudaMalloc(&work1, ldwork1 * 64 * sizeof(double)));            // 这两行是核函数中需要用到的
    CHECK_CUDA(cudaMalloc(&work2, ldwork2 * n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&work3, m * nb * sizeof(double)));
    
    CHECK_CUDA(cudaMemcpy(d_A_original, d_A, sizeof(double) * m * n, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_A_cusolver, d_A, sizeof(double) * m * n, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemset(d_W, 0, m * n * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_Y, 0, m * n * sizeof(double)));

    int lda = m;

    CudaHandles handles1;
    CudaHandles handles2;
        // //舍弃传入的handles，新创建handles1和handles2
    cublasHandle_t cublas_handle;  

    // 初始化 cuBLAS handle  
    cublasStatus_t status = cublasCreate(&cublas_handle);  

    if (status != CUBLAS_STATUS_SUCCESS) {  
        // 如果初始化失败，打印错误信息并退出  
        std::cerr << "cuBLAS initialization failed!" << std::endl;  
        return 0;  
    }  

#ifndef TIMEFUNC
    startTimer();
#endif
    IterativeQR(d_A, m, n, nb, b, d_W, d_Y, lda, d_Q, d_R, work1, work2, work3, ldwork1, ldwork2, cublas_handle); //这里的dR和dQ可以看作是work矩阵，并不是最后实际的R和Q
    cudaDeviceSynchronize();  
#ifndef TIMEFUNC
    float iterativeQR_time = stopTimer();
    printf("iterativeQR time = %.4f(s)\n", iterativeQR_time / 1000);
#endif
#ifdef TIMEFUNC
    printf("TSQR time = %.4f(s)\n", tsqrTime1 / 1000);
    printf("Reconstruct WY time = %.4f(s)\n", reconstructWYTime1 / 1000);
    printf("getU time = %.4f(s)\n", getUTime / 1000);
    // printf("tsqr+wy time = %.4f(s)\n", tsqrWYTime1 / 1000);
    printf("GEMM time = %.4f(s)\n", GemmTime1 / 1000);
    printf("WY time = %.4f(s)\n", WYTime1 / 1000);
    printf("TailMatrix time = %.4f(s)\n", TailMatrixTime1 / 1000);
    printf("fullWY time = %.4f(s)\n", fullWYTime1 / 1000);
    // printf("recursion time = %.4f(s)\n", recurTime / 1000);
    float totaltime = tsqrTime1 + reconstructWYTime1 + getUTime + GemmTime1 + WYTime1 + TailMatrixTime1 + TailMatrixTime2 + transWTime;
    printf("total time = %.4f(s)\n",  totaltime/ 1000);
    printf("tflops = %.4f\n", 2.0 * n * n *(m - 1.0/3 * n) / totaltime / 1e9);
#endif


#ifdef MYDEBUG4
    // std::string fileName = "matrix_output.csv";  
    MatrixName = "final W";  
    writeMatrixToCsvV2(d_W, m, m, n, fileName, MatrixName);
    MatrixName = "final Y";  
    writeMatrixToCsvV2(d_Y, m, m, n, fileName, MatrixName);
    // MatrixName = "A";  
    // writeMatrixToCsvV2(d_A, m, m, n, fileName, MatrixName);
#endif

#ifdef MYDEBUG1
    printf("最终的W和Y矩阵\n");
    printf("d_W\n");
    printDeviceMatrixV2(d_W,lda,m,n);
    printf("d_Y\n");
    printDeviceMatrixV2(d_Y,lda,m,n);
#endif 

    // 构造 Q 矩阵
    dim3 blockDim(threadsPerDim, threadsPerDim);
    dim3 gridDim((m + threadsPerDim - 1) / threadsPerDim, (m + threadsPerDim - 1) / threadsPerDim);
    setEye<<<gridDim, blockDim>>>(d_Q, m);
    // CHECK_CUDA(cudaDeviceSynchronize());

    double alpha = -1.0;
    double beta = 1.0;

    CHECK_CUBLAS(cublasDgemm(handles1.blas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             m, m, n,
                             &alpha, d_W, m,
                             d_Y, m,
                             &beta, d_Q, m));


    // 提取 R 矩阵
    dim3 gridDimMN((n + threadsPerDim - 1) / threadsPerDim, (n + threadsPerDim - 1) / threadsPerDim);
    getU<<<gridDimMN, blockDim>>>(n, n, d_A, m, d_R, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    // CHECK_CUDA(cudaDeviceSynchronize());
#ifdef MYDEBUG
    // std::string fileName = "matrix_output.csv";  
    MatrixName = "最终的R";  
    writeMatrixToCsvV2(d_R, n, n, n, fileName, MatrixName);
#endif
#ifdef MYDEBUG2
    printf("最终的Q和R矩阵\n");
    printf("d_Q\n");
    printDeviceMatrixV2(d_Q,lda,m,m);
    printf("d_R\n");
    printDeviceMatrixV2(d_R,n,n,n);
#endif

#ifdef MYDEBUG4
    // fileName = "matrix_output.csv";  
    MatrixName = "final Q";  
    writeMatrixToCsvV2(d_Q, m, m, m, fileName, MatrixName);
    MatrixName = "final R";  
    writeMatrixToCsvV2(d_R, n, n, n, fileName, MatrixName);
#endif


    // 检查正交性和重构误差
    checkOtho(m, n, d_Q, m);  
    checkBackwardError(m, n, d_A_original, m, d_Q, m, d_R, m);
    

    // cusolver计算qr分解计时
    int lwork = 0; // 工作区大小  
    double* d_tau;  // 存储 Householder 反射向量的数组  
    CHECK_CUDA(cudaMalloc((void**)&d_tau, sizeof(double) * n));  
    
    // 开始计时  
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);
    // QR分解
    CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(handles1.solver_handle, m, n, d_A_cusolver, lda, &lwork));  
    double* d_work; // 工作区  
    CHECK_CUDA(cudaMalloc((void**)&d_work, sizeof(double) * lwork));  
    startTimer();  
    cusolverDnDgeqrf(handle, m, n, d_A_cusolver, lda, d_tau, d_work, lwork, NULL);  
    // CHECK_CUDA(cudaDeviceSynchronize());
    float cusolverQR_time = stopTimer();  
    // 计算Q
    // 计算 Q 所需的工作区大小  
    int lwork_orgqr = 0;  
    CHECK_CUSOLVER(cusolverDnDorgqr_bufferSize(handles1.solver_handle, m, n, n, d_A_cusolver, lda, d_tau, &lwork_orgqr));  
    // 分配工作区  
    double* d_work_orgqr;  
    CHECK_CUDA(cudaMalloc((void**)&d_work_orgqr, sizeof(double) * lwork_orgqr));  
    CHECK_CUSOLVER(cusolverDnDorgqr(handles1.solver_handle, m, n, n, d_A_cusolver, lda, d_tau, d_work_orgqr, lwork_orgqr, NULL));  
   
    printf("cusolver QR time = %.4f(s)\n", cusolverQR_time / 1000);  
    printf("tflops = %.4f\n", 2.0 * n * n *(m - 1.0/3 * n) / cusolverQR_time / 1e9);


    // 释放内存
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_A_original));
    if (d_W) CHECK_CUDA(cudaFree(d_W));
    if (d_Y) CHECK_CUDA(cudaFree(d_Y));
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_R));
    CHECK_CUDA(cudaFree(work1));
    CHECK_CUDA(cudaFree(work2));
    CHECK_CUDA(cudaFree(work3));

    return 0;
}

