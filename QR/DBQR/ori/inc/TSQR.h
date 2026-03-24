#pragma once

#include <iostream>  
#include <vector>  
#include <cuda_runtime.h>  
#include <cusolverDn.h>  
#include <cublas_v2.h>  
#include <cassert>  
#include <cmath>  
#include <cstdlib>  

#include "myutils.h"
#include "errorcheck.h"


void tsqr(double* d_A, int lda, int m, int n, int blockSize,  
          double* d_Q, int ldq, double* d_R, int ldr,  
          CudaHandles& handles);

//float版本
void tsqr(float* d_A, int lda, int m, int n, int blockSize,  
          float* d_Q, int ldq, float* d_R, int ldr,  
          CudaHandles& handles);