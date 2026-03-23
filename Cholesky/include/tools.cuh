#ifndef UTILS_ALGO
#define UTILS_ALGO

#include "cublas_v2.h"
#include "../include/cuda_utils_check.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <string>
#include <utility>

namespace CBENTOOL {

// 取子矩阵row-major
double *getSubMatrix_row(double *matrix, size_t lda, size_t rows, size_t rowe,
                         size_t cols, size_t cole) {
  double *submatrix = NULL;
  submatrix = (double *)malloc(sizeof(double) * (rowe - rows) * (cole - cols));
  for (size_t i{rows}; i < (rowe); ++i) {
    for (size_t j{cols}; j < (cole); ++j) {
      submatrix[(i - rows) * (cole - cols) + (j - cols)] =
          matrix[(i)*lda + (j)];
    }
  }

  return submatrix;
}

// 取子矩阵col-major
double *getSubMatrix_col(double *matrix, size_t lda, size_t rows, size_t rowe,
                         size_t cols, size_t cole) {
  double *submatrix = NULL;
  submatrix = (double *)malloc(sizeof(double) * (rowe - rows) * (cole - cols));
  for (size_t i{rows}; i < (rowe); ++i) {
    for (size_t j{cols}; j < (cole); ++j) {
      submatrix[(i - rows) * (cole - cols) + (j - cols)] =
          matrix[(i) + (j)*lda];
    }
  }

  return submatrix;
}

// 特化模板声明
template <typename T>
int curandGenerateUniformWrapper(curandGenerator_t gen, T *matrx, size_t Sum);

// 模板版本的 curandGenerate 函数
template <typename T>
int curandGenerate(T *matrx, int m, int n, unsigned long long seed) {
  curandGenerator_t gen;
  size_t Sum = m * n;

  CHECK_Curand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CHECK_Curand(curandSetPseudoRandomGeneratorSeed(gen, seed));

  // 使用特化的模板函数来生成随机数
  if (curandGenerateUniformWrapper(gen, matrx, Sum) != CURAND_STATUS_SUCCESS) {
    CHECK_Curand(curandDestroyGenerator(gen));
    return EXIT_FAILURE;
  }

  CHECK_Curand(curandDestroyGenerator(gen));

  return EXIT_SUCCESS;
}

// 针对 float 类型的特化
template <>
int curandGenerateUniformWrapper<float>(curandGenerator_t gen, float *matrx,
                                        size_t Sum) {
  return curandGenerateUniform(gen, matrx, Sum);
}
// 针对 double 类型的特化
template <>
int curandGenerateUniformWrapper<double>(curandGenerator_t gen, double *matrx,
                                         size_t Sum) {
  return curandGenerateUniformDouble(gen, matrx, Sum);
}

// 初始化成ZERO矩阵
template <typename T> int init_zero(T *A, size_t m, size_t n) {
  for (size_t t{0}; t < m * n; ++t) {
    A[t] = 0.0;
  }
  return EXIT_SUCCESS;
}

// 初始化成eyes矩阵
template <typename T> int init_eyes(T *A, size_t m, size_t n) {
  init_zero(A, m, n);
  for (int i = 0; i < m; ++i) {
    A[i * n + i] += 1.0;
  }
  return EXIT_SUCCESS;
}

// col-major打印
template <typename T> int print_matrix_colmajor(T *matrix, int m, int n) {
  std::cout << std::fixed << std::setprecision(6);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      // printf(" %lf", matrix[j * m + i]);
      std::cout << " " << matrix[j * m + i];
    }
    printf("\n");
  }
  return EXIT_SUCCESS;
}

// row-major打印
template <typename T> int print_matrix_rowmajor(T *matrix, int m, int n) {
  std::cout << std::fixed << std::setprecision(6);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      // printf(" %lf", matrix[j * m + i]);
      std::cout << " " << matrix[i * n + j];
    }
    printf("\n");
  }
  return EXIT_SUCCESS;
}

// row-major-lda打印
template <typename T>
int print_matrix_colmajor_lda(T *matrix, int m, int n, int lda) {
  std::cout << std::fixed << std::setprecision(6);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      // printf(" %lf", matrix[j * m + i]);
      std::cout << " " << matrix[i + j * lda];
    }
    printf("\n");
  }
  return EXIT_SUCCESS;
}

// 行交换函数
int swapRows(double *matrix, int Cols, int row1, int row2) {
  for (int col{0}; col < Cols; ++col) {
    std::swap(matrix[row1 * Cols + col], matrix[row2 * Cols + col]);
  }
  return EXIT_SUCCESS;
}

/*  如果你要对
 *       | 1 2 3  |
 *   A = | 4 5 6  |
 *       | 7 8 10 |
 *
 *  进行lu分解，那么你要存入的A就应该是 A =
 * {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0};
 *  如果是方阵的话可以直接转置，即可以直接交换m和n的位置
 *  但是如果不是方阵的话，就需要进行没有这么简单的处理
 *
 *  或许应该这样说，如果你的A的模样是这样的{1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0,
 * 10.0} 那么getrf就是对 | 1 2 3  | A = | 4 5 6  | | 7 8 10 | 进行getrf
 *
 *  假如你生成了一个随机序列，然后，你rowprint输出的样子是这样的(即你A的样子是这样的{0.411109,0.694880,0.182897,0.021682......})
 *  0.411109 0.694880
 *  0.182897 0.021682
 *  0.555719 0.032696
 *  0.029398 0.785627
 *  0.907888 0.743238
 *  0.286248 0.018245
 *  即你想对这个矩阵进行getrf的话，你需要进行额外的colmajor处理,使得你输入的A是{0.411109,0.182897,0.555719,0.029398.....}这样的
 */
// row-major trans col-major
double *row2col(double *rowmatrix, size_t m, size_t n) {
  double *colmatrix = NULL;
  colmatrix = (double *)malloc(sizeof(double) * m * n);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      colmatrix[j * m + i] = rowmatrix[i * n + j];
    }
  }
  return colmatrix;
}

// col-major trans row-major
double *col2row(double *colmatrix, size_t m, size_t n) {
  double *rowmatrix = NULL;
  rowmatrix = (double *)malloc(sizeof(double) * m * n);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      rowmatrix[i * n + j] = colmatrix[j * m + i];
    }
  }
  return rowmatrix;
}

} // namespace CBENTOOL
#endif