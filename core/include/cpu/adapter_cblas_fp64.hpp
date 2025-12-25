#ifndef CPU_ADAPTER_CBLAS_FP64_H
#define CPU_ADAPTER_CBLAS_FP64_H

#include <hpx/future.hpp>
#include <vector>

using vector_future = hpx::shared_future<std::vector<double>>;

// Constants that are compatible with CBLAS

typedef enum BLAS_TRANSPOSE { Blas_no_trans = 111, Blas_trans = 112 } BLAS_TRANSPOSE;

typedef enum BLAS_SIDE { Blas_left = 141, Blas_right = 142 } BLAS_SIDE;

typedef enum BLAS_ALPHA { Blas_add = 1, Blas_substract = -1 } BLAS_ALPHA;

// typedef enum BLAS_UPLO { Blas_upper = 121,
//                          Blas_lower = 122 } BLAS_UPLO;

// typedef enum BLAS_ORDERING { Blas_row_major = 101,
//                              Blas_col_major = 102 } BLAS_ORDERING;

// BLAS level 3 operations

/**
 * @brief FP64 In-place Cholesky decomposition of A
 * @param f_A matrix to be factorized
 * @param N matrix dimension
 * @return factorized, lower triangular matrix f_L
 */
vector_future potrf(vector_future f_A, const int N);

/**
 * @brief FP64 In-place solve L(^T) * X = A or X * L(^T) = A where L lower triangular
 * @param f_L Cholesky factor matrix
 * @param f_A right hand side matrix
 * @param N first dimension
 * @param M second dimension
 * @return solution matrix f_X
 */
vector_future trsm(vector_future f_L,
                   vector_future f_A,
                   const int N,
                   const int M,
                   const BLAS_TRANSPOSE transpose_L,
                   const BLAS_SIDE side_L);

/**
 * @brief FP64 Symmetric rank-k update: A = A - B * B^T
 * @param f_A Base matrix
 * @param f_B Symmetric update matrix
 * @param N matrix dimension
 * @return updated matrix f_A
 */
vector_future syrk(vector_future f_A, vector_future f_B, const int N);

/**
 * @brief FP64 General matrix-matrix multiplication: C = C - A(^T) * B(^T)
 * @param f_C Base matrix
 * @param f_B Right update matrix
 * @param f_A Left update matrix
 * @param N first matrix dimension
 * @param M second matrix dimension
 * @param K third matrix dimension
 * @param transpose_A transpose left matrix
 * @param transpose_B transpose right matrix
 * @return updated matrix f_X
 */
vector_future
gemm(vector_future f_A,
     vector_future f_B,
     vector_future f_C,
     const int N,
     const int M,
     const int K,
     const BLAS_TRANSPOSE transpose_A,
     const BLAS_TRANSPOSE transpose_B);

#endif  // end of CPU_ADAPTER_CBLAS_FP64_H
