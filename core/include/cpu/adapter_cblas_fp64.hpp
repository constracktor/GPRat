#ifndef CPU_ADAPTER_CBLAS_FP64_H
#define CPU_ADAPTER_CBLAS_FP64_H
#include "tile_data.hpp"
#include <hpx/future.hpp>
#include <vector>

using vector_future = hpx::shared_future<std::vector<double>>;
using vector = std::vector<double>;

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
vector_future f_potrf(vector_future f_A, const int N);

/**
 * @brief FP64 In-place solve L(^T) * X = A or X * L(^T) = A where L lower triangular
 * @param f_L Cholesky factor matrix
 * @param f_A right hand side matrix
 * @param N first dimension
 * @param M second dimension
 * @return solution matrix f_X
 */
vector_future f_trsm(vector_future f_L,
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
vector_future f_syrk(vector_future f_A, vector_future f_B, const int N);

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
f_gemm(vector_future f_A,
       vector_future f_B,
       vector_future f_C,
       const int N,
       const int M,
       const int K,
       const BLAS_TRANSPOSE transpose_A,
       const BLAS_TRANSPOSE transpose_B);

///////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief FP64 In-place Cholesky decomposition of A
 * @param A matrix to be factorized
 * @param N matrix dimension
 * @return factorized, lower triangular matrix L
 */
vector potrf(vector A, const int N);

/**
 * @brief FP64 In-place solve L(^T) * X = A or X * L(^T) = A where L lower triangular
 * @param L Cholesky factor matrix
 * @param A right hand side matrix
 * @param N first dimension
 * @param M second dimension
 * @return solution matrix X
 */
vector
r_trsm(const vector &L, vector A, const int N, const int M, const BLAS_TRANSPOSE transpose_L, const BLAS_SIDE side_L);

/**
 * @brief FP64 Symmetric rank-k update: A = A - B * B^T
 * @param f_A Base matrix
 * @param f_B Symmetric update matrix
 * @param N matrix dimension
 * @return updated matrix f_A
 */
vector r_syrk(vector A, const vector &B, const int N);

/**
 * @brief FP64 General matrix-matrix multiplication: C = C - A(^T) * B(^T)
 * @param C Base matrix
 * @param B Right update matrix
 * @param A Left update matrix
 * @param N first matrix dimension
 * @param M second matrix dimension
 * @param K third matrix dimension
 * @param transpose_A transpose left matrix
 * @param transpose_B transpose right matrix
 * @return updated matrix X
 */
vector r_gemm(const vector &A,
              const vector &B,
              vector C,
              const int N,
              const int M,
              const int K,
              const BLAS_TRANSPOSE transpose_A,
              const BLAS_TRANSPOSE transpose_B);

///////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief FP64 In-place solve L(^T) * X = A or X * L(^T) = A where L lower triangular
 * @param L Cholesky factor matrix
 * @param A right hand side matrix
 * @param N first dimension
 * @param M second dimension
 * @return solution matrix X
 */
vector v_trsm(vector L, vector A, const int N, const int M, const BLAS_TRANSPOSE transpose_L, const BLAS_SIDE side_L);

/**
 * @brief FP64 Symmetric rank-k update: A = A - B * B^T
 * @param f_A Base matrix
 * @param f_B Symmetric update matrix
 * @param N matrix dimension
 * @return updated matrix f_A
 */
vector v_syrk(vector A, vector B, const int N);

/**
 * @brief FP64 General matrix-matrix multiplication: C = C - A(^T) * B(^T)
 * @param C Base matrix
 * @param B Right update matrix
 * @param A Left update matrix
 * @param N first matrix dimension
 * @param M second matrix dimension
 * @param K third matrix dimension
 * @param transpose_A transpose left matrix
 * @param transpose_B transpose right matrix
 * @return updated matrix X
 */
vector v_gemm(vector A,
              vector B,
              vector C,
              const int N,
              const int M,
              const int K,
              const BLAS_TRANSPOSE transpose_A,
              const BLAS_TRANSPOSE transpose_B);

//////////////////////////////////////////////////////////////////////

/**
 * @brief FP64 In-place Cholesky decomposition of A
 * @param A matrix to be factorized
 * @param N matrix dimension
 * @return factorized, lower triangular matrix f_L
 */
mutable_tile_data<double> m_potrf(const mutable_tile_data<double> &A, int N);

/**
 * @brief FP64 In-place solve L(^T) * X = A or X * L(^T) = A where L lower triangular
 * @param L Cholesky factor matrix
 * @param A right hand side matrix
 * @param N first dimension
 * @param M second dimension
 * @return solution matrix f_X
 */
mutable_tile_data<double>
m_trsm(const const_tile_data<double> &L,
       const mutable_tile_data<double> &A,
       int N,
       int M,
       BLAS_TRANSPOSE transpose_L,
       BLAS_SIDE side_L);

/**
 * @brief FP64 Symmetric rank-k update: A = A - B * B^T
 * @param A Base matrix
 * @param B Symmetric update matrix
 * @param N matrix dimension
 * @return updated matrix f_A
 */
mutable_tile_data<double> m_syrk(const mutable_tile_data<double> &A, const const_tile_data<double> &B, int N);

/**
 * @brief FP64 General matrix-matrix multiplication: C = C - A(^T) * B(^T)
 * @param C Base matrix
 * @param B Right update matrix
 * @param A Left update matrix
 * @param N first matrix dimension
 * @param M second matrix dimension
 * @param K third matrix dimension
 * @param transpose_A transpose left matrix
 * @param transpose_B transpose right matrix
 * @return updated matrix f_X
 */
mutable_tile_data<double>
m_gemm(const const_tile_data<double> &A,
       const const_tile_data<double> &B,
       const mutable_tile_data<double> &C,
       int N,
       int M,
       int K,
       BLAS_TRANSPOSE transpose_A,
       BLAS_TRANSPOSE transpose_B);

#endif  // end of CPU_ADAPTER_CBLAS_FP64_H
