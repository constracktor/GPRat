#ifndef ADAPTER_CUBLAS_H
#define ADAPTER_CUBLAS_H

#include <cusolverDn.h>
#include <hpx/future.hpp>
#include <hpx/modules/async_cuda.hpp>
#include <target.hpp>

// Constants, compatible with cuBLAS

/**
 * @brief BLAS operation types
 *
 * @see cublasOperation_t
 */
typedef enum BLAS_TRANSPOSE {
    Blas_no_trans = 0,  // CUBLAS_OP_N
    Blas_trans = 1      // CUBLAS_OP_T
} BLAS_TRANSPOSE;

/**
 * @brief BLAS side types
 *
 * @see cublasSideMode_t
 */
typedef enum BLAS_SIDE {
    Blas_left = 0,  // CUBLAS_SIDE_LEFT
    Blas_right = 1  // CUBLAS_SIDE_RIGHT
} BLAS_SIDE;

/**
 * @brief BLAS types for alpha scalar
 */
typedef enum BLAS_ALPHA { Blas_add = 1, Blas_substract = -1 } BLAS_ALPHA;

// BLAS level 3 operations

/**
 * @brief In-place Cholesky decomposition of A
 *
 * @param cusolver cuSolver handle, already created
 * @param stream CUDA stream, already created
 * @param f_A matrix to be factorized
 * @param N matrix dimension
 *
 * @return factorized, lower triangular matrix f_L, in-place update of f_A
 */
hpx::shared_future<double *>
potrf(cusolverDnHandle_t cusolver, cudaStream_t stream, hpx::shared_future<double *> f_A, const std::size_t N);

/**
 * @brief In-place solve A(^T) * X = B or X * A(^T) = B for lower triangular A
 *
 * @param cublas cuBLAS handle, already created
 * @param stream CUDA stream, already created
 * @param f_A lower triangular matrix
 * @param f_B right hand side matrix
 * @param M number of rows
 * @param N number of columns
 * @param transpose_A whether to transpose A
 * @param side_A whether to use A on the left or right side
 *
 * @return solution matrix f_X, in-place update of f_B
 */
hpx::shared_future<double *>
trsm(cublasHandle_t cublas,
     cudaStream_t stream,
     hpx::shared_future<double *> f_A,
     hpx::shared_future<double *> f_B,
     const std::size_t M,
     const std::size_t N,
     const BLAS_TRANSPOSE transpose_A,
     const BLAS_SIDE side_A);

/**
 * @brief Symmetric rank-k update: C = C - A * A^T
 *
 * @param cublas cuBLAS handle, already created
 * @param stream CUDA stream, already created
 * @param f_A matrix
 * @param f_C Symmetric matrix
 * @param N matrix dimension
 *
 * @return updated matrix f_A, inplace update
 */
hpx::shared_future<double *>
syrk(cublasHandle_t cublas,
     cudaStream_t stream,
     hpx::shared_future<double *> f_A,
     hpx::shared_future<double *> f_C,
     const std::size_t N);

/**
 * @brief General matrix-matrix multiplication: C = C - A(^T) * B(^T)
 *
 * @param cublas cuBLAS handle, already created
 * @param stream CUDA stream, already created
 * @param f_A Left update matrix
 * @param f_B Right update matrix
 * @param f_C Base matrix
 * @param M Number of rows of matrix A
 * @param N Number of columns of matrix B
 * @param K Number of columns of matrix A / rows of matrix B
 * @param transpose_A whether to transpose left matrix A
 * @param transpose_B whether to transpose right matrix B
 *
 * @return updated matrix f_C, in-place update
 */
hpx::shared_future<double *>
gemm(cublasHandle_t cublas,
     cudaStream_t stream,
     hpx::shared_future<double *> f_A,
     hpx::shared_future<double *> f_B,
     hpx::shared_future<double *> f_C,
     const std::size_t M,
     const std::size_t N,
     const std::size_t K,
     const BLAS_TRANSPOSE transpose_A,
     const BLAS_TRANSPOSE transpose_B);

// Helper functions

/**
 * @brief Return inverse of cublasOperation_t: transpose or no transpose
 *
 * @see BLAS_TRANSPOSE
 */
inline cublasOperation_t opposite(cublasOperation_t op) { return (op == CUBLAS_OP_N) ? CUBLAS_OP_T : CUBLAS_OP_N; }

/**
 * @brief Return inverse of cublasSideMode_t: left or right side
 *
 * @see BLAS_SIDE
 */
inline cublasSideMode_t opposite(cublasSideMode_t side)
{
    return (side == CUBLAS_SIDE_LEFT) ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;
}

#endif  // end of ADAPTER_CUBLAS_H
