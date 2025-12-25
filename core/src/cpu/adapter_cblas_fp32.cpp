#include "cpu/adapter_cblas_fp32.hpp"

#ifdef GPRAT_ENABLE_MKL
// MKL CBLAS and LAPACKE
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif

// BLAS level 3 operations

vector_future potrf(vector_future f_A, const int N)
{
    auto A = f_A.get();
    // POTRF: in-place Cholesky decomposition of A
    // use spotrf2 recursive version for better stability
    LAPACKE_spotrf2(LAPACK_ROW_MAJOR, 'L', N, A.data(), N);
    // return factorized matrix L
    return hpx::make_ready_future(A);
}

vector_future trsm(vector_future f_L,
                   vector_future f_A,
                   const int N,
                   const int M,
                   const BLAS_TRANSPOSE transpose_L,
                   const BLAS_SIDE side_L)

{
    auto L = f_L.get();
    auto A = f_A.get();
    // TRSM constants
    const float alpha = 1.0;
    // TRSM: in-place solve L(^T) * X = A or X * L(^T) = A where L lower triangular
    cblas_strsm(
        CblasRowMajor,
        static_cast<CBLAS_SIDE>(side_L),
        CblasLower,
        static_cast<CBLAS_TRANSPOSE>(transpose_L),
        CblasNonUnit,
        N,
        M,
        alpha,
        L.data(),
        N,
        A.data(),
        M);
    // return vector
    return hpx::make_ready_future(A);
}

vector_future syrk(vector_future f_A, vector_future f_B, const int N)
{
    auto B = f_B.get();
    auto A = f_A.get();
    // SYRK constants
    const float alpha = -1.0;
    const float beta = 1.0;
    // SYRK:A = A - B * B^T
    cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, N, alpha, B.data(), N, beta, A.data(), N);
    // return updated matrix A
    return hpx::make_ready_future(A);
}

vector_future
gemm(vector_future f_A,
     vector_future f_B,
     vector_future f_C,
     const int N,
     const int M,
     const int K,
     const BLAS_TRANSPOSE transpose_A,
     const BLAS_TRANSPOSE transpose_B)
{
    auto C = f_C.get();
    auto B = f_B.get();
    auto A = f_A.get();
    // GEMM constants
    const float alpha = -1.0;
    const float beta = 1.0;
    // GEMM: C = C - A(^T) * B(^T)
    cblas_sgemm(
        CblasRowMajor,
        static_cast<CBLAS_TRANSPOSE>(transpose_A),
        static_cast<CBLAS_TRANSPOSE>(transpose_B),
        K,
        M,
        N,
        alpha,
        A.data(),
        K,
        B.data(),
        M,
        beta,
        C.data(),
        M);
    // return updated matrix C
    return hpx::make_ready_future(C);
}
