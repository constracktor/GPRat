#include "gprat/cpu/adapter_cblas_fp32.hpp"

#ifdef GPRAT_ENABLE_MKL
// MKL CBLAS and LAPACKE
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif

GPRAT_NS_BEGIN

// BLAS level 3 operations

mutable_tile_data<float> potrf(const mutable_tile_data<float> &A, const int N)
{
    // POTRF: in-place Cholesky decomposition of A
    // use spotrf2 recursive version for better stability
    LAPACKE_spotrf2(LAPACK_ROW_MAJOR, 'L', N, A.data(), N);
    // return factorized matrix L
    return A;
}

mutable_tile_data<float>
trsm(const const_tile_data<float> &L,
     const mutable_tile_data<float> &A,
     const int N,
     const int M,
     const BLAS_TRANSPOSE transpose_L,
     const BLAS_SIDE side_L)

{
    // TRSM constants
    const float alpha = 1.0f;
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
    return A;
}

mutable_tile_data<float> syrk(const mutable_tile_data<float> &A, const const_tile_data<float> &B, const int N)
{
    // SYRK constants
    const float alpha = -1.0f;
    const float beta = 1.0f;
    // SYRK:A = A - B * B^T
    cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, N, alpha, B.data(), N, beta, A.data(), N);
    // return updated matrix A
    return A;
}

mutable_tile_data<float>
gemm(const const_tile_data<float> &A,
     const const_tile_data<float> &B,
     const mutable_tile_data<float> &C,
     const int N,
     const int M,
     const int K,
     const BLAS_TRANSPOSE transpose_A,
     const BLAS_TRANSPOSE transpose_B)
{
    // GEMM constants
    const float alpha = -1.0f;
    const float beta = 1.0f;
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
    return C;
}

// BLAS level 2 operations

mutable_tile_data<float>
trsv(const const_tile_data<float> &L, const mutable_tile_data<float> &a, const int N, const BLAS_TRANSPOSE transpose_L)
{
    // TRSV: In-place solve L(^T) * x = a where L lower triangular
    cblas_strsv(CblasRowMajor,
                CblasLower,
                static_cast<CBLAS_TRANSPOSE>(transpose_L),
                CblasNonUnit,
                N,
                L.data(),
                N,
                a.data(),
                1);
    // return solution vector x
    return a;
}

mutable_tile_data<float>
gemv(const const_tile_data<float> &A,
     const const_tile_data<float> &a,
     const mutable_tile_data<float> &b,
     const int N,
     const int M,
     const BLAS_ALPHA alpha,
     const BLAS_TRANSPOSE transpose_A)
{
    // GEMV constants
    // const float alpha = -1.0;
    const float beta = 1.0f;
    // GEMV:  b{N} = b{N} - A(^T){NxM} * a{M}
    cblas_sgemv(
        CblasRowMajor,
        static_cast<CBLAS_TRANSPOSE>(transpose_A),
        N,
        M,
        alpha,
        A.data(),
        M,
        a.data(),
        1,
        beta,
        b.data(),
        1);
    // return updated vector b
    return b;
}

mutable_tile_data<float>
dot_diag_syrk(const const_tile_data<float> &A, const mutable_tile_data<float> &r, const int N, const int M)
{
    auto r_p = r.data();
    auto A_p = A.data();
    // r = r + diag(A^T * A)
    for (std::size_t j = 0; j < static_cast<std::size_t>(M); ++j)
    {
        // Extract the j-th column and compute the dot product with itself
        r_p[j] += cblas_sdot(N, &A_p[j], M, &A_p[j], M);
    }
    return r;
}

mutable_tile_data<float>
dot_diag_gemm(const const_tile_data<float> &A,
              const const_tile_data<float> &B,
              const mutable_tile_data<float> &r,
              const int N,
              const int M)
{
    auto r_p = r.data();
    auto A_p = A.data();
    auto B_p = B.data();
    // r = r + diag(A * B)
    for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i)
    {
        r_p[i] += cblas_sdot(M, &A_p[i * static_cast<std::size_t>(M)], 1, &B_p[i], N);
    }
    return r;
}

// BLAS level 1 operations

mutable_tile_data<float> axpy(const mutable_tile_data<float> &y, const const_tile_data<float> &x, const int N)
{
    cblas_saxpy(N, -1.0, x.data(), 1, y.data(), 1);
    return y;
}

float dot(std::span<const float> a, std::span<const float> b, const int N)
{
    // DOT: a * b
    return cblas_sdot(N, a.data(), 1, b.data(), 1);
}

GPRAT_NS_END
