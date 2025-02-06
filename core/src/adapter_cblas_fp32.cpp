#include "../include/adapter_cblas_fp32.hpp"

// MKL CBLAS and LAPACKE
#include "mkl_cblas.h"
#include "mkl_lapacke.h"

// BLAS level 3 operations -------------------------------------- {{{

hpx::shared_future<std::vector<float>> potrf(hpx::shared_future<std::vector<float>> f_A, const int N)
{
    auto A = f_A.get();
    // POTRF: in-place Cholesky decomposition of A
    // use spotrf2 recursive version for better stability
    LAPACKE_spotrf2(LAPACK_ROW_MAJOR, 'L', N, A.data(), N);
    // return factorized matrix L
    return hpx::make_ready_future(A);
}

hpx::shared_future<std::vector<float>>
trsm(hpx::shared_future<std::vector<float>> f_L,
     hpx::shared_future<std::vector<float>> f_A,
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

hpx::shared_future<std::vector<float>>
syrk(hpx::shared_future<std::vector<float>> f_A, hpx::shared_future<std::vector<float>> f_B, const int N)
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

hpx::shared_future<std::vector<float>>
gemm(hpx::shared_future<std::vector<float>> f_A,
     hpx::shared_future<std::vector<float>> f_B,
     hpx::shared_future<std::vector<float>> f_C,
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

// }}} --------------------------------- end of BLAS level 3 operations

// BLAS level 2 operations ------------------------------- {{{

hpx::shared_future<std::vector<float>>
trsv(hpx::shared_future<std::vector<float>> f_L,
     hpx::shared_future<std::vector<float>> f_a,
     const int N,
     const BLAS_TRANSPOSE transpose_L)
{
    auto L = f_L.get();
    auto a = f_a.get();
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
    return hpx::make_ready_future(a);
}

hpx::shared_future<std::vector<float>>
gemv(hpx::shared_future<std::vector<float>> f_A,
     hpx::shared_future<std::vector<float>> f_a,
     hpx::shared_future<std::vector<float>> f_b,
     const int N,
     const int M,
     const BLAS_ALPHA alpha,
     const BLAS_TRANSPOSE transpose_A)
{
    auto A = f_A.get();
    auto a = f_a.get();
    auto b = f_b.get();
    // GEMV constants
    // const float alpha = -1.0;
    const float beta = 1.0;
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
    return hpx::make_ready_future(b);
}

hpx::shared_future<std::vector<float>>
ger(hpx::shared_future<std::vector<float>> f_A,
    hpx::shared_future<std::vector<float>> f_x,
    hpx::shared_future<std::vector<float>> f_y,
    const int N)
{
    auto A = f_A.get();
    auto x = f_x.get();
    auto y = f_y.get();
    // GER constants
    const float alpha = -1.0;
    // GER:  A = A - x*y^T
    cblas_sger(CblasRowMajor, N, N, alpha, x.data(), 1, y.data(), 1, A.data(), N);
    // return updated A
    return hpx::make_ready_future(A);
}

hpx::shared_future<std::vector<float>> dot_diag_syrk(
    hpx::shared_future<std::vector<float>> f_A, hpx::shared_future<std::vector<float>> f_r, const int N, const int M)
{
    auto A = f_A.get();
    auto r = f_r.get();
    // r = r + diag(A^T * A)
    for (std::size_t j = 0; j < static_cast<std::size_t>(M); ++j)
    {
        // Extract the j-th column and compute the dot product with itself
        r[j] += cblas_sdot(N, &A[j], M, &A[j], M);
    }
    return hpx::make_ready_future(r);
}

hpx::shared_future<std::vector<float>>
dot_diag_gemm(hpx::shared_future<std::vector<float>> f_A,
              hpx::shared_future<std::vector<float>> f_B,
              hpx::shared_future<std::vector<float>> f_r,
              const int N,
              const int M)
{
    auto A = f_A.get();
    auto B = f_B.get();
    auto r = f_r.get();
    // r = r + diag(A * B)
    for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i)
    {
        r[i] += cblas_sdot(M, &A[i * static_cast<std::size_t>(M)], 1, &B[i], N);
    }
    return hpx::make_ready_future(r);
}

// }}} --------------------------------- end of BLAS level 2 operations

// BLAS level 1 operations ------------------------------- {{{

float dot(std::vector<float> a, std::vector<float> b, const int N)
{
    // DOT: a * b
    return cblas_sdot(N, a.data(), 1, b.data(), 1);
}

// }}} --------------------------------- end of BLAS level 1 operations
