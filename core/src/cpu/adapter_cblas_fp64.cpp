#include "cpu/adapter_cblas_fp64.hpp"

#ifdef GPRAT_ENABLE_MKL
// MKL CBLAS and LAPACKE
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif

// BLAS level 3 operations

vector potrf(vector_future f_A, const int N)
{
    vector A = f_A.get();
    // POTRF: in-place Cholesky decomposition of A
    // use dpotrf2 recursive version for better stability
    LAPACKE_dpotrf2(LAPACK_ROW_MAJOR, 'L', N, A.data(), N);
    // return factorized matrix L
    return A;
}

vector trsm(vector_future f_L,
            vector_future f_A,
            const int N,
            const int M,
            const BLAS_TRANSPOSE transpose_L,
            const BLAS_SIDE side_L)

{
    const vector &L = f_L.get();
    vector A = f_A.get();
    // TRSM constants
    const double alpha = 1.0;
    // TRSM: in-place solve L(^T) * X = A or X * L(^T) = A where L lower triangular
    cblas_dtrsm(
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
    return A;
}

vector syrk(vector_future f_A, vector_future f_B, const int N)
{
    const vector &B = f_B.get();
    vector A = f_A.get();
    // SYRK constants
    const double alpha = -1.0;
    const double beta = 1.0;
    // SYRK:A = A - B * B^T
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, N, alpha, B.data(), N, beta, A.data(), N);
    // return updated matrix A
    return A;
}

vector gemm(vector_future f_A,
            vector_future f_B,
            vector_future f_C,
            const int N,
            const int M,
            const int K,
            const BLAS_TRANSPOSE transpose_A,
            const BLAS_TRANSPOSE transpose_B)
{
    vector C = f_C.get();
    const vector &B = f_B.get();
    const vector &A = f_A.get();
    // GEMM constants
    const double alpha = -1.0;
    const double beta = 1.0;
    // GEMM: C = C - A(^T) * B(^T)
    cblas_dgemm(
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

vector trsv(vector_future f_L, vector_future f_a, const int N, const BLAS_TRANSPOSE transpose_L)
{
    const vector &L = f_L.get();
    vector a = f_a.get();
    // TRSV: In-place solve L(^T) * x = a where L lower triangular
    cblas_dtrsv(CblasRowMajor,
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

vector gemv(vector_future f_A,
            vector_future f_a,
            vector_future f_b,
            const int N,
            const int M,
            const BLAS_ALPHA alpha,
            const BLAS_TRANSPOSE transpose_A)
{
    const vector &A = f_A.get();
    const vector &a = f_a.get();
    vector b = f_b.get();
    // GEMV constants
    // const double alpha = -1.0;
    const double beta = 1.0;
    // GEMV:  b{N} = b{N} - A(^T){NxM} * a{M}
    cblas_dgemv(
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

vector dot_diag_syrk(vector_future f_A, vector_future f_r, const int N, const int M)
{
    const vector &A = f_A.get();
    vector r = f_r.get();
    // r = r + diag(A^T * A)
    for (std::size_t j = 0; j < static_cast<std::size_t>(M); ++j)
    {
        // Extract the j-th column and compute the dot product with itself
        r[j] += cblas_ddot(N, &A[j], M, &A[j], M);
    }
    return r;
}

vector dot_diag_gemm(vector_future f_A, vector_future f_B, vector_future f_r, const int N, const int M)
{
    const vector &A = f_A.get();
    const vector &B = f_B.get();
    vector r = f_r.get();
    // r = r + diag(A * B)
    for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i)
    {
        r[i] += cblas_ddot(M, &A[i * static_cast<std::size_t>(M)], 1, &B[i], N);
    }
    return r;
}

// BLAS level 1 operations

vector axpy(vector_future f_y, vector_future f_x, const int N)
{
    vector y = f_y.get();
    const vector &x = f_x.get();
    cblas_daxpy(N, -1.0, x.data(), 1, y.data(), 1);
    return y;
}

double dot(std::vector<double> a, std::vector<double> b, const int N)
{
    // DOT: a * b
    return cblas_ddot(N, a.data(), 1, b.data(), 1);
}
