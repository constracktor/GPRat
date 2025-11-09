#include "gprat/cpu/adapter_cblas_fp64.hpp"

#include "gprat/performance_counters.hpp"

#ifdef HPX_HAVE_MODULE_PERFORMANCE_COUNTERS
#include <hpx/performance_counters/manage_counter_type.hpp>
#endif

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

mutable_tile_data<double> potrf(const mutable_tile_data<double> &A, const int N)
{
    GPRAT_BENCHMARK_FORCE_EVICT(A.as_span());
    GPRAT_TIME_FUNCTION(&potrf);
    // POTRF: in-place Cholesky decomposition of A
    // use dpotrf2 recursive version for better stability
    LAPACKE_dpotrf2(LAPACK_ROW_MAJOR, 'L', N, A.data(), N);
    // return factorized matrix L
    return A;
}

mutable_tile_data<double>
trsm(const const_tile_data<double> &L,
     const mutable_tile_data<double> &A,
     const int N,
     const int M,
     const BLAS_TRANSPOSE transpose_L,
     const BLAS_SIDE side_L)
{
    GPRAT_BENCHMARK_FORCE_EVICT(L.as_span());
    GPRAT_BENCHMARK_FORCE_EVICT(A.as_span());
    GPRAT_TIME_FUNCTION(&trsm);
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

mutable_tile_data<double> syrk(const mutable_tile_data<double> &A, const const_tile_data<double> &B, const int N)
{
    GPRAT_BENCHMARK_FORCE_EVICT(A.as_span());
    GPRAT_BENCHMARK_FORCE_EVICT(B.as_span());
    GPRAT_TIME_FUNCTION(&syrk);
    // SYRK constants
    const double alpha = -1.0;
    const double beta = 1.0;
    // SYRK:A = A - B * B^T
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, N, N, alpha, B.data(), N, beta, A.data(), N);
    // return updated matrix A
    return A;
}

mutable_tile_data<double>
gemm(const const_tile_data<double> &A,
     const const_tile_data<double> &B,
     const mutable_tile_data<double> &C,
     const int N,
     const int M,
     const int K,
     const BLAS_TRANSPOSE transpose_A,
     const BLAS_TRANSPOSE transpose_B)
{
    GPRAT_BENCHMARK_FORCE_EVICT(A.as_span());
    GPRAT_BENCHMARK_FORCE_EVICT(B.as_span());
    GPRAT_BENCHMARK_FORCE_EVICT(C.as_span());
    GPRAT_TIME_FUNCTION(&gemm);
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

mutable_tile_data<double> trsv(
    const const_tile_data<double> &L, const mutable_tile_data<double> &a, const int N, const BLAS_TRANSPOSE transpose_L)
{
    GPRAT_BENCHMARK_FORCE_EVICT(L.as_span());
    GPRAT_BENCHMARK_FORCE_EVICT(a.as_span());
    GPRAT_TIME_FUNCTION(&trsv);
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

mutable_tile_data<double>
gemv(const const_tile_data<double> &A,
     const const_tile_data<double> &a,
     const mutable_tile_data<double> &b,
     const int N,
     const int M,
     const BLAS_ALPHA alpha,
     const BLAS_TRANSPOSE transpose_A)
{
    GPRAT_BENCHMARK_FORCE_EVICT(A.as_span());
    GPRAT_BENCHMARK_FORCE_EVICT(a.as_span());
    GPRAT_BENCHMARK_FORCE_EVICT(b.as_span());
    GPRAT_TIME_FUNCTION(&gemv);
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

mutable_tile_data<double>
dot_diag_syrk(const const_tile_data<double> &A, const mutable_tile_data<double> &r, const int N, const int M)
{
    GPRAT_BENCHMARK_FORCE_EVICT(A.as_span());
    GPRAT_BENCHMARK_FORCE_EVICT(r.as_span());
    GPRAT_TIME_FUNCTION(&dot_diag_syrk);
    auto r_p = r.data();
    auto A_p = A.data();
    // r = r + diag(A^T * A)
    for (std::size_t j = 0; j < static_cast<std::size_t>(M); ++j)
    {
        // Extract the j-th column and compute the dot product with itself
        r_p[j] += cblas_ddot(N, &A_p[j], M, &A_p[j], M);
    }
    return r;
}

mutable_tile_data<double>
dot_diag_gemm(const const_tile_data<double> &A,
              const const_tile_data<double> &B,
              const mutable_tile_data<double> &r,
              const int N,
              const int M)
{
    GPRAT_BENCHMARK_FORCE_EVICT(A.as_span());
    GPRAT_BENCHMARK_FORCE_EVICT(B.as_span());
    GPRAT_BENCHMARK_FORCE_EVICT(r.as_span());
    GPRAT_TIME_FUNCTION(&dot_diag_gemm);
    auto r_p = r.data();
    auto A_p = A.data();
    auto B_p = B.data();
    // r = r + diag(A * B)
    for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i)
    {
        r_p[i] += cblas_ddot(M, &A_p[i * static_cast<std::size_t>(M)], 1, &B_p[i], N);
    }
    return r;
}

// BLAS level 1 operations

mutable_tile_data<double> axpy(const mutable_tile_data<double> &y, const const_tile_data<double> &x, const int N)
{
    GPRAT_BENCHMARK_FORCE_EVICT(y.as_span());
    GPRAT_BENCHMARK_FORCE_EVICT(x.as_span());
    GPRAT_TIME_FUNCTION(&axpy);
    cblas_daxpy(N, -1.0, x.data(), 1, y.data(), 1);
    return y;
}

double dot(std::span<const double> a, std::span<const double> b, const int N)
{
    GPRAT_BENCHMARK_FORCE_EVICT(a);
    GPRAT_BENCHMARK_FORCE_EVICT(b);
    GPRAT_TIME_FUNCTION(&dot);
    // DOT: a * b
    return cblas_ddot(N, a.data(), 1, b.data(), 1);
}

#ifdef HPX_HAVE_MODULE_PERFORMANCE_COUNTERS
namespace detail
{
void register_fp64_performance_counters()
{
    // XXX: you can do this with templates, but it's quite a bit more complicated
#define GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR(name, fn_expr)                                                              \
    hpx::performance_counters::install_counter_type(                                                                   \
        name "/time",                                                                                                  \
        get_and_reset_function_elapsed<fn_expr>,                                                                       \
        #fn_expr,                                                                                                      \
        "",                                                                                                            \
        hpx::performance_counters::counter_type::monotonically_increasing);                                            \
    hpx::performance_counters::install_counter_type(                                                                   \
        name "/calls",                                                                                                 \
        get_and_reset_function_calls<fn_expr>,                                                                         \
        #fn_expr,                                                                                                      \
        "",                                                                                                            \
        hpx::performance_counters::counter_type::monotonically_increasing)

    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/potrf64", &potrf);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/trsm64", &trsm);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/syrk64", &syrk);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/gemm64", &gemm);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/trsv64", &trsv);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/gemv64", &gemv);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/dot_diag_syrk64", &dot_diag_syrk);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/dot_diag_gemm64", &dot_diag_gemm);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/axpy64", &axpy);
    GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR("/gprat/dot64", &dot);

#undef GPRAT_MAKE_SIMPLE_COUNTER_ACCESSOR
}
}  // namespace detail
#endif

GPRAT_NS_END
