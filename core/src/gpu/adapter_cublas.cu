#include "gpu/adapter_cublas.cuh"

// frequently used names
using hpx::cuda::experimental::check_cuda_error;

// BLAS level 3 operations

hpx::shared_future<double *>
potrf(cusolverDnHandle_t cusolver, cudaStream_t stream, hpx::shared_future<double *> f_A, const std::size_t N)
{
    cusolverDnParams_t params;
    cusolverDnCreateParams(&params);

    int *d_info = nullptr;
    check_cuda_error(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    size_t workspaceInBytesOnDevice;
    void *d_work = nullptr;
    size_t workspaceInBytesOnHost;
    void *h_work = nullptr;

    double *d_A = f_A.get();

    cusolverDnXpotrf_bufferSize(
        cusolver,
        params,
        CUBLAS_FILL_MODE_UPPER,
        N,
        CUDA_R_64F,
        d_A,
        N,
        CUDA_R_64F,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost);

    check_cuda_error(cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));

    if (workspaceInBytesOnHost > 0)
    {
        h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));
        if (h_work == nullptr)
        {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    // row-major POTRF
    // A = potrf(A)
    // for LOWER part of symmetric positive semi-definite matrix A

    // column-major cuBLAS POTRF for row-major stored A
    // for UPPER part of symmetric positive semi-definite matrix A

    cusolverDnXpotrf(
        cusolver,
        params,
        CUBLAS_FILL_MODE_UPPER,
        N,
        CUDA_R_64F,
        d_A,
        N,
        CUDA_R_64F,
        d_work,
        workspaceInBytesOnDevice,
        h_work,
        workspaceInBytesOnHost,
        d_info);
    check_cuda_error(cudaStreamSynchronize(stream));

    check_cuda_error(cudaFree(d_work));
    if (h_work != nullptr)
    {
        free(h_work);
    }
    check_cuda_error(cudaFree(d_info));
    cusolverDnDestroyParams(params);

    return hpx::make_ready_future(d_A);
}

hpx::shared_future<double *>
trsm(cublasHandle_t cublas,
     cudaStream_t stream,
     hpx::shared_future<double *> f_A,
     hpx::shared_future<double *> f_B,
     const std::size_t M,
     const std::size_t N,
     const BLAS_TRANSPOSE transpose_A,
     const BLAS_SIDE side_A)
{
    // TRSM constants
    const double alpha = 1.0;
    double *d_A = f_A.get();
    double *d_B = f_B.get();

    // row-major TRSM solves for X
    //
    // for side_A == Blas_right:
    //   op(A) * X = alpha * B
    //     A^T * X = B
    //
    // for side_A == Blas_left:
    //   X * op(A) = alpha * B
    //     X * A^T = B
    //
    // for op: transpose_A

    // column-major cuBLAS TRSM for row-major stored A & B
    // for X on opposite side (opposite of side_A)

    cublasDtrsm(
        cublas,
        opposite(static_cast<cublasSideMode_t>(side_A)),
        CUBLAS_FILL_MODE_UPPER,
        static_cast<cublasOperation_t>(transpose_A),
        CUBLAS_DIAG_NON_UNIT,
        N,
        M,
        &alpha,
        d_A,
        M,
        d_B,
        N);

    check_cuda_error(cudaStreamSynchronize(stream));
    return hpx::make_ready_future(d_B);
}

hpx::shared_future<double *>
syrk(cublasHandle_t cublas,
     cudaStream_t stream,
     hpx::shared_future<double *> f_A,
     hpx::shared_future<double *> f_C,
     const std::size_t N)
{
    // SYRK constants
    const double alpha = -1.0;
    const double beta = 1.0;
    double *d_A = f_A.get();
    double *d_C = f_C.get();

    // row-major SYRK
    // C = alpha * op(A) * op(A)^T + beta * C
    //     C = - A * A^T + C
    // for LOWER part of symmetric matrix C
    // for op: NO transpose:

    // column-major cuBLAS SYRK for row-major stored A & C
    // C = - op(A) * op(A)^T + fm(C)
    //   = - A^T * A - C
    // for UPPER part of symmetric matrix C
    // for op: TRANSPOSE

    cublasDsyrk(cublas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, N, N, &alpha, d_A, N, &beta, d_C, N);

    check_cuda_error(cudaStreamSynchronize(stream));
    return hpx::make_ready_future(d_C);
}

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
     const BLAS_TRANSPOSE transpose_B)
{
    const double alpha = -1.0;
    const double beta = 1.0;
    double *d_A = f_A.get();
    double *d_B = f_B.get();
    double *d_C = f_C.get();

    // row-major GEMM
    // C = alpha * op(A) * op(B) + beta * C
    //   = op(A) * op(B) - C
    // for op(A): transpose_A
    // for op(B): transpose_B

    // column-major cuBLAS GEMM for row-major stored A, B, C
    // C = alpha * op(B) * op(A) + beta * C
    //   = op(B) * op(A) - C
    // for inverted ordering of matrices A, B

    cublasDgemm(
        cublas,
        static_cast<cublasOperation_t>(transpose_B),
        static_cast<cublasOperation_t>(transpose_A),
        N,
        M,
        K,
        &alpha,
        d_B,
        N,
        d_A,
        K,
        &beta,
        d_C,
        N);

    check_cuda_error(cudaStreamSynchronize(stream));
    return hpx::make_ready_future(d_C);
}
