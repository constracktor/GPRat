#include "gprat_c.hpp"

#include "cpu/gp_functions.hpp"
#include "utils_c.hpp"
#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <hpx/algorithm.hpp>
#include <hpx/assert.hpp>
#include <hpx/chrono.hpp>
#include <hpx/execution.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/async_cuda.hpp>
#include <hpx/modules/testing.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <sstream>
#include <utility>
#include <vector>
//
//std::mt19937 gen;

#if GPRAT_WITH_CUDA
#include "gpu/gp_functions.cuh"
#endif

// namespace for GPRat library entities
namespace gprat
{

GP_data::GP_data(const std::string &f_path, int n, int n_reg) :
    file_path(f_path),
    n_samples(n),
    n_regressors(n_reg)
{
    data = utils::load_data(f_path, n, n_reg - 1);
}

GP::GP(std::vector<double> input,
       std::vector<double> output,
       int n_tiles,
       int n_tile_size,
       int n_regressors,
       std::vector<double> kernel_hyperparams,
       std::vector<bool> trainable_bool,
       std::shared_ptr<Target> target) :
    training_input_(input),
    training_output_(output),
    n_tiles_(n_tiles),
    n_tile_size_(n_tile_size),
    trainable_params_(trainable_bool),
    target_(target),
    n_reg(n_regressors),
    kernel_params(kernel_hyperparams[0], kernel_hyperparams[1], kernel_hyperparams[2])
{ }

GP::GP(std::vector<double> input,
       std::vector<double> output,
       int n_tiles,
       int n_tile_size,
       int n_regressors,
       std::vector<double> kernel_hyperparams,
       std::vector<bool> trainable_bool) :
    training_input_(input),
    training_output_(output),
    n_tiles_(n_tiles),
    n_tile_size_(n_tile_size),
    trainable_params_(trainable_bool),
    target_(std::make_shared<CPU>()),
    n_reg(n_regressors),
    kernel_params(kernel_hyperparams[0], kernel_hyperparams[1], kernel_hyperparams[2])
{ }

GP::GP(std::vector<double> input,
       std::vector<double> output,
       int n_tiles,
       int n_tile_size,
       int n_regressors,
       std::vector<double> kernel_hyperparams,
       std::vector<bool> trainable_bool,
       int gpu_id,
       int n_streams) :
    training_input_(input),
    training_output_(output),
    n_tiles_(n_tiles),
    n_tile_size_(n_tile_size),
    trainable_params_(trainable_bool),
#if GPRAT_WITH_CUDA
    target_(std::make_shared<CUDA_GPU>(CUDA_GPU(gpu_id, n_streams))),
#else
    target_(std::make_shared<CPU>()),
#endif
    n_reg(n_regressors),
    kernel_params(kernel_hyperparams[0], kernel_hyperparams[1], kernel_hyperparams[2])
{
#if !GPRAT_WITH_CUDA
    throw std::runtime_error(
        "Cannot create GP object using CUDA for computation. "
        "CUDA is not available because GPRat has been compiled without CUDA. "
        "Remove arguments gpu_id ("
        + std::to_string(gpu_id) + ") and n_streams (" + std::to_string(n_streams)
        + ") to perform computations on the CPU.");
#endif
}

std::string GP::repr() const
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(12);
    oss << "Kernel_Params: [lengthscale=" << kernel_params.lengthscale << ", vertical_lengthscale="
        << kernel_params.vertical_lengthscale << ", noise_variance=" << kernel_params.noise_variance
        << ", n_regressors=" << n_reg << "], Trainable_Params: [trainable_params l=" << trainable_params_[0]
        << ", trainable_params v=" << trainable_params_[1] << ", trainable_params n=" << trainable_params_[2]
        << "], Target: [" << target_->repr() << "], n_tiles=" << n_tiles_ << ", n_tile_size=" << n_tile_size_;
    return oss.str();
}

std::vector<double> GP::get_training_input() const { return training_input_; }

std::vector<double> GP::get_training_output() const { return training_output_; }
struct sMatrixSize
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
};
//std::vector<std::vector<double>> 
void GP::cholesky()
{
    return hpx::async(hpx::launch::sync,
               [this]()
               {
                    hpx::async(hpx::launch::sync,
                        [this]()
                        {
/////////////////////////////////////////////////////////////////////////////
    const int m = 2, n = 2, k = 2;
    const double alpha = 1.0;
    const double beta  = 0.0;

    // Host matrices (column-major)
    double h_A[m * k] = {
        1.0, 3.0,
        2.0, 4.0
    };

    double h_B[k * n] = {
        5.0, 7.0,
        6.0, 8.0
    };

    double h_C[m * n] = {0.0, 0.0, 0.0, 0.0};

    // Device pointers
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * k * sizeof(double));
    cudaMalloc((void**)&d_B, k * n * sizeof(double));
    cudaMalloc((void**)&d_C, m * n * sizeof(double));

    cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, m * n * sizeof(double), cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // DGEMM: C = alpha * A * B + beta * C
    cublasDgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        &alpha,
        d_A, m,
        d_B, k,
        &beta,
        d_C, m
    );

    // Copy result back
    cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Print result
    printf("C = \n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", h_C[i + j * m]);
        }
        printf("\n");
    }

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
// /////////////////////////////////////////////////////////////////////////////
    // const int n = 2;   // C is n x n
    // const int k = 2;   // A is n x k
    // const double alpha = 1.0;
    // const double beta  = 0.0;
    //
    // // Host matrix A (column-major, n x k)
    // double h_A[n * k] = {
    //     1.0, 3.0,
    //     2.0, 4.0
    // };
    //
    // // Host matrix C (column-major, n x n)
    // double h_C[n * n] = {
    //     0.0, 0.0,
    //     0.0, 0.0
    // };
    //
    // // Device pointers
    // double *d_A, *d_C;
    // cudaMalloc((void**)&d_A, n * k * sizeof(double));
    // cudaMalloc((void**)&d_C, n * n * sizeof(double));
    //
    // cudaMemcpy(d_A, h_A, n * k * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_C, h_C, n * n * sizeof(double), cudaMemcpyHostToDevice);
    //
    // // Create cuBLAS handle
    // cublasHandle_t handle;
    // cublasCreate(&handle);
    //
    // // SYRK: C = alpha * A * A^T + beta * C
    // // Only the lower triangle of C is updated
    // cublasDsyrk(
    //     handle,
    //     CUBLAS_FILL_MODE_LOWER,  // update lower triangle of C
    //     CUBLAS_OP_N,             // A is not transposed
    //     n,                       // size of C (n x n)
    //     k,                       // rank-k update
    //     &alpha,
    //     d_A, n,                  // A and lda
    //     &beta,
    //     d_C, n                   // C and ldc
    // );
    //
    // // Copy result back
    // cudaMemcpy(h_C, d_C, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    //
    // // Print result (full matrix, even though only lower triangle is valid)
    // printf("C (lower triangle valid):\n");
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%f ", h_C[i + j * n]);
    //     }
    //     printf("\n");
    // }
    //
    // // Cleanup
    // cublasDestroy(handle);
    // cudaFree(d_A);
    // cudaFree(d_C);
/////////////////////////////////////////////////////////////////////////////
//     using hpx::execution::par;
//     // install cuda future polling handler
//     hpx::cuda::experimental::enable_user_polling poll("default");
//     //
//     std::size_t device = 0; 
//     std::size_t sizeMult = 4;
//     std::size_t iterations = 1;    //
//     unsigned int seed = std::random_device{}();
//     //
//     sizeMult = (std::min) (sizeMult, std::size_t(100));
//     sizeMult = (std::max) (sizeMult, std::size_t(1));
//     //
//     // use a larger block size for Fermi and above, query default cuda target properties
//     hpx::cuda::experimental::target target(device);
//
//     std::cout << "GPU Device " << target.native_handle().get_device() << ": \""
//               << target.native_handle().processor_name() << "\" "
//               << "with compute capability "
//               << target.native_handle().processor_family() << "\n";
//
//     int block_size = (target.native_handle().processor_family() < 2) ? 16 : 32;
// hpx::cuda::experimental::cublas_executor cublas(device,
//         CUBLAS_POINTER_MODE_HOST, hpx::cuda::experimental::event_mode{});
//     sMatrixSize matrix_size;
//     matrix_size.uiWA = 2 * block_size * sizeMult;
//     matrix_size.uiHA = 4 * block_size * sizeMult;
//     matrix_size.uiWB = 2 * block_size * sizeMult;
//     matrix_size.uiHB = 4 * block_size * sizeMult;
//     matrix_size.uiWC = 2 * block_size * sizeMult;
//     matrix_size.uiHC = 4 * block_size * sizeMult;
//
//     printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n\n",
//         matrix_size.uiWA, matrix_size.uiHA, matrix_size.uiWB, matrix_size.uiHB,
//         matrix_size.uiWC, matrix_size.uiHC);
//     // Allocate host memory for matrices A and B
//     unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
//     unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
//     unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
//
//     std::vector<double> h_A(size_A);
//     std::vector<double> h_B(size_B);
//     std::vector<double> h_C(size_C);
//     std::vector<double> h_CUBLAS(size_C);
//
//     // Fill A and B with random numbers
//     auto randfunc = [](double& x) { x = gen() / (double) RAND_MAX; };
//     hpx::for_each(par, h_A.begin(), h_A.end(), randfunc);
//     hpx::for_each(par, h_B.begin(), h_B.end(), randfunc);
//
//     // create a cublas executor we'll use to futurize cuda events
//     using namespace hpx::cuda::experimental;
//     using cublas_future = typename cuda_executor::future_type;
//
//     double *d_A, *d_B, *d_C;
//     hpx::cuda::experimental::check_cuda_error(
//         cudaMalloc((void**) &d_A, size_A * sizeof(double)));
//
//     hpx::cuda::experimental::check_cuda_error(
//         cudaMalloc((void**) &d_B, size_B * sizeof(double)));
//
//     hpx::cuda::experimental::check_cuda_error(
//         cudaMalloc((void**) &d_C, size_C * sizeof(double)));
//
//     // adding async copy operations into the stream before cublas calls puts
//     // the copies in the queue before the matrix operations.
//     hpx::post(cublas, cudaMemcpyAsync, d_A, h_A.data(), size_A * sizeof(double),
//         cudaMemcpyHostToDevice);
//
//     auto copy_future = hpx::async(cublas, cudaMemcpyAsync, d_B, h_B.data(),
//         size_B * sizeof(double), cudaMemcpyHostToDevice);
//
//     // we can call get_future multiple times on the cublas helper.
//     // Each one returns a new future that will be set ready when the stream event
//     // for this point is triggered
//     copy_future.then([](cublas_future&&) {
//         std::cout << "The async host->device copy operation completed"
//                   << std::endl;
//     });
//
//     std::cout << "Computing result using CUBLAS...\n";
//     double const alpha = 1.0;
//     double const beta = 0.0;
//
//     // Perform warmup operation with cublas
//     // note cublas is column major ordering : transpose the order
//     hpx::chrono::high_resolution_timer t1;
//     //
//     std::cout << "calling CUBLAS...\n";
//     auto fut = hpx::async(cublas, cublasDgemm, CUBLAS_OP_N, CUBLAS_OP_N,
//         matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B,
//         matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWA);
//
//     // wait until the operation completes
//     fut.get();
//
//     double us1 = t1.elapsed_microseconds();
//     std::cout << "warmup: elapsed_microseconds " << us1 << std::endl;
//
//     // once the future has been retrieved, the next call to
//     // get_future will create a new event attached to a new future
//     // so we can reuse the same cublas executor stream if we want
//
//     hpx::chrono::high_resolution_timer t2;
//     for (std::size_t j = 0; j < iterations; j++)
//     {
//         hpx::post(cublas, cublasDgemm, CUBLAS_OP_N, CUBLAS_OP_N,
//             matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B,
//             matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C,
//             matrix_size.uiWA);
//     }
//     // get a future for when the stream reaches this point (matrix operations complete)
//     auto matrix_finished = cublas.get_future();
//
//     // when the matrix operations complete, copy the result to the host
//     auto finished = hpx::async(cublas, cudaMemcpyAsync, h_CUBLAS.data(),
//         d_C, size_C * sizeof(double), cudaMemcpyDeviceToHost);
//
//     finished.get();
//     ::hpx::cuda::experimental::check_cuda_error(cudaFree(d_A));
//     ::hpx::cuda::experimental::check_cuda_error(cudaFree(d_B));
//     ::hpx::cuda::experimental::check_cuda_error(cudaFree(d_C));
//
// #if GPRAT_WITH_CUDA
//                    if (target_->is_gpu())
//                    {
//                        return gpu::cholesky(
//                            training_input_,
//                            kernel_params,
//                            n_tiles_,
//                            n_tile_size_,
//                            n_reg,
//                            *std::dynamic_pointer_cast<gprat::CUDA_GPU>(target_));
//                    }
//                    else
//                    {
//                        return cpu::cholesky(training_input_, kernel_params, n_tiles_, n_tile_size_, n_reg);
//                    }
// #else
//                    return cpu::cholesky(training_input_, kernel_params, n_tiles_, n_tile_size_, n_reg);
// #endif
                })
        .get();
  })
        .get();
}

}  // namespace gprat
