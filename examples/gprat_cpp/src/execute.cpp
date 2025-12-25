//  Copyright (c) 2017-2018 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// For compliance with the NVIDIA EULA:
// "This software contains source code provided by NVIDIA Corporation."

// This is a conversion of the NVIDIA cublas example matrixMulCUBLAS to use
// HPX style data structures, executors and futures and demonstrate a simple use
// of computing a number of iteration of a matrix multiply on a stream and returning
// a future when it completes. This can be used to chain/schedule other task
// in a manner consistent with the future based API of HPX.
//
// Example usage: bin/cublas_matmul --sizemult=10 --iterations=25 --hpx:threads=8
// NB. The hpx::threads param only controls how many parallel tasks to use for the CPU
// comparison/checks and makes no difference to the GPU execution.
//
// Note: The hpx::cuda::experimental::allocator makes use of device code and if used
// this example must be compiled with nvcc instead of c++ which requires the following
// cmake setting
// set_source_files_properties(cublas_matmul.cpp
//     PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
// Currently, nvcc does not handle lambda functions properly and it is simpler to use
// cudaMalloc/cudaMemcpy etc, so we do not #define HPX_CUBLAS_DEMO_WITH_ALLOCATOR
#include "utils_c.hpp"
#include <hpx/hpx_main.hpp>
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
std::mt19937 gen;

// -------------------------------------------------------------------------
// Optional Command-line multiplier for matrix sizes
struct sMatrixSize
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
};

// -------------------------------------------------------------------------
// Run a simple test matrix multiply using CUBLAS
// -------------------------------------------------------------------------
template <typename T>
void matrixMultiply(hpx::cuda::experimental::cublas_executor cublas,
                    std::size_t heightA, std::size_t widthA,
                    std::size_t widthB, std::size_t iterations){
// void matrixMultiply(
//                     std::size_t heightA, std::size_t widthA,
//                     std::size_t widthB, std::size_t iterations)
// {

//     hpx::cuda::experimental::cublas_executor cublas(0,
//         CUBLAS_POINTER_MODE_HOST, hpx::cuda::experimental::event_mode{});
     using hpx::execution::par;    std::size_t size_A = heightA * widthA;
    std::size_t size_B = widthA * widthB;
    std::size_t size_C = heightA * widthB;

    std::vector<T> h_A(size_A);
    std::vector<T> h_B(size_B);
    std::vector<T> h_C(size_C);
    std::vector<T> h_CUBLAS(size_C);

    // Fill matrices with random numbers
    auto randfunc = [](T& x) { x = static_cast<T>(rand()) / RAND_MAX; };
    hpx::for_each(par, h_A.begin(), h_A.end(), randfunc);
    hpx::for_each(par, h_B.begin(), h_B.end(), randfunc);

    // Device memory
    T *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    hpx::cuda::experimental::check_cuda_error(cudaMalloc(&d_A, size_A * sizeof(T)));
    hpx::cuda::experimental::check_cuda_error(cudaMalloc(&d_B, size_B * sizeof(T)));
    hpx::cuda::experimental::check_cuda_error(cudaMalloc(&d_C, size_C * sizeof(T)));

    // Async host->device copies
    hpx::post(cublas, cudaMemcpyAsync, d_A, h_A.data(), size_A * sizeof(T), cudaMemcpyHostToDevice);
    auto copy_future = hpx::async(cublas, cudaMemcpyAsync, d_B, h_B.data(), size_B * sizeof(T), cudaMemcpyHostToDevice);

    copy_future.then([](auto&&) {
        std::cout << "Async host->device copy completed\n";
    });

    std::cout << "Computing result using CUBLAS...\n";

    T alpha = 1.0f, beta = 0.0f;

    // Warmup
    hpx::chrono::high_resolution_timer t1;
    auto fut = hpx::async(hpx::launch::async,[=]{
            hpx::cuda::experimental::cublas_executor cublas2(0,
         CUBLAS_POINTER_MODE_HOST, hpx::cuda::experimental::event_mode{});         hpx::async(cublas2, cublasDgemm,
        CUBLAS_OP_N, CUBLAS_OP_N,
        widthB, heightA, widthA,
        &alpha, d_B, widthB,
        d_A, widthA, &beta, d_C, widthB);});

    fut.get();
    double elapsed = t1.elapsed_microseconds();
    // Device->host copy
    auto copy_finished = hpx::async(cublas, cudaMemcpyAsync, h_CUBLAS.data(), d_C, size_C * sizeof(T), cudaMemcpyDeviceToHost);

    auto new_future = copy_finished.then([&](auto&&) {
        std::cout << "Elapsed: " << elapsed << "\n";
        double flops = 2.0 * heightA * widthA * widthB;
        double gflops = (flops * 1.0e-9) / (elapsed / 1e6);
        printf("Performance = %.2f GFlop/s, Time = %.3f ms/iter\n", gflops, 1e-3 * (elapsed / iterations));
    });

    new_future.get();

    ::hpx::cuda::experimental::check_cuda_error(cudaFree(d_A));
    ::hpx::cuda::experimental::check_cuda_error(cudaFree(d_B));
    ::hpx::cuda::experimental::check_cuda_error(cudaFree(d_C));
}
// -------------------------------------------------------------------------
int main(int argc, char** argv)
{
            // Create new argc and argv to include the --hpx:threads argument
        std::vector<std::string> args(argv, argv + argc);
        args.push_back("--hpx:threads=1");

        // Convert the arguments to char* array
        std::vector<char *> cstr_args;
        for (auto &arg : args)
        {
            cstr_args.push_back(const_cast<char *>(arg.c_str()));
        }

        int new_argc = static_cast<int>(cstr_args.size());
        char **new_argv = cstr_args.data();

        // Initialize HPX with the new arguments, don't run hpx_main
        //utils::start_hpx_runtime(new_argc, new_argv);
    //utils::start_hpx_runtime(0, nullptr);

    //hpx::run_as_hpx_thread([]{
        // install cuda future polling handler
    hpx::cuda::experimental::enable_user_polling poll("default");
    //
    std::size_t device = 0;
    std::size_t sizeMult = 1;
    //
    unsigned int seed = std::random_device{}();
    gen.seed(seed);
    std::cout << "using seed: " << seed << std::endl;

    //
    sizeMult = (std::min) (sizeMult, std::size_t(100));
    sizeMult = (std::max) (sizeMult, std::size_t(1));
    //
    // use a larger block size for Fermi and above, query default cuda target properties
    hpx::cuda::experimental::target target(device);

    std::cout << "GPU Device " << target.native_handle().get_device() << ": \""
              << target.native_handle().processor_name() << "\" "
              << "with compute capability "
              << target.native_handle().processor_family() << "\n";

    int block_size = (target.native_handle().processor_family() < 2) ? 16 : 32;

    sMatrixSize matrix_size;
    matrix_size.uiWA = 2 * block_size * sizeMult;
    matrix_size.uiHA = 4 * block_size * sizeMult;
    matrix_size.uiWB = 2 * block_size * sizeMult;
    matrix_size.uiHB = 4 * block_size * sizeMult;
    matrix_size.uiWC = 2 * block_size * sizeMult;
    matrix_size.uiHC = 4 * block_size * sizeMult;

    printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n\n",
        matrix_size.uiWA, matrix_size.uiHA, matrix_size.uiWB, matrix_size.uiHB,
        matrix_size.uiWC, matrix_size.uiHC);

    // --------------------------------
    // test matrix multiply using cublas executor
    hpx::cuda::experimental::cublas_executor cublas(device,
        CUBLAS_POINTER_MODE_HOST, hpx::cuda::experimental::event_mode{});

    matrixMultiply<double>(cublas, 64,64,64,1);
/*     hpx::async(hpx::launch::async,[=]{matrixMultiply<double>(std::move(cublas), 64,64,64,1);}); */
    //});
    // Stop the HPX runtime
    //utils::stop_hpx_runtime();

    return 0;
}
