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
#include <future>

// -------------------------------------------------------------------------
// Run a simple test matrix multiply using CUBLAS
// -------------------------------------------------------------------------
void gemm(){
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
}
// -------------------------------------------------------------------------
int main(int argc, char** argv)
{
        //     // Create new argc and argv to include the --hpx:threads argument
        // std::vector<std::string> args(argv, argv + argc);
        // args.push_back("--hpx:threads=1");
        //
        // // Convert the arguments to char* array
        // std::vector<char *> cstr_args;
        // for (auto &arg : args)
        // {
        //     cstr_args.push_back(const_cast<char *>(arg.c_str()));
        // }
        //
        // int new_argc = static_cast<int>(cstr_args.size());
        // char **new_argv = cstr_args.data();

        // Initialize HPX with the new arguments, don't run hpx_main
        //utils::start_hpx_runtime(new_argc, new_argv);
    //utils::start_hpx_runtime(0, nullptr);
    std::async(&gemm);

auto fut = hpx::async([=] {
    cudaSetDevice(0);  // REQUIRED

    const int m = 2, n = 2, k = 2;
    const double alpha = 1.0;
    const double beta  = 0.0;

    double h_A[m * k] = {1.0, 3.0, 2.0, 4.0};
    double h_B[k * n] = {5.0, 7.0, 6.0, 8.0};
    double h_C[m * n] = {0.0, 0.0, 0.0, 0.0};

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(double));
    cudaMalloc(&d_B, k * n * sizeof(double));
    cudaMalloc(&d_C, m * n * sizeof(double));

    cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, m * n * sizeof(double), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
cudaDeviceSynchronize();
    cublasDgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k,
                &alpha,
                d_A, m,
                d_B, k,
                &beta,
                d_C, m);

    cudaDeviceSynchronize();  // IMPORTANT for correctness

    cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    printf("C =\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", h_C[i + j * m]);
        }
        printf("\n");
    }

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
});

fut.get();  // ensures lifetime & synchronization





    // Stop the HPX runtime
    //utils::stop_hpx_runtime();

    return 0;
}
