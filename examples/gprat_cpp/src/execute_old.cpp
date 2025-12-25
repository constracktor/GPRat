#include "gprat_c.hpp"
#include "utils_c.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include "hpx/hpx_main.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
int main(int argc, char *argv[])
{
hpx::async(hpx::launch::sync,
               []()
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
    cudaFree(d_C);}).get();
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






























    /////////////////////
    /////// configuration
    std::size_t START = 1024;
    std::size_t END = 1024;
    std::size_t STEP = 2;
    std::size_t LOOP = 1;
    const std::size_t OPT_ITER = 1;

    int n_test = 1024;
    const std::size_t N_CORES = 1;
    const std::size_t n_tiles = 16;
    const std::size_t n_reg = 8;

    std::string train_path = "../../../data/data_1024/training_input.txt";
    std::string out_path = "../../../data/data_1024/training_output.txt";
    std::string test_path = "../../../data/data_1024/test_input.txt";

    bool use_gpu = true;

       // utils::compiled_with_cuda() && gprat::gpu_count() > 0 && argc > 1 && std::strcmp(argv[1], "--use_gpu") == 0;

    for (std::size_t core = 1; core <= N_CORES; core = core * 2)
    {
        // Create new argc and argv to include the --hpx:threads argument
        std::vector<std::string> args(argv, argv + argc);
        if (use_gpu)
        {
            args.erase(args.begin() + 1);
        }
        args.push_back("--hpx:threads=" + std::to_string(core));

        // Convert the arguments to char* array
        std::vector<char *> cstr_args;
        for (auto &arg : args)
        {
            cstr_args.push_back(const_cast<char *>(arg.c_str()));
        }

        int new_argc = static_cast<int>(cstr_args.size());
        char **new_argv = cstr_args.data();

        for (std::size_t start = START; start <= END; start = start * STEP)
        {
            int n_train = static_cast<int>(start);
            for (std::size_t l = 0; l < LOOP; l++)
            {
                // Compute tile sizes and number of predict tiles
                int tile_size = utils::compute_train_tile_size(n_train, n_tiles);
                auto result = utils::compute_test_tiles(n_test, n_tiles, tile_size);
                /////////////////////
                ///// hyperparams
                gprat_hyper::AdamParams hpar = { 0.1, 0.9, 0.999, 1e-8, OPT_ITER };

                /////////////////////
                ////// data loading
                gprat::GP_data training_input(train_path, n_train, n_reg);
                gprat::GP_data training_output(out_path, n_train, n_reg);
                gprat::GP_data test_input(test_path, n_test, n_reg);

                auto start_total = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> init_time;
                std::chrono::duration<double> cholesky_time;
                std::chrono::duration<double> opt_time;
                std::chrono::duration<double> pred_uncer_time;
                std::chrono::duration<double> pred_full_cov_time;
                std::chrono::duration<double> pred_time;
                std::vector<bool> trainable = { true, true, true };
                std::string target;

                if (!use_gpu)
                {
                    target = "cpu";

                    /////////////////////
                    ///// GP
                    auto start_init = std::chrono::high_resolution_clock::now();
                    gprat::GP gp_cpu(training_input.data,
                                     training_output.data,
                                     n_tiles,
                                     tile_size,
                                     n_reg,
                                     { 1.0, 1.0, 0.1 },
                                     trainable);
                    auto end_init = std::chrono::high_resolution_clock::now();
                    init_time = end_init - start_init;

                    // Initialize HPX with the new arguments, don't run hpx_main
                    //utils::start_hpx_runtime(new_argc, new_argv);

                    // Measure the time taken to execute gp.cholesky();
                    auto start_cholesky = std::chrono::high_resolution_clock::now();
                    //std::vector<std::vector<double>> choleksy_cpu = 
                    gp_cpu.cholesky();
                    auto end_cholesky = std::chrono::high_resolution_clock::now();
                    cholesky_time = end_cholesky - start_cholesky;

                    // Measure the time taken to execute gp.optimize(hpar);
                    auto start_opt = std::chrono::high_resolution_clock::now();
                    //std::vector<double> losses = gp_cpu.optimize(hpar);
                    auto end_opt = std::chrono::high_resolution_clock::now();
                    opt_time = end_opt - start_opt;

                    auto start_pred_uncer = std::chrono::high_resolution_clock::now();
                    //std::vector<std::vector<double>> sum_cpu =
                      //  gp_cpu.predict_with_uncertainty(test_input.data, result.first, result.second);
                    auto end_pred_uncer = std::chrono::high_resolution_clock::now();
                    pred_uncer_time = end_pred_uncer - start_pred_uncer;

                    auto start_pred_full_cov = std::chrono::high_resolution_clock::now();
                    //std::vector<std::vector<double>> full_cpu =
                      //  gp_cpu.predict_with_full_cov(test_input.data, result.first, result.second);
                    auto end_pred_full_cov = std::chrono::high_resolution_clock::now();
                    pred_full_cov_time = end_pred_full_cov - start_pred_full_cov;

                    auto start_pred = std::chrono::high_resolution_clock::now();
                    //std::vector<double> pred_cpu = gp_cpu.predict(test_input.data, result.first, result.second);
                    auto end_pred = std::chrono::high_resolution_clock::now();
                    pred_time = end_pred - start_pred;
                }
                else
                {
                    target = "gpu";

                    auto start_init = std::chrono::high_resolution_clock::now();
                    gprat::GP gp_gpu(
                        training_input.data,
                        training_output.data,
                        n_tiles,
                        tile_size,
                        n_reg,
                        { 1.0, 1.0, 0.1 },
                        trainable,
                        0,
                        2);
                    auto end_init = std::chrono::high_resolution_clock::now();
                    init_time = end_init - start_init;

                    // Initialize HPX with the new arguments, don't run hpx_main
                    //utils::start_hpx_runtime(new_argc, new_argv);

                    auto start_cholesky = std::chrono::high_resolution_clock::now();
                    //std::vector<std::vector<double>> choleksy_gpu = 
                        gp_gpu.cholesky();
                    auto end_cholesky = std::chrono::high_resolution_clock::now();
                    cholesky_time = end_cholesky - start_cholesky;

                    // NOTE: optimization is not implemented for GPU
                    opt_time = std::chrono::seconds(-1);

                    auto start_pred_uncer = std::chrono::high_resolution_clock::now();
                    //std::vector<std::vector<double>> sum_gpu =
                        //gp_gpu.predict_with_uncertainty(test_input.data, result.first, result.second);
                    auto end_pred_uncer = std::chrono::high_resolution_clock::now();
                    pred_uncer_time = end_pred_uncer - start_pred_uncer;

                    auto start_pred_full_cov = std::chrono::high_resolution_clock::now();
                    //std::vector<std::vector<double>> full_gpu =
                      //  gp_gpu.predict_with_full_cov(test_input.data, result.first, result.second);
                    auto end_pred_full_cov = std::chrono::high_resolution_clock::now();
                    pred_full_cov_time = end_pred_full_cov - start_pred_full_cov;

                    auto start_pred = std::chrono::high_resolution_clock::now();
                    //std::vector<double> pred_gpu = gp_gpu.predict(test_input.data, result.first, result.second);
                    auto end_pred = std::chrono::high_resolution_clock::now();
                    pred_time = end_pred - start_pred;
                }

                // Stop the HPX runtime
                //utils::stop_hpx_runtime();

                auto end_total = std::chrono::high_resolution_clock::now();
                auto total_time = end_total - start_total;

                // Save parameters and times to a .txt file with a header
                std::ofstream outfile("../output.csv", std::ios::app);  // Append mode
                if (outfile.tellp() == 0)
                {
                    // If file is empty, write the header
                    outfile << "Target,Cores,N_train,N_test,N_tiles,N_regressor,Opt_iter,Total_time,Init_time,Cholesky_"
                               "time,Opt_time,Pred_Uncer_time,Pred_Full_time,Pred_time,N_loop\n";
                }
                outfile << target << "," << core << "," << n_train << "," << n_test << "," << n_tiles << "," << n_reg
                        << "," << OPT_ITER << "," << total_time.count() << "," << init_time.count() << ","
                        << cholesky_time.count() << "," << opt_time.count() << "," << pred_uncer_time.count() << ","
                        << pred_full_cov_time.count() << "," << pred_time.count() << "," << l << "\n";
                outfile.close();
            }
        }
    }

    return 0;
}
