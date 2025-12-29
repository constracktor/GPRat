#include "gprat_c.hpp"
#include "utils_c.hpp"
#include <hpx/hpx_main.hpp>
#include <iostream>
#include <vector>

bool are_identical(const std::vector<std::vector<double>> &A,
                   const std::vector<std::vector<double>> &B,
                   double tol = 1e-14)
{
    if (A.size() != B.size())
    {
        std::cout << "Size mismatch: rows " << A.size() << " vs " << B.size() << std::endl;
        return false;
    }

    for (std::size_t i = 0; i < A.size(); ++i)
    {
        if (A[i].size() != B[i].size())
        {
            std::cout << "Size mismatch at row " << i << ": cols " << A[i].size() << " vs " << B[i].size() << std::endl;
            return false;
        }

        for (std::size_t j = 0; j < A[i].size(); ++j)
        {
            double diff = std::abs(A[i][j] - B[i][j]);
            if (diff > tol)
            {
                std::cout << "Mismatch at (" << i << "," << j << ")  " << "cpu=" << B[i][j] << " gpu=" << A[i][j]
                          << " diff=" << diff << std::endl;
                return false;
            }
        }
    }

    return true;
}

int main(int argc, char *argv[])
{
    bool use_gpu =
        utils::compiled_with_cuda() && gprat::gpu_count() > 0 && argc > 1 && std::strcmp(argv[1], "--use_gpu") == 0;
    /////////////////////
    /////// configuration
    std::size_t START = 1024;
    std::size_t END = 65'536;
    std::size_t STEP = 2;
    std::size_t LOOP = 1;

    int n_test = 1024;
    const std::size_t N_CORES = 128;
    const std::size_t n_tiles = 32;
    const std::size_t n_reg = 8;

    std::string train_path = "../../../data/data_19/training_input_19.txt";

    for (std::size_t core = 128; core <= N_CORES; core = core * 2)
    {
        for (std::size_t start = START; start <= END; start = start * STEP)
        {
            int n_train = static_cast<int>(start);
            std::cout << "\n\nProblem size: " << start << std::endl;
            for (std::size_t l = 0; l < LOOP; l++)
            {
                // Compute tile sizes and number of predict tiles
                int tile_size = utils::compute_train_tile_size(n_train, n_tiles);
                /////////////////////
                ////// data loading
                gprat::GP_data training_input(train_path, n_train, n_reg);

                std::chrono::duration<double> init_time;
                std::chrono::duration<double> cholesky_async_time;
                std::chrono::duration<double> cholesky_sync_time;
                std::chrono::duration<double> cholesky_ref_time;
                std::chrono::duration<double> cholesky_val_time;
                std::chrono::duration<double> cholesky_mut_time;

                std::string target;

                auto start = std::chrono::high_resolution_clock::now();
                auto end = std::chrono::high_resolution_clock::now();

                if (!use_gpu)
                {
                    target = "cpu";

                    /////////////////////
                    ///// GP
                    start = std::chrono::high_resolution_clock::now();
                    gprat::GP gp_cpu(training_input.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 });
                    end = std::chrono::high_resolution_clock::now();
                    init_time = end - start;

                    start = std::chrono::high_resolution_clock::now();
                    std::vector<std::vector<double>> cholesky_cpu_async = gp_cpu.cholesky_async("async_future");
                    end = std::chrono::high_resolution_clock::now();
                    cholesky_async_time = end - start;
                    std::cout << "cpu async future cholesky time: " << cholesky_async_time.count() << std::endl;

                    start = std::chrono::high_resolution_clock::now();
                    cholesky_cpu_async = gp_cpu.cholesky_async("async_ref");
                    end = std::chrono::high_resolution_clock::now();
                    cholesky_async_time = end - start;
                    std::cout << "cpu async ref cholesky time: " << cholesky_async_time.count() << std::endl;

                    start = std::chrono::high_resolution_clock::now();
                    cholesky_cpu_async = gp_cpu.cholesky_async("async_val");
                    end = std::chrono::high_resolution_clock::now();
                    cholesky_async_time = end - start;
                    std::cout << "cpu async val cholesky time: " << cholesky_async_time.count() << std::endl;

                    ////

                    start = std::chrono::high_resolution_clock::now();
                    std::vector<std::vector<double>> cholesky_cpu_sync = gp_cpu.cholesky_sync("sync_future");
                    end = std::chrono::high_resolution_clock::now();
                    cholesky_sync_time = end - start;
                    std::cout << "cpu sync future cholesky time: " << cholesky_sync_time.count() << std::endl;

                    start = std::chrono::high_resolution_clock::now();
                    cholesky_cpu_sync = gp_cpu.cholesky_sync("sync_ref");
                    end = std::chrono::high_resolution_clock::now();
                    cholesky_sync_time = end - start;
                    std::cout << "cpu sync ref cholesky time: " << cholesky_sync_time.count() << std::endl;

                    start = std::chrono::high_resolution_clock::now();
                    cholesky_cpu_sync = gp_cpu.cholesky_sync("sync_val");
                    end = std::chrono::high_resolution_clock::now();
                    cholesky_sync_time = end - start;
                    std::cout << "cpu sync val cholesky time: " << cholesky_sync_time.count() << std::endl;

                    ////

                    start = std::chrono::high_resolution_clock::now();
                    std::vector<std::vector<double>> cholesky_cpu_ref = gp_cpu.cholesky_loop("loop_one");
                    end = std::chrono::high_resolution_clock::now();
                    cholesky_ref_time = end - start;
                    std::cout << "cpu ref cholesky time: " << cholesky_ref_time.count() << std::endl;

                    start = std::chrono::high_resolution_clock::now();
                    std::vector<std::vector<double>> cholesky_cpu_val = gp_cpu.cholesky_loop("loop_two");
                    end = std::chrono::high_resolution_clock::now();
                    cholesky_val_time = end - start;
                    std::cout << "cpu val cholesky time: " << cholesky_val_time.count() << std::endl;

                    start = std::chrono::high_resolution_clock::now();
                    std::vector<std::vector<double>> cholesky_cpu_mut = gp_cpu.cholesky_mutable();
                    end = std::chrono::high_resolution_clock::now();
                    cholesky_mut_time = end - start;
                    std::cout << "cpu mut cholesky time: " << cholesky_mut_time.count() << std::endl;
                    // bool ok_sync = are_identical(cholesky_cpu_async, cholesky_cpu_sync);
                    // bool ok_ref = are_identical(cholesky_cpu_async, cholesky_cpu_ref);
                    // bool ok_val = are_identical(cholesky_cpu_async, cholesky_cpu_val);
                    // if (ok_sync && ok_ref && ok_val)
                    //     std::cout << "Cholesky results are IDENTICAL (within tolerance)\n";
                    // else
                    //       std::cout << "Cholesky results differ!\n";
                }
                else
                {
                    target = "gpu";

                    auto start_init = std::chrono::high_resolution_clock::now();
                    gprat::GP gp_gpu(training_input.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, 0, 32);
                    auto end_init = std::chrono::high_resolution_clock::now();
                    init_time = end_init - start_init;

                    auto start_cholesky = std::chrono::high_resolution_clock::now();
                    std::vector<std::vector<double>> cholesky_gpu = gp_gpu.cholesky();
                    auto end_cholesky = std::chrono::high_resolution_clock::now();
                    cholesky_async_time = end_cholesky - start_cholesky;
                    std::cout << "GPU: Cholesky time: " << cholesky_async_time.count() << std::endl;

                    // compre agains CPU
                    target = "cpu";
                    gprat::GP gp_cpu(training_input.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 });
                    std::vector<std::vector<double>> cholesky_cpu = gp_cpu.cholesky();

                    // ---- call the check ----

                    bool ok = are_identical(cholesky_gpu, cholesky_cpu);

                    if (ok)
                    {
                        std::cout << "Cholesky results are IDENTICAL (within tolerance)\n";
                    }
                    else
                    {
                        std::cout << "Cholesky results differ!\n";
                    }
                }
            }
        }
    }

    return 0;
}
