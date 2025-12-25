#include "gprat_c.hpp"
#include "utils_c.hpp"
#include <hpx/hpx_main.hpp>

#include <iostream>
#include <vector>


int main(int argc, char *argv[])
{
    bool use_gpu =
        utils::compiled_with_cuda() && gprat::gpu_count() > 0 && argc > 1 && std::strcmp(argv[1], "--use_gpu") == 0;
    /////////////////////
    /////// configuration
    std::size_t START = 1024;
    std::size_t END = 1024;
    std::size_t STEP = 2;
    std::size_t LOOP = 10;

    int n_test = 1024;
    const std::size_t N_CORES = 4;
    const std::size_t n_tiles = 16;
    const std::size_t n_reg = 8;

    std::string train_path = "../../../data/data_1024/training_input.txt";

    for (std::size_t core = 2; core <= N_CORES; core = core * 2)
    {
        for (std::size_t start = START; start <= END; start = start * STEP)
        {
            int n_train = static_cast<int>(start);
            for (std::size_t l = 0; l < LOOP; l++)
            {
                // Compute tile sizes and number of predict tiles
                int tile_size = utils::compute_train_tile_size(n_train, n_tiles);
                auto result = utils::compute_test_tiles(n_test, n_tiles, tile_size);
                /////////////////////
                ////// data loading
                gprat::GP_data training_input(train_path, n_train, n_reg);

                auto start_total = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double> init_time;
                std::chrono::duration<double> cholesky_time;
                std::string target;

                if (!use_gpu)
                {
                    target = "cpu";

                    /////////////////////
                    ///// GP
                    auto start_init = std::chrono::high_resolution_clock::now();
                    gprat::GP gp_cpu(training_input.data,
                                     n_tiles,
                                     tile_size,
                                     n_reg,
                                     { 1.0, 1.0, 0.1 });
                    auto end_init = std::chrono::high_resolution_clock::now();
                    init_time = end_init - start_init;

                    auto start_cholesky = std::chrono::high_resolution_clock::now();
                    std::vector<std::vector<double>> choleksy_cpu = gp_cpu.cholesky();
                    auto end_cholesky = std::chrono::high_resolution_clock::now();
                    cholesky_time = end_cholesky - start_cholesky;
                    std::cout << "CPU: Cholesky time: " << cholesky_time.count() << std::endl;
                }
                else
                {
                    target = "gpu";

                    auto start_init = std::chrono::high_resolution_clock::now();
                    gprat::GP gp_gpu(
                        training_input.data,
                        n_tiles,
                        tile_size,
                        n_reg,
                        { 1.0, 1.0, 0.1 },
                        0,
                        32);
                    auto end_init = std::chrono::high_resolution_clock::now();
                    init_time = end_init - start_init;

                    auto start_cholesky = std::chrono::high_resolution_clock::now();
                    std::vector<std::vector<double>> cholesky_gpu = gp_gpu.cholesky();
                    auto end_cholesky = std::chrono::high_resolution_clock::now();
                    cholesky_time = end_cholesky - start_cholesky;
                    std::cout << "GPU: Cholesky time: " << cholesky_time.count() << std::endl;
                
                    // compre agains CPU
                    target = "cpu";
                    gprat::GP gp_cpu(training_input.data,
                                     n_tiles,
                                     tile_size,
                                     n_reg,
                                     { 1.0, 1.0, 0.1 });
                    std::vector<std::vector<double>> cholesky_cpu = gp_cpu.cholesky();

                    // ---- call the check ----
auto are_identical = [&](const auto& A, const auto& B,
                         double tol = 1e-14)
{
    if (A.size() != B.size()) {
        std::cout << "Size mismatch: rows "
                  << A.size() << " vs " << B.size() << std::endl;
        return false;
    }

    for (std::size_t i = 0; i < A.size(); ++i)
    {
        if (A[i].size() != B[i].size()) {
            std::cout << "Size mismatch at row " << i
                      << ": cols " << A[i].size()
                      << " vs " << B[i].size() << std::endl;
            return false;
        }

        for (std::size_t j = 0; j < A[i].size(); ++j)
        {
            double diff = std::abs(A[i][j] - B[i][j]);
            if (diff > tol) {
                std::cout << "Mismatch at (" << i << "," << j << ")  "
                          << "cpu=" << B[i][j]
                          << " gpu=" << A[i][j]
                          << " diff=" << diff << std::endl;
                return false;
            }
        }
    }

    return true;
};



bool ok = are_identical(cholesky_gpu, cholesky_cpu);

if (ok)
    std::cout << "Cholesky results are IDENTICAL (within tolerance)\n";
else
    std::cout << "Cholesky results differ!\n";
                }
            }
        }
    }

    return 0;
}
