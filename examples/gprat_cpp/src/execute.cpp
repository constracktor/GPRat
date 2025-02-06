#include "gprat_c.hpp"
#include "utils_c.hpp"
#include <chrono>
#include <fstream>
#include <iostream>

int main(int argc, char *argv[])
{
    /////////////////////
    /////// configuration
    std::size_t START = 512;
    std::size_t END = 1024;
    std::size_t STEP = 2;
    std::size_t LOOP = 2;
    const std::size_t OPT_ITER = 1;

    int n_test = 1024;
    const std::size_t N_CORES = 4;
    const std::size_t n_tiles = 16;
    const std::size_t n_reg = 8;

    std::string train_path = "../../../data/data_1024/training_input.txt";
    std::string out_path = "../../../data/data_1024/training_output.txt";
    std::string test_path = "../../../data/data_1024/test_input.txt";

    for (std::size_t core = 2; core <= N_CORES; core = core * 2)
    {
        // Create new argc and argv to include the --hpx:threads argument
        std::vector<std::string> args(argv, argv + argc);
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
                auto start_total = std::chrono::high_resolution_clock::now();

                // Compute tile sizes and number of predict tiles
                int tile_size = utils::compute_train_tile_size(n_train, n_tiles);
                auto result = utils::compute_test_tiles(n_test, n_tiles, tile_size);
                /////////////////////
                ///// hyperparams
                std::vector<double> M = { 0.0, 0.0, 0.0 };
                gprat_hyper::Hyperparameters hpar = { 0.1, 0.9, 0.999, 1e-8, OPT_ITER, M };

                /////////////////////
                ////// data loading
                gprat::GP_data training_input(train_path, n_train, n_reg);
                gprat::GP_data training_output(out_path, n_train, n_reg);
                gprat::GP_data test_input(test_path, n_test, n_reg);

                /////////////////////
                ///// GP
                auto start_init = std::chrono::high_resolution_clock::now();
                std::vector<bool> trainable = { false, false, true };
                gprat::GP gp(
                    training_input.data, training_output.data, n_tiles, tile_size, 1.0, 1.0, 0.1, n_reg, trainable);
                auto end_init = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> init_time = end_init - start_init;

                // Initialize HPX with the new arguments, don't run hpx_main
                utils::start_hpx_runtime(new_argc, new_argv);

                // Measure the time taken to execute gp.cholesky();
                auto start_cholesky = std::chrono::high_resolution_clock::now();
                std::vector<std::vector<double>> choleksy = gp.cholesky();
                auto end_cholesky = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> cholesky_time = end_cholesky - start_cholesky;

                // Measure the time taken to execute gp.optimize(hpar);
                auto start_opt = std::chrono::high_resolution_clock::now();
                std::vector<double> losses = gp.optimize(hpar);
                auto end_opt = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> opt_time = end_opt - start_opt;

                auto start_pred_uncer = std::chrono::high_resolution_clock::now();
                std::vector<std::vector<double>> sum =
                    gp.predict_with_uncertainty(test_input.data, result.first, result.second);
                auto end_pred_uncer = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> pred_uncer_time = end_pred_uncer - start_pred_uncer;

                auto start_pred_full_cov = std::chrono::high_resolution_clock::now();
                std::vector<std::vector<double>> full =
                    gp.predict_with_full_cov(test_input.data, result.first, result.second);
                auto end_pred_full_cov = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> pred_full_cov_time = end_pred_full_cov - start_pred_full_cov;

                auto start_pred = std::chrono::high_resolution_clock::now();
                std::vector<double> pred = gp.predict(test_input.data, result.first, result.second);
                auto end_pred = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> pred_time = end_pred - start_pred;

                // Stop the HPX runtime
                utils::stop_hpx_runtime();

                auto end_total = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> total_time = end_total - start_total;

                // Save parameters and times to a .txt file with a header
                std::ofstream outfile("../output.csv", std::ios::app);  // Append mode
                if (outfile.tellp() == 0)
                {
                    // If file is empty, write the header
                    outfile << "Cores,N_train,N_test,N_tiles,N_regressor,Opt_iter,Total_time,Init_time,Cholesky_time,"
                               "Opt_time,Pred_Uncer_time,Pred_Full_time,Pred_time,N_loop\n";
                }
                outfile << core << "," << n_train << "," << n_test << "," << n_tiles << "," << n_reg << "," << OPT_ITER
                        << "," << total_time.count() << "," << init_time.count() << "," << cholesky_time.count() << ","
                        << opt_time.count() << "," << pred_uncer_time.count() << "," << pred_full_cov_time.count()
                        << "," << pred_time.count() << "," << l << "\n";
                outfile.close();
            }
        }
    }
    return 0;
}
