#include "gprat/gprat.hpp"
#include "gprat/utils.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>

namespace gprat::example
{
struct Runtimes
{
    std::chrono::duration<double> init;
    std::chrono::duration<double> cholesky;
    std::chrono::duration<double> opt;
    std::chrono::duration<double> pred_uncer;
    std::chrono::duration<double> pred_full_cov;
    std::chrono::duration<double> pred;
};

struct GpratSettings
{
    std::string train_in_file;
    std::string train_out_file;
    std::string test_in_file;

    int train_size_start;
    int train_size_end;
    int train_size_step;

    int start_cores;
    int end_cores;

    int test_size;
    bool scale_test_with_train;

    int n_reg;
    int opt_iter;
    int loop;
    int n_tiles_start;
    int n_tiles_end;
    int step_tiles;

    bool cholesky;
};

template <typename T>
inline void extract(const boost::json::object &obj, T &t, std::string_view key)
{
    t = boost::json::value_to<T>(obj.at(key));
}

GpratSettings tag_invoke(boost::json::value_to_tag<GpratSettings>, const boost::json::value &jv)
{
    GpratSettings settings;
    const auto &obj = jv.as_object();
    extract(obj, settings.train_in_file, "TRAIN_IN_FILE");
    extract(obj, settings.train_out_file, "TRAIN_OUT_FILE");
    extract(obj, settings.test_in_file, "TEST_IN_FILE");
    extract(obj, settings.train_size_start, "TRAIN_SIZE_START");
    extract(obj, settings.train_size_end, "TRAIN_SIZE_END");
    extract(obj, settings.train_size_step, "STEP");
    extract(obj, settings.test_size, "TEST_SIZE");
    extract(obj, settings.scale_test_with_train, "SCALE_TEST_WITH_TRAIN");
    extract(obj, settings.n_reg, "N_REG");
    extract(obj, settings.opt_iter, "OPT_ITER");
    extract(obj, settings.loop, "LOOP");
    extract(obj, settings.start_cores, "START_CORES");
    extract(obj, settings.end_cores, "END_CORES");
    extract(obj, settings.n_tiles_start, "N_TILES_START");
    extract(obj, settings.n_tiles_end, "N_TILES_END");
    extract(obj, settings.step_tiles, "STEP_TILES");
    extract(obj, settings.cholesky, "CHOLESKY");

    return settings;
}

// GPU test settings
constexpr int device_id = 0;
constexpr int n_units = 1;

// Save parameters and times to a .txt file with a header
void append_to_output_file(
    std::string &target,
    int &core,
    int &n_tiles,
    int &n_train,
    int &n_test,
    int &n_reg,
    int &n_opt_iter,
    std::chrono::duration<double> &total_time,
    Runtimes &runtimes,
    int &l)
{
    const std::filesystem::path output_path = std::filesystem::path(GPRAT_CPP_CONFIG_PATH).parent_path() / "output.csv";
    std::ofstream outfile(output_path, std::ios::app);  // Append mode
    if (outfile.tellp() == 0)
    {
        // If file is empty, write the header
        outfile << "Target," << "Cores," << "N_tiles," << "N_train," << "N_test," << "N_regressor," << "Opt_iter,"
                << "Total_time," << "Init_time," << "Cholesky_time," << "Opt_Time," << "Predict_time,"
                << "Pred_uncer_time," << "Pred_Full_time," << "N_loop\n";
    }
    outfile << target << "," << core << "," << n_tiles << "," << n_train << "," << n_test << "," << n_reg << ","
            << n_opt_iter << "," << total_time.count() << "," << runtimes.init.count() << ","
            << runtimes.cholesky.count() << "," << runtimes.opt.count() << "," << runtimes.pred.count() << ","
            << runtimes.pred_uncer.count() << "," << runtimes.pred_full_cov.count() << "," << l << "\n";
    outfile.close();
}

void example_cpu(Runtimes &runtimes,
                 std::pair<int, int> &result,
                 gprat::GP_data &training_input,
                 gprat::GP_data &training_output,
                 gprat::GP_data &test_input,
                 const int n_tiles,
                 const int tile_size,
                 std::vector<bool> trainable,
                 GpratSettings &settings)
{
    gprat_hyper::AdamParams hpar = { 0.1, 0.9, 0.999, 1e-8, settings.opt_iter };

    auto start_init = std::chrono::high_resolution_clock::now();
    gprat::GP gp_cpu(
        training_input.data, training_output.data, n_tiles, tile_size, settings.n_reg, { 1.0, 1.0, 0.1 }, trainable);
    auto end_init = std::chrono::high_resolution_clock::now();
    runtimes.init = end_init - start_init;

    auto start_cholesky = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> cholesky_cpu;
    if (settings.cholesky)
    {
        cholesky_cpu = gp_cpu.cholesky();
    }
    auto end_cholesky = std::chrono::high_resolution_clock::now();
    runtimes.cholesky = settings.cholesky ? end_cholesky - start_cholesky : std::chrono::seconds(-1);

    auto start_opt = std::chrono::high_resolution_clock::now();
    std::vector<double> losses;
    if (!settings.cholesky)
    {
        losses = gp_cpu.optimize(hpar);
    }
    auto end_opt = std::chrono::high_resolution_clock::now();
    runtimes.opt = settings.cholesky ? std::chrono::seconds(-1) : end_opt - start_opt;

    auto start_pred_uncer = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> sum_cpu;
    if (!settings.cholesky)
    {
        sum_cpu = gp_cpu.predict_with_uncertainty(test_input.data, result.first, result.second);
    }
    auto end_pred_uncer = std::chrono::high_resolution_clock::now();
    runtimes.pred_uncer = settings.cholesky ? std::chrono::seconds(-1) : end_pred_uncer - start_pred_uncer;

    auto start_pred_full_cov = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> full_cpu;
    if (!settings.cholesky)
    {
        full_cpu = gp_cpu.predict_with_full_cov(test_input.data, result.first, result.second);
    }
    auto end_pred_full_cov = std::chrono::high_resolution_clock::now();
    runtimes.pred_full_cov = settings.cholesky ? std::chrono::seconds(-1) : end_pred_full_cov - start_pred_full_cov;

    auto start_pred = std::chrono::high_resolution_clock::now();
    std::vector<double> pred_cpu;
    if (!settings.cholesky)
    {
        pred_cpu = gp_cpu.predict(test_input.data, result.first, result.second);
    }
    auto end_pred = std::chrono::high_resolution_clock::now();
    runtimes.pred = settings.cholesky ? std::chrono::seconds(-1) : end_pred - start_pred;
}

void example_gpu(Runtimes &runtimes,
                 std::pair<int, int> &result,
                 gprat::GP_data &training_input,
                 gprat::GP_data &training_output,
                 gprat::GP_data &test_input,
                 const int n_tiles,
                 const int tile_size,
                 std::vector<bool> trainable,
                 int &n_reg,
                 bool &cholesky)
{
    auto start_init = std::chrono::high_resolution_clock::now();
    gprat::GP gp_gpu(
        training_input.data,
        training_output.data,
        n_tiles,
        tile_size,
        n_reg,
        std::vector<double>{ 1.0, 1.0, 0.1 },
        trainable,
        device_id,
        n_units);

    auto end_init = std::chrono::high_resolution_clock::now();
    runtimes.init = end_init - start_init;

    auto start_cholesky = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> cholesky_gpu;
    if (cholesky)
    {
        cholesky_gpu = gp_gpu.cholesky();
    }
    auto end_cholesky = std::chrono::high_resolution_clock::now();
    runtimes.cholesky = cholesky ? end_cholesky - start_cholesky : std::chrono::seconds(-1);

    // NOTE: optimization is not implemented for GPU
    runtimes.opt = std::chrono::seconds(-1);

    auto start_pred_uncer = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> sum_gpu;
    if (!cholesky)
    {
        sum_gpu = gp_gpu.predict_with_uncertainty(test_input.data, result.first, result.second);
    }
    auto end_pred_uncer = std::chrono::high_resolution_clock::now();
    runtimes.pred_uncer = cholesky ? std::chrono::seconds(-1) : end_pred_uncer - start_pred_uncer;

    auto start_pred_full_cov = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> full_gpu;
    if (!cholesky)
    {
        full_gpu = gp_gpu.predict_with_full_cov(test_input.data, result.first, result.second);
    }
    auto end_pred_full_cov = std::chrono::high_resolution_clock::now();
    runtimes.pred_full_cov = cholesky ? std::chrono::seconds(-1) : end_pred_full_cov - start_pred_full_cov;

    auto start_pred = std::chrono::high_resolution_clock::now();
    std::vector<double> pred_gpu;
    if (!cholesky)
    {
        pred_gpu = gp_gpu.predict(test_input.data, result.first, result.second);
    }
    auto end_pred = std::chrono::high_resolution_clock::now();
    runtimes.pred = cholesky ? std::chrono::seconds(-1) : end_pred - start_pred;
}

}  // namespace gprat::example

int main(int argc, char *argv[])
{
    namespace po = hpx::program_options;
    po::options_description desc("Allowed options");
    // clang-format off
    desc.add_options()
        ("help", "produce help message")
        ("train_x_path", po::value<std::string>()->default_value("../../../data/data_1024/training_input.txt"), "training data (x)")
        ("train_y_path", po::value<std::string>()->default_value("../../../data/data_1024/training_output.txt"), "training data (y)")
        ("test_path", po::value<std::string>()->default_value("../../../data/data_1024/test_input.txt"), "test data")
        ("tiles", po::value<std::size_t>()->default_value(16), "tiles per dimension")
        ("regressors", po::value<std::size_t>()->default_value(8), "num regressors")
        ("start-cores", po::value<std::size_t>()->default_value(2), "num CPUs to start with")
        ("end-cores", po::value<std::size_t>()->default_value(4), "num CPUs to end with")
        ("start", po::value<std::size_t>()->default_value(512), "Starting number of training samples")
        ("end", po::value<std::size_t>()->default_value(1024), "End number of training samples")
        ("step", po::value<std::size_t>()->default_value(2), "Increment of training samples")
        ("loop", po::value<std::size_t>()->default_value(2), "Number of iterations to be performed for each number of training samples")
        ("opt_iter", po::value<std::size_t>()->default_value(1), "Number of optimization iterations*/")
    ;
    // clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.contains("help"))
    {
        std::cout << desc << "\n";
        return 1;
    }

    /////////////////////
    /////// configuration
    std::size_t START = vm["start"].as<std::size_t>();
    std::size_t END = vm["end"].as<std::size_t>();
    std::size_t STEP = vm["step"].as<std::size_t>();
    std::size_t LOOP = vm["loop"].as<std::size_t>();
    const std::size_t OPT_ITER = vm["opt_iter"].as<std::size_t>();

    const std::size_t n_test = START;
    const std::size_t N_CORES = vm["end-cores"].as<std::size_t>();
    const std::size_t n_tiles = vm["tiles"].as<std::size_t>();
    const std::size_t n_reg = vm["regressors"].as<std::size_t>();

    std::string train_path = vm["train_x_path"].as<std::string>();
    std::string out_path = vm["train_y_path"].as<std::string>();
    std::string test_path = vm["test_path"].as<std::string>();

    bool use_gpu =
        gprat::compiled_with_cuda() && gprat::gpu_count() > 0 && argc > 1 && std::strcmp(argv[1], "--use_gpu") == 0;

    for (std::size_t core = vm["start-cores"].as<std::size_t>(); core <= N_CORES; core = core * 2)
    {
        // Create new argc and argv to include the --hpx:threads argument
        std::vector<std::string> args(argv, argv + argc);
        args.erase(args.begin() + argc - 1);
        args.push_back("--hpx:threads=" + std::to_string(core));

        // Convert the arguments to char* array
        std::vector<char *> cstr_args;
        for (auto &arg : args)
        {
            cstr_args.push_back(const_cast<char *>(arg.c_str()));
        }

        int new_argc = static_cast<int>(cstr_args.size());
        char **new_argv = cstr_args.data();

        utils::start_hpx_runtime(new_argc, new_argv);

        // Loop over tiles
        for (int n_tiles = settings.n_tiles_start; n_tiles <= settings.n_tiles_end; n_tiles *= settings.step_tiles)
        {
            const auto n_train = start;
            for (std::size_t l = 0; l < LOOP; l++)
            {
                // Compute tile sizes and number of predict tiles
                const auto tile_size = gprat::compute_train_tile_size(n_train, n_tiles);
                const auto result = gprat::compute_test_tiles(n_test, n_tiles, tile_size);
                /////////////////////
                ///// hyperparams
                gprat::AdamParams hpar = { 0.1, 0.9, 0.999, 1e-8, OPT_ITER };

                // Loop over repetitions
                for (int l = 0; l < settings.loop; l++)
                {
                    int tile_size = utils::compute_train_tile_size(train_size, n_tiles);
                    auto result = utils::compute_test_tiles(n_test, n_tiles, tile_size);

                    gprat::GP_data training_input(settings.train_in_file, train_size, settings.n_reg);
                    gprat::GP_data training_output(settings.train_out_file, train_size, settings.n_reg);
                    gprat::GP_data test_input(settings.test_in_file, n_test, settings.n_reg);

                    // Initialize HPX with the new arguments, don't run hpx_main
                    gprat::start_hpx_runtime(new_argc, new_argv);

                    // Measure the time taken to execute gp.cholesky();
                    auto start_cholesky = std::chrono::high_resolution_clock::now();
                    const auto choleksy_cpu = gp_cpu.cholesky();
                    auto end_cholesky = std::chrono::high_resolution_clock::now();
                    cholesky_time = end_cholesky - start_cholesky;

                    // Measure the time taken to execute gp.optimize(hpar);
                    auto start_opt = std::chrono::high_resolution_clock::now();
                    const auto losses = gp_cpu.optimize(hpar);
                    auto end_opt = std::chrono::high_resolution_clock::now();
                    opt_time = end_opt - start_opt;

                    auto start_pred_uncer = std::chrono::high_resolution_clock::now();
                    const auto sum_cpu = gp_cpu.predict_with_uncertainty(test_input.data, result.first, result.second);
                    auto end_pred_uncer = std::chrono::high_resolution_clock::now();
                    pred_uncer_time = end_pred_uncer - start_pred_uncer;

                    auto start_pred_full_cov = std::chrono::high_resolution_clock::now();
                    const auto full_cpu = gp_cpu.predict_with_full_cov(test_input.data, result.first, result.second);
                    auto end_pred_full_cov = std::chrono::high_resolution_clock::now();
                    pred_full_cov_time = end_pred_full_cov - start_pred_full_cov;

                    auto start_pred = std::chrono::high_resolution_clock::now();
                    const auto pred_cpu = gp_cpu.predict(test_input.data, result.first, result.second);
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
                    gprat::start_hpx_runtime(new_argc, new_argv);

                    auto start_cholesky = std::chrono::high_resolution_clock::now();
                    const auto choleksy_gpu = gp_gpu.cholesky();
                    auto end_cholesky = std::chrono::high_resolution_clock::now();
                    cholesky_time = end_cholesky - start_cholesky;

                    // NOTE: optimization is not implemented for GPU
                    opt_time = std::chrono::seconds(-1);

                    auto start_pred_uncer = std::chrono::high_resolution_clock::now();
                    const auto sum_gpu = gp_gpu.predict_with_uncertainty(test_input.data, result.first, result.second);
                    auto end_pred_uncer = std::chrono::high_resolution_clock::now();
                    pred_uncer_time = end_pred_uncer - start_pred_uncer;

                    auto start_pred_full_cov = std::chrono::high_resolution_clock::now();
                    const auto full_gpu = gp_gpu.predict_with_full_cov(test_input.data, result.first, result.second);
                    auto end_pred_full_cov = std::chrono::high_resolution_clock::now();
                    pred_full_cov_time = end_pred_full_cov - start_pred_full_cov;

                    auto start_pred = std::chrono::high_resolution_clock::now();
                    const auto pred_gpu = gp_gpu.predict(test_input.data, result.first, result.second);
                    auto end_pred = std::chrono::high_resolution_clock::now();
                    pred_time = end_pred - start_pred;
                }

                // Stop the HPX runtime
                gprat::stop_hpx_runtime();

                auto end_total = std::chrono::high_resolution_clock::now();
                auto total_time = end_total - start_total;

                // Save parameters and times to a .txt file with a header
                std::ofstream outfile("output.csv", std::ios::app);  // Append mode
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

        utils::stop_hpx_runtime();
    }

    return 0;
}
