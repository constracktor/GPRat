#include "../../test/src/test_data.hpp"
#include "distributed_blas.hpp"
#include "distributed_cholesky.hpp"
#include "distributed_tile.hpp"
#include "cpu/gp_functions.hpp"
#include "gp_kernels.hpp"
#include "gprat_c.hpp"
#include "cpu/tiled_algorithms.hpp"
#include "utils_c.hpp"
#include <fstream>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/hpx_init_params.hpp>
#include <span>

// This is a standalone test, so including this directly is fine.
// Better than having the whole project depend on compiled Boost.Json!
#include <boost/json/src.hpp>

namespace gprat_hyper
{

template <class Archive>
inline void save_construct_data(Archive &ar, const SEKParams *v, const unsigned int)
{
    ar << v->lengthscale;
    ar << v->vertical_lengthscale;
    ar << v->noise_variance;
}

template <class Archive>
inline void load_construct_data(Archive &ar, SEKParams *v, const unsigned int)
{
    double lengthscale, vertical_lengthscale, noise_variance;
    ar >> lengthscale;
    ar >> vertical_lengthscale;
    ar >> noise_variance;

    // ::new(ptr) construct new object at given address
    hpx::construct_at(v, lengthscale, vertical_lengthscale, noise_variance);
}

template <typename Archive>
void serialize(Archive &ar, SEKParams &pt, const unsigned int)
{
    ar & pt.m_T & pt.w_T;
}

}  // namespace gprat_hyper

/////////////////////////////////////////////////////////
// Tile generation
double compute_covariance_function(std::size_t n_regressors,
                                   const gprat_hyper::SEKParams &sek_params,
                                   std::span<const double> i_input,
                                   std::span<const double> j_input)
{
    // k(z_i,z_j) = vertical_lengthscale * exp(-0.5 / lengthscale^2 * (z_i - z_j)^2)
    double distance = 0.0;
    for (std::size_t k = 0; k < n_regressors; k++)
    {
        const double z_ik_minus_z_jk = i_input[k] - j_input[k];
        distance += z_ik_minus_z_jk * z_ik_minus_z_jk;
    }

    return sek_params.vertical_lengthscale * exp(-0.5 / (sek_params.lengthscale * sek_params.lengthscale) * distance);
}

tile_data<double> make_covariance_tile(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const gprat_hyper::SEKParams &sek_params,
    std::span<const double> input)
{
    tile_data<double> tile(N * N);
    for (std::size_t i = 0; i < N; i++)
    {
        std::size_t i_global = N * row + i;
        for (std::size_t j = 0; j < N; j++)
        {
            std::size_t j_global = N * col + j;

            // compute covariance function
            auto covariance_function = compute_covariance_function(
                n_regressors, sek_params, input.subspan(i_global, n_regressors), input.subspan(j_global, n_regressors));
            if (i_global == j_global)
            {
                // noise variance on diagonal
                covariance_function += sek_params.noise_variance;
            }

            tile.data()[i * N + j] = covariance_function;
        }
    }
    return tile;
}

tile_handle make_covariance_tile_distributed(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const gprat_hyper::SEKParams &sek_params,
    std::span<const double> input)
{
    return tile_handle(hpx::find_here(), make_covariance_tile(row, col, N, n_regressors, sek_params, input));
}

HPX_PLAIN_ACTION(make_covariance_tile_distributed, make_covariance_tile_action)

template <>
struct plain_action_for<&make_covariance_tile>
{
    using action_type = make_covariance_tile_action;
    constexpr static std::string_view name = "gen_tile_covariance";
};

template <typename Scheduler = tiled_cholesky_scheduler_distributed<>>
void right_looking_cholesky_tiled(
    Scheduler &sched, typename Scheduler::tiled_matrix_handles &ft_tiles, std::size_t N, std::size_t n_tiles)
{
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        // POTRF: Compute Cholesky factor L
        ft_tiles[k * n_tiles + k] = dataflow<inplace::potrf>(sched.for_POTRF(k), ft_tiles[k * n_tiles + k], N);
        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // TRSM:  Solve X * L^T = A
            ft_tiles[m * n_tiles + k] = dataflow<inplace::trsm>(
                sched.for_TRSM(k, m),
                ft_tiles[k * n_tiles + k],
                ft_tiles[m * n_tiles + k],
                N,
                N,
                Blas_trans,
                Blas_right);
        }
        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // SYRK:  A = A - B * B^T
            ft_tiles[m * n_tiles + m] =
                dataflow<inplace::syrk>(sched.for_SYRK(m), ft_tiles[m * n_tiles + m], ft_tiles[m * n_tiles + k], N);
            for (std::size_t n = k + 1; n < m; n++)
            {
                // GEMM: C = C - A * B^T
                ft_tiles[m * n_tiles + n] = dataflow<inplace::gemm>(
                    sched.for_GEMM(k, m, n),
                    ft_tiles[m * n_tiles + k],
                    ft_tiles[n * n_tiles + k],
                    ft_tiles[m * n_tiles + n],
                    N,
                    N,
                    N,
                    Blas_no_trans,
                    Blas_trans);
            }
        }
    }
}

template <typename Scheduler = tiled_cholesky_scheduler_distributed<>>
std::vector<tile_data<double>>
cholesky_hpx(Scheduler &sched,
             std::span<const double> training_input,
             const gprat_hyper::SEKParams &sek_params,
             std::size_t n_tiles,
             std::size_t n_tile_size,
             std::size_t n_regressors)
{
    typename Scheduler::tiled_matrix_handles tiles(n_tiles * n_tiles);  // Tiled covariance matrix

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous assembly
    // std::vector<hpx::future<hpx::id_type>> tile_objs;
    // tile_objs.reserve(n_tiles * n_tiles);

    for (std::size_t row = 0; row < n_tiles; row++)
    {
        for (std::size_t col = 0; col <= row; col++)
        {
            tiles[row * n_tiles + col] = dataflow<make_covariance_tile>(
                sched.for_tile(row, col), row, col, n_tile_size, n_regressors, sek_params, training_input);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled(sched, tiles, n_tile_size, n_tiles);

    ///////////////////////////////////////////////////////////////////////////
    // Synchronize
    std::vector<tile_data<double>> result(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            result[i * n_tiles + j] = tiles[i * n_tiles + j].get_data().get();
        }
    }
    return result;
}

gprat_results load_test_data_results(const std::string &filename)
{
    std::ifstream ifs(filename);
    if (!ifs.fail())
    {
        using iterator_type = std::istreambuf_iterator<char>;
        const std::string content(iterator_type{ ifs }, iterator_type{});
        return boost::json::value_to<gprat_results>(boost::json::parse(content));
    }
    throw std::runtime_error("Failed to load " + filename);
}

void check_data(const std::vector<std::vector<double>> &expected, const std::vector<tile_data<double>> &actual)
{
    if (expected.size() != actual.size())
    {
        throw std::runtime_error("expected.size() != actual.size()");
    }
    if (expected[0].size() != actual[0].size())
    {
        throw std::runtime_error("expected[0].size() != actual[0].size()");
    }

    constexpr double margin = 0.00001;
    for (std::size_t i = 0; i < expected.size(); i++)
    {
        const std::span<const double> actual_data = actual[i];
        for (std::size_t j = 0; j < expected[i].size(); j++)
        {
            const auto &expected_value = expected[i][j];
            const auto &actual_value = actual_data[j];

            const bool is_in_range =
                (expected_value + margin >= actual_value) && (actual_value + margin >= expected_value);
            if (!is_in_range)
            {
                std::cerr << "MISMATCH at " << i << " " << j << " " << expected_value << " !~= " << actual_value;
            }
        }
    }
}

void run(hpx::program_options::variables_map &vm)
{
    /////////////////////
    /////// configuration
    std::size_t START = vm["start"].as<std::size_t>();
    std::size_t END = vm["end"].as<std::size_t>();
    std::size_t STEP = vm["step"].as<std::size_t>();
    std::size_t LOOP = vm["loop"].as<std::size_t>();
    const int OPT_ITER = vm["opt_iter"].as<int>();

    int n_test = 1024;
    const std::size_t n_tiles = vm["tiles"].as<std::size_t>();
    const std::size_t n_reg = vm["regressors"].as<std::size_t>();

    const auto &train_path = vm["train_x_path"].as<std::string>();
    const auto &out_path = vm["train_y_path"].as<std::string>();
    const auto &test_path = vm["test_path"].as<std::string>();
    //const auto test_results = load_test_data_results(vm["test_results_path"].as<std::string>());

    tiled_cholesky_scheduler_distributed scheduler;

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
            gprat_hyper::AdamParams hpar = { 0.1, 0.9, 0.999, 1e-8, OPT_ITER };

            /////////////////////
            ////// data loading
            gprat::GP_data training_input(train_path, n_train, n_reg);
            gprat::GP_data training_output(out_path, n_train, n_reg);
            gprat::GP_data test_input(test_path, n_test, n_reg);

            /////////////////////
            ///// GP
            auto start_init = std::chrono::high_resolution_clock::now();
            std::vector<bool> trainable = { true, true, true };
            gprat::GP gp(
                training_input.data, training_output.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, trainable);
            auto end_init = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> init_time = end_init - start_init;

            // Measure the time taken to execute gp.cholesky();
            auto start_cholesky = std::chrono::high_resolution_clock::now();

            const auto cholesky =
                cholesky_hpx(scheduler, training_input.data, { 1.0, 1.0, 0.1 }, n_tiles, tile_size, n_reg);

            auto end_cholesky = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> cholesky_time = end_cholesky - start_cholesky;

            auto end_total = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> total_time = end_total - start_total;

            // Save parameters and times to a .txt file with a header
            std::ofstream outfile("output-distributed.csv", std::ios::app);  // Append mode
            if (outfile.tellp() == 0)
            {
                // If file is empty, write the header
                outfile << "Cores,N_train,N_test,N_tiles,N_regressor,Opt_iter,Total_time,Init_time,Cholesky_time,"
                           "Opt_time,Pred_Uncer_time,Pred_Full_time,Pred_time,N_loop\n";
            }
            outfile << hpx::get_locality_id() << "," << n_train << "," << n_test << "," << n_tiles << "," << n_reg
                    << "," << OPT_ITER << "," << total_time.count() << "," << init_time.count() << ","
                    << cholesky_time.count() << "," << 0 << "," << 0 << "," << 0 << "," << 0 << "," << l << "\n";
            outfile.close();

            //check_data(test_results.choleksy, cholesky);
        }
    }
    std::cerr << "DONE!" << std::endl;
}

int hpx_main(hpx::program_options::variables_map &vm)
{
    try
    {
        run(vm);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
    return hpx::finalize();
}

int main(int argc, char *argv[])
{
    namespace po = hpx::program_options;
    po::options_description desc("Allowed options");
#define BASE_DIR "../../../../"
    // clang-format off
    desc.add_options()
        ("help", "produce help message")
        ("train_x_path", po::value<std::string>()->default_value(BASE_DIR "data/data_1024/training_input.txt"), "training data (x)")
        ("train_y_path", po::value<std::string>()->default_value(BASE_DIR "data/data_1024/training_output.txt"), "training data (y)")
        ("test_path", po::value<std::string>()->default_value(BASE_DIR "data/data_1024/test_input.txt"), "test data")
        //("test_results_path", po::value<std::string>()->default_value(BASE_DIR "data/data_1024/output.json"), "test data results")
        ("tiles", po::value<std::size_t>()->default_value(16), "tiles per dimension")
        ("regressors", po::value<std::size_t>()->default_value(8), "num regressors")
        ("start", po::value<std::size_t>()->default_value(128), "Starting number of training samples")
        ("end", po::value<std::size_t>()->default_value(128), "End number of training samples")
        ("step", po::value<std::size_t>()->default_value(2), "Increment of training samples")
        ("loop", po::value<std::size_t>()->default_value(1), "Number of iterations to be performed for each number of training samples")
        ("opt_iter", po::value<int>()->default_value(3), "Number of optimization iterations*/")
    ;
// clang-format on
#undef BASE_DIR

    hpx::init_params init_args;
    init_args.desc_cmdline = desc;
    // If example requires to run hpx_main on all localities
    // std::vector<std::string> const cfg = {"hpx.run_hpx_main!=1"};
    // init_args.cfg = cfg;
    // Run HPX main
    return hpx::init(argc, argv, init_args);
}
