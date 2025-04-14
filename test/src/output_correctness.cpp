#include "gprat_c.hpp"
#include "utils_c.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// This is a standalone test, so including this directly is fine.
// Better than having the whole project depend on compiled Boost.Json!
#include <boost/json/src.hpp>

// std headers last
#include <fstream>
#include <string>
#include <string_view>

// Struct containing all results we'd like to compare
struct gprat_results
{
    std::vector<std::vector<double>> choleksy;
    std::vector<double> losses;
    std::vector<std::vector<double>> sum;
    std::vector<std::vector<double>> full;
    std::vector<double> pred;
    std::vector<std::vector<double>> sum_no_optimize;
    std::vector<std::vector<double>> full_no_optimize;
    std::vector<double> pred_no_optimize;
};

// The following two functions are for JSON (de-)serialization
void tag_invoke(boost::json::value_from_tag, boost::json::value &jv, const gprat_results &results)
{
    jv = {
        { "choleksy", boost::json::value_from(results.choleksy) },
        { "losses", boost::json::value_from(results.losses) },
        { "sum", boost::json::value_from(results.sum) },
        { "full", boost::json::value_from(results.full) },
        { "pred", boost::json::value_from(results.pred) },
        { "sum_no_optimize", boost::json::value_from(results.sum_no_optimize) },
        { "full_no_optimize", boost::json::value_from(results.full_no_optimize) },
        { "pred_no_optimize", boost::json::value_from(results.pred_no_optimize) },
    };
}

// This helper function deduces the type and assigns the value with the matching key
template <typename T>
inline void extract(const boost::json::object &obj, T &t, std::string_view key)
{
    t = boost::json::value_to<T>(obj.at(key));
}

gprat_results tag_invoke(boost::json::value_to_tag<gprat_results>, const boost::json::value &jv)
{
    gprat_results results;
    const auto &obj = jv.as_object();
    extract(obj, results.choleksy, "choleksy");
    extract(obj, results.losses, "losses");
    extract(obj, results.sum, "sum");
    extract(obj, results.full, "full");
    extract(obj, results.pred, "pred");
    extract(obj, results.sum_no_optimize, "sum_no_optimize");
    extract(obj, results.full_no_optimize, "full_no_optimize");
    extract(obj, results.pred_no_optimize, "pred_no_optimize");
    return results;
}

// This logic is basically equivalent to the GPRat C++ example (for now).
gprat_results run_on_data_cpu(const std::string &train_path, const std::string &out_path, const std::string &test_path)
{
    // configuration
    const std::size_t OPT_ITER = 3;
    const std::size_t n_test = 128;
    const std::size_t n_train = 128;
    const std::size_t n_tiles = 16;
    const std::size_t n_reg = 8;

    // Compute tile sizes and number of predict tiles
    const int tile_size = utils::compute_train_tile_size(n_train, n_tiles);
    const auto test_tiles = utils::compute_test_tiles(n_test, n_tiles, tile_size);

    // hyperparams
    gprat_hyper::AdamParams hpar = { 0.1, 0.9, 0.999, 1e-8, OPT_ITER };

    // data loading
    gprat::GP_data training_input(train_path, n_train, n_reg);
    gprat::GP_data training_output(out_path, n_train, n_reg);
    gprat::GP_data test_input(test_path, n_test, n_reg);

    // GP
    const std::vector<bool> trainable = { true, true, true };
    gprat::GP gp_cpu(
        training_input.data, training_output.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, trainable);

    // Initialize HPX with no arguments, don't run hpx_main
    utils::start_hpx_runtime(0, nullptr);

    gprat_results results_cpu;

    results_cpu.choleksy = gp_cpu.cholesky();

    results_cpu.losses = gp_cpu.optimize(hpar);

    results_cpu.sum = gp_cpu.predict_with_uncertainty(test_input.data, test_tiles.first, test_tiles.second);

    results_cpu.full = gp_cpu.predict_with_full_cov(test_input.data, test_tiles.first, test_tiles.second);

    results_cpu.pred = gp_cpu.predict(test_input.data, test_tiles.first, test_tiles.second);

    // Stop the HPX runtime
    utils::stop_hpx_runtime();

    return results_cpu;
}

// Add this helper function
gprat_results run_on_data_gpu(const std::string &train_path, const std::string &out_path, const std::string &test_path)
{
    const std::size_t n_test = 128;
    const std::size_t n_train = 128;
    const std::size_t n_tiles = 16;
    const std::size_t n_reg = 8;
    const int gpu_id = 0;
    const int n_streams = 1;

    const int tile_size = utils::compute_train_tile_size(n_train, n_tiles);
    const auto test_tiles = utils::compute_test_tiles(n_test, n_tiles, tile_size);

    gprat::GP_data training_input(train_path, n_train, n_reg);
    gprat::GP_data training_output(out_path, n_train, n_reg);
    gprat::GP_data test_input(test_path, n_test, n_reg);

    const std::vector<bool> trainable = { true, true, true };
    gprat::GP gp_gpu(
        training_input.data,
        training_output.data,
        n_tiles,
        tile_size,
        n_reg,
        { 1.0, 1.0, 0.1 },
        trainable,
        gpu_id,
        n_streams);

    utils::start_hpx_runtime(0, nullptr);

    gprat_results results_gpu;
    results_gpu.choleksy = gp_gpu.cholesky();
    // NOTE: optimize and optimize_step are currently not implemented for GPU
    results_gpu.sum_no_optimize = gp_gpu.predict_with_uncertainty(test_input.data, test_tiles.first, test_tiles.second);
    results_gpu.full_no_optimize = gp_gpu.predict_with_full_cov(test_input.data, test_tiles.first, test_tiles.second);
    results_gpu.pred_no_optimize = gp_gpu.predict(test_input.data, test_tiles.first, test_tiles.second);

    utils::stop_hpx_runtime();

    return results_gpu;
}

bool load_or_create_expected_results(
    const std::string &filename, const gprat_results &fallback_results, gprat_results &results)
{
    // First try to read our expected results file
    {
        std::ifstream ifs(filename);
        if (!ifs.fail())
        {
            using iterator_type = std::istreambuf_iterator<char>;
            const std::string content(iterator_type{ ifs }, iterator_type{});
            results = boost::json::value_to<gprat_results>(boost::json::parse(content));
            return true;
        }
    }

    // If that doesn't work, just write out the results we want
    std::ofstream fout(filename);
    fout << boost::json::value_from(fallback_results);
    return false;
}

std::string get_root_directory()
{
    const char *env_root = std::getenv("GPRAT_ROOT");
    if (env_root)
    {
        return env_root;
    }
    return "../data";
}

TEST_CASE("GP CPU results match known-good values", "[integration][cpu]")
{
    const std::string root = get_root_directory();
    const auto results = run_on_data_cpu(root + "/data_1024/training_input.txt",
                                         root + "/data_1024/training_output.txt",
                                         root + "/data_1024/test_input.txt");

    gprat_results expected_results;
    if (!load_or_create_expected_results(root + "/data_1024/output.json", results, expected_results))
    {
        std::cerr << "No previous results to compare to. The current results have been saved instead!" << std::endl;
        return;
    }

    // First we check for equal size
    REQUIRE(results.choleksy.size() == expected_results.choleksy.size());
    REQUIRE(results.losses.size() == expected_results.losses.size());
    REQUIRE(results.sum.size() == expected_results.sum.size());
    REQUIRE(results.sum[0].size() == expected_results.sum[0].size());
    REQUIRE(results.full.size() == expected_results.full.size());
    REQUIRE(results.full[0].size() == expected_results.full[0].size());
    REQUIRE(results.pred.size() == expected_results.pred.size());

    // Now we can compare content
    // The default-constructed WithinRel() matcher has a tolerance of epsilon * 100
    // see:
    // https://github.com/catchorg/Catch2/blob/914aeecfe23b1e16af6ea675a4fb5dbd5a5b8d0a/docs/comparing-floating-point-numbers.md#withinrel
    using Catch::Matchers::WithinRel;
    double eps = std::numeric_limits<double>::epsilon() * 1'000'000;
    for (std::size_t i = 0, n = results.choleksy.size(); i != n; ++i)
    {
        for (std::size_t j = 0, m = results.choleksy[i].size(); j != m; ++j)
        {
            INFO("CPU choleksy " << i << " " << j);
            REQUIRE_THAT(results.choleksy[i][j], WithinRel(expected_results.choleksy[i][j], eps));
        }
    }
    for (std::size_t i = 0, n = results.losses.size(); i != n; ++i)
    {
        INFO("CPU losses " << i);
        REQUIRE_THAT(results.losses[i], WithinRel(expected_results.losses[i], eps));
    }

    for (std::size_t i = 0, n = results.sum.size(); i != n; ++i)
    {
        for (std::size_t j = 0, m = results.sum[i].size(); j != m; ++j)
        {
            INFO("CPU sum " << i << " " << j);
            REQUIRE_THAT(results.sum[i][j], WithinRel(expected_results.sum[i][j], eps));
        }
    }

    for (std::size_t i = 0, n = results.full.size(); i != n; ++i)
    {
        for (std::size_t j = 0, m = results.full[i].size(); j != m; ++j)
        {
            INFO("CPU full " << i << " " << j);
            REQUIRE_THAT(results.full[i][j], WithinRel(expected_results.full[i][j], eps));
        }
    }

    for (std::size_t i = 0, n = results.pred.size(); i != n; ++i)
    {
        INFO("CPU pred " << i);
        REQUIRE_THAT(results.pred[i], WithinRel(expected_results.pred[i], eps));
    }
}

// Test for GPU
// NOTE: using higher tolerance than for CPU
TEST_CASE("GP GPU results match known-good values (no loss)", "[integration][gpu]")
{
    if (!utils::compiled_with_cuda())
    {
        WARN("CUDA not available — skipping GPU test.");
        return;
    }

    const std::string root = get_root_directory();
    const std::string train = root + "/data_1024/training_input.txt";
    const std::string out = root + "/data_1024/training_output.txt";
    const std::string test = root + "/data_1024/test_input.txt";

    const gprat_results results = run_on_data_gpu(train, out, test);

    gprat_results expected_results;
    const std::string ref_file = root + "/data_1024/output.json";

    if (!load_or_create_expected_results(ref_file, results, expected_results))
    {
        std::cerr << "No previous results to compare to. The current results have been saved instead!" << std::endl;
        return;
    }

    REQUIRE(results.choleksy.size() == expected_results.choleksy.size());
    REQUIRE(results.sum_no_optimize.size() == expected_results.sum_no_optimize.size());
    REQUIRE(results.sum_no_optimize[0].size() == expected_results.sum_no_optimize[0].size());
    REQUIRE(results.full_no_optimize.size() == expected_results.full_no_optimize.size());
    REQUIRE(results.full_no_optimize[0].size() == expected_results.full_no_optimize[0].size());
    REQUIRE(results.pred_no_optimize.size() == expected_results.pred_no_optimize.size());

    using Catch::Matchers::WithinRel;
    double eps = std::numeric_limits<double>::epsilon() * 1'000'000;
    for (std::size_t i = 0, n = results.choleksy.size(); i != n; ++i)
    {
        for (std::size_t j = 0, m = results.choleksy[i].size(); j != m; ++j)
        {
            INFO("GPU choleksy " << i << " " << j);
            REQUIRE_THAT(results.choleksy[i][j], WithinRel(expected_results.choleksy[i][j], eps));
        }
    }

    for (std::size_t i = 0, n = results.sum_no_optimize.size(); i != n; ++i)
    {
        for (std::size_t j = 0, m = results.sum_no_optimize[i].size(); j != m; ++j)
        {
            INFO("GPU sum_no_optimize " << i << " " << j);
            REQUIRE_THAT(results.sum_no_optimize[i][j], WithinRel(expected_results.sum_no_optimize[i][j], eps));
        }
    }

    for (std::size_t i = 0, n = results.full_no_optimize.size(); i != n; ++i)
    {
        for (std::size_t j = 0, m = results.full_no_optimize[i].size(); j != m; ++j)
        {
            INFO("GPU full " << i << " " << j);
            REQUIRE_THAT(results.full_no_optimize[i][j], WithinRel(expected_results.full_no_optimize[i][j], eps));
        }
    }

    for (std::size_t i = 0, n = results.pred_no_optimize.size(); i != n; ++i)
    {
        INFO("GPU pred_no_optimize " << i);
        REQUIRE_THAT(results.pred_no_optimize[i], WithinRel(expected_results.pred_no_optimize[i], eps));
    }
}
