#include "gprat/gprat.hpp"
#include "gprat/utils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
using Catch::Matchers::WithinRel;

// Boost
#include <boost/json/src.hpp>

// Standard library
#include <fstream>
#include <string>
#include <string_view>

namespace gprat::test
{

// Parameters /////////////////////////////////////////////////////////////////////////////////////

// Global test settings
constexpr std::size_t n_test = 128;
constexpr std::size_t n_train = 128;
constexpr std::size_t n_tiles = 4;
constexpr std::size_t n_reg = 8;

// CPU test settings
constexpr std::size_t OPT_ITER = 3;

// CUDA and SYCL test settings
constexpr int gpu_id = 0;
constexpr int n_units = 4;

// GPRat results structure ////////////////////////////////////////////////////////////////////////

/**
 * @brief   Struct containing all results we would like to compare
 */
struct GpratResults
{
    std::vector<std::vector<double>> cholesky;
    std::vector<double> losses;
    std::vector<std::vector<double>> sum;
    std::vector<std::vector<double>> full;
    std::vector<double> pred;
};

// JSON (de-)serialization ////////////////////////////////////////////////////////////////////////

/**
 * @brief Creates data for a JSON object from an existing results structure.
 *
 * @param jv the values held by the JSON file
 * @param results the GpratResults object that from which the values are read
 */
void tag_invoke(boost::json::value_from_tag, boost::json::value &jv, const GpratResults &results)
{
    jv = { { "cholesky", boost::json::value_from(results.cholesky) },
           { "losses", boost::json::value_from(results.losses) },
           { "sum", boost::json::value_from(results.sum) },
           { "full", boost::json::value_from(results.full) },
           { "pred", boost::json::value_from(results.pred) } };
}

/**
 * @brief Searches a specified JSON object for the property `key`, converts its type to `T`, and
 *        stores its value in `t`.
 *
 * @tparam T the target type of the read value
 *
 * @param obj the JSON object that is read from
 * @param t the variable to store the value in
 * @param key the key to search for in the JSON file
 */
template <typename T>
inline void extract(const boost::json::object &obj, T &t, std::string_view key)
{
    t = boost::json::value_to<T>(obj.at(key));
}

/**
 * @brief Returns a results structure with the contents of a loaded JSON file.
 *
 * @param jv the contents of the loaded JSON file
 *
 * @return a GpratResults structure filled with the loaded values
 */
GpratResults tag_invoke(boost::json::value_to_tag<GpratResults>, const boost::json::value &jv)
{
    GpratResults results;
    const auto &obj = jv.as_object();
    extract(obj, results.cholesky, "cholesky");
    extract(obj, results.losses, "losses");
    extract(obj, results.sum, "sum");
    extract(obj, results.full, "full");
    extract(obj, results.pred, "pred");
    return results;
}

/**
 * @brief Tries to read the contents of the specified filename to set them as the basis of the
 *        test for correctness. If that is not possible, a file with the specified name is created
 *        and filled with fallback results.
 *
 * @param filename the filename to read from in case of success or write to in case of failure
 * @param fallback_results the fallback results to fill the file with in case of failure
 * @param results the results object to fill up with the content of the file in case of success
 *
 * @return `true` if reading the specified file is successful, and `false` if it failed and has
 *         been created
 */
bool load_or_create_expected_results(
    const std::string &filename, const GpratResults &fallback_results, GpratResults &results)
{
    // First try to read our expected results file
    {
        std::ifstream ifs(filename);
        if (!ifs.fail())
        {
            using iterator_type = std::istreambuf_iterator<char>;
            const std::string content(iterator_type{ ifs }, iterator_type{});
            results = boost::json::value_to<GpratResults>(boost::json::parse(content));
            return true;
        }
    }

    // If that does not work, just write out the results we want
    std::ofstream fout(filename);
    fout << boost::json::value_from(fallback_results);
    return false;
}

/**
 * @brief Tries to load the environment variable `GPRAT_ROOT` as the directory pointing toward the
 *        test data, and sets `../data` if this is not possible.
 *
 * @return a string containing the location of the test data, potentially relative to the working
 *         directory
 */
std::string get_data_directory()
{
    const char *env_root = std::getenv("GPRAT_ROOT");
    if (env_root)
    {
        return env_root;
    }
    else
    {
        return "../data";
    }
}

// Test execution /////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Generates results for a test configuration using the CPU for computations.
 *
 * This logic is basically equivalent to the GPRat C++ example.
 *
 * @param train_path path to the text file containing the training data
 * @param out_path path to the text file containing the output data of the test
 * @param test_path path to the text file containing the input data for the test
 *
 * @return a GpratResults object holding the results generated during the test
 */
GpratResults run_on_data_cpu(const std::string &train_path, const std::string &out_path, const std::string &test_path)
{
    // Compute tile sizes and number of predict tiles
    const int tile_size = gprat::compute_train_tile_size(n_train, n_tiles);
    const auto test_tiles = gprat::compute_test_tiles(n_test, n_tiles, tile_size);

    // hyperparams
    gprat::AdamParams hpar = { 0.1, 0.9, 0.999, 1e-8, OPT_ITER };

    // data loading
    gprat::GP_data training_input(train_path, n_train, n_reg);
    gprat::GP_data training_output(out_path, n_train, n_reg);
    gprat::GP_data test_input(test_path, n_test, n_reg);

    // GP
    const std::vector<bool> trainable = { true, true, true };

    gprat::GP gp_cpu(
        training_input.data, training_output.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, trainable);

    // Initialize HPX with no arguments, don't run hpx_main
    gprat::start_hpx_runtime(0, nullptr);

    GpratResults results_cpu;

    // Cholesky decomposition
    results_cpu.cholesky = gp_cpu.cholesky();

    // Prediction
    results_cpu.sum = gp_cpu.predict_with_uncertainty(test_input.data, test_tiles.first, test_tiles.second);
    results_cpu.full = gp_cpu.predict_with_full_cov(test_input.data, test_tiles.first, test_tiles.second);
    results_cpu.pred = gp_cpu.predict(test_input.data, test_tiles.first, test_tiles.second);

    // Optimization
    results_cpu.losses = gp_cpu.optimize(hpar);

    // Stop the HPX runtime
    gprat::stop_hpx_runtime();

    return results_cpu;
}

/**
 * @brief Generates results for a test configuration using a CUDA GPU or a SYCL device for
 *        computations, depending on how GPRat was compiled.
 *
 * @param train_path path to the text file containing the training data
 * @param out_path path to the text file containing the output data of the test
 * @param test_path path to the text file containing the input data for the test
 *
 * @return a GpratResults object holding the results generated during the test
 */
GpratResults run_on_data_gpu(const std::string &train_path, const std::string &out_path, const std::string &test_path)
{
    const std::size_t n_test = 128;
    const std::size_t n_train = 128;
    const std::size_t n_tiles = 16;
    const std::size_t n_reg = 8;
    const int gpu_id = 0;
    const int n_streams = 1;

    const int tile_size = gprat::compute_train_tile_size(n_train, n_tiles);
    const auto test_tiles = gprat::compute_test_tiles(n_test, n_tiles, tile_size);

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
        n_units);

    gprat::start_hpx_runtime(0, nullptr);

    GpratResults results_gpu;

    // Cholesky
    results_gpu.cholesky = gp_gpu.cholesky();

    // Prediction
    results_gpu.sum = gp_gpu.predict_with_uncertainty(test_input.data, test_tiles.first, test_tiles.second);
    results_gpu.full = gp_gpu.predict_with_full_cov(test_input.data, test_tiles.first, test_tiles.second);
    results_gpu.pred = gp_gpu.predict(test_input.data, test_tiles.first, test_tiles.second);

    // GPUs do not support optimization

    gprat::stop_hpx_runtime();

    return results_gpu;
}

// Test cases /////////////////////////////////////////////////////////////////////////////////////

/*
 * CPU test case
 */
TEST_CASE("GP CPU results match known-good values", "[integration][cpu]")
{
    const std::string root = get_data_directory();

    const auto results = run_on_data_cpu(root + "/data_1024/training_input.txt",
                                         root + "/data_1024/training_output.txt",
                                         root + "/data_1024/test_input.txt");

    GpratResults expected_results;

    if (!load_or_create_expected_results(root + "/data_1024/output.json", results, expected_results))
    {
        std::cerr << "No previous results to compare to. The current results have been saved instead!\n";
        return;
    }

    /*
     * Compare content, see reference [1]
     * The default-constructed WithinRel() matcher has a tolerance of epsilon * 100
     */
    double eps = std::numeric_limits<double>::epsilon() * 1'000'000;

    /*
     * Require that the results of the Cholesky decomposition have a relative error below the
     * specified `eps`
     */
    for (std::size_t i = 0, n = results.cholesky.size(); i != n; ++i)
    {
        for (std::size_t j = 0, m = results.cholesky[i].size(); j != m; ++j)
        {
            INFO("CPU cholesky " << i << " " << j);
            REQUIRE_THAT(results.cholesky[i][j], WithinRel(expected_results.cholesky[i][j], eps));
        }
    }

    /*
     * Require that the losses after accessing `optimize` have a relative error below the
     * specified `eps`
     */
    for (std::size_t i = 0, n = results.losses.size(); i != n; ++i)
    {
        INFO("CPU losses " << i);
        REQUIRE_THAT(results.losses[i], WithinRel(expected_results.losses[i], eps));
    }

    /*
     * Require that the sums after predicting with uncertainty have a relative error below the
     * specified `eps`
     */
    for (std::size_t i = 0, n = results.sum.size(); i != n; ++i)
    {
        for (std::size_t j = 0, m = results.sum[i].size(); j != m; ++j)
        {
            INFO("CPU sum " << i << " " << j);
            REQUIRE_THAT(results.sum[i][j], WithinRel(expected_results.sum[i][j], eps));
        }
    }

    /*
     * Require that the results when predicting with the full covariance matrix have a relative
     * error below the specified `eps`
     */
    for (std::size_t i = 0, n = results.full.size(); i != n; ++i)
    {
        for (std::size_t j = 0, m = results.full[i].size(); j != m; ++j)
        {
            INFO("CPU full " << i << " " << j);
            REQUIRE_THAT(results.full[i][j], WithinRel(expected_results.full[i][j], eps));
        }
    }

    /*
     * Require that the results retrieved form a mere prediction have a relative error below the
     * specified `eps`
     */
    for (std::size_t i = 0, n = results.pred.size(); i != n; ++i)
    {
        INFO("CPU pred " << i);
        REQUIRE_THAT(results.pred[i], WithinRel(expected_results.pred[i], eps));
    }
}

/*
 * GPU test case for CUDA and SYCL
 */
TEST_CASE("GP GPU results match known-good values (no loss)", "[integration][gpu]")
{
    if (!gprat::compiled_with_cuda() && !gprat::compiled_with_sycl())
    {
        INFO("GPRat not compiled with GPU support — skipping GPU test.");
        return;
    }
    if (gprat::compiled_with_cuda())
    {
        INFO("Executing GPU test with CUDA support.");
    }
    else if (gprat::compiled_with_sycl())
    {
        INFO("Executing GPU test with SYCL support.");
    }
    else
    {
        INFO("GPRat not compiled with GPU support — skipping GPU test.");
        return;
    }

    const std::string root = get_data_directory();

    const auto results = run_on_data_gpu(root + "/data_1024/training_input.txt",
                                         root + "/data_1024/training_output.txt",
                                         root + "/data_1024/test_input.txt");

    GpratResults expected_results;
    const std::string ref_file = root + "/data_1024/output.json";

    if (!load_or_create_expected_results(ref_file, results, expected_results))
    {
        std::cerr << "No previous results to compare to. The current results have been saved instead!\n";
        return;
    }

    using Catch::Matchers::WithinRel;
    double eps = std::numeric_limits<double>::epsilon() * 1'000'000;

    for (std::size_t i = 0, n = results.cholesky.size(); i != n; ++i)
    {
        for (std::size_t j = 0, m = results.cholesky[i].size(); j != m; ++j)
        {
            INFO("GPU cholesky " << i << " " << j);
            REQUIRE_THAT(results.cholesky[i][j], WithinRel(expected_results.cholesky[i][j], eps));
        }
    }

    for (std::size_t i = 0, n = results.sum.size(); i != n; ++i)
    {
        for (std::size_t j = 0, m = results.sum[i].size(); j != m; ++j)
        {
            INFO("GPU sum " << i << " " << j);
            REQUIRE_THAT(results.sum[i][j], WithinRel(expected_results.sum[i][j], eps));
        }
    }

    for (std::size_t i = 0, n = results.full.size(); i != n; ++i)
    {
        for (std::size_t j = 0, m = results.full[i].size(); j != m; ++j)
        {
            INFO("GPU full " << i << " " << j);
            REQUIRE_THAT(results.full[i][j], WithinRel(expected_results.full[i][j], eps));
        }
    }

    for (std::size_t i = 0, n = results.pred.size(); i != n; ++i)
    {
        INFO("GPU pred " << i);
        REQUIRE_THAT(results.pred[i], WithinRel(expected_results.pred[i], eps));
    }
}

}  // namespace gprat::test
