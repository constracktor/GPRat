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
    return results;
}

// This logic is basically equivalent to the GPRat C++ example (for now).
gprat_results run_on_data(const std::string &train_path, const std::string &out_path, const std::string &test_path)
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
    gprat::GP gp(training_input.data, training_output.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, trainable);

    // Initialize HPX with no arguments, don't run hpx_main
    utils::start_hpx_runtime(0, nullptr);

    gprat_results results;
    results.choleksy = gp.cholesky();
    results.losses = gp.optimize(hpar);
    results.sum = gp.predict_with_uncertainty(test_input.data, test_tiles.first, test_tiles.second);
    results.full = gp.predict_with_full_cov(test_input.data, test_tiles.first, test_tiles.second);
    results.pred = gp.predict(test_input.data, test_tiles.first, test_tiles.second);

    // Stop the HPX runtime
    utils::stop_hpx_runtime();
    return results;
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

TEST_CASE("GP results match known-good values", "[integration]")
{
    const std::string root = get_root_directory();
    const auto results = run_on_data(root + "/data_1024/training_input.txt",
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
    REQUIRE(results.full.size() == expected_results.full.size());
    REQUIRE(results.pred.size() == expected_results.pred.size());

    // Now we can compare content
    // The default-constructed WithinRel() matcher has a tolerance of epsilon * 100
    // see:
    // https://github.com/catchorg/Catch2/blob/914aeecfe23b1e16af6ea675a4fb5dbd5a5b8d0a/docs/comparing-floating-point-numbers.md#withinrel
    using Catch::Matchers::WithinRel;
    for (std::size_t i = 0, n = results.choleksy.size(); i != n; ++i)
    {
        for (std::size_t j = 0, m = results.choleksy[i].size(); j != m; ++j)
        {
            INFO("choleksy " << i << " " << j);
            REQUIRE_THAT(results.choleksy[i][j], WithinRel(expected_results.choleksy[i][j]));
        }
    }
    for (std::size_t i = 0, n = results.losses.size(); i != n; ++i)
    {
        INFO("losses " << i);
        REQUIRE_THAT(results.losses[i], WithinRel(expected_results.losses[i]));
    }
    for (std::size_t i = 0, n = results.sum.size(); i != n; ++i)
    {
        for (std::size_t j = 0, m = results.sum[i].size(); j != m; ++j)
        {
            INFO("sum " << i << " " << j);
            REQUIRE_THAT(results.sum[i][j], WithinRel(expected_results.sum[i][j]));
        }
    }
    for (std::size_t i = 0, n = results.full.size(); i != n; ++i)
    {
        for (std::size_t j = 0, m = results.full[i].size(); j != m; ++j)
        {
            INFO("full " << i << " " << j);
            REQUIRE_THAT(results.full[i][j], WithinRel(expected_results.full[i][j]));
        }
    }
    for (std::size_t i = 0, n = results.pred.size(); i != n; ++i)
    {
        INFO("pred " << i);
        REQUIRE_THAT(results.pred[i], WithinRel(expected_results.pred[i]));
    }
}
