#include "gprat_c.hpp"
#include "target.hpp"
#include "utils_c.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdlib>
#include <limits>
#include <string>
using Catch::Matchers::WithinRel;

namespace
{
// Starts the HPX runtime on construction and stops it on destruction so that
// stop_hpx_runtime() is always called even when a test assertion fails mid-test.
struct hpx_runtime_guard
{
    hpx_runtime_guard() { utils::start_hpx_runtime(0, nullptr); }

    ~hpx_runtime_guard() { utils::stop_hpx_runtime(); }
};
}  // namespace

namespace gprat::test
{

static std::string gprat_data_root()
{
    const char *env = std::getenv("GPRAT_ROOT");
    return env ? env : "../data";
}

TEST_CASE("GP: results are tile-count invariant", "[unit][gp][predict][tiling]")
{
    // Tiling is purely a scheduling/decomposition detail: predictions, uncertainty,
    // and loss for fixed (untrained) hyperparameters must not depend on n_tiles.
    const std::string root = gprat_data_root();

    constexpr int n = 128, n_reg = 8, n_test = 64;
    const double eps = std::numeric_limits<double>::epsilon() * 1'000'000;

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, 1);
    gprat::GP_data test_in(root + "/data_1024/test_input.txt", n_test, n_reg);

    hpx_runtime_guard hpx_guard;

    // Baseline: a single tile, i.e. no decomposition at all.
    const int baseline_tile_size = utils::compute_train_tile_size(n, 1);
    const auto [baseline_m_tiles, baseline_m_tile_size] = utils::compute_test_tiles(n_test, 1, baseline_tile_size);
    gprat::GP baseline_gp(
        train_in.data, train_out.data, 1, baseline_tile_size, n_reg, { 1.0, 1.0, 0.1 }, { false, false, false });
    const auto baseline_pred =
        baseline_gp.predict_with_uncertainty(test_in.data, baseline_m_tiles, baseline_m_tile_size);
    const double baseline_loss = baseline_gp.calculate_loss();

    for (const int n_tiles : { 2, 4, 8 })
    {
        const int tile_size = utils::compute_train_tile_size(n, n_tiles);
        const auto [m_tiles, m_tile_size] = utils::compute_test_tiles(n_test, n_tiles, tile_size);

        gprat::GP gp(
            train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 }, { false, false, false });
        const auto pred = gp.predict_with_uncertainty(test_in.data, m_tiles, m_tile_size);

        for (int i = 0; i < n_test; ++i)
        {
            INFO("n_tiles=" << n_tiles << " mean[" << i << "]");
            REQUIRE_THAT(pred[0][static_cast<std::size_t>(i)],
                         WithinRel(baseline_pred[0][static_cast<std::size_t>(i)], eps));
            INFO("n_tiles=" << n_tiles << " variance[" << i << "]");
            REQUIRE_THAT(pred[1][static_cast<std::size_t>(i)],
                         WithinRel(baseline_pred[1][static_cast<std::size_t>(i)], eps));
        }

        INFO("n_tiles=" << n_tiles << " loss");
        REQUIRE_THAT(gp.calculate_loss(), WithinRel(baseline_loss, eps));
    }
}

}  // namespace gprat::test
