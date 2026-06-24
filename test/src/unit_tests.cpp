#include "gprat/cpu/adapter_cblas_fp32.hpp"
#include "gprat/gprat.hpp"
#include "gprat/hyperparameters.hpp"
#include "gprat/kernels.hpp"
#include "gprat/utils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <sstream>
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// Helper: build a tile_data from an initializer list
template <typename T>
static gprat::mutable_tile_data<T> make_tile(std::initializer_list<T> vals)
{
    gprat::mutable_tile_data<T> t(vals.size());
    std::size_t i = 0;
    for (const auto &v : vals)
        t.data()[i++] = v;
    return t;
}

template <typename T>
static gprat::const_tile_data<T> make_const_tile(std::initializer_list<T> vals)
{
    // const_tile_data has no mutable operator[], so build via mutable first
    gprat::mutable_tile_data<T> m = make_tile<T>(vals);
    return m;
}

namespace gprat::test
{

// GP_data ///////////////////////////////////////////////////////////////////////////////////////

TEST_CASE("GP_data loads correct number of samples", "[unit][gp_data]")
{
    const char *env_root = std::getenv("GPRAT_ROOT");
    const std::string root = env_root ? env_root : "../data";
    const std::string path = root + "/data_1024/training_input.txt";

    constexpr std::size_t n = 64;
    constexpr std::size_t n_reg = 8;

    gprat::GP_data d(path, n, n_reg);

    REQUIRE(d.n_samples == n);
    REQUIRE(d.n_regressors == n_reg);
    // load_data allocates n_samples + (n_reg - 1) elements: data starts at offset n_reg-1
    REQUIRE(d.data.size() == n + n_reg - 1);
    REQUIRE(d.file_path == path);
}

TEST_CASE("GP_data with n_reg=1 loads n samples", "[unit][gp_data]")
{
    const char *env_root = std::getenv("GPRAT_ROOT");
    const std::string root = env_root ? env_root : "../data";
    const std::string path = root + "/data_1024/training_input.txt";

    constexpr std::size_t n = 32;
    gprat::GP_data d(path, n, 1);

    REQUIRE(d.data.size() == n + 1 - 1);  // n + (n_reg - 1) with n_reg=1
}

// Tile utilities ////////////////////////////////////////////////////////////////////////////////

TEST_CASE("compute_train_tile_size divides evenly", "[unit][tiles]")
{
    REQUIRE(gprat::compute_train_tile_size(1024, 16) == 64);
    REQUIRE(gprat::compute_train_tile_size(512, 8) == 64);
    REQUIRE(gprat::compute_train_tile_size(256, 4) == 64);
}

TEST_CASE("compute_train_tiles divides evenly", "[unit][tiles]")
{
    REQUIRE(gprat::compute_train_tiles(1024, 64) == 16);
    REQUIRE(gprat::compute_train_tiles(512, 64) == 8);
}

TEST_CASE("compute_train_tile_size throws on zero tiles", "[unit][tiles]")
{
    REQUIRE_THROWS_AS(gprat::compute_train_tile_size(1024, 0), std::runtime_error);
}

TEST_CASE("compute_train_tiles throws on zero tile size", "[unit][tiles]")
{
    REQUIRE_THROWS_AS(gprat::compute_train_tiles(1024, 0), std::runtime_error);
}

TEST_CASE("compute_test_tiles: n_test divisible by tile_size uses same tile_size", "[unit][tiles]")
{
    // n_test=512, tile_size=64 → 512 % 64 == 0, so use m_tile_size=64, m_tiles=8
    const auto [m_tiles, m_tile_size] = gprat::compute_test_tiles(512, 16, 64);
    REQUIRE(m_tile_size == 64);
    REQUIRE(m_tiles == 8);
    REQUIRE(m_tiles * m_tile_size == 512);
}

TEST_CASE("compute_test_tiles: n_test not divisible by tile_size uses same n_tiles", "[unit][tiles]")
{
    // n_test=100, tile_size=64 → 100 % 64 != 0, so use m_tiles=16, m_tile_size=100/16
    const auto [m_tiles, m_tile_size] = gprat::compute_test_tiles(100, 16, 64);
    REQUIRE(m_tiles == 16);
    REQUIRE(m_tile_size == 100 / 16);
}

TEST_CASE("compute_train_tile_size and compute_train_tiles are inverses", "[unit][tiles]")
{
    constexpr std::size_t n = 1024;
    constexpr std::size_t tiles = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, tiles);
    const std::size_t recovered = gprat::compute_train_tiles(n, tile_size);
    REQUIRE(recovered == tiles);
}

// Optimizer (CPU) ////////////////////////////////////////////////////////////////////////////////

TEST_CASE("GP::optimize returns one loss per iteration", "[unit][optimizer][cpu]")
{
    const char *env_root = std::getenv("GPRAT_ROOT");
    const std::string root = env_root ? env_root : "../data";

    constexpr std::size_t n = 128;
    constexpr std::size_t n_tiles = 4;
    constexpr std::size_t n_reg = 8;
    constexpr int opt_iter = 5;

    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                 { true, true, true });

    gprat::start_hpx_runtime(0, nullptr);
    const gprat::AdamParams params{ 0.01, 0.9, 0.999, 1e-8, opt_iter };
    const auto losses = gp.optimize(params);
    gprat::stop_hpx_runtime();

    REQUIRE(losses.size() == static_cast<std::size_t>(opt_iter));
}

TEST_CASE("GP::optimize_step returns finite loss", "[unit][optimizer][cpu]")
{
    const char *env_root = std::getenv("GPRAT_ROOT");
    const std::string root = env_root ? env_root : "../data";

    constexpr std::size_t n = 128;
    constexpr std::size_t n_tiles = 4;
    constexpr std::size_t n_reg = 8;

    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                 { true, true, true });

    gprat::start_hpx_runtime(0, nullptr);
    gprat::AdamParams params{ 0.01, 0.9, 0.999, 1e-8, 1 };
    const double loss = gp.optimize_step(params, 1);
    gprat::stop_hpx_runtime();

    REQUIRE(std::isfinite(loss));
}

TEST_CASE("GP::calculate_loss returns finite value", "[unit][loss][cpu]")
{
    const char *env_root = std::getenv("GPRAT_ROOT");
    const std::string root = env_root ? env_root : "../data";

    constexpr std::size_t n = 128;
    constexpr std::size_t n_tiles = 4;
    constexpr std::size_t n_reg = 8;

    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                 { true, true, true });

    gprat::start_hpx_runtime(0, nullptr);
    const double loss = gp.calculate_loss();
    gprat::stop_hpx_runtime();

    REQUIRE(std::isfinite(loss));
}

TEST_CASE("GP::optimize reduces loss over iterations", "[unit][optimizer][cpu]")
{
    const char *env_root = std::getenv("GPRAT_ROOT");
    const std::string root = env_root ? env_root : "../data";

    constexpr std::size_t n = 128;
    constexpr std::size_t n_tiles = 4;
    constexpr std::size_t n_reg = 8;

    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                 { true, true, true });

    gprat::start_hpx_runtime(0, nullptr);
    const gprat::AdamParams params{ 0.1, 0.9, 0.999, 1e-8, 10 };
    const auto losses = gp.optimize(params);
    gprat::stop_hpx_runtime();

    // Loss should decrease from first to last iteration
    REQUIRE(losses.front() > losses.back());
}

// SEKParams /////////////////////////////////////////////////////////////////////////////////////

TEST_CASE("SEKParams::size returns 3", "[unit][sek]")
{
    gprat::SEKParams p(1.0, 2.0, 0.1);
    REQUIRE(p.size() == 3);
}

TEST_CASE("SEKParams::get_param returns correct fields", "[unit][sek]")
{
    gprat::SEKParams p(1.5, 2.5, 0.3);
    REQUIRE_THAT(p.get_param(0), WithinRel(1.5, 1e-12));
    REQUIRE_THAT(p.get_param(1), WithinRel(2.5, 1e-12));
    REQUIRE_THAT(p.get_param(2), WithinRel(0.3, 1e-12));
}

TEST_CASE("SEKParams::set_param mutates correct fields", "[unit][sek]")
{
    gprat::SEKParams p(1.0, 1.0, 0.1);
    p.set_param(0, 3.0);
    p.set_param(1, 4.0);
    p.set_param(2, 0.5);
    REQUIRE_THAT(p.lengthscale, WithinRel(3.0, 1e-12));
    REQUIRE_THAT(p.vertical_lengthscale, WithinRel(4.0, 1e-12));
    REQUIRE_THAT(p.noise_variance, WithinRel(0.5, 1e-12));
}

TEST_CASE("SEKParams::get_param throws on out-of-range index", "[unit][sek]")
{
    gprat::SEKParams p(1.0, 1.0, 0.1);
    REQUIRE_THROWS_AS(p.get_param(3), std::invalid_argument);
}

TEST_CASE("SEKParams::set_param throws on out-of-range index", "[unit][sek]")
{
    gprat::SEKParams p(1.0, 1.0, 0.1);
    REQUIRE_THROWS_AS(p.set_param(3, 0.0), std::invalid_argument);
}

TEST_CASE("SEKParams constructor resizes m_T and w_T to 3", "[unit][sek]")
{
    gprat::SEKParams p(1.0, 1.0, 0.1);
    REQUIRE(p.m_T.size() == 3);
    REQUIRE(p.w_T.size() == 3);
}

// AdamParams ////////////////////////////////////////////////////////////////////////////////////

TEST_CASE("AdamParams default constructor values", "[unit][adam]")
{
    gprat::AdamParams p;
    REQUIRE_THAT(p.learning_rate, WithinRel(0.001, 1e-12));
    REQUIRE_THAT(p.beta1, WithinRel(0.9, 1e-12));
    REQUIRE_THAT(p.beta2, WithinRel(0.999, 1e-12));
    REQUIRE_THAT(p.epsilon, WithinRel(1e-8, 1e-12));
    REQUIRE(p.opt_iter == 0);
}

TEST_CASE("AdamParams::repr contains all fields", "[unit][adam]")
{
    gprat::AdamParams p(0.01, 0.9, 0.999, 1e-8, 5);
    const auto s = p.repr();
    REQUIRE_THAT(s, ContainsSubstring("learning_rate"));
    REQUIRE_THAT(s, ContainsSubstring("beta1"));
    REQUIRE_THAT(s, ContainsSubstring("beta2"));
    REQUIRE_THAT(s, ContainsSubstring("epsilon"));
    REQUIRE_THAT(s, ContainsSubstring("opt_iter"));
}

// GP_data error handling /////////////////////////////////////////////////////////////////////////

TEST_CASE("GP_data throws on missing file", "[unit][gp_data]")
{
    REQUIRE_THROWS_AS(gprat::GP_data("/nonexistent/path/file.txt", 10, 4), std::runtime_error);
}

// GP accessors and repr //////////////////////////////////////////////////////////////////////////

TEST_CASE("GP::get_training_input round-trips data", "[unit][gp]")
{
    const char *env_root = std::getenv("GPRAT_ROOT");
    const std::string root = env_root ? env_root : "../data";

    constexpr std::size_t n = 64, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                 { true, true, true });

    REQUIRE(gp.get_training_input() == train_in.data);
    REQUIRE(gp.get_training_output() == train_out.data);
}

TEST_CASE("GP::repr contains key fields", "[unit][gp]")
{
    const char *env_root = std::getenv("GPRAT_ROOT");
    const std::string root = env_root ? env_root : "../data";

    constexpr std::size_t n = 64, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                 { true, true, true });

    const auto s = gp.repr();
    REQUIRE_THAT(s, ContainsSubstring("lengthscale"));
    REQUIRE_THAT(s, ContainsSubstring("n_tiles"));
}

// GP prediction shapes ///////////////////////////////////////////////////////////////////////////

TEST_CASE("GP::predict output has correct size", "[unit][gp][predict]")
{
    const char *env_root = std::getenv("GPRAT_ROOT");
    const std::string root = env_root ? env_root : "../data";

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8, n_test = 128;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);
    const auto [m_tiles, m_tile_size] = gprat::compute_test_tiles(n_test, n_tiles, tile_size);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);
    gprat::GP_data test_in(root + "/data_1024/test_input.txt", n_test, n_reg);

    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                 { true, true, true });

    gprat::start_hpx_runtime(0, nullptr);
    const auto pred = gp.predict(test_in.data, m_tiles, m_tile_size);
    const auto pred_unc = gp.predict_with_uncertainty(test_in.data, m_tiles, m_tile_size);
    const auto pred_cov = gp.predict_with_full_cov(test_in.data, m_tiles, m_tile_size);
    gprat::stop_hpx_runtime();

    REQUIRE(pred.size() == n_test);
    REQUIRE(pred_unc.size() == 2);
    REQUIRE(pred_unc[0].size() == n_test);
    REQUIRE(pred_unc[1].size() == n_test);
    // predict_with_full_cov returns {mean, diagonal(Sigma)} — same shape as predict_with_uncertainty
    REQUIRE(pred_cov.size() == 2);
    REQUIRE(pred_cov[0].size() == n_test);
    REQUIRE(pred_cov[1].size() == n_test);
}

TEST_CASE("GP::cholesky returns correct tile structure", "[unit][gp][cholesky]")
{
    const char *env_root = std::getenv("GPRAT_ROOT");
    const std::string root = env_root ? env_root : "../data";

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                 { true, true, true });

    gprat::start_hpx_runtime(0, nullptr);
    const auto L = gp.cholesky();
    gprat::stop_hpx_runtime();

    // n_tiles × n_tiles blocks stored as flat list of n_tiles^2 tiles
    REQUIRE(L.size() == n_tiles * n_tiles);
    REQUIRE(L[0].size() == tile_size * tile_size);
}

// GP trainable mask //////////////////////////////////////////////////////////////////////////////

TEST_CASE("GP::optimize with no trainable params leaves loss unchanged", "[unit][optimizer][cpu]")
{
    const char *env_root = std::getenv("GPRAT_ROOT");
    const std::string root = env_root ? env_root : "../data";

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                 { false, false, false });

    gprat::start_hpx_runtime(0, nullptr);
    const gprat::AdamParams params{ 0.1, 0.9, 0.999, 1e-8, 5 };
    const auto losses = gp.optimize(params);
    gprat::stop_hpx_runtime();

    // All losses should be equal — no parameters moved
    for (std::size_t i = 1; i < losses.size(); ++i)
        REQUIRE_THAT(losses[i], WithinRel(losses[0], 1e-10));
}

// GP kernel_params live mutation /////////////////////////////////////////////////////////////////

TEST_CASE("GP::calculate_loss changes when kernel_params is mutated", "[unit][gp][loss]")
{
    const char *env_root = std::getenv("GPRAT_ROOT");
    const std::string root = env_root ? env_root : "../data";

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                 { true, true, true });

    gprat::start_hpx_runtime(0, nullptr);
    const double loss_before = gp.calculate_loss();
    gp.kernel_params.lengthscale = 5.0;
    const double loss_after = gp.calculate_loss();
    gprat::stop_hpx_runtime();

    REQUIRE_THAT(std::abs(loss_before - loss_after), !WithinRel(0.0, 1e-10));
}

// guess_good_tile_count_per_dimension ////////////////////////////////////////////////////////////

TEST_CASE("guess_good_tile_count_per_dimension returns 1 for small n", "[unit][tiles]")
{
    // n < 2^8 = 256 → always returns 1
    REQUIRE(gprat::guess_good_tile_count_per_dimension(100) == 1);
    REQUIRE(gprat::guess_good_tile_count_per_dimension(1) == 1);
}

TEST_CASE("guess_good_tile_count_per_dimension returns positive count for large n", "[unit][tiles]")
{
    gprat::start_hpx_runtime(0, nullptr);
    const std::size_t count = gprat::guess_good_tile_count_per_dimension(1 << 14);
    gprat::stop_hpx_runtime();
    REQUIRE(count >= 1);
}

// compiled_with_cuda / compiled_with_sycl ////////////////////////////////////////////////////////

TEST_CASE("compiled_with_cuda returns false (no CUDA build)", "[unit][target]")
{
    REQUIRE_FALSE(gprat::compiled_with_cuda());
}

TEST_CASE("compiled_with_sycl returns false (no SYCL build)", "[unit][target]")
{
    REQUIRE_FALSE(gprat::compiled_with_sycl());
}

// GP GPU constructor throws without CUDA/SYCL ////////////////////////////////////////////////////

TEST_CASE("GP GPU constructor throws when built without CUDA/SYCL", "[unit][gp]")
{
    const char *env_root = std::getenv("GPRAT_ROOT");
    const std::string root = env_root ? env_root : "../data";

    constexpr std::size_t n = 64, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    REQUIRE_THROWS_AS(
        (gprat::GP(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                   { true, true, true }, 0, 1)),
        std::runtime_error);
}

// print_vector ///////////////////////////////////////////////////////////////////////////////////

TEST_CASE("print_vector: basic range prints to stdout", "[unit][utils]")
{
    const std::vector<double> v = { 1.0, 2.0, 3.0 };
    std::streambuf *old = std::cout.rdbuf();
    std::ostringstream buf;
    std::cout.rdbuf(buf.rdbuf());
    gprat::print_vector(v, 0, 3, ",");
    std::cout.rdbuf(old);
    REQUIRE_THAT(buf.str(), ContainsSubstring("1") && ContainsSubstring("2") && ContainsSubstring("3"));
}

TEST_CASE("print_vector: negative start wraps around", "[unit][utils]")
{
    const std::vector<double> v = { 10.0, 20.0, 30.0 };
    std::streambuf *old = std::cout.rdbuf();
    std::ostringstream buf;
    std::cout.rdbuf(buf.rdbuf());
    gprat::print_vector(v, -2, 3, " ");  // start = 3 - 2 = 1 → prints 20 30
    std::cout.rdbuf(old);
    REQUIRE_THAT(buf.str(), ContainsSubstring("20"));
}

TEST_CASE("print_vector: negative end wraps around", "[unit][utils]")
{
    const std::vector<double> v = { 10.0, 20.0, 30.0 };
    std::streambuf *old = std::cout.rdbuf();
    std::ostringstream buf;
    std::cout.rdbuf(buf.rdbuf());
    gprat::print_vector(v, 0, -1, " ");  // end = 3 + 1 - 1 = 3
    std::cout.rdbuf(old);
    REQUIRE_THAT(buf.str(), ContainsSubstring("10"));
}

TEST_CASE("print_vector: out-of-bound end is clamped", "[unit][utils]")
{
    const std::vector<double> v = { 5.0, 6.0 };
    std::streambuf *old_out = std::cout.rdbuf();
    std::streambuf *old_err = std::cerr.rdbuf();
    std::ostringstream buf_out, buf_err;
    std::cout.rdbuf(buf_out.rdbuf());
    std::cerr.rdbuf(buf_err.rdbuf());
    gprat::print_vector(v, 0, 100, ",");  // end clamped to 2
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    REQUIRE_THAT(buf_out.str(), ContainsSubstring("5"));
}

TEST_CASE("print_vector: invalid range prints to stderr", "[unit][utils]")
{
    const std::vector<double> v = { 1.0, 2.0, 3.0 };
    std::streambuf *old_err = std::cerr.rdbuf();
    std::ostringstream buf;
    std::cerr.rdbuf(buf.rdbuf());
    gprat::print_vector(v, 2, 1, ",");  // start >= end → invalid
    std::cerr.rdbuf(old_err);
    REQUIRE_THAT(buf.str(), ContainsSubstring("Invalid"));
}

// fp32 BLAS adapters /////////////////////////////////////////////////////////////////////////////

TEST_CASE("fp32 BLAS: potrf, dot, axpy, syrk, gemm, trsm, trsv, gemv", "[unit][blas][fp32]")
{
    gprat::start_hpx_runtime(0, nullptr);

    // potrf: Cholesky of 2x2 identity → L = I
    {
        auto A = make_tile<float>({ 1.0f, 0.0f, 0.0f, 1.0f });
        const auto L = gprat::potrf(A, 2);
        REQUIRE_THAT(L.data()[0], WithinAbs(1.0f, 1e-5f));
        REQUIRE_THAT(L.data()[3], WithinAbs(1.0f, 1e-5f));
    }

    // dot: 1*4 + 2*5 + 3*6 = 32
    {
        const std::vector<float> a = { 1.0f, 2.0f, 3.0f };
        const std::vector<float> b = { 4.0f, 5.0f, 6.0f };
        REQUIRE_THAT(gprat::dot(std::span<const float>(a), std::span<const float>(b), 3),
                     WithinAbs(32.0f, 1e-4f));
    }

    // axpy: y -= x  (alpha = -1 by convention in gprat)
    {
        auto y = make_tile<float>({ 10.0f, 20.0f, 30.0f });
        auto x = make_const_tile<float>({ 1.0f, 2.0f, 3.0f });
        const auto r = gprat::axpy(y, x, 3);
        REQUIRE_THAT(r.data()[0], WithinAbs(9.0f, 1e-5f));
        REQUIRE_THAT(r.data()[1], WithinAbs(18.0f, 1e-5f));
        REQUIRE_THAT(r.data()[2], WithinAbs(27.0f, 1e-5f));
    }

    // syrk: C -= B*B^T  (alpha = -1), C=0, B=diag(1,2) → C = -diag(1,4)
    {
        auto C = make_tile<float>({ 0.0f, 0.0f, 0.0f, 0.0f });
        auto B = make_const_tile<float>({ 1.0f, 0.0f, 0.0f, 2.0f });
        const auto r = gprat::syrk(C, B, 2);
        REQUIRE_THAT(r.data()[0], WithinAbs(-1.0f, 1e-5f));
        REQUIRE_THAT(r.data()[3], WithinAbs(-4.0f, 1e-5f));
    }

    // gemm: C -= A*B  (alpha=-1), C=0, A=I, B=diag(2,3) → C = -diag(2,3)
    {
        auto A = make_const_tile<float>({ 1.0f, 0.0f, 0.0f, 1.0f });
        auto B = make_const_tile<float>({ 2.0f, 0.0f, 0.0f, 3.0f });
        auto C = make_tile<float>({ 0.0f, 0.0f, 0.0f, 0.0f });
        const auto r = gprat::gemm(A, B, C, 2, 2, 2, gprat::Blas_no_trans, gprat::Blas_no_trans);
        REQUIRE_THAT(r.data()[0], WithinAbs(-2.0f, 1e-5f));
        REQUIRE_THAT(r.data()[3], WithinAbs(-3.0f, 1e-5f));
    }

    // trsm: I * X = B → X = B
    {
        auto L = make_const_tile<float>({ 1.0f, 0.0f, 0.0f, 1.0f });
        auto B = make_tile<float>({ 5.0f, 7.0f, 9.0f, 11.0f });
        const auto X = gprat::trsm(L, B, 2, 2, gprat::Blas_no_trans, gprat::Blas_left);
        REQUIRE_THAT(X.data()[0], WithinAbs(5.0f, 1e-5f));
        REQUIRE_THAT(X.data()[1], WithinAbs(7.0f, 1e-5f));
    }

    // trsv: I * x = b → x = b
    {
        auto L = make_const_tile<float>({ 1.0f, 0.0f, 0.0f, 1.0f });
        auto b = make_tile<float>({ 3.0f, 4.0f });
        const auto x = gprat::trsv(L, b, 2, gprat::Blas_no_trans);
        REQUIRE_THAT(x.data()[0], WithinAbs(3.0f, 1e-5f));
        REQUIRE_THAT(x.data()[1], WithinAbs(4.0f, 1e-5f));
    }

    // gemv: I * [1,2] = [1,2]
    {
        auto A = make_const_tile<float>({ 1.0f, 0.0f, 0.0f, 1.0f });
        auto x = make_const_tile<float>({ 1.0f, 2.0f });
        auto y = make_tile<float>({ 0.0f, 0.0f });
        const auto r = gprat::gemv(A, x, y, 2, 2, gprat::Blas_add, gprat::Blas_no_trans);
        REQUIRE_THAT(r.data()[0], WithinAbs(1.0f, 1e-5f));
        REQUIRE_THAT(r.data()[1], WithinAbs(2.0f, 1e-5f));
    }

    // dot_diag_syrk: r[j] += dot(col_j(A), col_j(A))
    // A = [[1,0],[2,0]] (col-major 2x2), M=2, N=2 → r[0] += 1²+2²=5, r[1] += 0
    {
        // A stored row-major 2x2: rows=[1,0],[2,0] → col 0 = [1,2], col 1 = [0,0]
        auto A = make_const_tile<float>({ 1.0f, 0.0f, 2.0f, 0.0f });
        auto r = make_tile<float>({ 0.0f, 0.0f });
        const auto out = gprat::dot_diag_syrk(A, r, 2, 2);
        REQUIRE_THAT(out.data()[0], WithinAbs(5.0f, 1e-4f));  // 1² + 2²
        REQUIRE_THAT(out.data()[1], WithinAbs(0.0f, 1e-4f));  // 0² + 0²
    }

    // dot_diag_gemm: r[i] += dot(row_i(A), col_i(B))
    // A=I2, B=I2 → r[i] += 1, so r = [1, 1]
    {
        auto A = make_const_tile<float>({ 1.0f, 0.0f, 0.0f, 1.0f });
        auto B = make_const_tile<float>({ 1.0f, 0.0f, 0.0f, 1.0f });
        auto r = make_tile<float>({ 0.0f, 0.0f });
        const auto out = gprat::dot_diag_gemm(A, B, r, 2, 2);
        REQUIRE_THAT(out.data()[0], WithinAbs(1.0f, 1e-4f));
        REQUIRE_THAT(out.data()[1], WithinAbs(1.0f, 1e-4f));
    }

    gprat::stop_hpx_runtime();
}

// HPX runtime suspend/resume /////////////////////////////////////////////////////////////////////

TEST_CASE("suspend_hpx_runtime and resume_hpx_runtime work correctly", "[unit][hpx]")
{
    gprat::start_hpx_runtime(0, nullptr);
    // Suspend pauses HPX worker threads without stopping the runtime.
    // Resume brings them back. A loss calculation after resume confirms the
    // runtime is fully functional again.
    gprat::suspend_hpx_runtime();
    gprat::resume_hpx_runtime();

    const char *env_root = std::getenv("GPRAT_ROOT");
    const std::string root = env_root ? env_root : "../data";
    constexpr std::size_t n = 64, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);
    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);
    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                 { true, true, true });
    REQUIRE(std::isfinite(gp.calculate_loss()));

    gprat::stop_hpx_runtime();
}

// gpu_algorithms coverage: gen_tile_identity, gen_tile_zeros, gen_tile_output //////////////////

TEST_CASE("GP::optimize exercises gen_tile_identity via noise gradient", "[unit][optimizer][cpu]")
{
    // Optimising with only noise_variance trainable triggers the identity-tile
    // assembly path in the gradient computation for the noise parameter.
    const char *env_root = std::getenv("GPRAT_ROOT");
    const std::string root = env_root ? env_root : "../data";
    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);
    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);
    gprat::GP gp(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                 { false, false, true });
    gprat::start_hpx_runtime(0, nullptr);
    const gprat::AdamParams params{ 0.01, 0.9, 0.999, 1e-8, 3 };
    const auto losses = gp.optimize(params);
    gprat::stop_hpx_runtime();
    REQUIRE(losses.size() == 3);
    REQUIRE(std::isfinite(losses.back()));
}

// GPU tests (NVIDIA only) ////////////////////////////////////////////////////////////////////////
//
// Each test calls SKIP() immediately if GPRat was compiled without CUDA or if
// no NVIDIA device is detected at runtime, so they are safe to include in
// every build.  When GPRAT_WITH_CUDA=ON and a GPU is present the tests run
// in full and compare GPU results against the CPU reference.

namespace
{
// Returns the number of visible CUDA devices (0 when CUDA is absent).
int cuda_device_count()
{
#if GPRAT_WITH_CUDA
    int n = 0;
    cudaGetDeviceCount(&n);
    return n;
#else
    return 0;
#endif
}
}  // namespace

// Macro that skips the test if CUDA is unavailable or no GPU is present.
#define GPRAT_SKIP_IF_NO_GPU()                                                                     \
    do {                                                                                           \
        if (!gprat::compiled_with_cuda())                                                          \
            SKIP("GPRat not compiled with CUDA support");                                          \
        if (cuda_device_count() == 0)                                                              \
            SKIP("No NVIDIA GPU detected");                                                        \
    } while (false)

TEST_CASE("GP GPU constructor succeeds when GPU is present", "[gpu][cuda]")
{
    GPRAT_SKIP_IF_NO_GPU();

    const char *env_root = std::getenv("GPRAT_ROOT");
    const std::string root = env_root ? env_root : "../data";

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    // Should not throw when a real GPU is present.
    REQUIRE_NOTHROW(
        (gprat::GP(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                   { true, true, true }, 0, 1)));
}

TEST_CASE("GP::predict GPU matches CPU result", "[gpu][cuda]")
{
    GPRAT_SKIP_IF_NO_GPU();

    const char *env_root = std::getenv("GPRAT_ROOT");
    const std::string root = env_root ? env_root : "../data";

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8, n_test = 64;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);
    const auto [m_tiles, m_tile_size] = gprat::compute_test_tiles(n_test, n_tiles, tile_size);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);
    gprat::GP_data test_in(root + "/data_1024/test_input.txt", n_test, n_reg);

    gprat::GP gp_cpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                     { true, true, true });
    gprat::GP gp_gpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                     { true, true, true }, 0, 1);

    gprat::start_hpx_runtime(0, nullptr);
    const auto cpu_pred = gp_cpu.predict(test_in.data, m_tiles, m_tile_size);
    const auto gpu_pred = gp_gpu.predict(test_in.data, m_tiles, m_tile_size);
    gprat::stop_hpx_runtime();

    REQUIRE(cpu_pred.size() == n_test);
    REQUIRE(gpu_pred.size() == n_test);
    for (std::size_t i = 0; i < n_test; ++i)
        REQUIRE_THAT(gpu_pred[i], WithinRel(cpu_pred[i], 1e-4));
}

TEST_CASE("GP::predict_with_uncertainty GPU matches CPU result", "[gpu][cuda]")
{
    GPRAT_SKIP_IF_NO_GPU();

    const char *env_root = std::getenv("GPRAT_ROOT");
    const std::string root = env_root ? env_root : "../data";

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8, n_test = 64;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);
    const auto [m_tiles, m_tile_size] = gprat::compute_test_tiles(n_test, n_tiles, tile_size);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);
    gprat::GP_data test_in(root + "/data_1024/test_input.txt", n_test, n_reg);

    gprat::GP gp_cpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                     { true, true, true });
    gprat::GP gp_gpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                     { true, true, true }, 0, 1);

    gprat::start_hpx_runtime(0, nullptr);
    const auto cpu_unc = gp_cpu.predict_with_uncertainty(test_in.data, m_tiles, m_tile_size);
    const auto gpu_unc = gp_gpu.predict_with_uncertainty(test_in.data, m_tiles, m_tile_size);
    gprat::stop_hpx_runtime();

    // cpu_unc[0] = mean, cpu_unc[1] = variance
    REQUIRE(gpu_unc.size() == 2);
    REQUIRE(gpu_unc[0].size() == n_test);
    REQUIRE(gpu_unc[1].size() == n_test);
    for (std::size_t i = 0; i < n_test; ++i)
    {
        REQUIRE_THAT(gpu_unc[0][i], WithinRel(cpu_unc[0][i], 1e-4));
        REQUIRE_THAT(gpu_unc[1][i], WithinRel(cpu_unc[1][i], 1e-4));
    }
}

TEST_CASE("GP::predict_with_full_cov GPU matches CPU result", "[gpu][cuda]")
{
    GPRAT_SKIP_IF_NO_GPU();

    const char *env_root = std::getenv("GPRAT_ROOT");
    const std::string root = env_root ? env_root : "../data";

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8, n_test = 64;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);
    const auto [m_tiles, m_tile_size] = gprat::compute_test_tiles(n_test, n_tiles, tile_size);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);
    gprat::GP_data test_in(root + "/data_1024/test_input.txt", n_test, n_reg);

    gprat::GP gp_cpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                     { true, true, true });
    gprat::GP gp_gpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                     { true, true, true }, 0, 1);

    gprat::start_hpx_runtime(0, nullptr);
    const auto cpu_cov = gp_cpu.predict_with_full_cov(test_in.data, m_tiles, m_tile_size);
    const auto gpu_cov = gp_gpu.predict_with_full_cov(test_in.data, m_tiles, m_tile_size);
    gprat::stop_hpx_runtime();

    REQUIRE(gpu_cov.size() == 2);
    REQUIRE(gpu_cov[0].size() == n_test);
    for (std::size_t i = 0; i < n_test; ++i)
        REQUIRE_THAT(gpu_cov[0][i], WithinRel(cpu_cov[0][i], 1e-4));
}

TEST_CASE("GP::calculate_loss GPU matches CPU result", "[gpu][cuda]")
{
    GPRAT_SKIP_IF_NO_GPU();

    const char *env_root = std::getenv("GPRAT_ROOT");
    const std::string root = env_root ? env_root : "../data";

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp_cpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                     { true, true, true });
    gprat::GP gp_gpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                     { true, true, true }, 0, 1);

    gprat::start_hpx_runtime(0, nullptr);
    const double cpu_loss = gp_cpu.calculate_loss();
    const double gpu_loss = gp_gpu.calculate_loss();
    gprat::stop_hpx_runtime();

    REQUIRE(std::isfinite(gpu_loss));
    REQUIRE_THAT(gpu_loss, WithinRel(cpu_loss, 1e-4));
}

TEST_CASE("GP::cholesky GPU tile count matches CPU", "[gpu][cuda]")
{
    GPRAT_SKIP_IF_NO_GPU();

    const char *env_root = std::getenv("GPRAT_ROOT");
    const std::string root = env_root ? env_root : "../data";

    constexpr std::size_t n = 128, n_tiles = 4, n_reg = 8;
    const std::size_t tile_size = gprat::compute_train_tile_size(n, n_tiles);

    gprat::GP_data train_in(root + "/data_1024/training_input.txt", n, n_reg);
    gprat::GP_data train_out(root + "/data_1024/training_output.txt", n, n_reg);

    gprat::GP gp_cpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                     { true, true, true });
    gprat::GP gp_gpu(train_in.data, train_out.data, n_tiles, tile_size, n_reg, { 1.0, 1.0, 0.1 },
                     { true, true, true }, 0, 1);

    gprat::start_hpx_runtime(0, nullptr);
    const auto cpu_L = gp_cpu.cholesky();
    const auto gpu_L = gp_gpu.cholesky();
    gprat::stop_hpx_runtime();

    REQUIRE(gpu_L.size() == cpu_L.size());
    REQUIRE(gpu_L[0].size() == cpu_L[0].size());
    // Diagonal tiles of L should match CPU within tolerance
    for (std::size_t t = 0; t < n_tiles; ++t)
    {
        const std::size_t diag = t * n_tiles + t;
        for (std::size_t e = 0; e < tile_size * tile_size; ++e)
            REQUIRE_THAT(gpu_L[diag].data()[e], WithinRel(cpu_L[diag].data()[e], 1e-4));
    }
}

}  // namespace gprat::test
