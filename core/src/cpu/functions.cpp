#include "cpu/functions.hpp"

#include "cpu/cholesky_factor.hpp"
#include "cpu/gp_kernel.hpp"
#include <hpx/algorithm.hpp>
#include <hpx/future.hpp>

using Tiled_matrix = std::vector<hpx::shared_future<std::vector<double>>>;
using Tiled_vector = std::vector<hpx::shared_future<std::vector<double>>>;

namespace cpu
{

std::vector<std::vector<double>> cholesky_asynchronous(
    std::string variant,
    const std::vector<double> &training_input,
    const SEKParams &sek_params,
    int n_tiles,
    int n_tile_size,
    int n_regressors)
{
    std::vector<std::vector<double>> result;
    // Tiled future data structures
    Tiled_matrix K_tiles;  // Tiled covariance matrix

    // Preallocate memory
    result.resize(static_cast<std::size_t>(n_tiles * n_tiles));
    K_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));  // No reserve because of triangular structure

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous assembly
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            K_tiles[i * static_cast<std::size_t>(n_tiles) + j] = hpx::async(
                hpx::annotated_function(gen_tile_covariance, "assemble_tiled_K"),
                i,
                j,
                static_cast<std::size_t>(n_tile_size),
                n_regressors,
                sek_params,
                training_input);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled(to_variant(variant), K_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Synchronize
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            result[i * static_cast<std::size_t>(n_tiles) + j] =
                K_tiles[i * static_cast<std::size_t>(n_tiles) + j].get();
        }
    }
    return result;
}

std::vector<std::vector<double>> cholesky_synchronous(
    std::string variant,
    const std::vector<double> &training_input,
    const SEKParams &sek_params,
    int n_tiles,
    int n_tile_size,
    int n_regressors)
{
    std::vector<std::vector<double>> result;
    // Tiled future data structures
    Tiled_matrix K_tiles;  // Tiled covariance matrix

    // Preallocate memory
    result.resize(static_cast<std::size_t>(n_tiles * n_tiles));
    K_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));  // No reserve because of triangular structure

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous assembly
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            K_tiles[i * static_cast<std::size_t>(n_tiles) + j] = hpx::async(
                hpx::annotated_function(gen_tile_covariance, "assemble_tiled_K"),
                i,
                j,
                static_cast<std::size_t>(n_tile_size),
                n_regressors,
                sek_params,
                training_input);
        }
    }
    // Synchronize
    hpx::wait_all(K_tiles);
    ///////////////////////////////////////////////////////////////////////////
    // Launch synchronous Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled(to_variant(variant), K_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Synchronize
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            result[i * static_cast<std::size_t>(n_tiles) + j] =
                K_tiles[i * static_cast<std::size_t>(n_tiles) + j].get();
        }
    }
    return result;
}

std::vector<std::vector<double>>
cholesky_loop(std::string variant,
              const std::vector<double> &training_input,
              const SEKParams &sek_params,
              int n_tiles,
              int n_tile_size,
              int n_regressors)
{
    // Tiled data structures
    std::vector<std::vector<double>> K_tiles;  // Tiled covariance matrix

    // Preallocate memory
    K_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));  // No reserve because of triangular structure

    ///////////////////////////////////////////////////////////////////////////
    // Launch synchronous assembly
    hpx::experimental::for_loop(
        hpx::execution::par,
        std::size_t{ 0 },
        std::size_t(n_tiles),
        [&](std::size_t i)
        {
            hpx::experimental::for_loop(
                hpx::execution::par,
                std::size_t{ 0 },
                i + 1,
                [&](std::size_t j)
                {
                    K_tiles[i * std::size_t(n_tiles) + j] = gen_tile_covariance(
                        i,
                        j,
                        static_cast<std::size_t>(n_tile_size),
                        static_cast<std::size_t>(n_regressors),
                        sek_params,
                        training_input);
                });
        });

    ///////////////////////////////////////////////////////////////////////////
    // Launch synchronous Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled_loop(to_variant(variant), K_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Synchronize
    return K_tiles;
}

std::vector<std::vector<double>>
cholesky_mutable(const std::vector<double> &training_input,
                 const SEKParams &sek_params,
                 std::size_t n_tiles,
                 std::size_t n_tile_size,
                 std::size_t n_regressors)
{
    // Tiled covariance matrix K_NxN
    auto K_tiles = std::vector<hpx::shared_future<mutable_tile_data<double>>>{ n_tiles * n_tiles };
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            K_tiles[i * static_cast<std::size_t>(n_tiles) + j] = hpx::async(
                hpx::annotated_function(gen_mutable_tile_covariance, "assemble_tiled_K"),
                i,
                j,
                static_cast<std::size_t>(n_tile_size),
                n_regressors,
                sek_params,
                training_input);
        }
    }
    // Launch asynchronous Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled_mutable(K_tiles, n_tile_size, n_tiles);

    ///////////////////////////////////////////////////////////////////////////
    // Synchronize
    std::vector<std::vector<double>> result(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            auto tile = K_tiles[i * n_tiles + j].get();
            result[i * n_tiles + j] = std::vector(tile.begin(), tile.end());
        }
    }
    return result;
}

}  // end of namespace cpu
