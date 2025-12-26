#ifndef CPU_CHOLESKY_FACTOR_H
#define CPU_CHOLESKY_FACTOR_H

#include "gp_hyperparameter.hpp"
#include <hpx/future.hpp>
#include <string>

using Tiled_matrix = std::vector<hpx::shared_future<std::vector<double>>>;

namespace cpu
{
enum class Variant
{
    async_future,
    async_ref,
    async_val,
    sync_future,
    sync_ref,
    sync_val,
    loop_ref,
    loop_val
};

inline Variant to_variant(std::string s)
{
    if (s == "async_future") return Variant::async_future;
    if (s == "async_ref")    return Variant::async_ref;
    if (s == "async_val")    return Variant::async_val;

    if (s == "sync_future")  return Variant::sync_future;
    if (s == "sync_ref")     return Variant::sync_ref;
    if (s == "sync_val")     return Variant::sync_val;

    if (s == "loop_ref")     return Variant::loop_ref;
    if (s == "loop_val")     return Variant::loop_val;

    throw std::invalid_argument("Unknown Variant: " + std::string(s));
}

// Tiled Cholesky Algorithm

/**
 * @brief Perform dataflow based right-looking tiled Cholesky decomposition.
 *
 * @param ft_tiles Tiled matrix represented as a vector of futurized tiles, containing the
 *        covariance matrix, afterwards the Cholesky decomposition.
 * @param N Tile size per dimension.
 * @param n_tiles Number of tiles per dimension.
 */
void right_looking_cholesky_tiled(Variant variant, Tiled_matrix &ft_tiles, int N, std::size_t n_tiles);

/**
 * @brief Performs future-free right-looking tiled Cholesky decomposition.
 *
 * @param ft_tiles Tiled matrix represented as a vector oftiles, containing the
 *        covariance matrix, afterwards the Cholesky decomposition.
 * @param N Tile size per dimension.
 * @param n_tiles Number of tiles per dimension.
 */
void right_looking_cholesky_tiled_loop(Variant variant, std::vector<std::vector<double>> &tiles, int N, std::size_t n_tiles);

}  // end of namespace cpu
#endif  // end of CPU_CHOLESKY_FACTOR_H
