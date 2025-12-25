#ifndef CPU_CHOLESKY_FACTOR_H
#define CPU_CHOLESKY_FACTOR_H

#include "gp_hyperparameter.hpp"
#include <hpx/future.hpp>

using Tiled_matrix = std::vector<hpx::shared_future<std::vector<double>>>;

namespace cpu
{

// Tiled Cholesky Algorithm

/**
 * @brief Perform right-looking tiled Cholesky decomposition.
 *
 * @param ft_tiles Tiled matrix represented as a vector of futurized tiles, containing the
 *        covariance matrix, afterwards the Cholesky decomposition.
 * @param N Tile size per dimension.
 * @param n_tiles Number of tiles per dimension.
 */
void right_looking_cholesky_tiled(Tiled_matrix &ft_tiles, int N, std::size_t n_tiles);
}  // end of namespace cpu
#endif  // end of CPU_CHOLESKY_FACTOR_H
