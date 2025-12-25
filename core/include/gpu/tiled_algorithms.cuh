#ifndef GPU_TILED_ALGORITHMS_H
#define GPU_TILED_ALGORITHMS_H

#include "gp_hyperparameters.hpp"
#include "target.hpp"
#include <cusolverDn.h>
#include <gp_kernels.hpp>
#include <hpx/modules/async_cuda.hpp>

namespace gpu
{

// Tiled Cholesky Algorithm

/**
 * @brief Perform right-looking Cholesky decomposition.
 *
 * @param n_streams Number of CUDA streams.
 * @param ft_tiles Matrix represented as a vector of tiles, containing the
 *        covariance matrix, afterwards the Cholesky decomposition.
 * @param n_tile_size Tile size per dimension.
 * @param n_tiles Number of tiles per dimension.
 * @param gpu GPU target for computations.
 * @param cusolver cuSolver handle, already created.
 */
void right_looking_cholesky_tiled(std::vector<hpx::shared_future<double *>> &ft_tiles,
                                  const std::size_t n_tile_size,
                                  const std::size_t n_tiles,
                                  gprat::CUDA_GPU &gpu,
                                  const cusolverDnHandle_t &cusolver);
}  // end of namespace gpu

#endif  // end of GPU_TILED_ALGORITHMS_H
