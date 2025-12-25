#ifndef GPU_GP_ALGORITHMS_H
#define GPU_GP_ALGORITHMS_H

#include "gp_kernels.hpp"
#include "target.hpp"
#include <hpx/future.hpp>
#include <vector>

namespace gpu
{

/**
 * @brief Generate a tile of the covariance matrix
 *
 * @param input The input data vector
 * @param row The row index of the tile in the tiled matrix
 * @param col The column index of the tile in the tiled matrix
 * @param N The dimension of the quadratic tile (N*N elements)
 * @param n_regressors The number of regressors
 * @param sek_params The kernel hyperparameters
 * @param gpu GPU target for computations
 *
 * @return A quadratic tile of the covariance matrix of size N x N
 * @note Does apply noise variance on the diagonal
 */
double *gen_tile_covariance(const double *d_input,
                            const std::size_t tile_row,
                            const std::size_t tile_column,
                            const std::size_t n_tile_size,
                            const std::size_t n_regressors,
                            const gprat_hyper::SEKParams sek_params,
                            gprat::CUDA_GPU &gpu);
/**
 * @brief Allocates the tiled covariance matrix on the device given the training
 *        data.
 *
 * @param d_training_input The training input data
 * @param n_tiles The number of tiles per dimension
 * @param n_tile_size The size of the tile
 * @param n_regressors The number of regressors
 * @param sek_params The kernel hyperparameters
 * @param gpu GPU target for computations
 */
std::vector<hpx::shared_future<double *>> assemble_tiled_covariance_matrix(
    const double *d_training_input,
    const std::size_t n_tiles,
    const std::size_t n_tile_size,
    const std::size_t n_regressors,
    const gprat_hyper::SEKParams sek_params,
    gprat::CUDA_GPU &gpu);

/**
 * @brief Allocates the tiled covariance matrix on the device given the training
 *        data.
 *
 * @param d_training_input The training input data
 * @param n_tile_size The size of the tile
 * @param n_tiles The number of tiles per dimension
 * @param gpu GPU target for computations
 */
std::vector<double> copy_tiled_vector_to_host_vector(std::vector<hpx::shared_future<double *>> &d_tiles,
                                                     std::size_t n_tile_size,
                                                     std::size_t n_tiles,
                                                     gprat::CUDA_GPU &gpu);

/**
 * @brief Moves lower triangular tiles of the covariance matrix to the host.
 *
 * Allocates host memory for the tiles on the host and free the device memory.
 *
 * @param d_tiles The tiles on the device
 * @param n_tile_size The size of the tile
 * @param n_tiles The number of tiles
 * @param gpu GPU target for computations
 */
std::vector<std::vector<double>> move_lower_tiled_matrix_to_host(
    const std::vector<hpx::shared_future<double *>> &d_tiles,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gprat::CUDA_GPU &gpu);

/**
 * @brief Frees the device memory of the lower triangular tiles of the covariance matrix.
 *
 * @param d_tiles The tiles on the device
 * @param n_tiles The number of tiles
 */
void free_lower_tiled_matrix(const std::vector<hpx::shared_future<double *>> &d_tiles, const std::size_t n_tiles);

}  // end of namespace gpu

#endif  // end of GPU_GP_ALGORITHMS_H
