#ifndef CPU_GP_KERNEL_H
#define CPU_GP_KERNEL_H

#include "gp_hyperparameter.hpp"
#include "tile_data.hpp"
#include <vector>

namespace cpu
{

/**
 * @brief Compute the squared exponential kernel of two feature vectors
 *
 * @param i_global The global index of the first feature vector
 * @param j_global The global index of the second feature vector
 * @param n_regressors The number of regressors
 * @param hyperparameters The kernel hyperparameters
 * @param i_input The first feature vector
 * @param j_input The second feature vector
 *
 * @return The entry of a covariance function at position i_global,j_global
 */
double compute_covariance_function(std::size_t i_global,
                                   std::size_t j_global,
                                   std::size_t n_regressors,
                                   const SEKParams &sek_params,
                                   const std::vector<double> &i_input,
                                   const std::vector<double> &j_input);

/**
 * @brief Generate a tile of the covariance matrix
 *
 * @param input The input data vector
 * @param row The row index of the tile in the tiled matrix
 * @param col The column index of the tile in the tiled matrix
 * @param N The dimension of the quadratic tile (N*N elements)
 * @param n_regressors The number of regressors
 * @param sek_params The kernel hyperparameters
 *
 * @return A quadratic tile of the covariance matrix of size N x N
 * @note Does apply noise variance on the diagonal
 */
std::vector<double> gen_tile_covariance(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    const std::vector<double> &input);

/**
 * @brief Generate a tile of the covariance matrix
 *
 * @param input The input data vector
 * @param row The row index of the tile in the tiled matrix
 * @param col The column index of the tile in the tiled matrix
 * @param N The dimension of the quadratic tile (N*N elements)
 * @param n_regressors The number of regressors
 * @param sek_params The kernel hyperparameters
 *
 * @return A quadratic tile of the covariance matrix of size N x N
 * @note Does apply noise variance on the diagonal
 */
mutable_tile_data<double> gen_mutable_tile_covariance(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    const std::vector<double> &input);

}  // end of namespace cpu

#endif  // end of CPU_GP_KERNEL_H
