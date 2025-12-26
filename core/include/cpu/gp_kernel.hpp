#ifndef CPU_GP_KERNEL_H
#define CPU_GP_KERNEL_H

#include "gp_hyperparameter.hpp"
#include "tile_data.hpp"
//#include <cmath>
//#include <span>
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


// /**
//  * @brief Compute the squared exponential kernel of two feature vectors
//  *
//  * @param n_regressors The number of regressors
//  * @param sek_params The kernel hyperparameters
//  * @param i_input The first feature vector
//  * @param j_input The second feature vector
//  *
//  * @return The entry of a covariance function
//  */
// double compute_covariance_function_span(std::size_t n_regressors,
//                                    const SEKParams &sek_params,
//                                    std::span<const double> i_input,
//                                    std::span<const double> j_input){    // k(z_i,z_j) = vertical_lengthscale * exp(-0.5 / lengthscale^2 * (z_i - z_j)^2)
//     double distance = 0.0;
//     for (std::size_t k = 0; k < n_regressors; k++)
//     {
//         const double z_ik_minus_z_jk = i_input[k] - j_input[k];
//         distance += z_ik_minus_z_jk * z_ik_minus_z_jk;
//     }
//
//     return sek_params.vertical_lengthscale * exp(-0.5 / (sek_params.lengthscale * sek_params.lengthscale) * distance);}

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
    //std::span<const double> input);

}  // end of namespace cpu

#endif  // end of CPU_GP_KERNEL_H
