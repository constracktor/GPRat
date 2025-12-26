#ifndef CPU_FUNCTIONS_H
#define CPU_FUNCTIONS_H

#include "gp_hyperparameter.hpp"
#include <vector>

namespace cpu
{

/**
 * @brief Perform asynchronous Cholesky decompositon (+ Assebmly)
 *
 * @param training_input The training input data
 * @param hyperparameters The kernel hyperparameters
 *
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param n_regressors The number of regressors
 *
 * @return The tiled Cholesky factor
 */
std::vector<std::vector<double>>
cholesky(const std::vector<double> &training_input,
         const SEKParams &sek_params,
         int n_tiles,
         int n_tile_size,
         int n_regressors);

/**
 * @brief Perform synchronous Cholesky decompositon (+ Assebmly)
 *
 * @param training_input The training input data
 * @param hyperparameters The kernel hyperparameters
 *
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param n_regressors The number of regressors
 *
 * @return The tiled Cholesky factor
 */
std::vector<std::vector<double>>
cholesky_synchronous(const std::vector<double> &training_input,
         const SEKParams &sek_params,
         int n_tiles,
         int n_tile_size,
         int n_regressors);

/**
 * @brief Perform call-by-reference synchronous Cholesky decompositon (+ Assebmly) without futures
 *
 * @param training_input The training input data
 * @param hyperparameters The kernel hyperparameters
 *
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param n_regressors The number of regressors
 *
 * @return The tiled Cholesky factor
 */
std::vector<std::vector<double>>
cholesky_loop_ref(const std::vector<double> &training_input,
         const SEKParams &sek_params,
         int n_tiles,
         int n_tile_size,
         int n_regressors);

/**
 * @brief Perform call-by-value synchronous Cholesky decompositon (+ Assebmly) without futures
 *
 * @param training_input The training input data
 * @param hyperparameters The kernel hyperparameters
 *
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param n_regressors The number of regressors
 *
 * @return The tiled Cholesky factor
 */
std::vector<std::vector<double>>
cholesky_loop_val(const std::vector<double> &training_input,
         const SEKParams &sek_params,
         int n_tiles,
         int n_tile_size,
         int n_regressors);
}
#endif  // end of CPU_FUNCTIONS_H
