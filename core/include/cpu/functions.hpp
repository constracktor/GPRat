#ifndef CPU_FUNCTIONS_H
#define CPU_FUNCTIONS_H

#include "gp_hyperparameter.hpp"
#include <vector>
#include <string>
namespace cpu
{

/**
 * @brief Perform future-based asynchronous Cholesky decompositon (+ Assebmly)
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
cholesky_asynchronous(std::string variant,
        const std::vector<double> &training_input,
         const SEKParams &sek_params,
         int n_tiles,
         int n_tile_size,
         int n_regressors);

/**
 * @brief Perform future-based synchronous Cholesky decompositon (+ Assebmly)
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
cholesky_synchronous(std::string variant,
        const std::vector<double> &training_input,
         const SEKParams &sek_params,
         int n_tiles,
         int n_tile_size,
         int n_regressors);

/**
 * @brief Perform loop-based synchronous Cholesky decompositon (+ Assebmly) without futures
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
cholesky_loop(std::string variant,
        const std::vector<double> &training_input,
         const SEKParams &sek_params,
         int n_tiles,
         int n_tile_size,
         int n_regressors);
}
#endif  // end of CPU_FUNCTIONS_H
