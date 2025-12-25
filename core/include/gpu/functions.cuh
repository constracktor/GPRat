#ifndef GPU_FUNCTIONS_H
#define GPU_FUNCTIONS_H

#include "gp_hyperparameter.hpp"
#include "target.hpp"

namespace gpu
{
/**
 * @brief Perform Cholesky decompositon (+ Assembly)
 *
 * @param training_input The training input data
 * @param hyperparameters The kernel hyperparameters
 *
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param n_regressors The number of regressors
 *
 * @param gpu GPU target for computations
 *
 * @return The tiled Cholesky factor
 */
std::vector<std::vector<double>>
cholesky(const std::vector<double> &training_input,
         const SEKParams &sek_params,
         int n_tiles,
         int n_tile_size,
         int n_regressors,
         gprat::CUDA_GPU &gpu);

}  // end of namespace gpu

#endif
