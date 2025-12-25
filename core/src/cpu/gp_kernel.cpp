#include "cpu/gp_kernel.hpp"

#include <cmath>

namespace cpu
{

// Tile generation

double compute_covariance_function(std::size_t i_global,
                                   std::size_t j_global,
                                   std::size_t n_regressors,
                                   const SEKParams &sek_params,
                                   const std::vector<double> &i_input,
                                   const std::vector<double> &j_input)
{
    // k(z_i,z_j) = vertical_lengthscale * exp(-0.5 / lengthscale^2 * (z_i - z_j)^2)
    double distance = 0.0;
    double z_ik_minus_z_jk;

    for (std::size_t k = 0; k < n_regressors; k++)
    {
        z_ik_minus_z_jk = i_input[i_global + k] - j_input[j_global + k];
        distance += z_ik_minus_z_jk * z_ik_minus_z_jk;
    }
    return sek_params.vertical_lengthscale * exp(-0.5 / (sek_params.lengthscale * sek_params.lengthscale) * distance);
}

std::vector<double> gen_tile_covariance(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const SEKParams &sek_params,
    const std::vector<double> &input)
{
    std::size_t i_global, j_global;
    double covariance_function;
    // Preallocate required memory
    std::vector<double> tile;
    tile.reserve(N * N);
    // Compute entries
    for (std::size_t i = 0; i < N; i++)
    {
        i_global = N * row + i;
        for (std::size_t j = 0; j < N; j++)
        {
            j_global = N * col + j;
            // compute covariance function
            covariance_function =
                compute_covariance_function(i_global, j_global, n_regressors, sek_params, input, input);
            if (i_global == j_global)
            {
                // noise variance on diagonal
                covariance_function += sek_params.noise_variance;
            }
            tile.push_back(covariance_function);
        }
    }
    return tile;
}
}  // end of namespace cpu
