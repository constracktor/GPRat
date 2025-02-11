#include "../include/gp_algorithms_cpu.hpp"
#include <cmath>

double compute_covariance_function(std::size_t i_global,
                                   std::size_t j_global,
                                   std::size_t n_regressors,
                                   const gprat_hyper::SEKParams &sek_params,
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
    const gprat_hyper::SEKParams &sek_params,
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

std::vector<double> gen_tile_full_prior_covariance(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const gprat_hyper::SEKParams &sek_params,
    const std::vector<double> &input)
{
    std::size_t i_global, j_global;
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
            tile.push_back(
                compute_covariance_function(i_global, j_global, n_regressors, sek_params, input, input));
        }
    }
    return tile;
}

std::vector<double> gen_tile_prior_covariance(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const gprat_hyper::SEKParams &sek_params,
    const std::vector<double> &input)
{
    std::size_t i_global, j_global;
    // Preallocate required memory
    std::vector<double> tile;
    tile.reserve(N);
    // Compute entries
    for (std::size_t i = 0; i < N; i++)
    {
        i_global = N * row + i;
        j_global = N * col + i;
        // compute covariance function
        tile.push_back(compute_covariance_function(i_global, j_global, n_regressors, sek_params, input, input));
    }
    return tile;
}

std::vector<double> gen_tile_cross_covariance(
    std::size_t row,
    std::size_t col,
    std::size_t N_row,
    std::size_t N_col,
    std::size_t n_regressors,
    const gprat_hyper::SEKParams &sek_params,
    const std::vector<double> &row_input,
    const std::vector<double> &col_input)
{
    std::size_t i_global, j_global;
    // Preallocate required memory
    std::vector<double> tile;
    tile.reserve(N_row * N_col);
    // Compute entries
    for (std::size_t i = 0; i < N_row; i++)
    {
        i_global = N_row * row + i;
        for (std::size_t j = 0; j < N_col; j++)
        {
            j_global = N_col * col + j;
            // compute covariance function
            tile.push_back(
                compute_covariance_function(i_global, j_global, n_regressors, sek_params, row_input, col_input));
        }
    }
    return tile;
}

std::vector<double> gen_tile_transpose(std::size_t N_row, std::size_t N_col, const std::vector<double> &tile)
{
    // Preallocate required memory
    std::vector<double> transposed;
    transposed.reserve(N_row * N_col);
    // Transpose entries
    for (std::size_t j = 0; j < N_col; j++)
    {
        for (std::size_t i = 0; i < N_row; ++i)
        {
            // Mapping (i, j) in the original tile to (j, i) in the transposed tile
            transposed.push_back(tile[i * N_col + j]);
        }
    }
    return transposed;
}

std::vector<double> gen_tile_output(std::size_t row, std::size_t N, const std::vector<double> &output)
{
    // Preallocate required memory
    std::vector<double> tile;
    tile.reserve(N);
    // Copy entries
    std::copy(output.begin() + static_cast<long int>(N * row),
              output.begin() + static_cast<long int>(N * (row + 1)),
              std::back_inserter(tile));
    return tile;
}

double compute_error_norm(std::size_t n_tiles,
                          std::size_t tile_size,
                          const std::vector<double> &b,
                          const std::vector<std::vector<double>> &tiles)
{
    double error = 0.0;
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        const std::vector<double> &a = tiles[k];
        for (std::size_t i = 0; i < tile_size; i++)
        {
            std::size_t i_global = tile_size * k + i;
            // ||a - b||_2
            error += (b[i_global] - a[i]) * (b[i_global] - a[i]);
        }
    }
    return sqrt(error);
}

std::vector<double> gen_tile_zeros(std::size_t N) { return std::vector<double>(N, 0.0); }

std::vector<double> gen_tile_identity(std::size_t N)
{
    // Initialize zero tile
    std::vector<double> tile(N * N, 0.0);
    // Fill diagonal with ones
    for (std::size_t i = 0; i < N; i++)
    {
        tile[i * N + i] = 1.0;
    }
    return tile;
}
