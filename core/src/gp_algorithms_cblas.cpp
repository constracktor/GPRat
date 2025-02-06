#include "../include/gp_algorithms_cpu.hpp"

/**
 * @brief Compute the squared exponential kernel of two feature vectors.
 *
 * @param i_global  global index of the first feature vector
 * @param j_global  global index of the second feature vector
 * @param n_regressors  number of regressors
 * @param hyperparameters  hyperparameters of the covariance function: expects
 *        lengthscale in hyperparameters[0]
 * @param i_input  first feature vector
 * @param j_input  second feature vector
 *
 * @return the covariance function value
 */
double compute_covariance_function(std::size_t i_global,
                                   std::size_t j_global,
                                   std::size_t n_regressors,
                                   const std::vector<double> &hyperparameters,
                                   const std::vector<double> &i_input,
                                   const std::vector<double> &j_input)
{
    // formula in papers:
    // C(z_i,z_j) = vertical_lengthscale * exp(-0.5/lengthscale^2*(z_i-z_j)^2)

    double lengthscale = hyperparameters[0];
    double vertical_lengthscale = hyperparameters[1];
    double distance = 0.0;
    double z_ik_minus_z_jk;

    for (std::size_t k = 0; k < n_regressors; k++)
    {
        z_ik_minus_z_jk = i_input[i_global + k] - j_input[j_global + k];
        distance += z_ik_minus_z_jk * z_ik_minus_z_jk;
    }
    return vertical_lengthscale * exp(-0.5 / (lengthscale * lengthscale) * distance);
}

/**
 * @brief Generate a tile of the covariance matrix
 *
 * @param row row index of the tile
 * @param col column index of the tile
 * @param N size of the tile
 * @param n_regressors number of regressors
 * @param hyperparameters hyperparameters of the covariance function
 * @param input input data
 */
std::vector<double> gen_tile_covariance(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const std::vector<double> &hyperparameters,
    const std::vector<double> &input)
{
    std::size_t i_global, j_global;
    double covariance_function;
    double noise_variance = hyperparameters[2];
    // Initialize tile
    std::vector<double> tile;
    tile.resize(N * N);
    for (std::size_t i = 0; i < N; i++)
    {
        i_global = N * row + i;
        for (std::size_t j = 0; j < N; j++)
        {
            j_global = N * col + j;
            // compute covariance function
            covariance_function =
                compute_covariance_function(i_global, j_global, n_regressors, hyperparameters, input, input);
            if (i_global == j_global)
            {
                // noise variance on diagonal
                covariance_function += noise_variance;
            }
            tile[i * N + j] = covariance_function;
        }
    }
    return tile;
}

// generate a tile of the prior covariance matrix
std::vector<double> gen_tile_full_prior_covariance(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const std::vector<double> &hyperparameters,
    const std::vector<double> &input)
{
    std::size_t i_global, j_global;
    double covariance_function;
    // Initialize tile
    std::vector<double> tile;
    tile.resize(N * N);
    for (std::size_t i = 0; i < N; i++)
    {
        i_global = N * row + i;
        for (std::size_t j = 0; j < N; j++)
        {
            j_global = N * col + j;
            // compute covariance function
            covariance_function =
                compute_covariance_function(i_global, j_global, n_regressors, hyperparameters, input, input);
            tile[i * N + j] = covariance_function;
        }
    }
    return tile;
}

// generate a tile of the prior covariance matrix
std::vector<double> gen_tile_prior_covariance(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const std::vector<double> &hyperparameters,
    const std::vector<double> &input)
{
    std::size_t i_global, j_global;
    double covariance_function;
    // Initialize tile
    std::vector<double> tile;
    tile.resize(N);
    for (std::size_t i = 0; i < N; i++)
    {
        i_global = N * row + i;
        j_global = N * col + i;
        // compute covariance function
        covariance_function =
            compute_covariance_function(i_global, j_global, n_regressors, hyperparameters, input, input);
        tile[i] = covariance_function;
    }
    return tile;
}

// generate a tile of the cross-covariance matrix
std::vector<double> gen_tile_cross_covariance(
    std::size_t row,
    std::size_t col,
    std::size_t N_row,
    std::size_t N_col,
    std::size_t n_regressors,
    const std::vector<double> &hyperparameters,
    const std::vector<double> &row_input,
    const std::vector<double> &col_input)
{
    std::size_t i_global, j_global;
    double covariance_function;
    // Initialize tile
    std::vector<double> tile;
    tile.resize(N_row * N_col);
    for (std::size_t i = 0; i < N_row; i++)
    {
        i_global = N_row * row + i;
        for (std::size_t j = 0; j < N_col; j++)
        {
            j_global = N_col * col + j;
            // compute covariance function
            covariance_function =
                compute_covariance_function(i_global, j_global, n_regressors, hyperparameters, row_input, col_input);
            tile[i * N_col + j] = covariance_function;
        }
    }
    return tile;
}

// generate a tile of the cross-covariance matrix
std::vector<double>
gen_tile_cross_cov_T(std::size_t N_row, std::size_t N_col, const std::vector<double> &cross_covariance_tile)
{
    std::vector<double> transposed;
    transposed.resize(N_row * N_col);
    for (std::size_t i = 0; i < N_row; ++i)
    {
        for (std::size_t j = 0; j < N_col; j++)
        {
            transposed[j * N_row + i] = cross_covariance_tile[i * N_col + j];
        }
    }
    return transposed;
}

// generate a tile containing the output observations
std::vector<double> gen_tile_output(std::size_t row, std::size_t N, const std::vector<double> &output)
{
    std::size_t i_global;
    // Initialize tile
    std::vector<double> tile;
    tile.resize(N);
    for (std::size_t i = 0; i < N; i++)
    {
        i_global = N * row + i;
        tile[i] = output[i_global];
    }
    return tile;
}

// compute the total 2-norm error
double compute_error_norm(std::size_t n_tiles,
                          std::size_t tile_size,
                          const std::vector<double> &b,
                          const std::vector<std::vector<double>> &tiles)
{
    double error = 0.0;
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        auto a = tiles[k];
        for (std::size_t i = 0; i < tile_size; i++)
        {
            std::size_t i_global = tile_size * k + i;
            // ||a - b||_2
            error += (b[i_global] - a[i]) * (b[i_global] - a[i]);
        }
    }
    return sqrt(error);
}

// generate an empty tile
std::vector<double> gen_tile_zeros(std::size_t N)
{
    // Initialize tile
    std::vector<double> tile;
    tile.resize(N);
    std::fill(tile.begin(), tile.end(), 0.0);
    return tile;
}
