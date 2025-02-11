#include "../include/gp_optimizer.hpp"

#include "../include/adapter_cblas_fp64.hpp"
#include <numeric>

///////////////////////////////////////////////////////////
// Parameter constraints
double to_constrained(double parameter, bool noise)
{
    if (noise)
    {
        return log(1.0 + exp(parameter)) + 1e-6;
    }
    else
    {
        return log(1.0 + exp(parameter));
    }
}

double to_unconstrained(double parameter, bool noise)
{
    if (noise)
    {
        return log(exp(parameter - 1e-6) - 1.0);
    }
    else
    {
        return log(exp(parameter) - 1.0);
    }
}

double compute_sigmoid(double parameter) { return 1.0 / (1.0 + exp(-parameter)); }

/////////////////////////////////////////////////////////
// Tile generation
double compute_covariance_dist_func(
    std::size_t i_global,
    std::size_t j_global,
    std::size_t n_regressors,
    const gprat_hyper::SEKParams &sek_params,
    const std::vector<double> &i_input,
    const std::vector<double> &j_input)
{
    // -0.5*lengthscale^2*(z_i-z_j)^2
    double distance = 0.0;
    double z_ik_minus_z_jk;

    for (std::size_t k = 0; k < n_regressors; k++)
    {
        z_ik_minus_z_jk = i_input[i_global + k] - j_input[j_global + k];
        distance += z_ik_minus_z_jk * z_ik_minus_z_jk;
    }
    return -0.5 / (sek_params.lengthscale * sek_params.lengthscale) * distance;
}

std::vector<double> compute_cov_dist_vec(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const gprat_hyper::SEKParams &sek_params,
    const std::vector<double> &input)
{
    std::size_t i_global, j_global;
    // Preallocate memory
    std::vector<double> tile;
    tile.reserve(N * N);
    for (std::size_t i = 0; i < N; i++)
    {
        i_global = N * row + i;
        for (std::size_t j = 0; j < N; j++)
        {
            j_global = N * col + j;
            // compute covariance function
            tile.push_back(
                compute_covariance_dist_func(i_global, j_global, n_regressors, sek_params, input, input));
        }
    }
    return tile;
}

std::vector<double> gen_tile_covariance_opt(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    const gprat_hyper::SEKParams &sek_params,
    const std::vector<double> &cov_dists)
{
    std::size_t i_global, j_global;
    double covariance;
    // Preallocate required memory
    std::vector<double> tile;
    tile.reserve(N * N);
    for (std::size_t i = 0; i < N; i++)
    {
        i_global = N * row + i;
        for (std::size_t j = 0; j < N; j++)
        {
            j_global = N * col + j;
            // compute covariance function
            covariance = sek_params.vertical_lengthscale * exp(cov_dists[i * N + j]);
            if (i_global == j_global)
            {
                // noise variance on diagonal
                covariance += sek_params.noise_variance;
            }
            tile.push_back(covariance);
        }
    }
    return tile;
}

std::vector<double>
gen_tile_grad_v(std::size_t N, const gprat_hyper::SEKParams &sek_params, const std::vector<double> &cov_dists)
{
    // Preallocate required memory
    std::vector<double> tile;
    tile.reserve(N * N);
    double hyperparam_der = compute_sigmoid(to_unconstrained(sek_params.vertical_lengthscale, false));
    for (std::size_t i = 0; i < N; i++)
    {
        for (std::size_t j = 0; j < N; j++)
        {
            // compute derivative
            tile.push_back(exp(cov_dists[i * N + j]) * hyperparam_der);
        }
    }
    return tile;
}

std::vector<double>
gen_tile_grad_l(std::size_t N, const gprat_hyper::SEKParams &sek_params, const std::vector<double> &cov_dists)
{
    // Preallocate required memory
    std::vector<double> tile;
    tile.reserve(N * N);
    double hyperparam_der = compute_sigmoid(to_unconstrained(sek_params.lengthscale, false));
    double factor = -2.0 * sek_params.vertical_lengthscale / sek_params.lengthscale;
    for (std::size_t i = 0; i < N; i++)
    {
        for (std::size_t j = 0; j < N; j++)
        {
            // compute derivative
            tile.push_back( factor * cov_dists[i * N + j] * exp(cov_dists[i * N + j]) * hyperparam_der);
        }
    }
    return tile;
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// adam stuff
/**
 * @brief Compute hyper-parameter beta_1 or beta_2 to power t.
 */
double gen_beta_T(int t, double parameter)
{
    return pow(parameter, t);
}

/**
 * @brief Update biased first raw moment estimate.
 */
double update_first_moment(double gradient, double m_T, double beta_1)
{
    return beta_1 * m_T + (1.0 - beta_1) * gradient;
}

/**
 * @brief Update biased second raw moment estimate.
 */
double update_second_moment(double gradient, double v_T, double beta_2)
{
    return beta_2 * v_T + (1.0 - beta_2) * gradient * gradient;
}

/**
 * @brief return zero - used to initialize moment vectors
 */
double gen_zero() { return 0.0; }

/**
 * @brief Update hyperparameter using gradient decent.
 */
double update_param(const double unconstrained_hyperparam,
                    const std::vector<double> &hyperparameters,
                    double m_T,
                    double v_T,
                    const std::vector<double> &beta1_T,
                    const std::vector<double> &beta2_T,
                    std::size_t iter)
{
    // Option 1:
    // double mhat = m_T / (1.0 - beta1_T[iter]);
    // double vhat = v_T / (1.0 - beta2_T[iter]);
    // return unconstrained_hyperparam - hyperparameters[3] * mhat / (sqrt(vhat)
    // + hyperparameters[6]);

    // Option 2:
    double alpha_T = hyperparameters[3] * sqrt(1.0 - beta2_T[iter]) / (1.0 - beta1_T[iter]);
    return unconstrained_hyperparam - alpha_T * m_T / (sqrt(v_T) + hyperparameters[6]);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// Losses
double compute_loss(const std::vector<double> &K_diag_tile,
                    const std::vector<double> &alpha_tile,
                    const std::vector<double> &y_tile,
                    std::size_t N)
{
    // l = y^T * alpha + \sum_i^N log(L_ii^2)
    double l;
    // Compute y^T * alpha
    l = dot(y_tile, alpha_tile, static_cast<int>(N));
    // Compute \sum_i^N log(L_ii^2)
    for (std::size_t i = 0; i < N; i++)
    {
        double diag_value = K_diag_tile[i * N + i];
        l += log(diag_value * diag_value);
    }
    return l;
}

double add_losses(const std::vector<double> &losses, std::size_t N, std::size_t n_tiles)
{
    // 0.5 * \sum losses + const
    double l = 0.0;
    double Nn = static_cast<double>(N * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        // Add the squared difference to the error
        l += losses[i];
    }

    l += Nn * log(2.0 * M_PI);
    return 0.5 * l / Nn;  // why /Nn?
}

/////////////////////////////////////////////////////////////
// Gradient stuff tbd.
/**
 * @brief
 */
double compute_gradient(double trace, double dot, std::size_t N, std::size_t n_tiles)
{
    return 0.5 / static_cast<double>(N * n_tiles) * (trace - dot);
}

double compute_trace(const std::vector<double> &diagonal, double trace)
{
    return trace + std::reduce(diagonal.begin(), diagonal.end());
}

double
compute_dot(const std::vector<double> &vector_T, const std::vector<double> &vector, double result)
{
    return result + dot(vector_T, vector, static_cast<int>(vector.size()));
}

/**
 * @brief Compute trace for noise variance.
 *
 * Same function as compute_trace with() the only difference that we only use
 * diag tiles multiplied by derivative of noise_variance.
 */
double compute_gradient_noise(const std::vector<std::vector<double>> &ft_tiles,
                              const std::vector<double> &hyperparameters,
                              std::size_t N,
                              std::size_t n_tiles)
{
    // Initialize tile
    double trace = 0.0;
    double hyperparam_der = compute_sigmoid(to_unconstrained(hyperparameters[2], true));
    for (std::size_t d = 0; d < n_tiles; d++)
    {
        auto tile = ft_tiles[d * n_tiles + d];
        for (std::size_t i = 0; i < N; ++i)
        {
            trace += (tile[i * N + i] * hyperparam_der);
        }
    }
    trace = 0.5 / static_cast<double>(N * n_tiles) * trace;
    return trace;
}



double sum_noise_gradleft(
    const std::vector<double> &ft_invK, double grad, const std::vector<double> &hyperparameters, std::size_t N)
{
    double noise_der = compute_sigmoid(to_unconstrained(hyperparameters[2], true));
    for (std::size_t i = 0; i < N; ++i)
    {
        grad += (ft_invK[i * N + i] * noise_der);
    }
    return grad;
}

double sum_noise_gradright(
    const std::vector<double> &alpha, double grad, const std::vector<double> &hyperparameters, std::size_t N)
{
    double noise_der = compute_sigmoid(to_unconstrained(hyperparameters[2], true));
    grad += (noise_der * dot(alpha, alpha, static_cast<int>(N)));
    return grad;
}
