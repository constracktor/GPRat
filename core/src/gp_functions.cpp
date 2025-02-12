#include "../include/gp_functions.hpp"

#include "../include/gp_algorithms_cpu.hpp"
#include "../include/gp_optimizer.hpp"
#include "../include/tiled_algorithms_cpu.hpp"
#include <hpx/future.hpp>

using Tiled_matrix = std::vector<hpx::shared_future<std::vector<double>>>;
using Tiled_vector = std::vector<hpx::shared_future<std::vector<double>>>;

///////////////////////////////////////////////////////////////////////////
// PREDICT
std::vector<std::vector<double>>
cholesky_hpx(const std::vector<double> &training_input,
             const gprat_hyper::SEKParams &sek_params,
             int n_tiles,
             int n_tile_size,
             int n_regressors)
{
    std::vector<std::vector<double>> result;
    // Tiled future data structures
    Tiled_matrix K_tiles;  // Tiled covariance matrix

    // Preallocate memory
    result.resize(static_cast<std::size_t>(n_tiles * n_tiles));
    K_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));  // No reserve because of triangular structure

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous assembly
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            K_tiles[i * static_cast<std::size_t>(n_tiles) + j] = hpx::async(
                hpx::annotated_function(gen_tile_covariance, "assemble_tiled_K"),
                i,
                j,
                n_tile_size,
                n_regressors,
                sek_params,
                training_input);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled(K_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Synchronize
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            result[i * static_cast<std::size_t>(n_tiles) + j] =
                K_tiles[i * static_cast<std::size_t>(n_tiles) + j].get();
        }
    }
    return result;
}

std::vector<double>
predict_hpx(const std::vector<double> &training_input,
            const std::vector<double> &training_output,
            const std::vector<double> &test_input,
            const gprat_hyper::SEKParams &sek_params,
            int n_tiles,
            int n_tile_size,
            int m_tiles,
            int m_tile_size,
            int n_regressors)
{
    /*
     * Prediction: hat(y)_M = cross(K)_MxN * K^-1_NxN * y_N
     * - Covariance matrix K_NxN
     * - Cross-covariance cross(K)_MxN
     * - Training ouput y_N
     * - Prediction output hat(y)_M
     *
     * Algorithm:
     * 1: Compute lower triangular part of covariance matrix K
     * 2: Compute Cholesky factor L of K
     * 3: Compute prediction hat(y):
     *    - triangular solve L * beta = y
     *    - triangular solve L^T * alpha = beta
     *    - compute hat(y) = cross(K) * alpha
     */

    std::vector<double> prediction_result;
    // Tiled future data structures
    Tiled_matrix K_tiles;                 // Tiled covariance matrix
    Tiled_matrix cross_covariance_tiles;  // Tiled cross_covariance matrix
    Tiled_vector prediction_tiles;        // Tiled solution
    Tiled_vector alpha_tiles;             // Tiled intermediate solution

    // Preallocate memory
    prediction_result.reserve(test_input.size());

    K_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));  // No reserve because of triangular structure
    alpha_tiles.reserve(static_cast<std::size_t>(n_tiles));
    cross_covariance_tiles.reserve(static_cast<std::size_t>(m_tiles) * static_cast<std::size_t>(n_tiles));
    prediction_tiles.reserve(static_cast<std::size_t>(m_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous assembly
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            K_tiles[i * static_cast<std::size_t>(n_tiles) + j] = hpx::async(
                hpx::annotated_function(gen_tile_covariance, "assemble_tiled_K"),
                i,
                j,
                n_tile_size,
                n_regressors,
                sek_params,
                training_input);
        }
    }

    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        alpha_tiles.push_back(hpx::async(
            hpx::annotated_function(gen_tile_output, "assemble_tiled_alpha"), i, n_tile_size, training_output));
    }

    for (std::size_t i = 0; i < static_cast<std::size_t>(m_tiles); i++)
    {
        for (std::size_t j = 0; j < static_cast<std::size_t>(n_tiles); j++)
        {
            cross_covariance_tiles.push_back(hpx::async(
                hpx::annotated_function(gen_tile_cross_covariance, "assemble_pred"),
                i,
                j,
                m_tile_size,
                n_tile_size,
                n_regressors,
                sek_params,
                test_input,
                training_input));
        }
    }

    for (std::size_t i = 0; i < static_cast<std::size_t>(m_tiles); i++)
    {
        prediction_tiles.push_back(hpx::async(hpx::annotated_function(gen_tile_zeros, "assemble_tiled"), m_tile_size));
    }

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled(K_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous triangular solve  L * (L^T * alpha) = y
    forward_solve_tiled(K_tiles, alpha_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));
    backward_solve_tiled(K_tiles, alpha_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous prediction computation solve: \hat{y} = K_cross_cov * alpha
    matrix_vector_tiled(
        cross_covariance_tiles,
        alpha_tiles,
        prediction_tiles,
        m_tile_size,
        n_tile_size,
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Synchronize prediction
    for (std::size_t i = 0; i < static_cast<std::size_t>(m_tiles); i++)
    {
        auto tile = prediction_tiles[i].get();
        std::copy(tile.begin(), tile.end(), std::back_inserter(prediction_result));
    }
    return prediction_result;
}

std::vector<std::vector<double>> predict_with_uncertainty_hpx(
    const std::vector<double> &training_input,
    const std::vector<double> &training_output,
    const std::vector<double> &test_input,
    const gprat_hyper::SEKParams &sek_params,
    int n_tiles,
    int n_tile_size,
    int m_tiles,
    int m_tile_size,
    int n_regressors)
{
    /*
     * Prediction: hat(y) = cross(K) * K^-1 * y
     * Uncertainty: diag(Sigma) = diag(prior(K)) * diag(cross(K)^T * K^-1 * cross(K))
     * - Covariance matrix K_NxN
     * - Cross-covariance cross(K)_MxN
     * - Prior covariance prior(K)_MxM
     * - Training ouput y_N
     * - Prediction output hat(y)_M
     * - Posterior covariance matrix Sigma_MxM
     *
     * Algorithm:
     * 1: Compute lower triangular part of covariance matrix K
     * 2: Compute Cholesky factor L of K
     * 3: Compute prediction hat(y):
     *    - triangular solve L * beta = y
     *    - triangular solve L^T * alpha = beta
     *    - compute hat(y) = cross(K) * alpha
     * 4: Compute uncertainty diag(Sigma):
     *    - triangular solve L * V = cross(K)^T
     *    - compute diag(W) = diag(V^T * V)
     *    - compute diag(Sigma) = diag(prior(K)) - diag(W)
     */

    std::vector<double> prediction_result;
    std::vector<double> uncertainty_result;
    // Tiled future data structures for prediction
    Tiled_matrix K_tiles;                 // Tiled covariance matrix K_NxN
    Tiled_matrix cross_covariance_tiles;  // Tiled cross_covariance matrix K_NxM
    Tiled_vector prediction_tiles;        // Tiled solution
    Tiled_vector alpha_tiles;             // Tiled intermediate solution
    // Tiled future data structures for uncertainty
    Tiled_matrix t_cross_covariance_tiles;  // Tiled transposed cross_covariance matrix K_MxN
    Tiled_vector prior_K_tiles;             // Tiled prior covariance matrix diagonal diag(K_MxM)
    Tiled_vector uncertainty_tiles;         // Tiled uncertainty solution

    // Preallocate memory
    prediction_result.reserve(test_input.size());
    uncertainty_result.reserve(test_input.size());

    K_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));  // No reserve because of triangular structure
    cross_covariance_tiles.reserve(static_cast<std::size_t>(m_tiles) * static_cast<std::size_t>(n_tiles));
    prediction_tiles.reserve(static_cast<std::size_t>(m_tiles));
    alpha_tiles.reserve(static_cast<std::size_t>(n_tiles));

    t_cross_covariance_tiles.reserve(static_cast<std::size_t>(n_tiles) * static_cast<std::size_t>(m_tiles));
    prior_K_tiles.reserve(static_cast<std::size_t>(m_tiles));
    uncertainty_tiles.reserve(static_cast<std::size_t>(m_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous assembly
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            K_tiles[i * static_cast<std::size_t>(n_tiles) + j] = hpx::async(
                hpx::annotated_function(gen_tile_covariance, "assemble_tiled_K"),
                i,
                j,
                n_tile_size,
                n_regressors,
                sek_params,
                training_input);
        }
    }

    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        alpha_tiles.push_back(hpx::async(
            hpx::annotated_function(gen_tile_output, "assemble_tiled_alpha"), i, n_tile_size, training_output));
    }

    for (std::size_t i = 0; i < static_cast<std::size_t>(m_tiles); i++)
    {
        for (std::size_t j = 0; j < static_cast<std::size_t>(n_tiles); j++)
        {
            cross_covariance_tiles.push_back(hpx::async(
                hpx::annotated_function(gen_tile_cross_covariance, "assemble_pred"),
                i,
                j,
                m_tile_size,
                n_tile_size,
                n_regressors,
                sek_params,
                test_input,
                training_input));
        }
    }

    for (std::size_t i = 0; i < static_cast<std::size_t>(m_tiles); i++)
    {
        prediction_tiles.push_back(hpx::async(hpx::annotated_function(gen_tile_zeros, "assemble_tiled"), m_tile_size));
    }

    for (std::size_t i = 0; i < static_cast<std::size_t>(m_tiles); i++)
    {
        prior_K_tiles.push_back(hpx::async(
            hpx::annotated_function(gen_tile_prior_covariance, "assemble_tiled"),
            i,
            i,
            m_tile_size,
            n_regressors,
            sek_params,
            test_input));
    }

    for (std::size_t j = 0; j < static_cast<std::size_t>(n_tiles); j++)
    {
        for (std::size_t i = 0; i < static_cast<std::size_t>(m_tiles); i++)
        {
            t_cross_covariance_tiles.push_back(hpx::dataflow(
                hpx::annotated_function(hpx::unwrapping(&gen_tile_transpose), "assemble_pred"),
                m_tile_size,
                n_tile_size,
                cross_covariance_tiles[i * static_cast<std::size_t>(n_tiles) + j]));
        }
    }

    for (std::size_t i = 0; i < static_cast<std::size_t>(m_tiles); i++)
    {
        uncertainty_tiles.push_back(
            hpx::async(hpx::annotated_function(gen_tile_zeros, "assemble_prior_inter"), m_tile_size));
    }

    // Prediction
    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled(K_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous triangular solve  L * (L^T * alpha) = y
    forward_solve_tiled(K_tiles, alpha_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));
    backward_solve_tiled(K_tiles, alpha_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous prediction computation solve: hat(y) = cross(K) * alpha
    matrix_vector_tiled(
        cross_covariance_tiles,
        alpha_tiles,
        prediction_tiles,
        m_tile_size,
        n_tile_size,
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous triangular solve L * V = cross(K)^T
    forward_solve_tiled_matrix(
        K_tiles,
        t_cross_covariance_tiles,
        n_tile_size,
        m_tile_size,
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous computation diag(W) = diag(V^T * V)
    symmetric_matrix_matrix_diagonal_tiled(
        t_cross_covariance_tiles,
        uncertainty_tiles,
        n_tile_size,
        m_tile_size,
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous computation diag(Sigma) = diag(prior(K)) - diag(W)
    vector_difference_tiled(prior_K_tiles, uncertainty_tiles, m_tile_size, static_cast<std::size_t>(m_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Synchronize prediction
    for (std::size_t i = 0; i < static_cast<std::size_t>(m_tiles); i++)
    {
        auto tile = prediction_tiles[i].get();
        std::copy(tile.begin(), tile.end(), std::back_inserter(prediction_result));
    }

    // Synchronize uncertainty
    for (std::size_t i = 0; i < static_cast<std::size_t>(m_tiles); i++)
    {
        auto tile = uncertainty_tiles[i].get();
        std::copy(tile.begin(), tile.end(), std::back_inserter(uncertainty_result));
    }

    return std::vector<std::vector<double>>{ std::move(prediction_result), std::move(uncertainty_result) };
}

std::vector<std::vector<double>> predict_with_full_cov_hpx(
    const std::vector<double> &training_input,
    const std::vector<double> &training_output,
    const std::vector<double> &test_input,
    const gprat_hyper::SEKParams &sek_params,
    int n_tiles,
    int n_tile_size,
    int m_tiles,
    int m_tile_size,
    int n_regressors)
{
    /*
     * Prediction: hat(y)_M = cross(K) * K^-1 * y
     * Full covariance: Sigma = prior(K) - cross(K)^T * K^-1 * cross(K)
     * - Covariance matrix K_NxN
     * - Cross-covariance cross(K)_MxN
     * - Prior covariance prior(K)_MxM
     * - Training ouput y_N
     * - Prediction output hat(y)_M
     * - Posterior covariance matrix Sigma_MxM
     *
     * Algorithm:
     * 1: Compute lower triangular part of covariance matrix K
     * 2: Compute Cholesky factor L of K
     * 3: Compute prediction hat(y):
     *    - triangular solve L * beta = y
     *    - triangular solve L^T * alpha = beta
     *    - compute hat(y) = cross(K) * alpha
     * 4: Compute full covariance matrix Sigma:
     *    - triangular solve L * V = cross(K)^T
     *    - compute W = V^T * V
     *    - compute Sigma = prior(K) - W
     * 5: Compute diag(Sigma)
     */

    std::vector<double> prediction_result;
    std::vector<double> uncertainty_result;
    // Tiled future data structures for prediction
    Tiled_matrix K_tiles;                 // Tiled covariance matrix K_NxN
    Tiled_matrix cross_covariance_tiles;  // Tiled cross_covariance matrix K_NxM
    Tiled_vector prediction_tiles;        // Tiled solution
    Tiled_vector alpha_tiles;             // Tiled intermediate solution
    // Tiled future data structures for uncertainty
    Tiled_matrix t_cross_covariance_tiles;  // Tiled transposed cross_covariance matrix K_MxN
    Tiled_matrix prior_K_tiles;             // Tiled prior covariance matrix K_MxM
    Tiled_vector uncertainty_tiles;         // Tiled uncertainty solution

    // Preallocate memory
    prediction_result.reserve(test_input.size());
    uncertainty_result.reserve(test_input.size());

    K_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));  // No reserve because of triangular structure
    cross_covariance_tiles.reserve(static_cast<std::size_t>(m_tiles) * static_cast<std::size_t>(n_tiles));
    prediction_tiles.reserve(static_cast<std::size_t>(m_tiles));
    alpha_tiles.reserve(static_cast<std::size_t>(n_tiles));

    t_cross_covariance_tiles.reserve(static_cast<std::size_t>(n_tiles) * static_cast<std::size_t>(m_tiles));
    prior_K_tiles.resize(static_cast<std::size_t>(m_tiles * m_tiles));
    uncertainty_tiles.reserve(static_cast<std::size_t>(m_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous assembly
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            K_tiles[i * static_cast<std::size_t>(n_tiles) + j] = hpx::async(
                hpx::annotated_function(gen_tile_covariance, "assemble_tiled_K"),
                i,
                j,
                n_tile_size,
                n_regressors,
                sek_params,
                training_input);
        }
    }

    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        alpha_tiles.push_back(hpx::async(
            hpx::annotated_function(gen_tile_output, "assemble_tiled_alpha"), i, n_tile_size, training_output));
    }

    for (std::size_t i = 0; i < static_cast<std::size_t>(m_tiles); i++)
    {
        for (std::size_t j = 0; j < static_cast<std::size_t>(n_tiles); j++)
        {
            cross_covariance_tiles.push_back(hpx::async(
                hpx::annotated_function(gen_tile_cross_covariance, "assemble_pred"),
                i,
                j,
                m_tile_size,
                n_tile_size,
                n_regressors,
                sek_params,
                test_input,
                training_input));
        }
    }

    for (std::size_t i = 0; i < static_cast<std::size_t>(m_tiles); i++)
    {
        prediction_tiles.push_back(hpx::async(hpx::annotated_function(gen_tile_zeros, "assemble_tiled"), m_tile_size));
    }

    // Assemble prior covariance matrix vector
    for (std::size_t i = 0; i < static_cast<std::size_t>(m_tiles); i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            prior_K_tiles[i * static_cast<std::size_t>(m_tiles) + j] = hpx::async(
                hpx::annotated_function(gen_tile_full_prior_covariance, "assemble_prior_tiled"),
                i,
                j,
                m_tile_size,
                n_regressors,
                sek_params,
                test_input);

            if (i != j)
            {
                prior_K_tiles[j * static_cast<std::size_t>(m_tiles) + i] = hpx::dataflow(
                    hpx::annotated_function(hpx::unwrapping(&gen_tile_transpose), "assemble_prior_tiled"),
                    m_tile_size,
                    m_tile_size,
                    prior_K_tiles[i * static_cast<std::size_t>(m_tiles) + j]);
            }
        }
    }

    for (std::size_t j = 0; j < static_cast<std::size_t>(n_tiles); j++)
    {
        for (std::size_t i = 0; i < static_cast<std::size_t>(m_tiles); i++)
        {
            t_cross_covariance_tiles.push_back(hpx::dataflow(
                hpx::annotated_function(hpx::unwrapping(&gen_tile_transpose), "assemble_pred"),
                m_tile_size,
                n_tile_size,
                cross_covariance_tiles[i * static_cast<std::size_t>(n_tiles) + j]));
        }
    }

    for (std::size_t i = 0; i < static_cast<std::size_t>(m_tiles); i++)
    {
        uncertainty_tiles.push_back(hpx::async(hpx::annotated_function(gen_tile_zeros, "assemble_tiled"), m_tile_size));
    }

    // Prediction
    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled(K_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous triangular solve  L * (L^T * alpha) = y
    forward_solve_tiled(K_tiles, alpha_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));
    backward_solve_tiled(K_tiles, alpha_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous prediction computation solve: hat(y) = K_cross_cov * alpha
    matrix_vector_tiled(
        cross_covariance_tiles,
        alpha_tiles,
        prediction_tiles,
        m_tile_size,
        n_tile_size,
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous triangular solve L * V = cross(K)^T
    forward_solve_tiled_matrix(
        K_tiles,
        t_cross_covariance_tiles,
        n_tile_size,
        m_tile_size,
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous computation of full covariance Sigma = prior(K) - V^T * V
    symmetric_matrix_matrix_tiled(
        t_cross_covariance_tiles,
        prior_K_tiles,
        n_tile_size,
        m_tile_size,
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tiles));
    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous computation of uncertainty diag(Sigma)
    matrix_diagonal_tiled(prior_K_tiles, uncertainty_tiles, m_tile_size, static_cast<std::size_t>(m_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Synchronize prediction
    for (std::size_t i = 0; i < static_cast<std::size_t>(m_tiles); i++)
    {
        auto tile = prediction_tiles[i].get();
        std::copy(tile.begin(), tile.end(), std::back_inserter(prediction_result));
    }

    // Synchronize uncertainty
    for (std::size_t i = 0; i < static_cast<std::size_t>(m_tiles); i++)
    {
        auto tile = uncertainty_tiles[i].get();
        std::copy(tile.begin(), tile.end(), std::back_inserter(uncertainty_result));
    }

    return std::vector<std::vector<double>>{ std::move(prediction_result), std::move(uncertainty_result) };
}

///////////////////////////////////////////////////////////////////////////
// OPTIMIZATION
double compute_loss_hpx(const std::vector<double> &training_input,
                        const std::vector<double> &training_output,
                        const gprat_hyper::SEKParams &sek_params,
                        int n_tiles,
                        int n_tile_size,
                        int n_regressors)
{
    /*
     * Negative log likelihood loss:
     * loss(theta) = 0.5 * ( log(det(K)) - y^T * K^-1 * y - N * log(2 * pi) )
     * - Covariance matrix K(theta)_NxN
     * - Training ouput y_N
     * - Hyperparameters theta ={ v, l, v_n }
     *
     * Algorithm:
     * 1: Compute lower triangular part of covariance matrix K
     * 2: Compute Cholesky factor L of K
     * 3: Compute prediction alpha = K^-1 * y:
     *    - triangular solve L * beta = y
     *    - triangular solve L^T * alpha = beta
     * 5: Compute beta = K^-1 * y
     * 6: Compute negative log likelihood loss
     *    - Calculate sum_i^N log(L_ii^2)
     *    - Calculate y^T * beta
     *    - Add constant N * log (2 * pi)
     */

    hpx::shared_future<double> loss_value;
    // Tiled future data structures
    Tiled_matrix K_tiles;      // Tiled covariance matrix K_NxN
    Tiled_vector y_tiles;      // Tiled output
    Tiled_vector alpha_tiles;  // Tiled intermediate solution

    // Preallocate memory
    K_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));  // No reserve because of triangular structure
    y_tiles.reserve(static_cast<std::size_t>(n_tiles));
    alpha_tiles.reserve(static_cast<std::size_t>(n_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous assembly
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            K_tiles[i * static_cast<std::size_t>(n_tiles) + j] = hpx::async(
                hpx::annotated_function(gen_tile_covariance, "assemble_tiled_K"),
                i,
                j,
                n_tile_size,
                n_regressors,
                sek_params,
                training_input);
        }
    }

    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        y_tiles.push_back(
            hpx::async(hpx::annotated_function(gen_tile_output, "assemble_tiled_y"), i, n_tile_size, training_output));
    }

    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        alpha_tiles.push_back(hpx::async(
            hpx::annotated_function(gen_tile_output, "assemble_tiled_alpha"), i, n_tile_size, training_output));
    }

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled(K_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous triangular solve  L * (L^T * alpha) = y
    forward_solve_tiled(K_tiles, alpha_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));
    backward_solve_tiled(K_tiles, alpha_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous loss computation
    compute_loss_tiled(K_tiles, alpha_tiles, y_tiles, loss_value, n_tile_size, static_cast<std::size_t>(n_tiles));

    return loss_value.get();
}

std::vector<double>
optimize_hpx(const std::vector<double> &training_input,
             const std::vector<double> &training_output,
             int n_tiles,
             int n_tile_size,
             int n_regressors,
             const gprat_hyper::AdamParams &adam_params,
             gprat_hyper::SEKParams &sek_params,
             std::vector<bool> trainable_params)
{
    /*
     * - Hyperparameters theta={v, l, v_n}
     * - Covariance matrix K(theta)
     * - Training ouput y
     *
     * Algorithm:
     * for opt_iter:
     *   1: Compute distance for entries of covariance matrix K
     *   2: Compute lower triangular part of K with distance
     *   3: Compute lower triangular gradients for delta(K)/delta(v), and delta(K)/delta(l) with distance
     *
     *   4: Compute Cholesky factor L of K
     *   5: Compute K^-1:
     *       - triangular solve L * {} = I
     *       - triangular solve L^T * K^-1 = {}
     *   6: Compute beta = K^-1 * y
     *
     *   7: Compute negative log likelihood loss
     *       - Calculate 0.5 sum_i^N log(L_ii^2)
     *       - Calculate 0.5 y^T * beta
     *       - Add constant N / 2 * log (2 * pi)
     *
     *   8: Compute delta(loss)/delta(param_i)
     *       - Compute trace(K^-1 * delta(K)/delta(theta_i))
     *       - Compute beta^T *  delta(K)/delta(theta_i) * beta
     *   9: Update hyperparameters theta with Adam optimizer
     *       - m_T = beta1 * m_T-1 + (1 - beta1) * g_T
     *       - w_T = beta2 + w_T-1 + (1 - beta2) * g_T^2
     *       - nu_T = nu * sqrt(1 - beta2_T) / (1 - beta1_T)
     *       - theta_T = theta_T-1 - nu_T * m_T / (sqrt(w_T) + epsilon)
     * endfor
     */

    // data holder for loss
    hpx::shared_future<double> loss_value;
    // data holder for computed loss values
    std::vector<double> losses;

    // Tiled future data structures
    Tiled_matrix K_tiles;      // Tiled covariance matrix K_NxN
    Tiled_vector y_tiles;      // Tiled output
    Tiled_vector alpha_tiles;  // Tiled intermediate solution
    Tiled_matrix K_inv_tiles;  // Tiled inversed covariance matrix K^-1_NxN
    // Tiled future data structures for gradients
    Tiled_matrix grad_v_tiles;  // Tiled covariance with gradient v
    Tiled_matrix grad_l_tiles;  // Tiled covariance with gradient l

    // Preallocate memory
    losses.reserve(static_cast<std::size_t>(adam_params.opt_iter));
    y_tiles.reserve(static_cast<std::size_t>(n_tiles));

    alpha_tiles.resize(static_cast<std::size_t>(n_tiles));            // for now resize since reset in loop
    K_inv_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));  // for now resize since reset in loop

    K_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));       // No reserve because of triangular structure
    grad_v_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));  // No reserve because of triangular structure
    grad_l_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));  // No reserve because of triangular structure

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous assembly of output y
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        y_tiles.push_back(
            hpx::async(hpx::annotated_function(gen_tile_output, "assemble_y"), i, n_tile_size, training_output));
    }

    //////////////////////////////////////////////////////////////////////////////
    // Perform optimization
    for (std::size_t iter = 0; iter < static_cast<std::size_t>(adam_params.opt_iter); iter++)
    {
        ///////////////////////////////////////////////////////////////////////////
        // Launch asynchronous assembly of tiled covariance matrix, derivative of covariance matrix
        // vector w.r.t. to vertical lengthscale and derivative of covariance
        // matrix vector w.r.t. to lengthscale
        for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
        {
            for (std::size_t j = 0; j <= i; j++)
            {
                // Compute the distance (z_i - z_j) of K entries to reuse
                hpx::shared_future<std::vector<double>> cov_dists = hpx::async(
                    hpx::annotated_function(gen_tile_distance, "assemble_cov_dist"),
                    i,
                    j,
                    n_tile_size,
                    n_regressors,
                    sek_params,
                    training_input);

                K_tiles[i * static_cast<std::size_t>(n_tiles) + j] = hpx::dataflow(
                    hpx::annotated_function(hpx::unwrapping(&gen_tile_covariance_with_distance), "assemble_K"),
                    i,
                    j,
                    n_tile_size,
                    sek_params,
                    cov_dists);
                if (trainable_params[0])
                {
                    grad_l_tiles[i * static_cast<std::size_t>(n_tiles) + j] = hpx::dataflow(
                        hpx::annotated_function(hpx::unwrapping(&gen_tile_grad_l), "assemble_gradl"),
                        n_tile_size,
                        sek_params,
                        cov_dists);
                    if (i != j)
                    {
                        grad_l_tiles[j * static_cast<std::size_t>(n_tiles) + i] = hpx::dataflow(
                            hpx::annotated_function(hpx::unwrapping(&gen_tile_transpose), "assemble_gradl_t"),
                            n_tile_size,
                            n_tile_size,
                            grad_l_tiles[i * static_cast<std::size_t>(n_tiles) + j]);
                    }
                }

                if (trainable_params[1])
                {
                    grad_v_tiles[i * static_cast<std::size_t>(n_tiles) + j] = hpx::dataflow(
                        hpx::annotated_function(hpx::unwrapping(&gen_tile_grad_v), "assemble_gradv"),
                        n_tile_size,
                        sek_params,
                        cov_dists);
                    if (i != j)
                    {
                        grad_v_tiles[j * static_cast<std::size_t>(n_tiles) + i] = hpx::dataflow(
                            hpx::annotated_function(hpx::unwrapping(&gen_tile_transpose), "assemble_gradv_t"),
                            n_tile_size,
                            n_tile_size,
                            grad_v_tiles[i * static_cast<std::size_t>(n_tiles) + j]);
                    }
                }
            }
        }

        // Assembly with reallocation -> optimize to only set existing values
        for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
        {
            alpha_tiles[i] = hpx::async(hpx::annotated_function(gen_tile_zeros, "assemble_tiled"), n_tile_size);
        }

        for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
        {
            for (std::size_t j = 0; j < static_cast<std::size_t>(n_tiles); j++)
            {
                if (i == j)
                {
                    K_inv_tiles[i * static_cast<std::size_t>(n_tiles) + j] =
                        hpx::async(hpx::annotated_function(gen_tile_identity, "assemble_identity_matrix"), n_tile_size);
                }
                else
                {
                    K_inv_tiles[i * static_cast<std::size_t>(n_tiles) + j] = hpx::async(
                        hpx::annotated_function(gen_tile_zeros, "assemble_identity_matrix"), n_tile_size * n_tile_size);
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////////
        // Launch asynchronous Cholesky decomposition: K = L * L^T
        right_looking_cholesky_tiled(K_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));

        ///////////////////////////////////////////////////////////////////////////
        // Launch asynchronous compute K^-1 through L* (L^T * X) = I
        forward_solve_tiled_matrix(
            K_tiles,
            K_inv_tiles,
            n_tile_size,
            n_tile_size,
            static_cast<std::size_t>(n_tiles),
            static_cast<std::size_t>(n_tiles));
        backward_solve_tiled_matrix(
            K_tiles,
            K_inv_tiles,
            n_tile_size,
            n_tile_size,
            static_cast<std::size_t>(n_tiles),
            static_cast<std::size_t>(n_tiles));

        ///////////////////////////////////////////////////////////////////////////
        // Launch asynchronous compute beta = inv(K) * y
        matrix_vector_tiled(
            K_inv_tiles,
            y_tiles,
            alpha_tiles,
            n_tile_size,
            n_tile_size,
            static_cast<std::size_t>(n_tiles),
            static_cast<std::size_t>(n_tiles));

        ///////////////////////////////////////////////////////////////////////////
        // Launch asynchronous loss computation where
        // loss(theta) = 0.5 * ( log(det(K)) - y^T * K^-1 * y - N * log(2 * pi) )
        compute_loss_tiled(K_tiles, alpha_tiles, y_tiles, loss_value, n_tile_size, static_cast<std::size_t>(n_tiles));

        ///////////////////////////////////////////////////////////////////////////
        // Launch asynchronous update of the hyperparameters
        if (trainable_params[0])
        {  // lengthscale
            update_hyperparameter_tiled(
                K_inv_tiles,
                grad_l_tiles,
                alpha_tiles,
                adam_params,
                sek_params,
                n_tile_size,
                static_cast<std::size_t>(n_tiles),
                iter,
                0);
        }
        if (trainable_params[1])
        {  // vertical_lengthscale
            update_hyperparameter_tiled(
                K_inv_tiles,
                grad_v_tiles,
                alpha_tiles,
                adam_params,
                sek_params,
                n_tile_size,
                static_cast<std::size_t>(n_tiles),
                iter,
                1);
        }
        if (trainable_params[2])
        {  // noise_variance
            update_hyperparameter_tiled(
                K_inv_tiles,
                Tiled_matrix{},  // no tiled gradient matrix required
                alpha_tiles,
                adam_params,
                sek_params,
                n_tile_size,
                static_cast<std::size_t>(n_tiles),
                iter,
                2);
        }
        // Synchronize after iteration
        losses.push_back(loss_value.get());
    }
    // Return losses
    return losses;
}

double optimize_step_hpx(
    const std::vector<double> &training_input,
    const std::vector<double> &training_output,
    int n_tiles,
    int n_tile_size,
    int n_regressors,
    gprat_hyper::AdamParams &adam_params,
    gprat_hyper::SEKParams &sek_params,
    std::vector<bool> trainable_params,
    int iter)
{
    /*
     * - Hyperparameters theta={v, l, v_n}
     * - Covariance matrix K(theta)
     * - Training ouput y
     *
     * Algorithm:
     *   1: Compute distance for entries of covariance matrix K
     *   2: Compute lower triangular part of K with distance
     *   3: Compute lower triangular gradients for delta(K)/delta(v), and delta(K)/delta(l) with distance
     *
     *   4: Compute Cholesky factor L of K
     *   5: Compute K^-1:
     *       - triangular solve L * {} = I
     *       - triangular solve L^T * K^-1 = {}
     *   6: Compute beta = K^-1 * y
     *
     *   7: Compute negative log likelihood loss
     *       - Calculate 0.5 sum_i^N log(L_ii^2)
     *       - Calculate 0.5 y^T * beta
     *       - Add constant N / 2 * log (2 * pi)
     *
     *   8: Compute delta(loss)/delta(param_i)
     *       - Compute trace(K^-1 * delta(K)/delta(theta_i))
     *       - Compute beta^T *  delta(K)/delta(theta_i) * beta
     *   9: Update hyperparameters theta with Adam optimizer
     *       - m_T = beta1 * m_T-1 + (1 - beta1) * g_T
     *       - w_T = beta2 + w_T-1 + (1 - beta2) * g_T^2
     *       - nu_T = nu * sqrt(1 - beta2_T) / (1 - beta1_T)
     *       - theta_T = theta_T-1 - nu_T * m_T / (sqrt(w_T) + epsilon)
     */

    // data holder for loss
    hpx::shared_future<double> loss_value;

    // Tiled future data structures
    Tiled_matrix K_tiles;      // Tiled covariance matrix K_NxN
    Tiled_vector y_tiles;      // Tiled output
    Tiled_vector alpha_tiles;  // Tiled intermediate solution
    Tiled_matrix K_inv_tiles;  // Tiled inversed covariance matrix K^-1_NxN
    // Tiled future data structures for gradients
    Tiled_matrix grad_v_tiles;  // Tiled covariance with gradient v
    Tiled_matrix grad_l_tiles;  // Tiled covariance with gradient l

    // Preallocate memory
    y_tiles.reserve(static_cast<std::size_t>(n_tiles));

    alpha_tiles.resize(static_cast<std::size_t>(n_tiles));            // for now resize since reset in loop
    K_inv_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));  // for now resize since reset in loop

    K_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));       // No reserve because of triangular structure
    grad_v_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));  // No reserve because of triangular structure
    grad_l_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));  // No reserve because of triangular structure

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous assembly of output y
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        y_tiles.push_back(
            hpx::async(hpx::annotated_function(gen_tile_output, "assemble_y"), i, n_tile_size, training_output));
    }

    //////////////////////////////////////////////////////////////////////////////
    // Perform one optimization step
    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous assembly of tiled covariance matrix, derivative of covariance matrix
    // vector w.r.t. to vertical lengthscale and derivative of covariance
    // matrix vector w.r.t. to lengthscale
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            // Compute the distance (z_i - z_j) of K entries to reuse
            hpx::shared_future<std::vector<double>> cov_dists = hpx::async(
                hpx::annotated_function(gen_tile_distance, "assemble_cov_dist"),
                i,
                j,
                n_tile_size,
                n_regressors,
                sek_params,
                training_input);

            K_tiles[i * static_cast<std::size_t>(n_tiles) + j] = hpx::dataflow(
                hpx::annotated_function(hpx::unwrapping(&gen_tile_covariance_with_distance), "assemble_K"),
                i,
                j,
                n_tile_size,
                sek_params,
                cov_dists);

            if (trainable_params[0])
            {
                grad_l_tiles[i * static_cast<std::size_t>(n_tiles) + j] = hpx::dataflow(
                    hpx::annotated_function(hpx::unwrapping(&gen_tile_grad_l), "assemble_gradl"),
                    n_tile_size,
                    sek_params,
                    cov_dists);
                if (i != j)
                {
                    grad_l_tiles[j * static_cast<std::size_t>(n_tiles) + i] = hpx::dataflow(
                        hpx::annotated_function(hpx::unwrapping(&gen_tile_transpose), "assemble_gradl_t"),
                        n_tile_size,
                        n_tile_size,
                        grad_l_tiles[i * static_cast<std::size_t>(n_tiles) + j]);
                }
            }

            if (trainable_params[1])
            {
                grad_v_tiles[i * static_cast<std::size_t>(n_tiles) + j] = hpx::dataflow(
                    hpx::annotated_function(hpx::unwrapping(&gen_tile_grad_v), "assemble_gradv"),
                    n_tile_size,
                    sek_params,
                    cov_dists);
                if (i != j)
                {
                    grad_v_tiles[j * static_cast<std::size_t>(n_tiles) + i] = hpx::dataflow(
                        hpx::annotated_function(hpx::unwrapping(&gen_tile_transpose), "assemble_gradv_t"),
                        n_tile_size,
                        n_tile_size,
                        grad_v_tiles[i * static_cast<std::size_t>(n_tiles) + j]);
                }
            }
        }
    }

    // Assembly with reallocation -> optimize to only set existing values
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        alpha_tiles[i] = hpx::async(hpx::annotated_function(gen_tile_zeros, "assemble_tiled"), n_tile_size);
    }

    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        for (std::size_t j = 0; j < static_cast<std::size_t>(n_tiles); j++)
        {
            if (i == j)
            {
                K_inv_tiles[i * static_cast<std::size_t>(n_tiles) + j] =
                    hpx::async(hpx::annotated_function(gen_tile_identity, "assemble_identity_matrix"), n_tile_size);
            }
            else
            {
                K_inv_tiles[i * static_cast<std::size_t>(n_tiles) + j] = hpx::async(
                    hpx::annotated_function(gen_tile_zeros, "assemble_identity_matrix"), n_tile_size * n_tile_size);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous Cholesky decomposition: K = L * L^T
    right_looking_cholesky_tiled(K_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous compute K^-1 through L* (L^T * X) = I
    forward_solve_tiled_matrix(
        K_tiles,
        K_inv_tiles,
        n_tile_size,
        n_tile_size,
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(n_tiles));
    backward_solve_tiled_matrix(
        K_tiles,
        K_inv_tiles,
        n_tile_size,
        n_tile_size,
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(n_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous compute beta = inv(K) * y
    matrix_vector_tiled(
        K_inv_tiles,
        y_tiles,
        alpha_tiles,
        n_tile_size,
        n_tile_size,
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(n_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous loss computation where
    // loss(theta) = 0.5 * ( log(det(K)) - y^T * K^-1 * y - N * log(2 * pi) )
    compute_loss_tiled(K_tiles, alpha_tiles, y_tiles, loss_value, n_tile_size, static_cast<std::size_t>(n_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous update of the hyperparameters
    if (trainable_params[0])
    {  // lengthscale
        update_hyperparameter_tiled(
            K_inv_tiles,
            grad_l_tiles,
            alpha_tiles,
            adam_params,
            sek_params,
            n_tile_size,
            static_cast<std::size_t>(n_tiles),
            static_cast<std::size_t>(iter),
            0);
    }
    if (trainable_params[1])
    {  // vertical_lengthscale
        update_hyperparameter_tiled(
            K_inv_tiles,
            grad_v_tiles,
            alpha_tiles,
            adam_params,
            sek_params,
            n_tile_size,
            static_cast<std::size_t>(n_tiles),
            static_cast<std::size_t>(iter),
            1);
    }
    if (trainable_params[2])
    {  // noise_variance
        update_hyperparameter_tiled(
            K_inv_tiles,
            Tiled_matrix{},  // no tiled gradient matrix required
            alpha_tiles,
            adam_params,
            sek_params,
            n_tile_size,
            static_cast<std::size_t>(n_tiles),
            static_cast<std::size_t>(iter),
            2);
    }
    return loss_value.get();
}
