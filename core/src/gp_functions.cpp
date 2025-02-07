#include "../include/gp_functions.hpp"

#include "../include/gp_algorithms_cpu.hpp"
#include "../include/gp_optimizer.hpp"
#include "../include/tiled_algorithms_cpu.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using Tiled_matrix = std::vector<hpx::shared_future<std::vector<double>>>;
using Tiled_vector = std::vector<hpx::shared_future<std::vector<double>>>;

///////////////////////////////////////////////////////////////////////////
// PARAMETER STRUCT
namespace gprat_hyper
{

Hyperparameters::Hyperparameters(
    double lr,
    double b1,
    double b2,
    double eps,
    int opt_i,
    std::vector<double> M_T_init,
    std::vector<double> V_T_init) :
    learning_rate(lr),
    beta1(b1),
    beta2(b2),
    epsilon(eps),
    opt_iter(opt_i),
    M_T(M_T_init),
    V_T(V_T_init)
{ }

std::string Hyperparameters::repr() const
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(8);
    oss << "Hyperparameters: [learning_rate=" << learning_rate << ", beta1=" << beta1 << ", beta2=" << beta2
        << ", epsilon=" << epsilon << ", opt_iter=" << opt_iter << "]";
    return oss.str();
}

}  // namespace gprat_hyper

///////////////////////////////////////////////////////////////////////////
// PREDICT
std::vector<std::vector<double>>
cholesky_hpx(const std::vector<double> &training_input,
             const std::vector<double> &hyperparameters,
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
                hyperparameters,
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
            const std::vector<double> &hyperparameters,
            int n_tiles,
            int n_tile_size,
            int m_tiles,
            int m_tile_size,
            int n_regressors)
{
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
                hyperparameters,
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
                hyperparameters,
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
    prediction_tiled(cross_covariance_tiles,
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
    const std::vector<double> &hyperparameters,
    int n_tiles,
    int n_tile_size,
    int m_tiles,
    int m_tile_size,
    int n_regressors)
{
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
    Tiled_vector prior_inter_tiles;         // Tiled intermediate diagonal diag(K_MxN * (K^-1_NxN * K_NxM))
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
    prior_inter_tiles.reserve(static_cast<std::size_t>(m_tiles));
    uncertainty_tiles.resize(static_cast<std::size_t>(m_tiles));

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
                hyperparameters,
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
                hyperparameters,
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
            hyperparameters,
            test_input));
    }

    for (std::size_t j = 0; j < static_cast<std::size_t>(n_tiles); j++)
    {
        for (std::size_t i = 0; i < static_cast<std::size_t>(m_tiles); i++)
        {
            t_cross_covariance_tiles.push_back(hpx::dataflow(
                hpx::annotated_function(hpx::unwrapping(&gen_tile_cross_cov_T), "assemble_pred"),
                m_tile_size,
                n_tile_size,
                cross_covariance_tiles[i * static_cast<std::size_t>(n_tiles) + j]));
        }
    }

    for (std::size_t i = 0; i < static_cast<std::size_t>(m_tiles); i++)
    {
        prior_inter_tiles.push_back(
            hpx::async(hpx::annotated_function(gen_tile_zeros_diag, "assemble_prior_inter"), m_tile_size));
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
    // Launch asynchronous prediction computation solve: \hat{y} = K_cross_cov * alpha
    prediction_tiled(cross_covariance_tiles,
                     alpha_tiles,
                     prediction_tiles,
                     m_tile_size,
                     n_tile_size,
                     static_cast<std::size_t>(n_tiles),
                     static_cast<std::size_t>(m_tiles));

    // Uncertainty - todo: look into the computations
    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous triangular solve A_M,N * K_NxN = K_MxN -> A_MxN = K_MxN * K^-1_NxN
    forward_solve_KcK_tiled(
        K_tiles,
        t_cross_covariance_tiles,
        n_tile_size,
        m_tile_size,
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous posterior covariance matrix - (K_MxN * K^-1_NxN) * K_NxM???
    posterior_covariance_tiled(
        t_cross_covariance_tiles,
        prior_inter_tiles,
        n_tile_size,
        m_tile_size,
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous uncertainty computation equation??
    prediction_uncertainty_tiled(
        prior_K_tiles, prior_inter_tiles, uncertainty_tiles, m_tile_size, static_cast<std::size_t>(m_tiles));

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
    const std::vector<double> &hyperparameters,
    int n_tiles,
    int n_tile_size,
    int m_tiles,
    int m_tile_size,

    int n_regressors)
{
    std::vector<double> prediction_result;
    std::vector<double> full_covariance_result;
    // Tiled future data structures for prediction
    Tiled_matrix K_tiles;                 // Tiled covariance matrix K_NxN
    Tiled_matrix cross_covariance_tiles;  // Tiled cross_covariance matrix K_NxM
    Tiled_vector prediction_tiles;        // Tiled solution
    Tiled_vector alpha_tiles;             // Tiled intermediate solution
    // Tiled future data structures for uncertainty
    Tiled_matrix t_cross_covariance_tiles;  // Tiled transposed cross_covariance matrix K_MxN
    Tiled_matrix prior_K_tiles;             // Tiled prior covariance matrix K_MxM
    Tiled_vector full_covariance_tiles;     // Tiled uncertainty solution

    // Preallocate memory
    prediction_result.reserve(test_input.size());
    full_covariance_result.reserve(test_input.size());

    K_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));  // No reserve because of triangular structure
    cross_covariance_tiles.reserve(static_cast<std::size_t>(m_tiles) * static_cast<std::size_t>(n_tiles));
    prediction_tiles.reserve(static_cast<std::size_t>(m_tiles));
    alpha_tiles.reserve(static_cast<std::size_t>(n_tiles));

    t_cross_covariance_tiles.reserve(static_cast<std::size_t>(n_tiles) * static_cast<std::size_t>(m_tiles));
    prior_K_tiles.resize(static_cast<std::size_t>(m_tiles * m_tiles));
    full_covariance_tiles.reserve(static_cast<std::size_t>(m_tiles));

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
                hyperparameters,
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
                hyperparameters,
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
                hyperparameters,
                test_input);

            if (i != j)
            {
                prior_K_tiles[j * static_cast<std::size_t>(m_tiles) + i] = hpx::dataflow(
                    hpx::annotated_function(hpx::unwrapping(&gen_tile_grad_l_trans), "assemble_prior_tiled"),
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
                hpx::annotated_function(hpx::unwrapping(&gen_tile_cross_cov_T), "assemble_pred"),
                m_tile_size,
                n_tile_size,
                cross_covariance_tiles[i * static_cast<std::size_t>(n_tiles) + j]));
        }
    }

    for (std::size_t i = 0; i < static_cast<std::size_t>(m_tiles); i++)
    {
        full_covariance_tiles.push_back(
            hpx::async(hpx::annotated_function(gen_tile_zeros, "assemble_tiled"), m_tile_size));
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
    // Launch asynchronous prediction computation solve: \hat{y} = K_cross_cov * alpha
    prediction_tiled(cross_covariance_tiles,
                     alpha_tiles,
                     prediction_tiles,
                     m_tile_size,
                     n_tile_size,
                     static_cast<std::size_t>(n_tiles),
                     static_cast<std::size_t>(m_tiles));

    // Full covariance - todo: look into the computations
    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous triangular solve A_M,N * K_NxN = K_MxN -> A_MxN = K_MxN * K^-1_NxN
    forward_solve_KcK_tiled(
        K_tiles,
        t_cross_covariance_tiles,
        n_tile_size,
        m_tile_size,
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(m_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous posterior covariance matrix K_MxM - (K_MxN * K^-1_NxN) * K_NxM
    full_cov_tiled(t_cross_covariance_tiles,
                   prior_K_tiles,
                   n_tile_size,
                   m_tile_size,
                   static_cast<std::size_t>(n_tiles),
                   static_cast<std::size_t>(m_tiles));
    ///////////////////////////////////////////////////////////////////////////
    // Launch asynchronous full covriance computation equation??
    pred_uncer_tiled(prior_K_tiles, full_covariance_tiles, m_tile_size, static_cast<std::size_t>(m_tiles));

    ///////////////////////////////////////////////////////////////////////////
    // Synchronize prediction
    for (std::size_t i = 0; i < static_cast<std::size_t>(m_tiles); i++)
    {
        auto tile = prediction_tiles[i].get();
        std::copy(tile.begin(), tile.end(), std::back_inserter(prediction_result));
    }

    // Synchronize full covariance
    for (std::size_t i = 0; i < static_cast<std::size_t>(m_tiles); i++)
    {
        auto tile = full_covariance_tiles[i].get();
        std::copy(tile.begin(), tile.end(), std::back_inserter(full_covariance_result));
    }

    return std::vector<std::vector<double>>{ std::move(prediction_result), std::move(full_covariance_result) };
}

///////////////////////////////////////////////////////////////////////////
// OPTIMIZATION
double compute_loss_hpx(const std::vector<double> &training_input,
                        const std::vector<double> &training_output,
                        const std::vector<double> &hyperparameters,
                        int n_tiles,
                        int n_tile_size,
                        int n_regressors)
{
    // Tiled future data structures for prediction
    Tiled_matrix K_tiles;      // Tiled covariance matrix K_NxN
    Tiled_vector y_tiles;      // Tiled output
    Tiled_vector alpha_tiles;  // Tiled intermediate solution
    hpx::shared_future<double> loss_value;

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
                hyperparameters,
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
             const gprat_hyper::Hyperparameters &hyperparams,
             std::vector<double> &kernel_hyperparams,
             std::vector<bool> trainable_params)
{
    std::vector<double> hyperparameters(7);
    hyperparameters[0] = kernel_hyperparams[0];      // lengthscale
    hyperparameters[1] = kernel_hyperparams[1];      // vertical_lengthscale
    hyperparameters[2] = kernel_hyperparams[2];      // noise_variance
    hyperparameters[3] = hyperparams.learning_rate;  // learning rate
    hyperparameters[4] = hyperparams.beta1;          // beta1
    hyperparameters[5] = hyperparams.beta2;          // beta2
    hyperparameters[6] = hyperparams.epsilon;        // epsilon
    // declare data structures
    // tiled future data structures
    std::vector<hpx::shared_future<std::vector<double>>> K_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_v_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_l_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_K_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_I_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> alpha_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> y_tiles;
    // data holders for Adam
    std::vector<hpx::shared_future<double>> m_T;
    std::vector<hpx::shared_future<double>> v_T;
    std::vector<hpx::shared_future<double>> beta1_T;
    std::vector<hpx::shared_future<double>> beta2_T;
    // data holder for loss
    hpx::shared_future<double> loss_value;
    // data holder for computed loss values
    std::vector<double> losses;
    losses.resize(static_cast<std::size_t>(hyperparams.opt_iter));
    //////////////////////////////////////////////////////////////////////////////
    // Assemble beta1_t and beta2_t
    beta1_T.resize(static_cast<std::size_t>(hyperparams.opt_iter));
    for (std::size_t i = 0; i < static_cast<std::size_t>(hyperparams.opt_iter); i++)
    {
        beta1_T[i] = hpx::async(hpx::annotated_function(gen_beta_T, "assemble_tiled"), i + 1, hyperparameters, 4);
    }
    beta2_T.resize(static_cast<std::size_t>(hyperparams.opt_iter));
    for (std::size_t i = 0; i < static_cast<std::size_t>(hyperparams.opt_iter); i++)
    {
        beta2_T[i] = hpx::async(hpx::annotated_function(gen_beta_T, "assemble_tiled"), i + 1, hyperparameters, 5);
    }
    // Assemble first and second momemnt vectors: m_T and v_T
    m_T.resize(3);
    v_T.resize(3);
    for (std::size_t i = 0; i < 3; i++)
    {
        m_T[i] = hpx::async(hpx::annotated_function(gen_zero, "assemble_tiled"));
        v_T[i] = hpx::async(hpx::annotated_function(gen_zero, "assemble_tiled"));
    }
    // Assemble y
    y_tiles.resize(static_cast<std::size_t>(n_tiles));
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        y_tiles[i] =
            hpx::async(hpx::annotated_function(gen_tile_output, "assemble_y"), i, n_tile_size, training_output);
    }

    //////////////////////////////////////////////////////////////////////////////
    // Perform optimization
    for (std::size_t iter = 0; iter < static_cast<std::size_t>(hyperparams.opt_iter); iter++)
    {
        // Assemble covariance matrix vector, derivative of covariance matrix
        // vector w.r.t. to vertical lengthscale and derivative of covariance
        // matrix vector w.r.t. to lengthscale
        K_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));
        grad_v_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));
        grad_l_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));
        for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
        {
            for (std::size_t j = 0; j <= i; j++)
            {
                hpx::shared_future<std::vector<double>> cov_dists = hpx::async(
                    hpx::annotated_function(compute_cov_dist_vec, "assemble_cov_dist"),
                    i,
                    j,
                    n_tile_size,
                    n_regressors,
                    hyperparameters,
                    training_input);

                K_tiles[i * static_cast<std::size_t>(n_tiles) + j] = hpx::dataflow(
                    hpx::annotated_function(hpx::unwrapping(&gen_tile_covariance_opt), "assemble_K"),
                    i,
                    j,
                    n_tile_size,
                    hyperparameters,
                    cov_dists);

                grad_v_tiles[i * static_cast<std::size_t>(n_tiles) + j] = hpx::dataflow(
                    hpx::annotated_function(hpx::unwrapping(&gen_tile_grad_v), "assemble_gradv"),
                    n_tile_size,
                    hyperparameters,
                    cov_dists);

                grad_l_tiles[i * static_cast<std::size_t>(n_tiles) + j] = hpx::dataflow(
                    hpx::annotated_function(hpx::unwrapping(&gen_tile_grad_l), "assemble_gradl"),
                    n_tile_size,
                    hyperparameters,
                    cov_dists);

                if (i != j)
                {
                    grad_v_tiles[j * static_cast<std::size_t>(n_tiles) + i] = hpx::dataflow(
                        hpx::annotated_function(hpx::unwrapping(&gen_tile_grad_v_trans), "assemble_gradv_t"),
                        n_tile_size,
                        grad_v_tiles[i * static_cast<std::size_t>(n_tiles) + j]);

                    grad_l_tiles[j * static_cast<std::size_t>(n_tiles) + i] = hpx::dataflow(
                        hpx::annotated_function(hpx::unwrapping(&gen_tile_grad_l_trans), "assemble_gradl_t"),
                        n_tile_size,
                        grad_l_tiles[i * static_cast<std::size_t>(n_tiles) + j]);
                }
            }
        }
        // Assemble placeholder matrix for K^-1 * (I - y*y^T*K^-1)
        grad_K_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));
        for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
        {
            for (std::size_t j = 0; j < static_cast<std::size_t>(n_tiles); j++)
            {
                grad_K_tiles[i * static_cast<std::size_t>(n_tiles) + j] =
                    hpx::async(hpx::annotated_function(gen_tile_identity, "assemble_tiled"), i, j, n_tile_size);
            }
        }
        // Assemble alpha
        alpha_tiles.resize(static_cast<std::size_t>(n_tiles));
        for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
        {
            alpha_tiles[i] = hpx::async(hpx::annotated_function(gen_tile_zeros, "assemble_tiled"), n_tile_size);
        }
        // Assemble placeholder matrix for K^-1
        grad_I_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));
        for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
        {
            for (std::size_t j = 0; j < static_cast<std::size_t>(n_tiles); j++)
            {
                grad_I_tiles[i * static_cast<std::size_t>(n_tiles) + j] = hpx::async(
                    hpx::annotated_function(gen_tile_identity, "assemble_identity_matrix"), i, j, n_tile_size);
            }
        }

        //////////////////////////////////////////////////////////////////////////////
        // Cholesky decomposition
        right_looking_cholesky_tiled(K_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));
        // Compute K^-1 through L*L^T*X = I
        forward_solve_tiled_matrix(
            K_tiles,
            grad_I_tiles,
            n_tile_size,
            n_tile_size,
            static_cast<std::size_t>(n_tiles),
            static_cast<std::size_t>(n_tiles));
        backward_solve_tiled_matrix(
            K_tiles,
            grad_I_tiles,
            n_tile_size,
            n_tile_size,
            static_cast<std::size_t>(n_tiles),
            static_cast<std::size_t>(n_tiles));

        // Triangular solve K_NxN * alpha = y
        // forward_solve_tiled(grad_I_tiles, alpha_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));
        // backward_solve_tiled(grad_I_tiles, alpha_tiles, n_tile_size,
        // static_cast<std::size_t>(n_tiles));

        // inv(K)*y
        compute_gemm_of_invK_y(grad_I_tiles, y_tiles, alpha_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));

        // Compute loss
        compute_loss_tiled(K_tiles, alpha_tiles, y_tiles, loss_value, n_tile_size, static_cast<std::size_t>(n_tiles));
        losses[iter] = loss_value.get();

        // Compute I-y*y^T*inv(K) -> NxN matrix
        // update_grad_K_tiled(grad_K_tiles, y_tiles, alpha_tiles, n_tile_size,
        // static_cast<std::size_t>(n_tiles));

        // Compute K^-1 *(I - y*y^T*K^-1)
        // forward_solve_tiled_matrix(K_tiles, grad_K_tiles, n_tile_size,
        // n_tile_size, static_cast<std::size_t>(n_tiles), static_cast<std::size_t>(n_tiles));
        // backward_solve_tiled_matrix(K_tiles, grad_K_tiles, n_tile_size, n_tile_size,
        // static_cast<std::size_t>(n_tiles), static_cast<std::size_t>(n_tiles));

        // Update the hyperparameters
        if (trainable_params[0])
        {  // lengthscale
            update_hyperparameter(
                grad_I_tiles,
                grad_l_tiles,
                alpha_tiles,
                hyperparameters,
                n_tile_size,
                static_cast<std::size_t>(n_tiles),
                m_T,
                v_T,
                beta1_T,
                beta2_T,
                0,
                0);
        }
        if (trainable_params[1])
        {  // vertical_lengthscale
            update_hyperparameter(
                grad_I_tiles,
                grad_v_tiles,
                alpha_tiles,
                hyperparameters,
                n_tile_size,
                static_cast<std::size_t>(n_tiles),
                m_T,
                v_T,
                beta1_T,
                beta2_T,
                0,
                1);
        }
        if (trainable_params[2])
        {  // noise_variance
            update_noise_variance(
                grad_I_tiles,
                alpha_tiles,
                hyperparameters,
                n_tile_size,
                static_cast<std::size_t>(n_tiles),
                m_T,
                v_T,
                beta1_T,
                beta2_T,
                iter);
        }
    }
    // Update hyperparameter attributes in Gaussian process model
    kernel_hyperparams[0] = hyperparameters[0];
    kernel_hyperparams[1] = hyperparameters[1];
    kernel_hyperparams[2] = hyperparameters[2];
    // Return losses
    return losses;
}

double optimize_step_hpx(
    const std::vector<double> &training_input,
    const std::vector<double> &training_output,
    int n_tiles,
    int n_tile_size,
    int n_regressors,
    gprat_hyper::Hyperparameters &hyperparams,
    std::vector<double> &kernel_hyperparams,
    std::vector<bool> trainable_params,
    int iter)
{
    std::vector<double> hyperparameters(7);
    hyperparameters[0] = kernel_hyperparams[0];      // lengthscale
    hyperparameters[1] = kernel_hyperparams[1];      // vertical_lengthscale
    hyperparameters[2] = kernel_hyperparams[2];      // noise_variance
    hyperparameters[3] = hyperparams.learning_rate;  // learning rate
    hyperparameters[4] = hyperparams.beta1;          // beta1
    hyperparameters[5] = hyperparams.beta2;          // beta2
    hyperparameters[6] = hyperparams.epsilon;        // epsilon
    // declare data structures
    // tiled future data structures
    std::vector<hpx::shared_future<std::vector<double>>> K_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_v_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_l_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_K_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_I_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> alpha_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> y_tiles;
    // data holders for Adam
    std::vector<hpx::shared_future<double>> m_T;
    std::vector<hpx::shared_future<double>> v_T;
    std::vector<hpx::shared_future<double>> beta1_T;
    std::vector<hpx::shared_future<double>> beta2_T;
    // data holder for loss
    hpx::shared_future<double> loss_value;
    // make shared future
    for (std::size_t i = 0; i < 3; i++)
    {
        hpx::shared_future<double> m = hpx::make_ready_future(hyperparams.M_T[i]);  //.share();
        m_T.push_back(m);
        hpx::shared_future<double> v = hpx::make_ready_future(hyperparams.V_T[i]);  //.share();
        v_T.push_back(v);
    }
    //////////////////////////////////////////////////////////////////////////////
    // Assemble beta1_t and beta2_t
    beta1_T.resize(1);
    beta1_T[0] = hpx::async(hpx::annotated_function(gen_beta_T, "assemble_tiled"), iter + 1, hyperparameters, 4);
    beta2_T.resize(1);
    beta2_T[0] = hpx::async(hpx::annotated_function(gen_beta_T, "assemble_tiled"), iter + 1, hyperparameters, 5);
    // Assemble covariance matrix vector
    K_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            K_tiles[i * static_cast<std::size_t>(n_tiles) + j] = hpx::async(
                hpx::annotated_function(gen_tile_covariance, "assemble_tiled"),
                i,
                j,
                n_tile_size,
                n_regressors,
                hyperparameters,
                training_input);
        }
    }
    // Assemble derivative of covariance matrix vector w.r.t. to vertical
    // lengthscale
    grad_v_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        for (std::size_t j = 0; j < static_cast<std::size_t>(n_tiles); j++)
        {
            grad_v_tiles[i * static_cast<std::size_t>(n_tiles) + j] =
                hpx::async(hpx::annotated_function(gen_tile_grad_v, "assemble_tiled"),
                           n_tile_size,
                           hyperparameters,
                           training_input);
        }
    }
    // Assemble derivative of covariance matrix vector w.r.t. to lengthscale
    grad_l_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        for (std::size_t j = 0; j < static_cast<std::size_t>(n_tiles); j++)
        {
            grad_l_tiles[i * static_cast<std::size_t>(n_tiles) + j] =
                hpx::async(hpx::annotated_function(gen_tile_grad_l, "assemble_tiled"),
                           n_tile_size,
                           hyperparameters,
                           training_input);
        }
    }
    // Assemble matrix that will be multiplied with derivates
    grad_K_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        for (std::size_t j = 0; j < static_cast<std::size_t>(n_tiles); j++)
        {
            grad_K_tiles[i * static_cast<std::size_t>(n_tiles) + j] =
                hpx::async(hpx::annotated_function(gen_tile_identity, "assemble_tiled"), i, j, n_tile_size);
        }
    }
    // Assemble alpha
    alpha_tiles.resize(static_cast<std::size_t>(n_tiles));
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        alpha_tiles[i] =
            hpx::async(hpx::annotated_function(gen_tile_output, "assemble_tiled"), i, n_tile_size, training_output);
    }
    // Assemble y
    y_tiles.resize(static_cast<std::size_t>(n_tiles));
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        y_tiles[i] =
            hpx::async(hpx::annotated_function(gen_tile_output, "assemble_tiled"), i, n_tile_size, training_output);
    }
    // Assemble placeholder matrix for K^-1
    grad_I_tiles.resize(static_cast<std::size_t>(n_tiles * n_tiles));
    for (std::size_t i = 0; i < static_cast<std::size_t>(n_tiles); i++)
    {
        for (std::size_t j = 0; j < static_cast<std::size_t>(n_tiles); j++)
        {
            grad_I_tiles[i * static_cast<std::size_t>(n_tiles) + j] =
                hpx::async(hpx::annotated_function(gen_tile_identity, "assemble_identity_matrix"), i, j, n_tile_size);
        }
    }
    //////////////////////////////////////////////////////////////////////////////
    // Cholesky decomposition
    right_looking_cholesky_tiled(K_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));
    // Triangular solve K_NxN * alpha = y
    forward_solve_tiled(K_tiles, alpha_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));
    backward_solve_tiled(K_tiles, alpha_tiles, n_tile_size, static_cast<std::size_t>(n_tiles));

    // Compute K^-1 through L*L^T*X = I
    forward_solve_tiled_matrix(
        K_tiles,
        grad_I_tiles,
        n_tile_size,
        n_tile_size,
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(n_tiles));
    backward_solve_tiled_matrix(
        K_tiles,
        grad_I_tiles,
        n_tile_size,
        n_tile_size,
        static_cast<std::size_t>(n_tiles),
        static_cast<std::size_t>(n_tiles));

    // Compute loss
    compute_loss_tiled(K_tiles, alpha_tiles, y_tiles, loss_value, n_tile_size, static_cast<std::size_t>(n_tiles));

    // // Fill I-y*y^T*inv(K)
    // update_grad_K_tiled(grad_K_tiles, y_tiles, alpha_tiles, n_tile_size,
    // static_cast<std::size_t>(n_tiles));

    // // Compute K^-1 * (I-y*y^T*K^-1)
    // forward_solve_tiled_matrix(K_tiles, grad_K_tiles, n_tile_size,
    // n_tile_size, static_cast<std::size_t>(n_tiles), static_cast<std::size_t>(n_tiles));
    // backward_solve_tiled_matrix(K_tiles, grad_K_tiles, n_tile_size, n_tile_size, static_cast<std::size_t>(n_tiles),
    // static_cast<std::size_t>(n_tiles));

    // Update the hyperparameters
    if (trainable_params[0])
    {  // lengthscale
        update_hyperparameter(
            grad_I_tiles,
            grad_l_tiles,
            alpha_tiles,
            hyperparameters,
            n_tile_size,
            static_cast<std::size_t>(n_tiles),
            m_T,
            v_T,
            beta1_T,
            beta2_T,
            0,
            0);
    }

    if (trainable_params[1])
    {  // vertical_lengthscale
        update_hyperparameter(
            grad_K_tiles,
            grad_v_tiles,
            alpha_tiles,
            hyperparameters,
            n_tile_size,
            static_cast<std::size_t>(n_tiles),
            m_T,
            v_T,
            beta1_T,
            beta2_T,
            0,
            1);
    }

    if (trainable_params[2])
    {  // noise_variance
        update_noise_variance(
            grad_K_tiles,
            alpha_tiles,
            hyperparameters,
            n_tile_size,
            static_cast<std::size_t>(n_tiles),
            m_T,
            v_T,
            beta1_T,
            beta2_T,
            0);
    }

    // Update hyperparameter attributes in Gaussian process model
    kernel_hyperparams[0] = hyperparameters[0];
    kernel_hyperparams[1] = hyperparameters[1];
    kernel_hyperparams[2] = hyperparameters[2];
    // Update hyperparameter attributes (first and second moment) for Adam
    for (std::size_t i = 0; i < 3; i++)
    {
        hyperparams.M_T[i] = m_T[i].get();
        hyperparams.V_T[i] = v_T[i].get();
    }

    return loss_value.get();
}
