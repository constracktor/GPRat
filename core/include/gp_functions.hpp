#ifndef GP_FUNCTIONS_H
#define GP_FUNCTIONS_H

#include <hpx/future.hpp>
#include <vector>

namespace gprat_hyper
{

/**
 * @brief Data structure to hold the hyperparameters of the Adam optimizer
 */
struct Hyperparameters
{
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    int opt_iter;
    std::vector<double> M_T;
    std::vector<double> V_T;

    /**
     * @brief Constructor for Adam hyperparameters
     *
     * @param lr Learning rate
     * @param b1 Beta1
     * @param b2 Beta2
     * @param eps Epsilon
     * @param opt_i The Number of optimization iterations
     * @param M_T_init The initial values for first moment vector
     * @param V_T_init The initial values for second moment vector
     */
    Hyperparameters(double lr = 0.001,
                    double b1 = 0.9,
                    double b2 = 0.999,
                    double eps = 1e-8,
                    int opt_i = 0,
                    std::vector<double> M_T = { 0.0, 0.0, 0.0 },
                    std::vector<double> V_T = { 0.0, 0.0, 0.0 });

    /**
     * @brief Returns a string representation of the hyperparameters
     */
    std::string repr() const;
};
}  // namespace gprat_hyper

/**
 * @brief Compute the predictions without uncertainties.
 *
 * @param training_input The training input data
 * @param training_output The raining output data
 * @param test_input The test input data
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param m_tiles The number of test tiles
 * @param m_tile_size The size of each test tile
 * @param hyperparameters The kernel hyperparameters
 * @param n_regressors The number of regressors
 *
 * @return A vector containing the predictions
 */
std::vector<double>
predict_hpx(const std::vector<double> &training_input,
            const std::vector<double> &training_output,
            const std::vector<double> &test_input,
            const std::vector<double> &hyperparameters,
            int n_tiles,
            int n_tile_size,
            int m_tiles,
            int m_tile_size,
            int n_regressors);

// Compute the predictions and uncertainties
std::vector<std::vector<double>> predict_with_uncertainty_hpx(
    const std::vector<double> &training_input,
    const std::vector<double> &training_output,
    const std::vector<double> &test_input,
    const std::vector<double> &hyperparameters,
    int n_tiles,
    int n_tile_size,
    int m_tiles,
    int m_tile_size,
    int n_regressors);

// Compute the predictions and full covariance matrix
std::vector<std::vector<double>> predict_with_full_cov_hpx(
    const std::vector<double> &training_input,
    const std::vector<double> &training_output,
    const std::vector<double> &test_data,
    const std::vector<double> &hyperparameters,
    int n_tiles,
    int n_tile_size,
    int m_tiles,
    int m_tile_size,
    int n_regressors);

// Compute loss for given data and Gaussian process model
double compute_loss_hpx(const std::vector<double> &training_input,
                        const std::vector<double> &training_output,
                        const std::vector<double> &hyperparameters,
                        int n_tiles,
                        int n_tile_size,
                        int n_regressors);

// Perform optimization for a given number of iterations
std::vector<double>
optimize_hpx(const std::vector<double> &training_input,
             const std::vector<double> &training_output,
             int n_tiles,
             int n_tile_size,
             int n_regressors,
             const gprat_hyper::Hyperparameters &hyperparams,
             std::vector<double> &hyperparameters,
             std::vector<bool> trainable_params);

// Perform a single optimization step
double optimize_step_hpx(
    const std::vector<double> &training_input,
    const std::vector<double> &training_output,
    int n_tiles,
    int n_tile_size,
    int n_regressors,
    gprat_hyper::Hyperparameters &hyperparams,
    std::vector<double> &hyperparameters,
    std::vector<bool> trainable_params,
    int iter);

// Compute Cholesky decomposition
std::vector<std::vector<double>>
cholesky_hpx(const std::vector<double> &training_input,
             const std::vector<double> &hyperparamaters,
             int n_tiles,
             int n_tile_size,
             int n_regressors);

#endif
