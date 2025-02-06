#ifndef GP_FUNCTIONS_H
#define GP_FUNCTIONS_H

#include <hpx/future.hpp>
#include <vector>

namespace gprat_hyper
{
struct Hyperparameters
{
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    int opt_iter;
    std::vector<double> M_T;
    std::vector<double> V_T;

    // Initialize Hyperparameter constructor
    Hyperparameters(double lr = 0.001,
                    double b1 = 0.9,
                    double b2 = 0.999,
                    double eps = 1e-8,
                    int opt_i = 0,
                    std::vector<double> M_T = { 0.0, 0.0, 0.0 },
                    std::vector<double> V_T = { 0.0, 0.0, 0.0 });

    // Print Hyperparameter attributes
    std::string repr() const;
};
}  // namespace gprat_hyper

// Compute the predictions
std::vector<double> predict_hpx(
    const std::vector<double> &training_input,
    const std::vector<double> &training_output,
    const std::vector<double> &test_data,
    int n_tiles,
    int n_tile_size,
    int m_tiles,
    int m_tile_size,
    double lengthscale,
    double vertical_lengthscale,
    double noise_variance,
    int n_regressors);

// Compute the predictions and uncertainties
std::vector<std::vector<double>> predict_with_uncertainty_hpx(
    const std::vector<double> &training_input,
    const std::vector<double> &training_output,
    const std::vector<double> &test_input,
    int n_tiles,
    int n_tile_size,
    int m_tiles,
    int m_tile_size,
    double lengthscale,
    double vertical_lengthscale,
    double noise_variance,
    int n_regressors);

// Compute the predictions and full covariance matrix
std::vector<std::vector<double>> predict_with_full_cov_hpx(
    const std::vector<double> &training_input,
    const std::vector<double> &training_output,
    const std::vector<double> &test_data,
    int n_tiles,
    int n_tile_size,
    int m_tiles,
    int m_tile_size,
    double lengthscale,
    double vertical_lengthscale,
    double noise_variance,
    int n_regressors);

// Compute loss for given data and Gaussian process model
double compute_loss_hpx(const std::vector<double> &training_input,
                        const std::vector<double> &training_output,
                        int n_tiles,
                        int n_tile_size,
                        int n_regressors,
                        const std::vector<double> &hyperparameters);

// Perform optimization for a given number of iterations
std::vector<double> optimize_hpx(
    const std::vector<double> &training_input,
    const std::vector<double> &training_output,
    int n_tiles,
    int n_tile_size,
    double &lengthscale,
    double &vertical_lengthscale,
    double &noise_variance,
    int n_regressors,
    const gprat_hyper::Hyperparameters &hyperparams,
    std::vector<bool> trainable_params);

// Perform a single optimization step
double optimize_step_hpx(
    const std::vector<double> &training_input,
    const std::vector<double> &training_output,
    int n_tiles,
    int n_tile_size,
    double &lengthscale,
    double &vertical_lengthscale,
    double &noise_variance,
    int n_regressors,
    gprat_hyper::Hyperparameters &hyperparams,
    std::vector<bool> trainable_params,
    int iter);

// Compute Cholesky decomposition
std::vector<std::vector<double>>
cholesky_hpx(const std::vector<double> &training_input,
             int n_tiles,
             int n_tile_size,
             double lengthscale,
             double vertical_lengthscale,
             double noise_variance,
             int n_regressors);

#endif
