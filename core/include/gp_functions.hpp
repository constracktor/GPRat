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
 * @brief Perform Cholesky decompositon (+Assebmly)
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
cholesky_hpx(const std::vector<double> &training_input,
             const std::vector<double> &hyperparamaters,
             int n_tiles,
             int n_tile_size,
             int n_regressors);

/**
 * @brief Compute the predictions without uncertainties.
 *
 * @param training_input The training input data
 * @param training_output The raining output data
 * @param test_input The test input data
 * @param hyperparameters The kernel hyperparameters
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param m_tiles The number of test tiles
 * @param m_tile_size The size of each test tile
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

/**
 * @brief Compute the predictions with uncertainties.
 *
 * @param training_input The training input data
 * @param training_output The raining output data
 * @param test_input The test input data
 * @param hyperparameters The kernel hyperparameters
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param m_tiles The number of test tiles
 * @param m_tile_size The size of each test tile
 * @param n_regressors The number of regressors
 *
 * @return A vector containing the prediction vector and the uncertainty vector
 */
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

/**
 * @brief Compute the predictions with full covariance matrix.
 *
 * @param training_input The training input data
 * @param training_output The raining output data
 * @param test_input The test input data
 * @param hyperparameters The kernel hyperparameters
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param m_tiles The number of test tiles
 * @param m_tile_size The size of each test tile
 * @param n_regressors The number of regressors
 *
 * @return A vector containing the prediction vector and the full posterior covariance matrix
 */
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

/**
 * @brief Compute loss for given data and Gaussian process model
 *
 * @param training_input The training input data
 * @param training_output The raining output data
 * @param hyperparameters The kernel hyperparameters
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param n_regressors The number of regressors
 *
 * @return The loss
 */
double compute_loss_hpx(const std::vector<double> &training_input,
                        const std::vector<double> &training_output,
                        const std::vector<double> &hyperparameters,
                        int n_tiles,
                        int n_tile_size,
                        int n_regressors);

/**
 * @brief Perform optimization for a given number of iterations
 *
 * @param training_input The training input data
 * @param training_output The raining output data
 *
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param n_regressors The number of regressors
 *
 * @param hyperparams The Adam optimizer hyperparameters
 * @param hyperparameters The kernel hyperparameters
 * @param trainable_params The vector containing a bool wheather to train a hyperparameter
 *
 * @return A vector containing the loss values of each iteration
 */
std::vector<double>
optimize_hpx(const std::vector<double> &training_input,
             const std::vector<double> &training_output,
             int n_tiles,
             int n_tile_size,
             int n_regressors,
             const gprat_hyper::Hyperparameters &hyperparams,
             std::vector<double> &hyperparameters,
             std::vector<bool> trainable_params);

/**
 * @brief Perform a single optimization step
 *
 * @param training_input The training input data
 * @param training_output The raining output data
 *
 * @param n_tiles The number of training tiles
 * @param n_tile_size The size of each training tile
 * @param n_regressors The number of regressors
 *
 * @param hyperparams The Adam optimizer hyperparameters
 * @param hyperparameters The kernel hyperparameters
 * @param trainable_params The vector containing a bool wheather to train a hyperparameter
 *
 * @param iter The current optimization iteration
 *
 * @return The loss value
 */
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
#endif
