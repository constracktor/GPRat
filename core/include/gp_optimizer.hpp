#ifndef GP_OPTIMIZER_H
#define GP_OPTIMIZER_H

#include "gp_kernels.hpp"
#include <vector>

/**
 * @brief Transform hyperparameter to enforce constraints using softplus.
 *
 * @param parameter The parameter to constrain
 * @param noise A flag to apply noise
 *
 * @return The constrained parameter
 */
double to_constrained(double parameter, bool noise);

/**
 * @brief Transform hyperparmeter to entire real line using inverse of softplus.
 *
 * Optimizers, such as gradient descent or Adam, work better with unconstrained parameters.
 *
 * @param parameter The parameter to constrain
 * @param noise A flag to apply noise
 *
 * @return The unconstrained parameter
 */
double to_unconstrained(double parameter, bool noise);

/**
 * @brief Calculate the sigmoid function for a given value
 *
 * @param parameter The parameter to input into the function
 *
 * @return The sigmoid value for the given parameter
 */
double compute_sigmoid(double parameter);

/**
 * @brief Compute the distance between two feature vectors divided by the lengthscale
 *
 * @param i_global The global index of the first feature vector
 * @param j_global The global index of the second feature vector
 * @param n_regressors The number of regressors
 * @param hyperparameters The kernel hyperparameters
 * @param i_input The first feature vector
 * @param j_input The second feature vector
 *
 * @return The distance between two features at position i_global,j_global
 */
double compute_covariance_dist_func(
    std::size_t i_global,
    std::size_t j_global,
    std::size_t n_regressors,
    const gprat_hyper::SEKParams &sek_params,
    const std::vector<double> &i_input,
    const std::vector<double> &j_input);

/**
 * @brief Generate a tile of distances devided by the lengthscale
 *
 * @param row The row index of the tile in the tiled matrix
 * @param col The column index of the tile in the tiled matrix
 * @param N The dimension of the quadratic tile (N*N elements)
 * @param n_regressors The number of regressors
 * @param hyperparameters The kernel hyperparameters
 * @param input The input data vector
 *
 * @return A quadratic tile containing the distance between the features of size N x N
 */
// NAME: gen tile distance
std::vector<double> compute_cov_dist_vec(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    const gprat_hyper::SEKParams &sek_params,
    const std::vector<double> &input);

/* @brief Generate a tile of the covariance matrix with given distances
 *
 * @param row The row index of the tile in the tiled matrix
 * @param col The column index of the tile in the tiled matrix
 * @param N The dimension of the quadratic tile (N*N elements)
 * @param hyperparameters The kernel hyperparameters
 * @param cov_dists The pre-computed distances for the tile
 *
 * @return A quadratic tile of the covariance matrix of size N x N
 */
std::vector<double> gen_tile_covariance_opt(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    const gprat_hyper::SEKParams &sek_params,
    const std::vector<double> &cov_dists);

/* @brief  Generate a derivative tile w.r.t. vertical_lengthscale v
 *
 * @param N The dimension of the quadratic tile (N*N elements)
 * @param hyperparameters The kernel hyperparameters
 * @param cov_dists The pre-computed distances for the tile
 *
 * @return A quadratic tile of the derivative of v of size N x N
 */
std::vector<double>
gen_tile_grad_v(std::size_t N, const gprat_hyper::SEKParams &sek_params, const std::vector<double> &cov_dists);

/* @brief  Generate a derivative tile w.r.t. lengthscale l
 *
 * @param N The dimension of the quadratic tile (N*N elements)
 * @param hyperparameters The kernel hyperparameters
 * @param cov_dists The pre-computed distances for the tile
 *
 * @return A quadratic tile of the derivative of l of size N x N
 */
std::vector<double>
gen_tile_grad_l(std::size_t N, const gprat_hyper::SEKParams &sek_params, const std::vector<double> &cov_dists);

/**
 * @brief Compute hyper-parameter beta_1 or beta_2 to power t.
 */
double gen_beta_T(int t, double parameter);

/**
 * @brief Update biased first raw moment estimate.
 */
double update_first_moment(double gradient, double m_T, double beta_1);

/**
 * @brief Update biased second raw moment estimate.
 */
double update_second_moment(double gradient, double v_T, double beta_2);

/**
 * @brief Update hyperparameter using gradient decent.
 */
double update_param(double unconstrained_hyperparam,
                    const std::vector<double> &hyperparameters,
                    double m_T,
                    double v_T,
                    const std::vector<double> &beta1_T,
                    const std::vector<double> &beta2_T,
                    std::size_t iter);
/**
 * @brief return zero - used to initialize moment vectors
 */
double gen_zero();

/**
 * @brief Compute negative-log likelihood on one tile
 *
 * @param K_diag_tile The Cholesky factor L (in a diagonal tile)
 * @param alpha_tile The tiled solution of K * alpha = y
 * @param y_tile The output tile
 *
 * @return Return l = y^T * alpha + \sum_i^N log(L_ii^2)
 */
double compute_loss(const std::vector<double> &K_diag_tile,
                    const std::vector<double> &alpha_tile,
                    const std::vector<double> &y_tile,
                    std::size_t N);

/**
 * @brief Add up negative-log likelihood loss for all tiles
 *
 * @param losses A vector contianing the loss per tile
 * @param N The size of a tile
 * @param n_tiles The number of tiles
 *
 * @return The added up loss plus the constant factor
 */
double add_losses(const std::vector<double> &losses, std::size_t N, std::size_t n);

/**
 * @brief Compute trace of (K^-1 - K^-1*y*y^T*K^-1)* del(K)/del(hyperparam) =
 *        gradient(K) w.r.t. hyperparam.
 */
double compute_gradient(double grad_l, double grad_r, std::size_t N, std::size_t n_tiles);

/**
 * @brief Compute trace for noise variance.
 *
 * Same function as compute_trace with() the only difference that we only use
 * diag tiles multiplied by derivative of noise_variance.
 */
double compute_gradient_noise(const std::vector<std::vector<double>> &ft_tiles,
                              const std::vector<double> &hyperparameters,
                              std::size_t N,
                              std::size_t n_tiles);

double sum_gradleft(const std::vector<double> &diagonal, double grad);

double
sum_gradright(const std::vector<double> &inter_alpha, const std::vector<double> &alpha, double grad, std::size_t N);

double sum_noise_gradleft(
    const std::vector<double> &ft_invK, double grad, const std::vector<double> &hyperparameters, std::size_t N);

double sum_noise_gradright(
    const std::vector<double> &alpha, double grad, const std::vector<double> &hyperparameters, std::size_t N);

#endif  // end of GP_OPTIMIZER_H
