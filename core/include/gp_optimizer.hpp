#ifndef GP_OPTIMIZER_H
#define GP_OPTIMIZER_H

#include "gp_hyperparameters.hpp"
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
double compute_covariance_distance(std::size_t i_global,
                                   std::size_t j_global,
                                   std::size_t n_regressors,
                                   const gprat_hyper::SEKParams &sek_params,
                                   const std::vector<double> &i_input,
                                   const std::vector<double> &j_input);

/**
 * @brief Generate a tile of distances divided by the lengthscale
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
std::vector<double> gen_tile_distance(
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
std::vector<double> gen_tile_covariance_with_distance(
    std::size_t row,
    std::size_t col,
    std::size_t N,
    const gprat_hyper::SEKParams &sek_params,
    const std::vector<double> &distance);

/* @brief  Generate a derivative tile w.r.t. vertical_lengthscale v
 *
 * @param N The dimension of the quadratic tile (N*N elements)
 * @param hyperparameters The kernel hyperparameters
 * @param cov_dists The pre-computed distances for the tile
 *
 * @return A quadratic tile of the derivative of v of size N x N
 */
std::vector<double>
gen_tile_grad_v(std::size_t N, const gprat_hyper::SEKParams &sek_params, const std::vector<double> &distance);

/* @brief  Generate a derivative tile w.r.t. lengthscale l
 *
 * @param N The dimension of the quadratic tile (N*N elements)
 * @param hyperparameters The kernel hyperparameters
 * @param cov_dists The pre-computed distances for the tile
 *
 * @return A quadratic tile of the derivative of l of size N x N
 */
std::vector<double>
gen_tile_grad_l(std::size_t N, const gprat_hyper::SEKParams &sek_params, const std::vector<double> &distance);

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
double adam_step(const double unconstrained_hyperparam,
                 const gprat_hyper::AdamParams adam_params,
                 double m_T,
                 double v_T,
                 std::size_t iter);

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

double compute_gradient(double trace, double dot, std::size_t N, std::size_t n_tiles);

double compute_trace(const std::vector<double> &diagonal, double trace);

double compute_dot(const std::vector<double> &vector_T, const std::vector<double> &vector, double result);

double
compute_trace_noise(const std::vector<double> &ft_invK, double trace, const double noise_variance, std::size_t N);

double compute_dot_noise(const std::vector<double> &vector, double result, const double noise_variance);

#endif  // end of GP_OPTIMIZER_H
