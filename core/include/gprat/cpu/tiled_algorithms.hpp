#ifndef GPRAT_CPU_TILED_ALGORITHMS_H
#define GPRAT_CPU_TILED_ALGORITHMS_H

#pragma once

#include "gprat/detail/config.hpp"
#include "gprat/hyperparameters.hpp"
#include "gprat/kernels.hpp"
#include "gprat/tile_data.hpp"

#include <hpx/future.hpp>

GPRAT_NS_BEGIN

using Tiled_matrix = std::vector<hpx::shared_future<mutable_tile_data<double>>>;
using Tiled_vector = std::vector<hpx::shared_future<mutable_tile_data<double>>>;

namespace cpu
{

// Tiled Cholesky Algorithm

/**
 * @brief Perform right-looking tiled Cholesky decomposition.
 *
 * @param ft_tiles Tiled matrix represented as a vector of futurized tiles, containing the
 *        covariance matrix, afterwards the Cholesky decomposition.
 * @param N Tile size per dimension.
 * @param n_tiles Number of tiles per dimension.
 */
void right_looking_cholesky_tiled(Tiled_matrix &ft_tiles, std::size_t N, std::size_t n_tiles);

// Tiled Triangular Solve Algorithms

/**
 * @brief Perform tiled forward triangular matrix-vector solve.
 *
 * @param ft_tiles Tiled triangular matrix represented as a vector of futurized tiles.
 * @param ft_rhs Tiled right-hand side vector, afterwards containing the tiled solution vector
 * @param N Tile size per dimension.
 * @param n_tiles Number of tiles per dimension.
 */
void forward_solve_tiled(Tiled_matrix &ft_tiles, Tiled_vector &ft_rhs, std::size_t N, std::size_t n_tiles);

/**
 * @brief Perform tiled backward triangular matrix-vector solve.
 *
 * @param ft_tiles Tiled triangular matrix represented as a vector of futurized tiles.
 * @param ft_rhs Tiled right-hand side vector, afterwards containing the tiled solution vector
 * @param N Tile size per dimension.
 * @param n_tiles Number of tiles per dimension.
 */
void backward_solve_tiled(Tiled_matrix &ft_tiles, Tiled_vector &ft_rhs, std::size_t N, std::size_t n_tiles);

/**
 * @brief Perform tiled forward triangular matrix-matrix solve.
 *
 * @param ft_tiles Tiled triangular matrix represented as a vector of futurized tiles.
 * @param ft_rhs Tiled right-hand side matrix, afterwards containing the tiled solution matrix.
 * @param N Tile size of first dimension.
 * @param M Tile size of second dimension.
 * @param n_tiles Number of tiles in first dimension.
 * @param m_tiles Number of tiles in second dimension.
 */
void forward_solve_tiled_matrix(Tiled_matrix &ft_tiles,
                                Tiled_matrix &ft_rhs,
                                std::size_t N,
                                std::size_t M,
                                std::size_t n_tiles,
                                std::size_t m_tiles);

/**
 * @brief Perform tiled backward triangular matrix-matrix solve.
 *
 * @param ft_tiles Tiled triangular matrix represented as a vector of futurized tiles.
 * @param ft_rhs Tiled right-hand side matrix, afterwards containing the tiled solution matrix.
 * @param N Tile size of first dimension.
 * @param M Tile size of second dimension.
 * @param n_tiles Number of tiles in first dimension.
 * @param m_tiles Number of tiles in second dimension.
 */
void backward_solve_tiled_matrix(Tiled_matrix &ft_tiles,
                                 Tiled_matrix &ft_rhs,
                                 std::size_t N,
                                 std::size_t M,
                                 std::size_t n_tiles,
                                 std::size_t m_tiles);

/**
 * @brief Perform tiled matrix-vector multiplication
 *
 * @param ft_tiles Tiled matrix represented as a vector of futurized tiles.
 * @param ft_vector Tiled vector represented as a vector of futurized tiles.
 * @param ft_rhs Tiled solution represented as a vector of futurized tiles.
 * @param N_row Tile size of first dimension.
 * @param N_col Tile size of second dimension.
 * @param n_tiles Number of tiles in first dimension.
 * @param m_tiles Number of tiles in second dimension.
 */
void matrix_vector_tiled(Tiled_matrix &ft_tiles,
                         Tiled_vector &ft_vector,
                         Tiled_vector &ft_rhs,
                         std::size_t N_row,
                         std::size_t N_col,
                         std::size_t n_tiles,
                         std::size_t m_tiles);

/**
 * @brief Perform tiled symmetric k-rank update on diagonal tiles
 *
 * @param ft_tiles Tiled matrix represented as a vector of futurized tiles.
 * @param ft_vector Tiled vector holding the diagonal tile results
 * @param N Tile size of first dimension.
 * @param M Tile size of second dimension.
 * @param n_tiles Number of tiles in first dimension.
 * @param m_tiles Number of tiles in second dimension.
 */
void symmetric_matrix_matrix_diagonal_tiled(
    Tiled_matrix &ft_tiles,
    Tiled_vector &ft_vector,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles);

/**
 * @brief Perform tiled symmetric k-rank update (ft_tiles^T * ft_tiles)
 *
 * @param ft_tiles Tiled matrix represented as a vector of futurized tiles.
 * @param ft_result Tiled matrix holding the result of the computationi.
 * @param N Tile size of first dimension.
 * @param M Tile size of second dimension.
 * @param n_tiles Number of tiles in first dimension.
 * @param m_tiles Number of tiles in second dimension.
 */
void symmetric_matrix_matrix_tiled(Tiled_matrix &ft_tiles,
                                   Tiled_matrix &ft_result,
                                   std::size_t N,
                                   std::size_t M,
                                   std::size_t n_tiles,
                                   std::size_t m_tiles);

/**
 * @brief Compute the difference between two tiled vectors
 * @param ft_minuend Tiled vector that is being subtracted from.
 * @param ft_subtrahend Tiled vector that is being subtracted.
 * @param M Tile size dimension.
 * @param m_tiles Number of tiles.
 */
void vector_difference_tiled(Tiled_vector &ft_minuend, Tiled_vector &ft_subtrahend, std::size_t M, std::size_t m_tiles);

/**
 * @brief Extract the tiled diagonals of a tiled matrix
 * @param ft_tiles Tiled matrix represented as a vector of futurized tiles.
 * @param ft_vector Tiled vector containing the diagonals of the matrix tiles
 * @param M Tile size per dimension.
 * @param m_tiles Number of tiles per dimension.
 */
void matrix_diagonal_tiled(Tiled_matrix &ft_tiles, Tiled_vector &ft_vector, std::size_t M, std::size_t m_tiles);

/**
 * @brief Compute the negative log likelihood loss with a tiled covariance matrix K.
 *
 *  Computes l = 0.5 * ( log(det(K)) + y^T * K^-1 * y) + const.)
 *
 * @param ft_tiles Tiled Cholesky factor matrix represented as a vector of futurized tiles.
 * @param ft_alpha Tiled vector containing the solution of K^-1 * y
 * @param ft_y Tiled vector containing the the training output y
 * @param loss The loss value to be computed
 * @param N Tile size per dimension.
 * @param n_tiles Number of tiles per dimension.
 */
void compute_loss_tiled(Tiled_matrix &ft_tiles,
                        Tiled_vector &ft_alpha,
                        Tiled_vector &ft_y,
                        hpx::shared_future<double> &loss,
                        std::size_t N,
                        std::size_t n_tiles);

/**
 * @brief Updates a hyperparameter of the SEK kernel using Adam
 *
 * @param ft_invK Tiled inverse of the covariance matrix K represented as a vector of futurized tiles.
 * @param ft_gradK_param Tiled covariance matrix gradient w.r.t. a hyperparameter.
 * @param ft_alpha Tiled vector containing the precomputed inv(K) * y where y is the training output.
 * @param adam_params Hyperparameter of the Adam optimizer
 * @param sek_params Hyperparameters of the SEK kernel
 * @param N Tile size per dimension.
 * @param n_tiles Number of tiles per dimension.
 * @param iter Current iteration.
 * @param param_idx Index of the hyperparameter to optimize.
 */
void update_hyperparameter_tiled(
    const Tiled_matrix &ft_invK,
    const Tiled_matrix &ft_gradK_param,
    const Tiled_vector &ft_alpha,
    const AdamParams &adam_params,
    SEKParams &sek_params,
    std::size_t N,
    std::size_t n_tiles,
    std::size_t iter,
    std::size_t param_idx);

}  // end of namespace cpu

GPRAT_NS_END

#endif
