#ifndef GP_UNCERTAINTY_H
#define GP_UNCERTAINTY_H
#pragma once
#include <hpx/future.hpp>
#include <vector>

/**
 * @brief Compute difference of two vectors a - b
 *
 * @param a Diagonal elements matrix A
 * @param b Diagonal elements matrix B
 * @param M Number of rows in the matrix
 *
 * @return Resulting vector of size M
 */
std::vector<double>
compute_vector_difference(const std::vector<double> &a, const std::vector<double> &b, std::size_t M);

/**
 * @brief Extract diagonal elements of the matrix A.
 *
 * @param A The matrix
 * @param M The rumber of rows in the matrix
 *
 * @return Diagonal element vector of the matrix A of size M
 */
// std::vector<double> get_matrix_diagonal(const std::vector<double> &A, std::size_t M);
hpx::shared_future<std::vector<double>> get_matrix_diagonal(hpx::shared_future<std::vector<double>> f_A, std::size_t M);
#endif  // GP_UNCERTAINTY_H
