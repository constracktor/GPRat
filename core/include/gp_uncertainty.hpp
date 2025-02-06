#ifndef GP_UNCERTAINTY_H
#define GP_UNCERTAINTY_H
#pragma once
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
// Is this function really necessary? why not replace with saxpy blas a-b inside
// NAME: compute_diagonal_difference?
std::vector<double> diag_posterior(const std::vector<double> &a, const std::vector<double> &b, std::size_t M);

/**
 * @brief Retrieve diagonal elements of the posterior covariance matrix A.
 *
 * @param A The posterior covariance matrix
 * @param M The rumber of rows in the matrix
 *
 * @return Diagonal element vector of the posterior covariance matrix A of size M
 */
// NAME: get_diagonal?
std::vector<double> diag_tile(const std::vector<double> &A, std::size_t M);

#endif  // GP_UNCERTAINTY_H
