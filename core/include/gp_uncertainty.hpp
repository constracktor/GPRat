#ifndef GP_UNCERTAINTY_H
#define GP_UNCERTAINTY_H
#pragma once
#include <cmath>
#include <vector>

/**
 * @brief Retrieve diagonal elements of posterior covariance matrix.
 *
 * @param A Diagonal elements matrix A
 * @param B Diagonal elements matrix B
 * @param M Number of rows in the matrix
 *
 * @return Diagonal elements of posterior covariance matrix
 */
std::vector<double> diag_posterior(const std::vector<double> &A, const std::vector<double> &B, std::size_t M);

/**
 * @brief Retrieve diagonal elements of posterior covariance matrix.
 *
 * @param A Posterior covariance matrix
 * @param M Number of rows in the matrix
 *
 * @return Diagonal elements of posterior covariance matrix
 */
std::vector<double> diag_tile(const std::vector<double> &A, std::size_t M);

#endif  // GP_UNCERTAINTY_H
