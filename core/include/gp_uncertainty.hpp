#ifndef GP_UNCERTAINTY_H
#define GP_UNCERTAINTY_H
#pragma once
#include <hpx/future.hpp>
#include <vector>

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
