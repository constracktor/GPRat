#ifndef GPRAT_CPU_GP_UNCERTAINTY_HPP
#define GPRAT_CPU_GP_UNCERTAINTY_HPP

#pragma once

#include "gprat/detail/config.hpp"

#include <hpx/future.hpp>
#include <vector>

GPRAT_NS_BEGIN

namespace cpu
{

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

}  // end of namespace cpu

GPRAT_NS_END

#endif  // end of CPU_GP_UNCERTAINTY_H
