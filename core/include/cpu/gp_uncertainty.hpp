#ifndef CPU_GP_UNCERTAINTY_H
#define CPU_GP_UNCERTAINTY_H

#include <hpx/future.hpp>
#include <vector>

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

#endif  // end of CPU_GP_UNCERTAINTY_H
