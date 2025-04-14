#ifndef GPU_GP_UNCERTAINTY_H
#define GPU_GP_UNCERTAINTY_H

#include "target.hpp"

namespace gpu
{

/**
 * @brief Retrieve diagonal elements of posterior covariance matrix.
 *
 * @param A Diagonal elements matrix A
 * @param B Diagonal elements matrix B
 * @param M Number of rows in the matrix
 *
 * @return Diagonal elements of posterior covariance matrix
 */
hpx::shared_future<double *> diag_posterior(
    const hpx::shared_future<double *> A, const hpx::shared_future<double *> B, std::size_t M, gprat::CUDA_GPU &gpu);

/**
 * @brief Retrieve diagonal elements of posterior covariance matrix.
 *
 * @param A Posterior covariance matrix
 * @param M Number of rows in the matrix
 *
 * @return Diagonal elements of posterior covariance matrix
 */
hpx::shared_future<double *> diag_tile(const hpx::shared_future<double *> A, std::size_t M, gprat::CUDA_GPU &gpu);

}  // end of namespace gpu

#endif  // end of GPU_GP_UNCERTAINTY_H
