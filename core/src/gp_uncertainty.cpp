#include "../include/gp_uncertainty.hpp"

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
std::vector<double> diag_posterior(const std::vector<double> &A, const std::vector<double> &B, std::size_t M)
{
    // Initialize tile
    std::vector<double> tile;
    tile.reserve(M);

    for (std::size_t i = 0; i < M; ++i)
    {
        tile.push_back(A[i] - B[i]);
    }

    return tile;
}

/**
 * @brief Retrieve diagonal elements of posterior covariance matrix.
 *
 * @param A Posterior covariance matrix
 * @param M Number of rows in the matrix
 *
 * @return Diagonal elements of posterior covariance matrix
 */
std::vector<double> diag_tile(const std::vector<double> &A, std::size_t M)
{
    // Initialize tile
    std::vector<double> tile;
    tile.reserve(M);

    for (std::size_t i = 0; i < M; ++i)
    {
        tile.push_back(A[i * M + i]);
    }

    return tile;
}
