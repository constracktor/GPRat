#include "../include/gp_uncertainty.hpp"

std::vector<double> diag_posterior(const std::vector<double> &a, const std::vector<double> &b, std::size_t M)
{
    // Preallocate memory
    std::vector<double> tile;
    tile.reserve(M);
    // Add elements
    for (std::size_t i = 0; i < M; ++i)
    {
        tile.push_back(a[i] - b[i]);
    }

    return tile;
}


std::vector<double> diag_tile(const std::vector<double> &A, std::size_t M)
{
    // Preallocate memory
    std::vector<double> tile;
    tile.reserve(M);
    // Add elements
    for (std::size_t i = 0; i < M; ++i)
    {
        tile.push_back(A[i * M + i]);
    }

    return tile;
}
