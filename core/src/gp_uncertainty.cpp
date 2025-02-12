#include "../include/gp_uncertainty.hpp"

hpx::shared_future<std::vector<double>> get_matrix_diagonal(hpx::shared_future<std::vector<double>> f_A, std::size_t M)
{
    auto A = f_A.get();
    // Preallocate memory
    std::vector<double> tile;
    tile.reserve(M);
    // Add elements
    for (std::size_t i = 0; i < M; ++i)
    {
        tile.push_back(A[i * M + i]);
    }

    return hpx::make_ready_future(std::move(tile));
}
