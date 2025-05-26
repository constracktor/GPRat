#pragma once

#include "distributed_tile.hpp"
#include "scheduling.hpp"
#include <hpx/runtime_distributed/find_localities.hpp>

struct tiled_cholesky_distribution_policy_paap12
{
    constexpr std::size_t locality_for_tile(std::size_t row, std::size_t col) const
    {
        return (row + col) % num_localities;
    }

    constexpr std::size_t locality_for_POTRF(std::size_t k) const { return (2 * k) % num_localities; }

    constexpr std::size_t locality_for_SYRK(std::size_t m) const { return (2 * m) % num_localities; }

    constexpr std::size_t locality_for_TRSM(std::size_t k, std::size_t m) const { return (k + m) % num_localities; }

    constexpr std::size_t locality_for_GEMM(std::size_t /*k*/, std::size_t m, std::size_t n) const
    {
        return (m + n) % num_localities;
    }

    std::size_t num_localities;
};

template <typename DistPolicy = tiled_cholesky_distribution_policy_paap12>
struct tiled_cholesky_scheduler_distributed
{
    using tiled_matrix_handles = std::vector<tile_handle>;

    tiled_cholesky_scheduler_distributed() = default;

    [[nodiscard]] schedule_on_locality for_tile(std::size_t row, std::size_t col) const
    {
        return localities[policy.locality_for_tile(row, col)];
    }

    [[nodiscard]] schedule_on_locality for_POTRF(std::size_t k) const
    {
        return localities[policy.locality_for_POTRF(k)];
    }

    [[nodiscard]] schedule_on_locality for_SYRK(std::size_t m) const { return localities[policy.locality_for_SYRK(m)]; }

    [[nodiscard]] schedule_on_locality for_TRSM(std::size_t k, std::size_t m) const
    {
        return localities[policy.locality_for_TRSM(k, m)];
    }

    [[nodiscard]] schedule_on_locality for_GEMM(std::size_t k, std::size_t m, std::size_t n) const
    {
        return localities[policy.locality_for_GEMM(k, m, n)];
    }

    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    DistPolicy policy{ localities.size() };
};
