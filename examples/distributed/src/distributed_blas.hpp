#pragma once

#include "cpu/adapter_cblas_fp64.hpp"
#include "distributed_tile.hpp"
#include "scheduling.hpp"
#include <hpx/actions_base/plain_action.hpp>

tile_handle potrf_distributed(const tile_handle &A, int N);
tile_handle trsm_distributed(
    const tile_handle &L, const tile_handle &A, int N, int M, BLAS_TRANSPOSE transpose_L, BLAS_SIDE side_L);
tile_handle syrk_distributed(const tile_handle &A, const tile_handle &B, int N);
tile_handle gemm_distributed(
    const tile_handle &A,
    const tile_handle &B,
    const tile_handle &C,
    int N,
    int M,
    int K,
    BLAS_TRANSPOSE transpose_A,
    BLAS_TRANSPOSE transpose_B);

HPX_DEFINE_PLAIN_ACTION(potrf_distributed);
HPX_DEFINE_PLAIN_ACTION(trsm_distributed);
HPX_DEFINE_PLAIN_ACTION(syrk_distributed);
HPX_DEFINE_PLAIN_ACTION(gemm_distributed);

template <>
struct plain_action_for<&inplace::potrf>
{
    using action_type = potrf_distributed_action;
    constexpr static std::string_view name = "POTRF";
};

template <>
struct plain_action_for<&inplace::trsm>
{
    using action_type = trsm_distributed_action;
    constexpr static std::string_view name = "TRSM";
};

template <>
struct plain_action_for<&inplace::syrk>
{
    using action_type = syrk_distributed_action;
    constexpr static std::string_view name = "SYRK";
};

template <>
struct plain_action_for<&inplace::gemm>
{
    using action_type = gemm_distributed_action;
    constexpr static std::string_view name = "GEMM";
};
