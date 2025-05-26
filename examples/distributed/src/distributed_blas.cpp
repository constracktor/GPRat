#include "distributed_blas.hpp"

#include "cpu/adapter_cblas_fp64.hpp"
#include <hpx/distribution_policies/colocating_distribution_policy.hpp>

HPX_REGISTER_ACTION_DECLARATION(potrf_distributed_action);
HPX_REGISTER_ACTION_DECLARATION(trsm_distributed_action);
HPX_REGISTER_ACTION_DECLARATION(syrk_distributed_action);
HPX_REGISTER_ACTION_DECLARATION(gemm_distributed_action);

tile_handle potrf_distributed(const tile_handle &A, int N)
{
    return hpx::dataflow(
        hpx::launch::async,
        hpx::unwrapping(
            [A, N](tile_data<double> tile)
            {
                inplace::potrf(tile, N);
                return tile_handle(hpx::colocated(A.get_id()), tile);
            }),
        A.get_data());
}

tile_handle
trsm_distributed(const tile_handle &L, const tile_handle &A, int N, int M, BLAS_TRANSPOSE transpose_L, BLAS_SIDE side_L)
{
    return hpx::dataflow(
        hpx::launch::async,
        hpx::unwrapping(
            [L, A, N, M, transpose_L, side_L](const tile_data<double> &Ld, tile_data<double> Ad)
            {
                inplace::trsm(Ld, Ad, N, M, transpose_L, side_L);
                return tile_handle(hpx::colocated(A.get_id()), Ad);
            }),
        L.get_data(),
        A.get_data());
}

tile_handle syrk_distributed(const tile_handle &A, const tile_handle &B, int N)
{
    return hpx::dataflow(
        hpx::launch::async,
        hpx::unwrapping(
            [A, B, N](tile_data<double> Ad, const tile_data<double> &Bd)
            {
                inplace::syrk(Ad, Bd, N);
                return tile_handle(hpx::colocated(A.get_id()), Ad);
            }),
        A.get_data(),
        B.get_data());
}

tile_handle gemm_distributed(
    const tile_handle &A,
    const tile_handle &B,
    const tile_handle &C,
    int N,
    int M,
    int K,
    BLAS_TRANSPOSE transpose_A,
    BLAS_TRANSPOSE transpose_B)
{
    return hpx::dataflow(
        hpx::launch::async,
        hpx::unwrapping(
            [A, B, C, N, M, K, transpose_A, transpose_B](
                const tile_data<double> &Ad, const tile_data<double> &Bd, tile_data<double> Cd)
            {
                inplace::gemm(Ad, Bd, Cd, N, M, K, transpose_A, transpose_B);
                return tile_handle(hpx::colocated(C.get_id()), Cd);
            }),
        A.get_data(),
        B.get_data(),
        C.get_data());
}
