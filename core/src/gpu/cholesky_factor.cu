#include "gpu/cholesky_factor.cuh"

#include "gpu/adapter_cublas.cuh"
#include <hpx/algorithm.hpp>

namespace gpu
{

// Tiled Cholesky Algorithm

void right_looking_cholesky_tiled(std::vector<hpx::shared_future<double *>> &ft_tiles,
                                  const std::size_t n_tile_size,
                                  const std::size_t n_tiles,
                                  gprat::CUDA_GPU &gpu,
                                  const cusolverDnHandle_t &cusolver)
{
    for (std::size_t k = 0; k < n_tiles; ++k)
    {
        cudaStream_t stream = gpu.next_stream();
        cusolverDnSetStream(cusolver, stream);

        // POTRF
        ft_tiles[k * n_tiles + k] = hpx::dataflow(
            hpx::launch::sync,
            hpx::annotated_function(&potrf, "Cholesky POTRF"),
            cusolver,
            stream,
            ft_tiles[k * n_tiles + k],
            n_tile_size);

        // NOTE: The result is immediately needed by TRSM. Also TRSM may throw
        // an error otherwise.
        ft_tiles[k * n_tiles + k].get();

        for (std::size_t m = k + 1; m < n_tiles; ++m)
        {
            auto [cublas, stream] = gpu.next_cublas_handle();

            // TRSM
            ft_tiles[m * n_tiles + k] = hpx::dataflow(
                hpx::launch::sync,
                &trsm,
                cublas,
                stream,
                ft_tiles[k * n_tiles + k],
                ft_tiles[m * n_tiles + k],
                n_tile_size,
                n_tile_size,
                Blas_trans,
                Blas_right);
        }

        for (std::size_t m = k + 1; m < n_tiles; ++m)
        {
            auto [cublas, stream] = gpu.next_cublas_handle();

            // SYRK
            ft_tiles[m * n_tiles + m] = hpx::dataflow(
                hpx::launch::sync,
                &syrk,
                cublas,
                stream,
                ft_tiles[m * n_tiles + k],
                ft_tiles[m * n_tiles + m],
                n_tile_size);

            for (std::size_t n = k + 1; n < m; ++n)
            {
                auto [cublas, stream] = gpu.next_cublas_handle();

                // GEMM
                ft_tiles[m * n_tiles + n] = hpx::dataflow(
                    hpx::launch::sync,
                    &gemm,
                    cublas,
                    stream,
                    ft_tiles[m * n_tiles + k],
                    ft_tiles[n * n_tiles + k],
                    ft_tiles[m * n_tiles + n],
                    n_tile_size,
                    n_tile_size,
                    n_tile_size,
                    Blas_no_trans,
                    Blas_trans);
            }
        }
    }
}

}  // end of namespace gpu
