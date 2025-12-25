#include "gpu/gp_algorithms.cuh"

#include "gp_kernels.hpp"
#include "gpu/cuda_kernels.cuh"
#include "gpu/cuda_utils.cuh"
#include "target.hpp"
#include <cuda_runtime.h>
#include <hpx/algorithm.hpp>
#include <hpx/async_cuda/cuda_exception.hpp>

namespace gpu
{

// Kernel function to compute covariance
__global__ void gen_tile_covariance_kernel(
    double *d_tile,
    const double *d_input,
    const std::size_t n_tile_size,
    const std::size_t n_regressors,
    const std::size_t tile_row,
    const std::size_t tile_column,
    const gprat_hyper::SEKParams sek_params)
{
    // Compute the global indices of the thread
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_tile_size && j < n_tile_size)
    {
        std::size_t i_global = n_tile_size * tile_row + i;
        std::size_t j_global = n_tile_size * tile_column + j;

        double distance = 0.0;
        double z_ik_minus_z_jk;

        for (std::size_t k = 0; k < n_regressors; ++k)
        {
            z_ik_minus_z_jk = d_input[i_global + k] - d_input[j_global + k];
            distance += z_ik_minus_z_jk * z_ik_minus_z_jk;
        }

        // Compute the covariance value
        double covariance =
            sek_params.vertical_lengthscale * exp(-0.5 * distance / (sek_params.lengthscale * sek_params.lengthscale));

        // Add noise variance if diagonal
        if (i_global == j_global)
        {
            covariance += sek_params.noise_variance;
        }

        d_tile[i * n_tile_size + j] = covariance;
    }
}

double *gen_tile_covariance(const double *d_input,
                            const std::size_t tile_row,
                            const std::size_t tile_column,
                            const std::size_t n_tile_size,
                            const std::size_t n_regressors,
                            const gprat_hyper::SEKParams sek_params,
                            gprat::CUDA_GPU &gpu)
{
    double *d_tile;

    dim3 threads_per_block(16, 16);
    dim3 n_blocks((n_tile_size + 15) / 16, (n_tile_size + 15) / 16);

    cudaStream_t stream = gpu.next_stream();

    check_cuda_error(cudaMalloc(&d_tile, n_tile_size * n_tile_size * sizeof(double)));
    gen_tile_covariance_kernel<<<n_blocks, threads_per_block, gpu.shared_memory_size, stream>>>(
        d_tile, d_input, n_tile_size, n_regressors, tile_row, tile_column, sek_params);

    check_cuda_error(cudaStreamSynchronize(stream));

    return d_tile;
}

std::vector<hpx::shared_future<double *>> assemble_tiled_covariance_matrix(
    const double *d_training_input,
    const std::size_t n_tiles,
    const std::size_t n_tile_size,
    const std::size_t n_regressors,
    const gprat_hyper::SEKParams sek_params,
    gprat::CUDA_GPU &gpu)
{
    std::vector<hpx::shared_future<double *>> d_tiles(n_tiles * n_tiles);

    for (std::size_t tile_row = 0; tile_row < n_tiles; ++tile_row)
    {
        for (std::size_t tile_column = 0; tile_column < tile_row + 1; ++tile_column)
        {
            d_tiles[tile_row * n_tiles + tile_column] = hpx::async(
                &gen_tile_covariance,
                d_training_input,
                tile_row,
                tile_column,
                n_tile_size,
                n_regressors,
                sek_params,
                std::ref(gpu));
        }
    }

    return d_tiles;
}

std::vector<double> copy_tiled_vector_to_host_vector(std::vector<hpx::shared_future<double *>> &d_tiles,
                                                     std::size_t n_tile_size,
                                                     std::size_t n_tiles,
                                                     gprat::CUDA_GPU &gpu)
{
    std::vector<double> h_vector(n_tiles * n_tile_size);
    std::vector<cudaStream_t> streams(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        streams[i] = gpu.next_stream();
        check_cuda_error(cudaMemcpyAsync(
            h_vector.data() + i * n_tile_size,
            d_tiles[i].get(),
            n_tile_size * sizeof(double),
            cudaMemcpyDeviceToHost,
            streams[i]));
    }
    gpu.sync_streams(streams);
    return h_vector;
}

std::vector<std::vector<double>> move_lower_tiled_matrix_to_host(
    const std::vector<hpx::shared_future<double *>> &d_tiles,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gprat::CUDA_GPU &gpu)
{
    std::vector<std::vector<double>> h_tiles(n_tiles * n_tiles);

    std::vector<cudaStream_t> streams(n_tiles * (n_tiles + 1) / 2);
    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            streams[i] = gpu.next_stream();
            h_tiles[i * n_tiles + j].resize(n_tile_size * n_tile_size);
            check_cuda_error(cudaMemcpyAsync(
                h_tiles[i * n_tiles + j].data(),
                d_tiles[i * n_tiles + j].get(),
                n_tile_size * n_tile_size * sizeof(double),
                cudaMemcpyDeviceToHost,
                streams[i]));
            check_cuda_error(cudaFree(d_tiles[i * n_tiles + j].get()));
        }
    }
    gpu.sync_streams(streams);

    return h_tiles;
}

void free_lower_tiled_matrix(const std::vector<hpx::shared_future<double *>> &d_tiles, const std::size_t n_tiles)
{
    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            check_cuda_error(cudaFree(d_tiles[i * n_tiles + j].get()));
        }
    }
}
}  // end of namespace gpu
