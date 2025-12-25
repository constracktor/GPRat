#include "gpu/gp_functions.cuh"

#include "gp_kernels.hpp"
#include "gpu/cuda_utils.cuh"
#include "gpu/gp_algorithms.cuh"
#include "gpu/tiled_algorithms.cuh"
#include "target.hpp"
#include <cuda_runtime.h>
#include <hpx/algorithm.hpp>
#include <hpx/async_cuda/cuda_exception.hpp>

namespace gpu
{
std::vector<std::vector<double>>
cholesky(const std::vector<double> &h_training_input,
         const gprat_hyper::SEKParams &sek_params,
         int n_tiles,
         int n_tile_size,
         int n_regressors,
         gprat::CUDA_GPU &gpu)
{
    gpu.create();

    double *d_training_input = copy_to_device(h_training_input, gpu);
    // Assemble tiled covariance matrix on GPU.
    std::vector<hpx::shared_future<double *>> d_tiles =
        assemble_tiled_covariance_matrix(d_training_input, n_tiles, n_tile_size, n_regressors, sek_params, gpu);

    // Compute Tiled Cholesky decomposition on device
    cusolverDnHandle_t cusolver = create_cusolver_handle();
    right_looking_cholesky_tiled(d_tiles, n_tile_size, n_tiles, gpu, cusolver);

    // Copy tiled matrix to host
    std::vector<std::vector<double>> h_tiles = move_lower_tiled_matrix_to_host(d_tiles, n_tile_size, n_tiles, gpu);

    cudaFree(d_training_input);
    destroy(cusolver);
    gpu.destroy();

    return h_tiles;
}

}  // end of namespace gpu
