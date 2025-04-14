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

std::vector<double>
predict(const std::vector<double> &h_training_input,
        const std::vector<double> &h_training_output,
        const std::vector<double> &h_test_input,
        const gprat_hyper::SEKParams &sek_params,
        int n_tiles,
        int n_tile_size,
        int m_tiles,
        int m_tile_size,
        int n_regressors,
        gprat::CUDA_GPU &gpu)
{
    gpu.create();

    double *d_training_input = copy_to_device(h_training_input, gpu);
    double *d_training_output = copy_to_device(h_training_output, gpu);
    double *d_test_input = copy_to_device(h_test_input, gpu);

    auto d_tiles =
        assemble_tiled_covariance_matrix(d_training_input, n_tiles, n_tile_size, n_regressors, sek_params, gpu);

    auto alpha_tiles = assemble_alpha_tiles(d_training_output, n_tiles, n_tile_size, gpu);
    auto cross_covariance_tiles = assemble_cross_covariance_tiles(
        d_test_input, d_training_input, m_tiles, n_tiles, m_tile_size, n_tile_size, n_regressors, sek_params, gpu);

    auto prediction_tiles = assemble_tiles_with_zeros(m_tile_size, m_tiles, gpu);

    cusolverDnHandle_t cusolver = create_cusolver_handle();
    right_looking_cholesky_tiled(d_tiles, n_tile_size, n_tiles, gpu, cusolver);

    // Triangular solve K_NxN * alpha = y
    forward_solve_tiled(d_tiles, alpha_tiles, n_tile_size, n_tiles, gpu);
    backward_solve_tiled(d_tiles, alpha_tiles, n_tile_size, n_tiles, gpu);

    matrix_vector_tiled(
        cross_covariance_tiles, alpha_tiles, prediction_tiles, m_tile_size, n_tile_size, n_tiles, m_tiles, gpu);
    std::vector<double> prediction = copy_tiled_vector_to_host_vector(prediction_tiles, m_tile_size, m_tiles, gpu);

    free_lower_tiled_matrix(d_tiles, n_tiles);
    free(alpha_tiles);
    free(cross_covariance_tiles);
    free(prediction_tiles);
    destroy(cusolver);

    gpu.destroy();

    return prediction;
}

std::vector<std::vector<double>> predict_with_uncertainty(
    const std::vector<double> &h_training_input,
    const std::vector<double> &h_training_output,
    const std::vector<double> &h_test_input,
    const gprat_hyper::SEKParams &sek_params,
    int n_tiles,
    int n_tile_size,
    int m_tiles,
    int m_tile_size,
    int n_regressors,
    gprat::CUDA_GPU &gpu)
{
    gpu.create();

    double *d_training_input = copy_to_device(h_training_input, gpu);
    double *d_training_output = copy_to_device(h_training_output, gpu);
    double *d_test_input = copy_to_device(h_test_input, gpu);

    // Assemble tiled covariance matrix on GPU.
    auto d_K_tiles =
        assemble_tiled_covariance_matrix(d_training_input, n_tiles, n_tile_size, n_regressors, sek_params, gpu);

    auto d_alpha_tiles = assemble_alpha_tiles(d_training_output, n_tiles, n_tile_size, gpu);

    auto d_prior_K_tiles = assemble_prior_K_tiles(d_test_input, m_tiles, m_tile_size, n_regressors, sek_params, gpu);

    auto d_cross_covariance_tiles = assemble_cross_covariance_tiles(
        d_test_input, d_training_input, m_tiles, n_tiles, m_tile_size, n_tile_size, n_regressors, sek_params, gpu);

    auto d_t_cross_covariance_tiles =
        assemble_t_cross_covariance_tiles(d_cross_covariance_tiles, n_tiles, m_tiles, n_tile_size, m_tile_size, gpu);

    // Assemble placeholder matrix for diag(K_MxN * (K^-1_NxN * K_NxM))
    auto d_prior_inter_tiles = assemble_tiles_with_zeros(m_tile_size, m_tiles, gpu);

    auto d_prediction_tiles = assemble_tiles_with_zeros(m_tile_size, m_tiles, gpu);

    // Assemble placeholder for uncertainty
    auto d_prediction_uncertainty_tiles = assemble_tiles_with_zeros(m_tile_size, m_tiles, gpu);

    cusolverDnHandle_t cusolver = create_cusolver_handle();
    right_looking_cholesky_tiled(d_K_tiles, n_tile_size, n_tiles, gpu, cusolver);

    // Triangular solve K_NxN * alpha = y
    forward_solve_tiled(d_K_tiles, d_alpha_tiles, n_tile_size, n_tiles, gpu);
    backward_solve_tiled(d_K_tiles, d_alpha_tiles, n_tile_size, n_tiles, gpu);

    // Triangular solve A_M,N * K_NxN = K_MxN -> A_MxN = K_MxN * K^-1_NxN
    forward_solve_tiled_matrix(d_K_tiles, d_t_cross_covariance_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles, gpu);

    // Compute predictions
    matrix_vector_tiled(
        d_cross_covariance_tiles, d_alpha_tiles, d_prediction_tiles, m_tile_size, n_tile_size, n_tiles, m_tiles, gpu);

    // posterior covariance matrix - (K_MxN * K^-1_NxN) * K_NxM
    symmetric_matrix_matrix_diagonal_tiled(
        d_t_cross_covariance_tiles, d_prior_inter_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles, gpu);

    // Compute predicition uncertainty
    vector_difference_tiled(
        d_prior_K_tiles, d_prior_inter_tiles, d_prediction_uncertainty_tiles, m_tile_size, m_tiles, gpu);

    // Get predictions and uncertainty to return them
    std::vector<double> prediction = copy_tiled_vector_to_host_vector(d_prediction_tiles, m_tile_size, m_tiles, gpu);
    std::vector<double> pred_var_full =
        copy_tiled_vector_to_host_vector(d_prediction_uncertainty_tiles, m_tile_size, m_tiles, gpu);

    check_cuda_error(cudaFree(d_training_input));
    check_cuda_error(cudaFree(d_training_output));
    check_cuda_error(cudaFree(d_test_input));
    free_lower_tiled_matrix(d_K_tiles, n_tiles);
    free(d_alpha_tiles);
    free(d_prior_K_tiles);
    free(d_cross_covariance_tiles);
    free(d_t_cross_covariance_tiles);
    free(d_prior_inter_tiles);
    free(d_prediction_tiles);
    free(d_prediction_uncertainty_tiles);
    destroy(cusolver);

    gpu.destroy();

    return std::vector<std::vector<double>>{ prediction, pred_var_full };
}

std::vector<std::vector<double>> predict_with_full_cov(
    const std::vector<double> &h_training_input,
    const std::vector<double> &h_training_output,
    const std::vector<double> &h_test_input,
    const gprat_hyper::SEKParams &sek_params,
    int n_tiles,
    int n_tile_size,
    int m_tiles,
    int m_tile_size,
    int n_regressors,
    gprat::CUDA_GPU &gpu)
{
    gpu.create();

    double *d_training_input = copy_to_device(h_training_input, gpu);
    double *d_training_output = copy_to_device(h_training_output, gpu);
    double *d_test_input = copy_to_device(h_test_input, gpu);

    // Assemble tiled covariance matrix on GPU.
    auto d_K_tiles =
        assemble_tiled_covariance_matrix(d_training_input, n_tiles, n_tile_size, n_regressors, sek_params, gpu);

    auto d_alpha_tiles = assemble_alpha_tiles(d_training_output, n_tiles, n_tile_size, gpu);

    auto d_prior_K_tiles =
        assemble_prior_K_tiles_full(d_test_input, m_tiles, m_tile_size, n_regressors, sek_params, gpu);

    auto d_cross_covariance_tiles = assemble_cross_covariance_tiles(
        d_test_input, d_training_input, m_tiles, n_tiles, m_tile_size, n_tile_size, n_regressors, sek_params, gpu);

    auto d_t_cross_covariance_tiles =
        assemble_t_cross_covariance_tiles(d_cross_covariance_tiles, n_tiles, m_tiles, n_tile_size, m_tile_size, gpu);

    auto d_prediction_tiles = assemble_tiles_with_zeros(m_tile_size, m_tiles, gpu);

    // Assemble placeholder for uncertainty
    auto d_prediction_uncertainty_tiles = assemble_tiles_with_zeros(m_tile_size, m_tiles, gpu);

    cusolverDnHandle_t cusolver = create_cusolver_handle();
    right_looking_cholesky_tiled(d_K_tiles, n_tile_size, n_tiles, gpu, cusolver);

    // Triangular solve K_NxN * alpha = y
    forward_solve_tiled(d_K_tiles, d_alpha_tiles, n_tile_size, n_tiles, gpu);
    backward_solve_tiled(d_K_tiles, d_alpha_tiles, n_tile_size, n_tiles, gpu);

    // Triangular solve A_M,N * K_NxN = K_MxN -> A_MxN = K_MxN * K^-1_NxN
    forward_solve_tiled_matrix(d_K_tiles, d_t_cross_covariance_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles, gpu);

    // Compute predictions
    matrix_vector_tiled(
        d_cross_covariance_tiles, d_alpha_tiles, d_prediction_tiles, m_tile_size, n_tile_size, n_tiles, m_tiles, gpu);

    // posterior covariance matrix - (K_MxN * K^-1_NxN) * K_NxM
    symmetric_matrix_matrix_tiled(
        d_t_cross_covariance_tiles, d_prior_K_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles, gpu);

    // Compute predicition uncertainty
    matrix_diagonal_tiled(d_prior_K_tiles, d_prediction_uncertainty_tiles, m_tile_size, m_tiles, gpu);

    // Get predictions and uncertainty to return them
    std::vector<double> prediction = copy_tiled_vector_to_host_vector(d_prediction_tiles, m_tile_size, m_tiles, gpu);
    std::vector<double> pred_var_full =
        copy_tiled_vector_to_host_vector(d_prediction_uncertainty_tiles, m_tile_size, m_tiles, gpu);

    check_cuda_error(cudaFree(d_training_input));
    check_cuda_error(cudaFree(d_training_output));
    check_cuda_error(cudaFree(d_test_input));
    free_lower_tiled_matrix(d_K_tiles, n_tiles);
    free(d_alpha_tiles);
    free_lower_tiled_matrix(d_prior_K_tiles, m_tiles);
    free(d_cross_covariance_tiles);
    free(d_t_cross_covariance_tiles);
    free(d_prediction_tiles);
    free(d_prediction_uncertainty_tiles);
    destroy(cusolver);

    gpu.destroy();

    return std::vector<std::vector<double>>{ prediction, pred_var_full };
}

double compute_loss(const std::vector<double> &h_training_input,
                    const std::vector<double> &h_training_output,
                    const gprat_hyper::SEKParams &sek_params,
                    int n_tiles,
                    int n_tile_size,
                    int n_regressors,
                    gprat::CUDA_GPU &gpu)
{
    gpu.create();

    double *d_training_input = copy_to_device(h_training_input, gpu);
    double *d_training_output = copy_to_device(h_training_output, gpu);

    // Assemble tiled covariance matrix on GPU.
    auto d_K_tiles =
        assemble_tiled_covariance_matrix(d_training_input, n_tiles, n_tile_size, n_regressors, sek_params, gpu);

    auto d_alpha_tiles = assemble_alpha_tiles(d_training_output, n_tiles, n_tile_size, gpu);

    auto d_y_tiles = assemble_y_tiles(d_training_output, n_tiles, n_tile_size, gpu);

    cusolverDnHandle_t cusolver = create_cusolver_handle();
    right_looking_cholesky_tiled(d_K_tiles, n_tile_size, n_tiles, gpu, cusolver);

    // Triangular solve K_NxN * alpha = y
    forward_solve_tiled(d_K_tiles, d_alpha_tiles, n_tile_size, n_tiles, gpu);
    backward_solve_tiled(d_K_tiles, d_alpha_tiles, n_tile_size, n_tiles, gpu);

    // Compute loss
    hpx::shared_future<double> loss_value =
        compute_loss_tiled(d_K_tiles, d_alpha_tiles, d_y_tiles, n_tile_size, n_tiles, gpu);

    check_cuda_error(cudaFree(d_training_input));
    check_cuda_error(cudaFree(d_training_output));

    loss_value.get();
    free_lower_tiled_matrix(d_K_tiles, n_tiles);
    free(d_alpha_tiles);
    free(d_y_tiles);
    destroy(cusolver);

    gpu.destroy();

    return loss_value.get();
}

std::vector<double>
optimize(const std::vector<double> &training_input,
         const std::vector<double> &training_output,
         int n_tiles,
         int n_tile_size,
         int n_regressors,
         const gprat_hyper::AdamParams &adam_params,
         const gprat_hyper::SEKParams &sek_params,
         std::vector<bool> trainable_params,
         gprat::CUDA_GPU &gpu)
{
    throw std::logic_error("Function not implemented for GPU");
    // return std::vector<double>>();
}

double optimize_step(const std::vector<double> &training_input,
                     const std::vector<double> &training_output,
                     int n_tiles,
                     int n_tile_size,
                     int n_regressors,
                     gprat_hyper::AdamParams &adam_params,
                     gprat_hyper::SEKParams &sek_params,
                     std::vector<bool> trainable_params,
                     int iter,
                     gprat::CUDA_GPU &gpu)
{
    throw std::logic_error("Function not implemented for GPU");
    // return 0.0;
}

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
