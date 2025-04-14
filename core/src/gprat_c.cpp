#include "gprat_c.hpp"

#include "cpu/gp_functions.hpp"
#include "utils_c.hpp"
#include <cstdio>

#if GPRAT_WITH_CUDA
#include "gpu/gp_functions.cuh"
#endif

// namespace for GPRat library entities
namespace gprat
{

GP_data::GP_data(const std::string &f_path, int n, int n_reg) :
    file_path(f_path),
    n_samples(n),
    n_regressors(n_reg)
{
    data = utils::load_data(f_path, n, n_reg - 1);
}

GP::GP(std::vector<double> input,
       std::vector<double> output,
       int n_tiles,
       int n_tile_size,
       int n_regressors,
       std::vector<double> kernel_hyperparams,
       std::vector<bool> trainable_bool,
       std::shared_ptr<Target> target) :
    training_input_(input),
    training_output_(output),
    n_tiles_(n_tiles),
    n_tile_size_(n_tile_size),
    trainable_params_(trainable_bool),
    target_(target),
    n_reg(n_regressors),
    kernel_params(kernel_hyperparams[0], kernel_hyperparams[1], kernel_hyperparams[2])
{ }

GP::GP(std::vector<double> input,
       std::vector<double> output,
       int n_tiles,
       int n_tile_size,
       int n_regressors,
       std::vector<double> kernel_hyperparams,
       std::vector<bool> trainable_bool) :
    training_input_(input),
    training_output_(output),
    n_tiles_(n_tiles),
    n_tile_size_(n_tile_size),
    trainable_params_(trainable_bool),
    target_(std::make_shared<CPU>()),
    n_reg(n_regressors),
    kernel_params(kernel_hyperparams[0], kernel_hyperparams[1], kernel_hyperparams[2])
{ }

GP::GP(std::vector<double> input,
       std::vector<double> output,
       int n_tiles,
       int n_tile_size,
       int n_regressors,
       std::vector<double> kernel_hyperparams,
       std::vector<bool> trainable_bool,
       int gpu_id,
       int n_streams) :
    training_input_(input),
    training_output_(output),
    n_tiles_(n_tiles),
    n_tile_size_(n_tile_size),
    trainable_params_(trainable_bool),
#if GPRAT_WITH_CUDA
    target_(std::make_shared<CUDA_GPU>(CUDA_GPU(gpu_id, n_streams))),
#else
    target_(std::make_shared<CPU>()),
#endif
    n_reg(n_regressors),
    kernel_params(kernel_hyperparams[0], kernel_hyperparams[1], kernel_hyperparams[2])
{
#if !GPRAT_WITH_CUDA
    throw std::runtime_error(
        "Cannot create GP object using CUDA for computation. "
        "CUDA is not available because GPRat has been compiled without CUDA. "
        "Remove arguments gpu_id ("
        + std::to_string(gpu_id) + ") and n_streams (" + std::to_string(n_streams)
        + ") to perform computations on the CPU.");
#endif
}

std::string GP::repr() const
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(12);
    oss << "Kernel_Params: [lengthscale=" << kernel_params.lengthscale << ", vertical_lengthscale="
        << kernel_params.vertical_lengthscale << ", noise_variance=" << kernel_params.noise_variance
        << ", n_regressors=" << n_reg << "], Trainable_Params: [trainable_params l=" << trainable_params_[0]
        << ", trainable_params v=" << trainable_params_[1] << ", trainable_params n=" << trainable_params_[2]
        << "], Target: [" << target_->repr() << "], n_tiles=" << n_tiles_ << ", n_tile_size=" << n_tile_size_;
    return oss.str();
}

std::vector<double> GP::get_training_input() const { return training_input_; }

std::vector<double> GP::get_training_output() const { return training_output_; }

std::vector<double> GP::predict(const std::vector<double> &test_input, int m_tiles, int m_tile_size)
{
    return hpx::async(
               [this, &test_input, m_tiles, m_tile_size]()
               {
#if GPRAT_WITH_CUDA
                   if (target_->is_gpu())
                   {
                       return gpu::predict(
                           training_input_,
                           training_output_,
                           test_input,
                           kernel_params,
                           n_tiles_,
                           n_tile_size_,
                           m_tiles,
                           m_tile_size,
                           n_reg,
                           *std::dynamic_pointer_cast<gprat::CUDA_GPU>(target_));
                   }
                   else
                   {
                       return cpu::predict(
                           training_input_,
                           training_output_,
                           test_input,
                           kernel_params,
                           n_tiles_,
                           n_tile_size_,
                           m_tiles,
                           m_tile_size,
                           n_reg);
                   }
#else
                   return cpu::predict(
                       training_input_,
                       training_output_,
                       test_input,
                       kernel_params,
                       n_tiles_,
                       n_tile_size_,
                       m_tiles,
                       m_tile_size,
                       n_reg);
#endif
               })
        .get();
}

std::vector<std::vector<double>>
GP::predict_with_uncertainty(const std::vector<double> &test_input, int m_tiles, int m_tile_size)
{
    return hpx::async(
               [this, &test_input, m_tiles, m_tile_size]()
               {
#if GPRAT_WITH_CUDA
                   if (target_->is_gpu())
                   {
                       return gpu::predict_with_uncertainty(
                           training_input_,
                           training_output_,
                           test_input,
                           kernel_params,
                           n_tiles_,
                           n_tile_size_,
                           m_tiles,
                           m_tile_size,
                           n_reg,
                           *std::dynamic_pointer_cast<gprat::CUDA_GPU>(target_));
                   }
                   else
                   {
                       return cpu::predict_with_uncertainty(
                           training_input_,
                           training_output_,
                           test_input,
                           kernel_params,
                           n_tiles_,
                           n_tile_size_,
                           m_tiles,
                           m_tile_size,
                           n_reg);
                   }
#else
                   return cpu::predict_with_uncertainty(
                       training_input_,
                       training_output_,
                       test_input,
                       kernel_params,
                       n_tiles_,
                       n_tile_size_,
                       m_tiles,
                       m_tile_size,
                       n_reg);
#endif
               })
        .get();
}

std::vector<std::vector<double>>
GP::predict_with_full_cov(const std::vector<double> &test_input, int m_tiles, int m_tile_size)
{
    return hpx::async(
               [this, &test_input, m_tiles, m_tile_size]()
               {
#if GPRAT_WITH_CUDA
                   if (target_->is_gpu())
                   {
                       return gpu::predict_with_full_cov(
                           training_input_,
                           training_output_,
                           test_input,
                           kernel_params,
                           n_tiles_,
                           n_tile_size_,
                           m_tiles,
                           m_tile_size,
                           n_reg,
                           *std::dynamic_pointer_cast<gprat::CUDA_GPU>(target_));
                   }
                   else
                   {
                       return cpu::predict_with_full_cov(
                           training_input_,
                           training_output_,
                           test_input,
                           kernel_params,
                           n_tiles_,
                           n_tile_size_,
                           m_tiles,
                           m_tile_size,
                           n_reg);
                   }
#else
                   return cpu::predict_with_full_cov(
                       training_input_,
                       training_output_,
                       test_input,
                       kernel_params,
                       n_tiles_,
                       n_tile_size_,
                       m_tiles,
                       m_tile_size,
                       n_reg);
#endif
               })
        .get();
}

std::vector<double> GP::optimize(const gprat_hyper::AdamParams &adam_params)
{
    return hpx::async(
               [this, &adam_params]()
               {
#if GPRAT_WITH_CUDA
                   if (target_->is_gpu())
                   {
                       std::cerr << "GP::optimze_step has not been implemented for the GPU.\n"
                                 << "Instead, this operation executes the CPU implementation." << std::endl;
                   }
#endif
                   return cpu::optimize(
                       training_input_,
                       training_output_,
                       n_tiles_,
                       n_tile_size_,
                       n_reg,
                       adam_params,
                       kernel_params,
                       trainable_params_);
               })
        .get();
}

double GP::optimize_step(gprat_hyper::AdamParams &adam_params, int iter)
{
    return hpx::async(
               [this, &adam_params, iter]()
               {
#if GPRAT_WITH_CUDA
                   if (target_->is_gpu())
                   {
                       std::cerr << "GP::optimze_step has not been implemented for the GPU.\n"
                                 << "Instead, this operation executes the CPU implementation." << std::endl;
                   }
#endif
                   return cpu::optimize_step(
                       training_input_,
                       training_output_,
                       n_tiles_,
                       n_tile_size_,
                       n_reg,
                       adam_params,
                       kernel_params,
                       trainable_params_,
                       iter);
               })
        .get();
}

double GP::calculate_loss()
{
    return hpx::async(
               [this]()
               {
#if GPRAT_WITH_CUDA
                   if (target_->is_gpu())
                   {
                       return gpu::compute_loss(
                           training_input_,
                           training_output_,
                           kernel_params,
                           n_tiles_,
                           n_tile_size_,
                           n_reg,
                           *std::dynamic_pointer_cast<gprat::CUDA_GPU>(target_));
                   }
                   else
                   {
                       return cpu::compute_loss(
                           training_input_, training_output_, kernel_params, n_tiles_, n_tile_size_, n_reg);
                   }
#else
                   return cpu::compute_loss(
                       training_input_, training_output_, kernel_params, n_tiles_, n_tile_size_, n_reg);
#endif
               })
        .get();
}

std::vector<std::vector<double>> GP::cholesky()
{
    return hpx::async(
               [this]()
               {
#if GPRAT_WITH_CUDA
                   if (target_->is_gpu())
                   {
                       return gpu::cholesky(
                           training_input_,
                           kernel_params,
                           n_tiles_,
                           n_tile_size_,
                           n_reg,
                           *std::dynamic_pointer_cast<gprat::CUDA_GPU>(target_));
                   }
                   else
                   {
                       return cpu::cholesky(training_input_, kernel_params, n_tiles_, n_tile_size_, n_reg);
                   }
#else
                   return cpu::cholesky(training_input_, kernel_params, n_tiles_, n_tile_size_, n_reg);
#endif
               })
        .get();
}

}  // namespace gprat
