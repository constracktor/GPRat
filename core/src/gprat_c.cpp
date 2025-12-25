#include "gprat_c.hpp"

#include "cpu/functions.hpp"
#include "utils_c.hpp"

#if GPRAT_WITH_CUDA
#include "gpu/functions.cuh"
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
       int n_tiles,
       int n_tile_size,
       int n_regressors,
       std::vector<double> kernel_hyperparams,
       std::shared_ptr<Target> target) :
    training_input_(input),
    n_tiles_(n_tiles),
    n_tile_size_(n_tile_size),
    target_(target),
    n_reg(n_regressors),
    kernel_params(kernel_hyperparams[0], kernel_hyperparams[1], kernel_hyperparams[2])
{ }

GP::GP(std::vector<double> input,
       int n_tiles,
       int n_tile_size,
       int n_regressors,
       std::vector<double> kernel_hyperparams) :
    training_input_(input),
    n_tiles_(n_tiles),
    n_tile_size_(n_tile_size),
    target_(std::make_shared<CPU>()),
    n_reg(n_regressors),
    kernel_params(kernel_hyperparams[0], kernel_hyperparams[1], kernel_hyperparams[2])
{ }

GP::GP(std::vector<double> input,
       int n_tiles,
       int n_tile_size,
       int n_regressors,
       std::vector<double> kernel_hyperparams,
       int gpu_id,
       int n_streams) :
    training_input_(input),
    n_tiles_(n_tiles),
    n_tile_size_(n_tile_size),
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
        << ", n_regressors=" << n_reg << "], Target: [" << target_->repr() << "], n_tiles=" << n_tiles_ << ", n_tile_size=" << n_tile_size_;
    return oss.str();
}

std::vector<double> GP::get_training_input() const { return training_input_; }

std::vector<double> GP::get_training_output() const { return training_output_; }

std::vector<std::vector<double>> GP::cholesky()
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
}

std::vector<std::vector<double>> GP::cholesky_synchronous()
{
                return cpu::cholesky_synchronous(training_input_, kernel_params, n_tiles_, n_tile_size_, n_reg);
}

std::vector<std::vector<double>> GP::cholesky_loop()
{
                return cpu::cholesky_loop(training_input_, kernel_params, n_tiles_, n_tile_size_, n_reg);
}
}  // namespace gprat
