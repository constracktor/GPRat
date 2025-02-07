#include "gprat_c.hpp"

#include "utils_c.hpp"
#include <cstdio>
#include <iomanip>
#include <sstream>

// namespace for GPRat library entities
namespace gprat
{

/**
 * @brief Initialize of Gaussian process data by loading data from a file.
 *
 * The file specified by `f_path` must contain `n` samples.
 *
 * @param f_path Path to the file
 * @param n Number of samples
 */
GP_data::GP_data(const std::string &f_path, int n, int n_reg) :
    file_path(f_path),
    n_samples(n),
    n_regressors(n_reg)
{
    data = utils::load_data(f_path, n, n_reg - 1);
}

/**
 * @brief Initialize of Gaussian process.
 *
 * @param input Training input data
 * @param output Training output data
 * @param n_tiles Number of tiles
 * @param n_tile_size Size of each tile
 * @param l Lengthscale
 * @param v Vertical lengthscale
 * @param n Noise variance
 * @param n_r Number of regressors
 * @param trainable_bool Boolean vector indicating which hyperparameters are
 * trainable
 */
GP::GP(std::vector<double> input,
       std::vector<double> output,
       int n_tiles,
       int n_tile_size,
       int n_r,
       std::vector<double> hyperparams,
       std::vector<bool> trainable_bool) :
    _training_input(input),
    _training_output(output),
    _n_tiles(n_tiles),
    _n_tile_size(n_tile_size),
    n_regressors(n_r),
    kernel_hyperparams(hyperparams),
    trainable_params(trainable_bool)
{ }

/**
 * Returns Gaussian process attributes as string.
 */
std::string GP::repr() const
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(12);
    oss << "Kernel_Params: [lengthscale=" << kernel_hyperparams[0] << ", vertical_lengthscale=" << kernel_hyperparams[1]
        << ", noise_variance=" << kernel_hyperparams[2] << ", n_regressors=" << n_regressors
        << ", trainable_params l=" << trainable_params[0] << ", trainable_params v=" << trainable_params[1]
        << ", trainable_params n=" << trainable_params[2] << "]";
    return oss.str();
}

/**
 * @brief Returns training input data
 */
std::vector<double> GP::get_training_input() const { return _training_input; }

/**
 * @brief Returns training output data
 */
std::vector<double> GP::get_training_output() const { return _training_output; }

/**
 * @brief Predict output for test input
 *
 * @param test_data Test input data
 * @param m_tiles Number of tiles
 * @param m_tile_size Size of each tile
 *
 * @return Predicted output
 */
std::vector<double> GP::predict(const std::vector<double> &test_input, int m_tiles, int m_tile_size)
{
    return hpx::async(
               [this, &test_input, m_tiles, m_tile_size]()
               {
                   return predict_hpx(
                       _training_input,
                       _training_output,
                       test_input,
                       kernel_hyperparams,
                       _n_tiles,
                       _n_tile_size,
                       m_tiles,
                       m_tile_size,
                       n_regressors);
               })
        .get();
}

/**
 * @brief Predict output for test input and additionally provide
 *        uncertainty for the predictions.
 *
 * @param test_input Test input data
 * @param m_tiles Number of tiles
 * @param m_tile_size Size of each tile
 *
 * @return
 */
std::vector<std::vector<double>>
GP::predict_with_uncertainty(const std::vector<double> &test_input, int m_tiles, int m_tile_size)
{
    return hpx::async(
               [this, &test_input, m_tiles, m_tile_size]()
               {
                   return predict_with_uncertainty_hpx(
                       _training_input,
                       _training_output,
                       test_input,
                       kernel_hyperparams,
                       _n_tiles,
                       _n_tile_size,
                       m_tiles,
                       m_tile_size,
                       n_regressors);
               })
        .get();
}

/**
 * @brief Predict output for test input and additionally compute full
 * posterior covariance matrix.
 *
 * @param test_input Test input data
 * @param m_tiles Number of tiles
 * @param m_tile_size Size of each tile
 *
 * @return Full covariance matrix
 */
std::vector<std::vector<double>>
GP::predict_with_full_cov(const std::vector<double> &test_input, int m_tiles, int m_tile_size)
{
    return hpx::async(
               [this, &test_input, m_tiles, m_tile_size]()
               {
                   return predict_with_full_cov_hpx(
                       _training_input,
                       _training_output,
                       test_input,
                       kernel_hyperparams,
                       _n_tiles,
                       _n_tile_size,
                       m_tiles,
                       m_tile_size,
                       n_regressors);
               })
        .get();
}

/**
 * @brief Optimize hyperparameters
 *
 * @param hyperparams Hyperparameters of squared exponential kernel:
 *        lengthscale, vertical_lengthscale, noise_variance
 *
 * @return losses
 */
std::vector<double> GP::optimize(const gprat_hyper::Hyperparameters &hyperparams)
{
    return hpx::async(
               [this, &hyperparams]()
               {
                   return optimize_hpx(
                       _training_input,
                       _training_output,
                       _n_tiles,
                       _n_tile_size,
                       n_regressors,
                       hyperparams,
                       kernel_hyperparams,
                       trainable_params);
               })
        .get();
}

/**
 * @brief Perform a single optimization step
 *
 * @param hyperparams Hyperparameters of squared exponential kernel:
 *        lengthscale, vertical_lengthscale, noise_variance
 * @param iter number of iterations
 *
 * @return loss
 */
double GP::optimize_step(gprat_hyper::Hyperparameters &hyperparams, int iter)
{
    return hpx::async(
               [this, &hyperparams, iter]()
               {
                   return optimize_step_hpx(
                       _training_input,
                       _training_output,
                       _n_tiles,
                       _n_tile_size,
                       n_regressors,
                       hyperparams,
                       kernel_hyperparams,
                       trainable_params,
                       iter);
               })
        .get();
}

/**
 * @brief Calculate loss for given data and Gaussian process model
 */
double GP::calculate_loss()
{
    return hpx::async(
               [this]()
               {
                   return compute_loss_hpx(
                       _training_input, _training_output, kernel_hyperparams, _n_tiles, _n_tile_size, n_regressors);
               })
        .get();
}

/**
 * @brief Computes & returns cholesky decomposition
 */
std::vector<std::vector<double>> GP::cholesky()
{
    return hpx::async(
               [this]()
               { return cholesky_hpx(_training_input, kernel_hyperparams, _n_tiles, _n_tile_size, n_regressors); })
        .get();
}

}  // namespace gprat
