#ifndef GPRAT_C_H
#define GPRAT_C_H

#include "gp_hyperparameters.hpp"
#include "gp_kernels.hpp"
#include "target.hpp"
#include <memory>
#include <string>
#include <vector>

// namespace for GPRat library entities
namespace gprat
{

/**
 * @brief Data structure for Gaussian Process data
 *
 * It includes the file path to the data, the number of samples, and the
 * data itself which contains this many samples.
 */
struct GP_data
{
    /** @brief Path to the file containing the data */
    std::string file_path;

    /** @brief Number of samples in the data */
    int n_samples;

    /** @brief Number of GP regressors */
    int n_regressors;

    /** @brief Vector containing the data */
    std::vector<double> data;

    /**
     * @brief Initialize of Gaussian process data by loading data from a
     * file.
     *
     * The file specified by `f_path` must contain `n` samples.
     *
     * @param f_path Path to the file
     * @param n Number of samples
     */
    GP_data(const std::string &file_path, int n, int n_reg);
};

/**
 * @brief Gaussian Process class for regression tasks
 *
 * This class provides methods for training a Gaussian Process model, making
 * predictions, optimizing hyperparameters, and calculating loss. It also
 * includes methods for computing the Cholesky decomposition.
 */
class GP
{
  private:
    /** @brief Input data for training */
    std::vector<double> training_input_;

    /** @brief Output data for given input data */
    std::vector<double> training_output_;

    /** @brief Number of tiles */
    int n_tiles_;

    /** @brief Size of each tile in each dimension */
    int n_tile_size_;

    /**
     * @brief List of bools indicating trainable parameters: lengthscale,
     * vertical lengthscale, noise variance
     */
    std::vector<bool> trainable_params_;

    /**
     * @brief Target handle pointing to the unit used for computation.
     */
    std::shared_ptr<Target> target_;

  public:
    /** @brief Number of regressors */
    int n_reg;

    /**
     * @brief Hyperarameters of the squared exponential kernel
     */
    gprat_hyper::SEKParams kernel_params;

    /**
     * @brief Constructs a Gaussian Process (GP)
     *
     * @param input Input data for training of the GP
     * @param output Expected output data for training of the GP
     * @param n_tiles Number of tiles
     * @param n_tile_size Size of each tile in each dimension
     * @param n_regressors Number of regressors
     * @param kernel_hyperparams Vector including lengthscale,
     *                           vertical lengthscale, and noise variance
     *                           parameter of squared exponential kernel
     * @param trainable_bool Vector indicating which parameters are trainable
     * @param target Target for computations
     */
    GP(std::vector<double> input,
       std::vector<double> output,
       int n_tiles,
       int n_tile_size,
       int n_regressors,
       std::vector<double> kernel_hyperparams,
       std::vector<bool> trainable_bool,
       std::shared_ptr<Target> target);

    /**
     * @brief Constructs a Gaussian Process (GP) for CPU computations
     *
     * @param input Input data for training of the GP
     * @param output Expected output data for training of the GP
     * @param n_tiles Number of tiles
     * @param n_tile_size Size of each tile in each dimension
     * @param n_regressors Number of regressors
     * @param kernel_hyperparams Vector including lengthscale,
     *                           vertical lengthscale, and noise variance
     *                           parameter of squared exponential kernel
     * @param trainable_bool Vector indicating which parameters are trainable
     */
    GP(std::vector<double> input,
       std::vector<double> output,
       int n_tiles,
       int n_tile_size,
       int n_regressors,
       std::vector<double> kernel_hyperparams,
       std::vector<bool> trainable_bool);

    /**
     * @brief Constructs a Gaussian Process (GP) for GPU computations
     *
     * @param input Input data for training of the GP
     * @param output Expected output data for training of the GP
     * @param n_tiles Number of tiles
     * @param n_tile_size Size of each tile in each dimension
     * @param n_regressors Number of regressors
     * @param kernel_hyperparams Vector including lengthscale,
     *                           vertical lengthscale, and noise variance
     *                           parameter of squared exponential kernel
     * @param trainable_bool Vector indicating which parameters are trainable
     * @param gpu_id GPU identifier
     * @param n_streams Number of CUDA streams for GPU computations
     */
    GP(std::vector<double> input,
       std::vector<double> output,
       int n_tiles,
       int n_tile_size,
       int n_regressors,
       std::vector<double> kernel_hyperparams,
       std::vector<bool> trainable_bool,
       int gpu_id,
       int n_streams);

    /**
     * Returns Gaussian Process attributes as string.
     */
    std::string repr() const;

    /**
     * @brief Returns training input data
     */
    std::vector<double> get_training_input() const;

    /**
     * @brief Returns training output data
     */
    std::vector<double> get_training_output() const;

    /**
     * @brief Predict output for test input
     */
    std::vector<double> predict(const std::vector<double> &test_data, int m_tiles, int m_tile_size);

    /**
     * @brief Predict output for test input and additionally provide
     * uncertainty for the predictions.
     */
    std::vector<std::vector<double>>
    predict_with_uncertainty(const std::vector<double> &test_data, int m_tiles, int m_tile_size);

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
    predict_with_full_cov(const std::vector<double> &test_data, int m_tiles, int m_tile_size);

    /**
     * @brief Optimize hyperparameters
     *
     * @param hyperparams Hyperparameters of squared exponential kernel:
     *        lengthscale, vertical_lengthscale, noise_variance
     *
     * @return losses
     */
    std::vector<double> optimize(const gprat_hyper::AdamParams &adam_params);

    /**
     * @brief Perform a single optimization step
     *
     * @param hyperparams Hyperparameters of squared exponential kernel:
     *        lengthscale, vertical_lengthscale, noise_variance
     * @param iter number of iterations
     *
     * @return loss
     */
    double optimize_step(gprat_hyper::AdamParams &adam_params, int iter);

    /**
     * @brief Calculate loss for given data and Gaussian process model
     */
    double calculate_loss();

    /**
     * @brief Computes & returns cholesky decomposition
     */
    std::vector<std::vector<double>> cholesky();
};
}  // namespace gprat

#endif  // end of GPRAT_C_H
