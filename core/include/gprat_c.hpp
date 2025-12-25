#ifndef GPRAT_C_H
#define GPRAT_C_H

#include "gp_hyperparameter.hpp"
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
     * @brief Target handle pointing to the unit used for computation.
     */
    std::shared_ptr<Target> target_;

  public:
    /** @brief Number of regressors */
    int n_reg;

    /**
     * @brief Hyperarameters of the squared exponential kernel
     */
    SEKParams kernel_params;

    /**
     * @brief Constructs a Gaussian Process (GP)
     *
     * @param input Input data for training of the GP
     * @param n_tiles Number of tiles
     * @param n_tile_size Size of each tile in each dimension
     * @param n_regressors Number of regressors
     * @param kernel_hyperparams Vector including lengthscale,
     *                           vertical lengthscale, and noise variance
     *                           parameter of squared exponential kernel
     * @param target Target for computations
     */
    GP(std::vector<double> input,
       int n_tiles,
       int n_tile_size,
       int n_regressors,
       std::vector<double> kernel_hyperparams,
       std::shared_ptr<Target> target);

    /**
     * @brief Constructs a Gaussian Process (GP) for CPU computations
     *
     * @param input Input data for training of the GP
     * @param n_tiles Number of tiles
     * @param n_tile_size Size of each tile in each dimension
     * @param n_regressors Number of regressors
     * @param kernel_hyperparams Vector including lengthscale,
     *                           vertical lengthscale, and noise variance
     *                           parameter of squared exponential kernel
     */
    GP(std::vector<double> input,
       int n_tiles,
       int n_tile_size,
       int n_regressors,
       std::vector<double> kernel_hyperparams);

    /**
     * @brief Constructs a Gaussian Process (GP) for GPU computations
     *
     * @param input Input data for training of the GP
     * @param n_tiles Number of tiles
     * @param n_tile_size Size of each tile in each dimension
     * @param n_regressors Number of regressors
     * @param kernel_hyperparams Vector including lengthscale,
     *                           vertical lengthscale, and noise variance
     *                           parameter of squared exponential kernel
     * @param gpu_id GPU identifier
     * @param n_streams Number of CUDA streams for GPU computations
     */
    GP(std::vector<double> input,
       int n_tiles,
       int n_tile_size,
       int n_regressors,
       std::vector<double> kernel_hyperparams,
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
     * @brief Computes (asynchronous) & returns cholesky decomposition
     */
    std::vector<std::vector<double>> cholesky();

    /**
     * @brief Computes (synchronous) & returns cholesky decomposition
     */
    std::vector<std::vector<double>> cholesky_synchronous();

        /**
     * @brief Computes (asynchronous) & returns cholesky decomposition
     */
    std::vector<std::vector<double>> cholesky_loop();
};
}  // namespace gprat

#endif  // end of GPRAT_C_H
