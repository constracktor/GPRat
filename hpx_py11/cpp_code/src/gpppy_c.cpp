#include "../include/automobile_bits/gpppy_c.hpp"
#include "../include/automobile_bits/utils_c.hpp"

#include <stdexcept>
#include <iomanip>
#include <cstdio>
#include <sstream>

namespace gpppy
{
    // Implementation of the Car constructor
    GP_data::GP_data(const std::string &f_path, int n)
    {
        n_samples = n;
        file_path = f_path;
        data = utils::load_data(f_path, n);
    }

    Kernel_Params::Kernel_Params(double l, double v, double n, int n_r)
        : lengthscale(l), vertical_lengthscale(v), noise_variance(n), n_regressors(n_r) {}

    std::string Kernel_Params::repr() const
    {
        std::ostringstream oss;
        oss << "Kernel_Params: [lengthscale=" << lengthscale
            << ", vertical_lengthscale=" << vertical_lengthscale
            << ", noise_variance=" << noise_variance
            << ", n_regressors=" << n_regressors << "]";
        return oss.str();
    }

    GP::GP(std::vector<double> input, std::vector<double> output, int n_tiles, int n_tile_size, double l, double v, double n, int n_r, std::vector<bool> trainable_bool)
    {
        _training_input = input;
        _training_output = output;
        _n_tiles = n_tiles;
        _n_tile_size = n_tile_size;
        lengthscale = l;
        vertical_lengthscale = v;
        noise_variance = n;
        n_regressors = n_r;
        trainable_params = trainable_bool;
    }

    std::vector<double> GP::get_training_input() const
    {
        return _training_input;
    }

    std::vector<double> GP::get_training_output() const
    {
        return _training_output;
    }

    std::string GP::repr() const
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(8);
        oss << "Kernel_Params: [lengthscale=" << lengthscale
            << ", vertical_lengthscale=" << vertical_lengthscale
            << ", noise_variance=" << noise_variance
            << ", n_regressors=" << n_regressors 
            << ", trainable_params l=" << trainable_params[0]
            << ", trainable_params v=" << trainable_params[1]
            << ", trainable_params n=" << trainable_params[2]
            << "]";
        return oss.str();
    }

    std::vector<std::vector<double>> GP::predict(const std::vector<double> &test_data, int m_tiles, int m_tile_size)
    {
        hpx::shared_future<std::vector<std::vector<double>>> fut = predict_hpx(_training_input, _training_output, test_data,
                                                                               _n_tiles, _n_tile_size, m_tiles, m_tile_size,
                                                                               lengthscale, vertical_lengthscale, noise_variance, n_regressors);

        // hpx::async([input, output]()
        //  { return add_vectors(input, output); });

        std::vector<std::vector<double>> result;
        hpx::threads::run_as_hpx_thread([&result, &fut]()
                                        {
                                            result = fut.get(); // Wait for and get the result from the future
                                        });
        return result;
    }

    std::vector<double> GP::optimize(const gpppy_hyper::Hyperparameters &hyperparams)
    {
        hpx::shared_future<std::vector<double>> fut = optimize_hpx(_training_input, _training_output, _n_tiles, _n_tile_size,
                                                                   lengthscale, vertical_lengthscale, noise_variance, n_regressors, 
                                                                   hyperparams, trainable_params);

        // hpx::async([input, output]()
        //  { return add_vectors(input, output); });

        std::vector<double> losses;
        hpx::threads::run_as_hpx_thread([&losses, &fut]()
                                        {
                                            losses = fut.get(); // Wait for and get the result from the future
                                        });

        // Set new hyperparams

        return losses;
    }

}
