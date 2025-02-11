#ifndef GP_KERNELS_H
#define GP_KERNELS_H
#include <vector>
//#include <cstddef>

namespace gprat_hyper
{
/**
 * @brief Squared Exponential Kernel Parameters
 */
struct SEKParams
{
    /**
     * @brief Lengthscale: variance of training output
     *
     * Sometimes denoted with index 0.
     */
    double lengthscale;

    /**
     * @brief Vertical Lengthscale: standard deviation of training input
     *
     * Sometimes denoted with index 1.
     */
    double vertical_lengthscale;

    /**
     * @brief Noise Variance: small value
     *
     * Sometimes denoted with index 2.
     */
    double noise_variance;


    std::vector<double> m_T;

    std::vector<double> w_T;

    /**
     * @brief Construct a new SEKParams object
     *
     * @param lengthscale Lengthscale: variance of training output
     * @param vertical_lengthscale Vertical Lengthscale: standard deviation
     * of training input
     * @param noise_variance Noise Variance: small value
     */
    SEKParams(double lengthscale_,
              double vertical_lengthscale_,
              double noise_variance_);

    /**
     * @brief return the number of parameters
     */
    std::size_t size();

    void set_param(std::size_t index, double value);

    const double& get_param(std::size_t index) const;


};

}  // namespace gprat_hyper

#endif  // end of GP_KERNELS_H
