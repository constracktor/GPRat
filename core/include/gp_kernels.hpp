#ifndef GP_KERNELS_H
#define GP_KERNELS_H

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

    /**
     * @brief Construct a new SEKParams object
     *
     * @param lengthscale Lengthscale: variance of training output
     * @param vertical_lengthscale Vertical Lengthscale: standard deviation
     * of training input
     * @param noise_variance Noise Variance: small value
     */
    SEKParams(double lengthscale,
              double vertical_lengthscale,
              double noise_variance);
};

}  // namespace gprat_hyper

#endif  // end of GP_KERNELS_H
