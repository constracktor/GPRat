#include "../include/gp_kernels.hpp"

namespace gprat_hyper
{
SEKParams::SEKParams(double lengthscale,
                     double vertical_lengthscale,
                     double noise_variance) :
    lengthscale(lengthscale),
    vertical_lengthscale(vertical_lengthscale),
    noise_variance(noise_variance) {};

}
