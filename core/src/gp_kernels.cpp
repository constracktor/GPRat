#include "../include/gp_kernels.hpp"
#include <stdexcept>
namespace gprat_hyper
{
SEKParams::SEKParams(double lengthscale_,
                     double vertical_lengthscale_,
                     double noise_variance_) :
    lengthscale(lengthscale_),
    vertical_lengthscale(vertical_lengthscale_),
    noise_variance(noise_variance_) {
        m_T.resize(this->size());
        w_T.resize(this->size());
    };

std::size_t SEKParams::size(){return 3;}

void SEKParams::set_param(std::size_t index, double value) {
        if (index == 0) lengthscale = value;
        else if (index == 1) vertical_lengthscale = value;
        else if (index == 2) noise_variance = value;
        else throw std::invalid_argument("Set Invalid param_idx");
    }

const double& SEKParams::get_param(std::size_t index) const {
        if (index == 0) return lengthscale;
        if (index == 1) return vertical_lengthscale;
        if (index == 2) return noise_variance;
        throw std::invalid_argument("Get Invalid param_idx");
    }
}
