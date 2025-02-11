#include "../include/gp_hyperparameters.hpp"
#include <iomanip>

namespace gprat_hyper
{

AdamParams::AdamParams(double lr,
                       double b1,
                       double b2,
                       double eps,
                       int opt_i,
                       std::vector<double> M_T_init,
                       std::vector<double> V_T_init) :
    learning_rate(lr),
    beta1(b1),
    beta2(b2),
    epsilon(eps),
    opt_iter(opt_i),
    M_T(M_T_init),
    V_T(V_T_init)
{ }

std::string AdamParams::repr() const
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(8);

    // clang-format off
        oss << "Hyperparameters: [learning_rate=" << learning_rate
                            << ", beta1=" << beta1
                            << ", beta2=" << beta2
                            << ", epsilon=" << epsilon
                            << ", opt_iter=" << opt_iter << "]";
    // clang-format on

    return oss.str();
}

}  // namespace gprat_hyper
