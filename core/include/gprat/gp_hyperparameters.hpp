#ifndef GP_HYPERPARAMETERS_H
#define GP_HYPERPARAMETERS_H

#include <string>

namespace gprat_hyper
{

/**
 * @brief Hyperparameters for the Adam optimizer
 */
struct AdamParams
{
    /**
     * @brief Learning rate is step size per iteration
     */
    double learning_rate;

    /**
     * @brief Beta1 is the exponential decay rate for the first moment estimates
     */
    double beta1;

    /**
     * @brief Beta2 is the exponential decay rate for the second moment estimates
     */
    double beta2;

    /**
     * @brief Epsilon is a small constant to prevent division by zero
     */
    double epsilon;

    /**
     * @brief Number of optimization iterations
     */
    int opt_iter;

    /**
     * @brief Initialize hyperparameters
     *
     * @param lr learning rate
     * @param b1 beta1
     * @param b2 beta2
     * @param eps epsilon
     * @param opt_i number of optimization iterationsgp op
     * @param M_T_init initial values for first moment vector
     * @param V_T_init initial values for second moment vector
     */
    AdamParams(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8, int opt_i = 0);

    /**
     * @brief Returns a string representation of the hyperparameters
     */
    std::string repr() const;
};

}  // namespace gprat_hyper

#endif  // GP_HYPERPARAMETERS_H
