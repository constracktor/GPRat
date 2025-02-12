#ifndef GP_HYPERPARAMETERS_H
#define GP_HYPERPARAMETERS_H

#include <string>
#include <vector>

namespace gprat_hyper
{

/**
 * @brief Hyperparameters for the Adam optimizer
 */
struct AdamParams
{
    /** @brief TODO: documentation */
    double learning_rate;

    /** @brief TODO: documentation */
    double beta1;

    /** @brief TODO: documentation */
    double beta2;

    /** @brief TODO: documentation */
    double epsilon;

    /** @brief TODO: documentation */
    int opt_iter;

    /** @brief TODO: documentation */
    std::vector<double> M_T;

    /** @brief TODO: documentation */
    std::vector<double> V_T;

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
