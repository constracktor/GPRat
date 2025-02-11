#include "../include/tiled_algorithms_cpu.hpp"

#include "../include/adapter_cblas_fp64.hpp"
#include "../include/gp_algorithms_cpu.hpp"
#include "../include/gp_optimizer.hpp"
#include "../include/gp_uncertainty.hpp"
#include <hpx/future.hpp>

// Tiled Cholesky Algorithm ------------------------------------------------ {{{

void right_looking_cholesky_tiled(Tiled_matrix &ft_tiles, int N, std::size_t n_tiles)
{
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        // POTRF: Compute Cholesky factor L
        ft_tiles[k * n_tiles + k] =
            hpx::dataflow(hpx::annotated_function(potrf, "cholesky_tiled"), ft_tiles[k * n_tiles + k], N);
        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // TRSM:  Solve X * L^T = A
            ft_tiles[m * n_tiles + k] = hpx::dataflow(
                hpx::annotated_function(trsm, "cholesky_tiled"),
                ft_tiles[k * n_tiles + k],
                ft_tiles[m * n_tiles + k],
                N,
                N,
                Blas_trans,
                Blas_right);
        }
        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // SYRK:  A = A - B * B^T
            ft_tiles[m * n_tiles + m] = hpx::dataflow(
                hpx::annotated_function(syrk, "cholesky_tiled"),
                ft_tiles[m * n_tiles + m],
                ft_tiles[m * n_tiles + k],
                N);
            for (std::size_t n = k + 1; n < m; n++)
            {
                // GEMM: C = C - A * B^T
                ft_tiles[m * n_tiles + n] = hpx::dataflow(
                    hpx::annotated_function(gemm, "cholesky_tiled"),
                    ft_tiles[m * n_tiles + k],
                    ft_tiles[n * n_tiles + k],
                    ft_tiles[m * n_tiles + n],
                    N,
                    N,
                    N,
                    Blas_no_trans,
                    Blas_trans);
            }
        }
    }
}

// }}} ----------------------------------------- end of Tiled Cholesky Algorithm

// Tiled Triangular Solve Algorithms --------------------------------------- {{{

void forward_solve_tiled(Tiled_matrix &ft_tiles, Tiled_vector &ft_rhs, int N, std::size_t n_tiles)
{
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        // TRSM: Solve L * x = a
        ft_rhs[k] = hpx::dataflow(
            hpx::annotated_function(trsv, "triangular_solve_tiled"),
            ft_tiles[k * n_tiles + k],
            ft_rhs[k],
            N,
            Blas_no_trans);
        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // GEMV: b = b - A * a
            ft_rhs[m] = hpx::dataflow(
                hpx::annotated_function(gemv, "triangular_solve_tiled"),
                ft_tiles[m * n_tiles + k],
                ft_rhs[k],
                ft_rhs[m],
                N,
                N,
                Blas_substract,
                Blas_no_trans);
        }
    }
}

void backward_solve_tiled(Tiled_matrix &ft_tiles, Tiled_vector &ft_rhs, int N, std::size_t n_tiles)
{
    for (int k_ = static_cast<int>(n_tiles) - 1; k_ >= 0; k_--)  // int instead of std::size_t for last comparison
    {
        std::size_t k = static_cast<std::size_t>(k_);
        // TRSM: Solve L^T * x = a
        ft_rhs[k] = hpx::dataflow(
            hpx::annotated_function(trsv, "triangular_solve_tiled"),
            ft_tiles[k * n_tiles + k],
            ft_rhs[k],
            N,
            Blas_trans);
        for (int m_ = k_ - 1; m_ >= 0; m_--)  // int instead of std::size_t for last comparison
        {
            std::size_t m = static_cast<std::size_t>(m_);
            // GEMV:b = b - A^T * a
            ft_rhs[m] = hpx::dataflow(
                hpx::annotated_function(gemv, "triangular_solve_tiled"),
                ft_tiles[k * n_tiles + m],
                ft_rhs[k],
                ft_rhs[m],
                N,
                N,
                Blas_substract,
                Blas_trans);
        }
    }
}

void forward_solve_tiled_matrix(
    Tiled_matrix &ft_tiles, Tiled_matrix &ft_rhs, int N, int M, std::size_t n_tiles, std::size_t m_tiles)
{
    for (std::size_t c = 0; c < m_tiles; c++)
    {
        for (std::size_t k = 0; k < n_tiles; k++)
        {
            // TRSM: solve L * X = A
            ft_rhs[k * m_tiles + c] = hpx::dataflow(
                hpx::annotated_function(trsm, "triangular_solve_tiled_matrix"),
                ft_tiles[k * n_tiles + k],
                ft_rhs[k * m_tiles + c],
                N,
                M,
                Blas_no_trans,
                Blas_left);
            for (std::size_t m = k + 1; m < n_tiles; m++)
            {
                // GEMM: C = C - A * B
                ft_rhs[m * m_tiles + c] = hpx::dataflow(
                    hpx::annotated_function(gemm, "triangular_solve_tiled_matrix"),
                    ft_tiles[m * n_tiles + k],
                    ft_rhs[k * m_tiles + c],
                    ft_rhs[m * m_tiles + c],
                    N,
                    M,
                    N,
                    Blas_no_trans,
                    Blas_no_trans);
            }
        }
    }
}

void backward_solve_tiled_matrix(
    Tiled_matrix &ft_tiles, Tiled_matrix &ft_rhs, int N, int M, std::size_t n_tiles, std::size_t m_tiles)
{
    for (std::size_t c = 0; c < m_tiles; c++)
    {
        for (int k_ = static_cast<int>(n_tiles) - 1; k_ >= 0; k_--)  // int instead of std::size_t for last comparison
        {
            std::size_t k = static_cast<std::size_t>(k_);
            // TRSM: solve L^T * X = A
            ft_rhs[k * m_tiles + c] = hpx::dataflow(
                hpx::annotated_function(trsm, "triangular_solve_tiled_matrix"),
                ft_tiles[k * n_tiles + k],
                ft_rhs[k * m_tiles + c],
                N,
                M,
                Blas_trans,
                Blas_left);
            for (int m_ = k_ - 1; m_ >= 0; m_--)  // int instead of std::size_t for last comparison
            {
                std::size_t m = static_cast<std::size_t>(m_);
                // GEMM: C = C - A^T * B
                ft_rhs[m * m_tiles + c] = hpx::dataflow(
                    hpx::annotated_function(gemm, "triangular_solve_tiled_matrix"),
                    ft_tiles[k * n_tiles + m],
                    ft_rhs[k * m_tiles + c],
                    ft_rhs[m * m_tiles + c],
                    N,
                    M,
                    N,
                    Blas_trans,
                    Blas_no_trans);
            }
        }
    }
}

// }}} -------------------------------- end of Tiled Triangular Solve Algorithms

void matrix_vector_tiled(Tiled_matrix &ft_tiles,
                         Tiled_vector &ft_vector,
                         Tiled_vector &ft_rhs,
                         int N_row,
                         int N_col,
                         std::size_t n_tiles,
                         std::size_t m_tiles)
{
    for (std::size_t k = 0; k < m_tiles; k++)
    {
        for (std::size_t m = 0; m < n_tiles; m++)
        {
            ft_rhs[k] = hpx::dataflow(
                hpx::annotated_function(gemv, "prediction_tiled"),
                ft_tiles[k * n_tiles + m],
                ft_vector[m],
                ft_rhs[k],
                N_row,
                N_col,
                Blas_add,
                Blas_no_trans);
        }
    }
}

void symmetric_matrix_matrix_diagonal_tiled(
    Tiled_matrix &ft_tiles, Tiled_vector &ft_vector, int N, int M, std::size_t n_tiles, std::size_t m_tiles)
{
    for (std::size_t i = 0; i < m_tiles; ++i)
    {
        for (std::size_t n = 0; n < n_tiles; ++n)
        {  // Compute inner product to obtain diagonal elements of
           // V^T * V  <=> cross(K) * K^-1 * cross(K)^T
            ft_vector[i] = hpx::dataflow(
                hpx::annotated_function(dot_diag_syrk, "posterior_tiled"),
                ft_tiles[n * m_tiles + i],
                ft_vector[i],
                N,
                M);
        }
    }
}

void symmetric_matrix_matrix_tiled(
    Tiled_matrix &ft_tiles, Tiled_matrix &ft_result, int N, int M, std::size_t n_tiles, std::size_t m_tiles)
{
    for (std::size_t c = 0; c < m_tiles; c++)
    {
        for (std::size_t k = 0; k < m_tiles; k++)
        {
            for (std::size_t m = 0; m < n_tiles; m++)
            {
                // (SYRK for (c == k) possible)
                // GEMM:  C = C - A^T * B
                ft_result[c * m_tiles + k] = hpx::dataflow(
                    hpx::annotated_function(&gemm, "triangular_solve_tiled_matrix"),
                    ft_tiles[m * m_tiles + c],
                    ft_tiles[m * m_tiles + k],
                    ft_result[c * m_tiles + k],
                    N,
                    M,
                    M,
                    Blas_trans,
                    Blas_no_trans);
            }
        }
    }
}

void vector_difference_tiled(Tiled_vector &ft_minuend, Tiled_vector &ft_subtrahend, int M, std::size_t m_tiles)
{
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        ft_subtrahend[i] =
            hpx::dataflow(hpx::annotated_function(&axpy, "uncertainty_tiled"), ft_minuend[i], ft_subtrahend[i], M);
    }
}

void matrix_diagonal_tiled(Tiled_matrix &ft_tiles, Tiled_vector &ft_vector, int M, std::size_t m_tiles)
{
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        ft_vector[i] = hpx::dataflow(
            hpx::annotated_function(get_matrix_diagonal, "uncertainty_tiled"), ft_tiles[i * m_tiles + i], M);
    }
}

void compute_loss_tiled(Tiled_matrix &ft_tiles,
                        Tiled_vector &ft_alpha,
                        Tiled_vector &ft_y,
                        hpx::shared_future<double> &loss,
                        int N,
                        std::size_t n_tiles)
{
    std::vector<hpx::shared_future<double>> loss_tiled;
    loss_tiled.reserve(n_tiles);
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        loss_tiled.push_back(hpx::dataflow(
            hpx::annotated_function(hpx::unwrapping(&compute_loss), "loss_tiled"),
            ft_tiles[k * n_tiles + k],
            ft_alpha[k],
            ft_y[k],
            N));
    }

    loss = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&add_losses), "loss_tiled"), loss_tiled, N, n_tiles);
}

///////////////////////////////////////////////////////////////////////////////
// optimization stuff

// Perform a gradient descent step for selected hyperparameter using Adam
// algorithm
void update_hyperparameter(
    const Tiled_matrix &ft_invK,
    const Tiled_matrix &ft_gradK_param,
    const Tiled_vector &ft_alpha,
    const gprat_hyper::AdamParams &adam_params,
    gprat_hyper::SEKParams &sek_params,
    int N,
    std::size_t n_tiles,
    std::size_t iter,
    std::size_t param_idx)
{
    /*
     * PART 1:
     * Compute gradient = 0.5 * ( trace(inv(K) * grad(K)_param) + y^T * inv(K) * grad(K)_param * inv(K) * y )
     *
     * 1: Compute   trace(inv(K) * grad(K)_param)
     * 2: Compute   y^T * inv(K) * grad(K)_param * inv(K) * y
     *
     * Update parameter:
     * 3: Update moments
     *      - m_T = beta1 * m_T-1 + (1 - beta1) * g_T
     *      - w_T = beta2 + w_T-1 + (1 - beta2) * g_T^2
     * 4: Adam step:
     *      - nu_T = nu * sqrt(1 - beta2_T) / (1 - beta1_T) 
     *      - theta_T = theta_T-1 - nu_T * m_T / (sqrt(w_T) + epsilon)
    */
    hpx::shared_future<double> trace = hpx::make_ready_future(0.0);
    hpx::shared_future<double> dot = hpx::make_ready_future(0.0);
    bool jitter = false;
    if (param_idx == 0 || param_idx == 1)  // 0: lengthscale; 1: vertical-lengthscale
    {

        Tiled_vector diag_tiles;    // Diagonal tiles
        Tiled_vector inter_alpha; // Intermediate result
        // Preallocate memory
        inter_alpha.reserve(n_tiles);
        diag_tiles.reserve(n_tiles);
        // Asynchrnonous assembly
        for (std::size_t d = 0; d < n_tiles; d++)
        {
            diag_tiles.push_back(hpx::async(hpx::annotated_function(gen_tile_zeros, "assemble"), N));
            inter_alpha.push_back(hpx::async(hpx::annotated_function(gen_tile_zeros, "assemble"), N));
        }

        ////////////////////////////////////
        // PART 1: Compute gradient
        // trace(inv(K)*grad_param)
        // Compute diagonal elements of inv(K) * grad(K)_param
        for (std::size_t i = 0; i < n_tiles; ++i)
        {
            for (std::size_t j = 0; j < n_tiles; ++j)
            {
                diag_tiles[i] = hpx::dataflow(
                    hpx::annotated_function(dot_diag_gemm, "trace"),
                    ft_invK[i * n_tiles + j],
                    ft_gradK_param[j * n_tiles + i],
                    diag_tiles[i],
                    N,
                    N);
            }
        }
        // Compute trace(inv(K) * grad(K)_param)
        for (std::size_t j = 0; j < n_tiles; ++j)
        {
            trace = hpx::dataflow(
                hpx::annotated_function(hpx::unwrapping(&compute_trace), "trace"), diag_tiles[j], trace);
        }
        // Not sure if can be done this way
        // alpha^T * grad(K)_param * alpha (with alpha = inv(K) * y)
        // Compute grad(K)_param * alpha
        for (std::size_t k = 0; k < n_tiles; k++)
        {
            for (std::size_t m = 0; m < n_tiles; m++)
            {
                inter_alpha[k] = hpx::dataflow(
                    hpx::annotated_function(gemv, "gemv"),
                    ft_gradK_param[k * n_tiles + m],
                    ft_alpha[m],
                    inter_alpha[k],
                    N,
                    N,
                    Blas_add,
                    Blas_no_trans);
            }
        }
        // ??
        for (std::size_t j = 0; j < n_tiles; ++j)
        { 
            dot = hpx::dataflow(
                hpx::annotated_function(hpx::unwrapping(&compute_dot), "grad_right_tiled"),
                inter_alpha[j],
                ft_alpha[j],
                dot);
        }
    }
    else if(param_idx == 2)
    {
    jitter =true;
    ///////////////////////////////////////
    // part1: compute trace(inv(K) * )
    for (std::size_t j = 0; j < n_tiles; ++j)
    {
        trace = hpx::dataflow(
            hpx::annotated_function(hpx::unwrapping(&compute_trace_noise), "grad_left_tiled"),
            ft_invK[j * n_tiles + j],
            trace,
            sek_params.noise_variance,
            N);
    }
    ///////////////////////////////////////
    /// part 2: alpha^T * grad_param * alpha
    for (std::size_t j = 0; j < n_tiles; ++j)
    {
        dot = hpx::dataflow(
            hpx::annotated_function(hpx::unwrapping(&compute_dot_noise), "grad_right_tiled"),
            ft_alpha[j],
            dot,
            sek_params.noise_variance);
    }
    }
    else
    {
        // Throw an exception for invalid param_idx
        throw std::invalid_argument("Invalid param_idx");
    }

          // get values and compute sequentially
        // compute gradient = trace + dot
        double gradient = compute_gradient(trace.get(), dot.get(), static_cast<std::size_t>(N), n_tiles);
        ////////////////////////////////////
        // PART 2: Update parameter
        // Update moments
        // m_T = beta1 * m_T-1 + (1 - beta1) * g_T
        sek_params.m_T[param_idx] = update_first_moment(gradient,
            sek_params.m_T[param_idx],
            adam_params.beta1);
        // w_T = beta2 + w_T-1 + (1 - beta2) * g_T^2
        sek_params.w_T[param_idx] = update_second_moment(gradient,
            sek_params.w_T[param_idx],
            adam_params.beta2);

        // Transform hyperparameter to unconstrained form
        double unconstrained_param = to_unconstrained(sek_params.get_param(param_idx), jitter);
        // Adam step update with unconstrained parameter
        // compute beta_t inside
        double updated_param = adam_step(unconstrained_param,
            adam_params,
            sek_params.m_T[param_idx],
            sek_params.w_T[param_idx],
            static_cast<std::size_t>(iter));
        // Transform hyperparameter back to constrained form
        sek_params.set_param(param_idx, to_constrained(updated_param, jitter));
}
