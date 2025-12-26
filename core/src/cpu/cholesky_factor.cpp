#include "cpu/cholesky_factor.hpp"

#include "cpu/adapter_cblas_fp64.hpp"
#include "cpu/gp_kernel.hpp"
#include <hpx/future.hpp>
#include <hpx/algorithm.hpp>
#include <hpx/functional.hpp>

namespace cpu
{

// Tiled Cholesky Algorithm
void right_looking_cholesky_tiled(Variant variant, Tiled_matrix &ft_tiles, int N, std::size_t n_tiles)
{
    switch (variant)
    {
    // Asynchronous variants
        case Variant::async_future:
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        // POTRF: Compute Cholesky factor L
        ft_tiles[k * n_tiles + k] =
            hpx::dataflow(hpx::annotated_function(f_potrf, "cholesky_potrf"), ft_tiles[k * n_tiles + k], N);
        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // TRSM:  Solve X * L^T = A
            ft_tiles[m * n_tiles + k] = hpx::dataflow(
                hpx::annotated_function(f_trsm, "cholesky_trsm"),
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
                hpx::annotated_function(f_syrk, "cholesky_syrk"),
                ft_tiles[m * n_tiles + m],
                ft_tiles[m * n_tiles + k],
                N);
            for (std::size_t n = k + 1; n < m; n++)
            {
                // GEMM: C = C - A * B^T
                ft_tiles[m * n_tiles + n] = hpx::dataflow(
                    hpx::annotated_function(f_gemm, "cholesky_gemm"),
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
        break;

        case Variant::async_ref:
for (std::size_t k = 0; k < n_tiles; k++)
  {
    // POTRF
    ft_tiles[k * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&(potrf)), "cholesky_tiled"), ft_tiles[k * n_tiles + k], N);
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // TRSM
      ft_tiles[m * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&r_trsm), "cholesky_tiled"), ft_tiles[k * n_tiles + k], ft_tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
    }
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // SYRK
      ft_tiles[m * n_tiles + m] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&r_syrk), "cholesky_tiled"), ft_tiles[m * n_tiles + m], ft_tiles[m * n_tiles + k], N);
      for (std::size_t n = k + 1; n < m; n++)
      {
        // GEMM
        ft_tiles[m * n_tiles + n] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&r_gemm), "cholesky_tiled"), ft_tiles[m * n_tiles + k], ft_tiles[n * n_tiles + k], ft_tiles[m * n_tiles + n], N, N, N, Blas_no_trans, Blas_trans);
      }
    }
  }
        break;

        case Variant::async_val:
 for (std::size_t k = 0; k < n_tiles; k++)
  {
    // POTRF
    ft_tiles[k * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&(potrf)), "cholesky_tiled"), ft_tiles[k * n_tiles + k], N);
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // TRSM
      ft_tiles[m * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&v_trsm), "cholesky_tiled"), ft_tiles[k * n_tiles + k], ft_tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
    }
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // SYRK
      ft_tiles[m * n_tiles + m] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&v_syrk), "cholesky_tiled"), ft_tiles[m * n_tiles + m], ft_tiles[m * n_tiles + k], N);
      for (std::size_t n = k + 1; n < m; n++)
      {
        // GEMM
        ft_tiles[m * n_tiles + n] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&v_gemm), "cholesky_tiled"), ft_tiles[m * n_tiles + k], ft_tiles[n * n_tiles + k], ft_tiles[m * n_tiles + n], N, N, N, Blas_no_trans, Blas_trans);
      }
    }
  }
 // Synchronous variants
        case Variant::sync_future:
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        // POTRF: Compute Cholesky factor L
        ft_tiles[k * n_tiles + k] =
            hpx::dataflow(hpx::annotated_function(f_potrf, "cholesky_potrf"), ft_tiles[k * n_tiles + k], N);
        // Synchronize
        ft_tiles[k * n_tiles + k].get();

        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // TRSM:  Solve X * L^T = A
            ft_tiles[m * n_tiles + k] = hpx::dataflow(
                hpx::annotated_function(f_trsm, "cholesky_trsm"),
                ft_tiles[k * n_tiles + k],
                ft_tiles[m * n_tiles + k],
                N,
                N,
                Blas_trans,
                Blas_right);
        }

        // Synchronize
        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // TRSM:  Solve X * L^T = A
            ft_tiles[m * n_tiles + k].get();
        }


        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // SYRK:  A = A - B * B^T
            ft_tiles[m * n_tiles + m] = hpx::dataflow(
                hpx::annotated_function(f_syrk, "cholesky_syrk"),
                ft_tiles[m * n_tiles + m],
                ft_tiles[m * n_tiles + k],
                N);
            for (std::size_t n = k + 1; n < m; n++)
            {
                // GEMM: C = C - A * B^T
                ft_tiles[m * n_tiles + n] = hpx::dataflow(
                    hpx::annotated_function(f_gemm, "cholesky_gemm"),
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
        // Synchronize
        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            for (std::size_t n = k + 1; n <= m; n++)
            {
                ft_tiles[m * n_tiles + n].get();
            }
        }

    }
break;

        case Variant::sync_ref:
for (std::size_t k = 0; k < n_tiles; k++)
  {
    // POTRF
    ft_tiles[k * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&(potrf)), "cholesky_tiled"), ft_tiles[k * n_tiles + k], N);
            // Synchronize
        ft_tiles[k * n_tiles + k].get();
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // TRSM
      ft_tiles[m * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&r_trsm), "cholesky_tiled"), ft_tiles[k * n_tiles + k], ft_tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
    }
            // Synchronize
        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // TRSM:  Solve X * L^T = A
            ft_tiles[m * n_tiles + k].get();
        }
    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // SYRK
      ft_tiles[m * n_tiles + m] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&r_syrk), "cholesky_tiled"), ft_tiles[m * n_tiles + m], ft_tiles[m * n_tiles + k], N);
      for (std::size_t n = k + 1; n < m; n++)
      {
        // GEMM
        ft_tiles[m * n_tiles + n] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&r_gemm), "cholesky_tiled"), ft_tiles[m * n_tiles + k], ft_tiles[n * n_tiles + k], ft_tiles[m * n_tiles + n], N, N, N, Blas_no_trans, Blas_trans);
      }
    }
            // Synchronize
        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            for (std::size_t n = k + 1; n <= m; n++)
            {
                ft_tiles[m * n_tiles + n].get();
            }
        }
  }
        break;

        case Variant::sync_val:
 for (std::size_t k = 0; k < n_tiles; k++)
  {
    // POTRF
    ft_tiles[k * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&(potrf)), "cholesky_tiled"), ft_tiles[k * n_tiles + k], N);
          // Synchronize
    hpx::wait_all(ft_tiles[k * n_tiles + k]);

    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // TRSM
      ft_tiles[m * n_tiles + k] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&v_trsm), "cholesky_tiled"), ft_tiles[k * n_tiles + k], ft_tiles[m * n_tiles + k], N, N, Blas_trans, Blas_right);
    }
                // Synchronize
        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // TRSM:  Solve X * L^T = A
            hpx::wait_all(ft_tiles[m * n_tiles + k]);
        }

    for (std::size_t m = k + 1; m < n_tiles; m++)
    {
      // SYRK
      ft_tiles[m * n_tiles + m] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&v_syrk), "cholesky_tiled"), ft_tiles[m * n_tiles + m], ft_tiles[m * n_tiles + k], N);
      for (std::size_t n = k + 1; n < m; n++)
      {
        // GEMM
        ft_tiles[m * n_tiles + n] = hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&v_gemm), "cholesky_tiled"), ft_tiles[m * n_tiles + k], ft_tiles[n * n_tiles + k], ft_tiles[m * n_tiles + n], N, N, N, Blas_no_trans, Blas_trans);
      }
    }
    // Synchronize
        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            for (std::size_t n = k + 1; n <= m; n++)
            {
                hpx::wait_all(ft_tiles[m * n_tiles + n]);
            }
        }

  }
  break;
    default:
        std::cout << "Variant not supported.\n";
        break;
}
}



void right_looking_cholesky_tiled_loop(Variant variant, std::vector<std::vector<double>> &tiles, int N, std::size_t n_tiles)
{
switch (variant){
    case Variant::loop_ref:
      for (std::size_t k = 0; k < n_tiles; k++) {
        // POTRF: Compute Cholesky factor L
        tiles[k * n_tiles + k] = potrf(std::move(tiles[k * n_tiles + k]), N);

        hpx::experimental::for_loop( hpx::execution::par, k + 1, n_tiles, [&](std::size_t m)
        {
            // TRSM:  Solve X * L^T = A
            tiles[m * n_tiles + k] = r_trsm(tiles[k * n_tiles + k],
                std::move(tiles[m * n_tiles + k]),
                N,
                N,
                Blas_trans,
                Blas_right);
        });

        hpx::experimental::for_loop(hpx::execution::par, k + 1, n_tiles, [&](std::size_t m)
        {
            hpx::experimental::for_loop(hpx::execution::par, k + 1, m + 1, [&](std::size_t n)
            {
                if (n == m)
                {
                    // SYRK: A = A - B * B^T
                    tiles[m * n_tiles + m] =
                        r_syrk(std::move(tiles[m * n_tiles + m]),
                             tiles[m * n_tiles + k],
                             N);
                }
                else
                {
                    // GEMM: C = C - A * B^T
                    tiles[m * n_tiles + n] =
                        r_gemm(tiles[m * n_tiles + k],
                             tiles[n * n_tiles + k],
                             std::move(tiles[m * n_tiles + n]),
                             N, N, N,
                             Blas_no_trans,
                             Blas_trans);
                }
            });
        });
    }
  break;
    case Variant::loop_val:
   for (std::size_t k = 0; k < n_tiles; k++)
    {
        // POTRF: Compute Cholesky factor L
        tiles[k * n_tiles + k] = potrf(std::move(tiles[k * n_tiles + k]), N);

        hpx::experimental::for_loop( hpx::execution::par, k + 1, n_tiles, [&](std::size_t m)
        {
            // TRSM:  Solve X * L^T = A
            tiles[m * n_tiles + k] = v_trsm(tiles[k * n_tiles + k],
                std::move(tiles[m * n_tiles + k]),
                N,
                N,
                Blas_trans,
                Blas_right);
        });

        hpx::experimental::for_loop(hpx::execution::par, k + 1, n_tiles, [&](std::size_t m)
        {
            hpx::experimental::for_loop(hpx::execution::par, k + 1, m + 1, [&](std::size_t n)
            {
                if (n == m)
                {
                    // SYRK: A = A - B * B^T
                    tiles[m * n_tiles + m] =
                        v_syrk(std::move(tiles[m * n_tiles + m]),
                             tiles[m * n_tiles + k],
                             N);
                }
                else
                {
                    // GEMM: C = C - A * B^T
                    tiles[m * n_tiles + n] =
                        v_gemm(tiles[m * n_tiles + k],
                             tiles[n * n_tiles + k],
                             std::move(tiles[m * n_tiles + n]),
                             N, N, N,
                             Blas_no_trans,
                             Blas_trans);
                }
            });
        });
    }
break;

    default:
        std::cout << "Variant not supported.\n";
        break;
}}

}  // end of namespace cpu
