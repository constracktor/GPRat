#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

/**
 * @brief Kernel to transpose a matrix.
 *
 * @param transposed Pointer to the transposed output matrix.
 * @param original Pointer to the original input matrix.
 * @param width Width of the original matrix.
 * @param height Height of the original matrix.
 */
__global__ void transpose(double *transposed, double *original, std::size_t width, std::size_t height);

#endif  // CUDA_KERNELS_H
