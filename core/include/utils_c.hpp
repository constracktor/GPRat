#ifndef UTILS_C_H
#define UTILS_C_H

#include <hpx/future.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/hpx_suspend.hpp>
#include <string>
#include <vector>

namespace utils
{
/**
 * @brief Compute the number of tiles for training data, given the number of
 * samples and the size of each tile.
 *
 * @param n_samples Number of samples
 * @param n_tile_size Size of each tile
 */
int compute_train_tiles(int n_samples, int n_tile_size);

/**
 * @brief Compute the number of tiles for training data, given the number of
 * samples and the size of each tile.
 *
 * @param n_samples Number of samples
 * @param n_tile_size Size of each tile
 */
int compute_train_tile_size(int n_samples, int n_tiles);

/**
 * @brief Compute the number of test tiles and the size of a test tile.
 *
 * Uses n_tiles_size if n_test is divisible by n_tile_size. Otherwise uses
 * n_tiles for calculation.
 *
 * @param n_test Number of test samples
 * @param n_tiles Number of tiles
 * @param n_tile_size Size of each tile
 */
std::pair<int, int> compute_test_tiles(int n_test, int n_tiles, int n_tile_size);

/**
 * @brief Load data from file
 *
 * @param file_path Path to the file
 * @param n_samples Number of samples to load
 */
std::vector<double> load_data(const std::string &file_path, int n_samples, int offset);

/**
 * @brief Print a vector
 *
 * @param vec Vector to print
 * @param start Start index
 * @param end End index
 * @param separator Separator between elements
 */
void print_vector(const std::vector<double> &vec, int start, int end, const std::string &separator);

/**
 * @brief Start HPX runtime
 *
 * @param argc Number of arguments
 * @param argv Arguments as array of strings
 */
void start_hpx_runtime(int argc, char **argv);

/**
 * @brief Resume HPX runtime
 */
void resume_hpx_runtime();

/**
 * @brief Wait for all tasks to finish, and suspend the HPX runtime
 */
void suspend_hpx_runtime();

/**
 * @brief Stop HPX runtime
 */
void stop_hpx_runtime();

/**
 * @brief Returns whether the code was compiled with CUDA support.
 */
bool compiled_with_cuda();

}  // namespace utils

#endif
