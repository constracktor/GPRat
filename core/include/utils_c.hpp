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

// Compute number of test tiles and the size of a test tile
std::pair<int, int> compute_test_tiles(int m_samples, int n_tiles, int n_tile_size);

// Load data from file
std::vector<double> load_data(const std::string &file_path, int n_samples, int offset = 0);

// Print a vector
void print_vector(const std::vector<double> &vec, int start = 0, int end = -1, const std::string &separator = " ");

// Start HPX runtime
void start_hpx_runtime(int argc, char **argv);

// Resume HPX runtime
void resume_hpx_runtime();

// Wait for all tasks to finish, and suspend the HPX runtime
void suspend_hpx_runtime();

// Stop HPX runtime
void stop_hpx_runtime();
}  // namespace utils

#endif
