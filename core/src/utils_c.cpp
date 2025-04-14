#include "utils_c.hpp"

#include <cstdio>

namespace utils
{

int compute_train_tiles(int n_samples, int n_tile_size)
{
    if (n_tile_size > 0)
    {
        // n_tiles
        return n_samples / n_tile_size;
    }
    else
    {
        throw std::runtime_error("Error: Please specify a valid value for train_tile_size.\n");
    }
}

int compute_train_tile_size(int n_samples, int n_tiles)
{
    if (n_tiles > 0)
    {
        // n_tile_size
        return n_samples / n_tiles;
    }
    else
    {
        throw std::runtime_error("Error: Please specify a valid value for train_tiles.\n");
    }
}

std::pair<int, int> compute_test_tiles(int n_test, int n_tiles, int n_tile_size)
{
    int m_tiles;
    int m_tile_size;

    // if n_test is not divisible by (incl. smaller than) n_tile_size, use the same number of tiles
    if ((n_test % n_tile_size) > 0)
    {
        m_tiles = n_tiles;
        m_tile_size = n_test / m_tiles;
    }
    // if n_test is divisible by n_tile_size, use the same tile size
    else
    {
        m_tiles = n_test / n_tile_size;
        m_tile_size = n_tile_size;
    }
    return { m_tiles, m_tile_size };
}

std::vector<double> load_data(const std::string &file_path, int n_samples, int offset)
{
    std::vector<double> _data;
    _data.resize(static_cast<std::size_t>(n_samples + offset), 0.0);

    FILE *input_file = fopen(file_path.c_str(), "r");
    if (input_file == NULL)
    {
        throw std::runtime_error("Error: File not found: " + file_path);
    }

    // load data
    int scanned_elements = 0;
    for (int i = 0; i < n_samples; i++)
    {
        scanned_elements +=
            fscanf(input_file, "%lf", &_data[static_cast<std::size_t>(i + offset)]);  // scanned_elements++;
    }

    fclose(input_file);

    if (scanned_elements != n_samples)
    {
        throw std::runtime_error("Error: Data not correctly read. Expected " + std::to_string(n_samples)
                                 + " elements, but read " + std::to_string(scanned_elements));
    }
    return _data;
}

void print_vector(const std::vector<double> &vec, int start, int end, const std::string &separator)
{
    // Convert negative indices to positive
    if (start < 0)
    {
        start += static_cast<int>(vec.size());
    }
    if (end < 0)
    {
        end += static_cast<int>(vec.size()) + 1;
    }

    // Ensure the indices are within bounds
    if (start < 0)
    {
        start = 0;
    }
    if (end > static_cast<int>(vec.size()))
    {
        end = static_cast<int>(vec.size());
    }

    // Validate the range
    if (start >= static_cast<int>(vec.size()) || start >= end)
    {
        std::cerr << "Invalid range" << std::endl;
        return;
    }

    for (int i = start; i < end; i++)
    {
        std::cout << vec[static_cast<std::size_t>(i)];
        if (i < end - 1)
        {
            std::cout << separator;
        }
    }
    std::cout << std::endl;
}

void start_hpx_runtime(int argc, char **argv) { hpx::start(nullptr, argc, argv); }

void resume_hpx_runtime() { hpx::resume(); }

void suspend_hpx_runtime() { hpx::suspend(); }

void stop_hpx_runtime()
{
    hpx::post([]() { hpx::finalize(); });
    hpx::stop();
}

bool compiled_with_cuda()
{
#if GPRAT_WITH_CUDA
    return true;
#else
    return false;
#endif
}

}  // namespace utils
