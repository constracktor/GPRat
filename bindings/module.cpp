#include <pybind11/pybind11.h>  // PYBIND11_MODULE

namespace py = pybind11;

void init_gprat(py::module &);  // See gprat_py.cpp
void init_utils(py::module &);  // See utils_py.cpp

// Define Python module with name gprat and handle m
PYBIND11_MODULE(gprat, m)
{
    m.doc() = "GPRat library";

    // NOTE: order of operations matters

    init_gprat(m);  // Adds classes: `GP_data`, `AdamParams`, `GP`

    init_utils(m);  // adds module functions: `compute_train_tiles`,
                    // `compute_train_tile_size`, `compute_test_tiles`, `print`,
                    // `start_hpx`, `resume_hpx`, `suspend_hpx`, `stop_hpx`,
                    // `compiled_with_cuda`, `print_available_gpus`, and
                    // `gpu_count`
}
