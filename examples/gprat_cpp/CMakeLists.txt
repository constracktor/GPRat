cmake_minimum_required(VERSION 3.16)

project(gprat_cpp)

# This project can be built as part of GPRat itself or as a standalone project
# in which case below find_package() call is used.
if(TARGET GPRat::core)
  message("Using in-tree GPRat::core")
else()
  message("Using out-of-tree GPRat::core")
  find_package(GPRat REQUIRED)
  # Try to find Intel oneMKL
  set(MKL_INTERFACE_FULL "intel_lp64")
  set(MKL_THREADING "sequential")
  find_package(MKL CONFIG REQUIRED)
endif()

# Enables/disables computation with CUDA on test system
option(GPRAT_WITH_CUDA
       "Enable additional computation with CUDA, cuSolver, cuBLAS" OFF)
# Pass variable to C++ code
add_compile_definitions(GPRAT_WITH_CUDA=$<BOOL:${GPRAT_WITH_CUDA}>)

# Add the executable
add_executable(gprat_cpp src/execute.cpp)

target_compile_features(gprat_cpp PUBLIC cxx_std_17)

# Link the libraries
target_link_libraries(gprat_cpp PUBLIC GPRat::core)
