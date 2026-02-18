if(GPRAT_WITH_CUDA)
  enable_language(CUDA)
endif()

# Option for GPU support with CUDA, cuSolver, cuBLAS
option(GPRAT_WITH_CUDA "Enable GPU support with CUDA, cuSolver, cuBLAS" OFF)
# Pass variable to C++ code
add_compile_definitions(GPRAT_WITH_CUDA=$<BOOL:${GPRAT_WITH_CUDA}>)

# Option for steps duration measurement with APEX
option(GPRAT_APEX_STEPS "Enable measuring duration of steps with APEX" OFF)
# Pass variable to C++ code
add_compile_definitions(GPRAT_APEX_STEPS=$<BOOL:${GPRAT_APEX_STEPS}>)

# Option for measuring duration of assembly of covariance matrix and right
# looking cholesky in the cholesky function using APEX. Technically, the
# cholesky function also returns a large matrix but this adds significant
# overhead to the whole function call and does not provide much useful
# information on the optimized assembly and right looking cholesky steps which
# are used by other GP operations.
option(
  GPRAT_APEX_CHOLESKY
  "Enable measuring duration of cholesky assembly and computation with APEX"
  OFF)
# Pass variable to C++ code
add_compile_definitions(GPRAT_APEX_CHOLESKY=$<BOOL:${GPRAT_APEX_CHOLESKY}>)

set(SOURCE_FILES
    src/gprat_c.cpp
    src/utils_c.cpp
    src/target.cpp
    src/gp_kernels.cpp
    src/gp_hyperparameters.cpp
    src/cpu/gp_functions.cpp
    src/cpu/gp_algorithms.cpp
    src/cpu/gp_uncertainty.cpp
    src/cpu/gp_optimizer.cpp
    src/cpu/tiled_algorithms.cpp
    src/cpu/adapter_cblas_fp32.cpp
    src/cpu/adapter_cblas_fp64.cpp)

if(GPRAT_WITH_CUDA)
  list(
    APPEND
    SOURCE_FILES
    src/gpu/gp_functions.cu
    src/gpu/gp_algorithms.cu
    src/gpu/gp_uncertainty.cu
    src/gpu/gp_optimizer.cu
    src/gpu/tiled_algorithms.cu
    src/gpu/adapter_cublas.cu
    src/gpu/cuda_kernels.cu)
endif()

add_library(gprat_core STATIC ${SOURCE_FILES})

# We'd like to consume it just as core under the GPRat namespace
set_property(TARGET gprat_core PROPERTY EXPORT_NAME core)
add_library(GPRat::core ALIAS gprat_core)

# Add them as PRIVATE sources here so they show up in project files Can't use
# PUBLIC etc., see: https://stackoverflow.com/a/62465051
file(GLOB_RECURSE header_files CONFIGURE_DEPENDS include/*.hpp
     include/cpu/*.hpp)
if(GPRAT_WITH_CUDA)
  file(GLOB_RECURSE cuda_headers CONFIGURE_DEPENDS include/gpu/*.cuh)
  list(APPEND header_files ${cuda_headers})
endif()
target_sources(gprat_core PRIVATE ${header_files})

# Link HPX libraries
target_link_libraries(gprat_core PUBLIC HPX::hpx)

if(GPRAT_WITH_CUDA)
  target_link_libraries(gprat_core PUBLIC CUDA::cusolver CUDA::cublas)
endif()

# Include directories
target_include_directories(
  gprat_core PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/core/include>")

# Link BLAS
if(GPRAT_ENABLE_MKL)
  # Link Intel oneMKL
  target_link_libraries(gprat_core PUBLIC MKL::mkl_intel_lp64 MKL::mkl_core
                                          MKL::MKL MKL::mkl_sequential)
  target_compile_definitions(gprat_core PUBLIC GPRAT_ENABLE_MKL)
else()
  # Link OpenBLAS
  target_link_libraries(gprat_core PUBLIC ${OpenBLAS_LIB})
endif()

target_compile_features(gprat_core PUBLIC cxx_std_20)

set_property(TARGET gprat_core PROPERTY POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_SKIP_INSTALL_RULES)
  # We need to manually install those into CMAKE_INSTALL_INCLUDEDIR. Below
  # install(TARGETS ...) only setups the paths for the exported targets.
  install(
    DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT Development)

  install(
    TARGETS gprat_core
    EXPORT GPRatTargets
    RUNTIME COMPONENT Runtime
    LIBRARY COMPONENT Runtime NAMELINK_COMPONENT Development
    ARCHIVE COMPONENT Development
    INCLUDES
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endif()
