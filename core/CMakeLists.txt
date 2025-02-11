add_library(
  gprat_core STATIC
  src/gp_optimizer.cpp
  src/gp_functions.cpp
  src/gp_hyperparameters.cpp
  src/gp_algorithms_cblas.cpp
  src/gprat_c.cpp
  src/adapter_cblas_fp32.cpp
  src/adapter_cblas_fp64.cpp
  src/tiled_algorithms_cpu.cpp
  src/gp_uncertainty.cpp
  src/utils_c.cpp)

# We'd like to consume it just as core under the GPRat namespace
set_property(TARGET gprat_core PROPERTY EXPORT_NAME core)
add_library(GPRat::core ALIAS gprat_core)

# Add them as PRIVATE sources here so they show up in project files Can't use
# PUBLIC etc., see: https://stackoverflow.com/a/62465051
file(GLOB_RECURSE header_files CONFIGURE_DEPENDS include/*.hpp)
target_sources(gprat_core PRIVATE ${header_files})

# Link HPX libraries
target_link_libraries(
  gprat_core PUBLIC HPX::hpx MKL::mkl_intel_lp64 MKL::mkl_core MKL::MKL
                    MKL::mkl_sequential)

# Include directories
target_include_directories(
  gprat_core PUBLIC "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/core/include>")

target_compile_features(gprat_core PUBLIC cxx_std_17)

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
