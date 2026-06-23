add_executable(gprat_distributed src/main.cpp)
target_compile_features(gprat_distributed PUBLIC cxx_std_20)

include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.8.0)

FetchContent_MakeAvailable(Catch2)

find_package(Boost REQUIRED)
target_link_libraries(gprat_distributed PUBLIC GPRat::core HPX::hpx
                                               Catch2::Catch2 Boost::boost)

set_target_properties(gprat_distributed PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY
                                                   "${CMAKE_SOURCE_DIR}")
