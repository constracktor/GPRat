add_executable(gprat_distributed src/main.cpp src/distributed_blas.cpp src/distributed_tile.cpp)
target_compile_features(gprat_distributed PUBLIC cxx_std_20)

find_package(Boost REQUIRED)
target_link_libraries(gprat_distributed PUBLIC GPRat::core HPX::hpx Boost::boost)
