#!/bin/bash
# $1: python/cpp
################################################################################
set -e  # Exit immediately if a command exits with a non-zero status.
#set -x  # Print each command before executing it.

################################################################################
# Configurations
################################################################################
# Load GCC compiler
if [[ "$2" == "arm" ]]
then
    spack load gcc@14.2.0
    export USE_MKL=OFF
elif [[ "$2" == "riscv" ]]
then
    echo "TBD"
else
    module load gcc@14.2.0
    export USE_MKL=ON
fi

# Activate spack environment
spack env activate gprat_cpu_gcc

# Bindings and examples
if [[ "$1" == "python" ]]
then
	export EXAMPLES=OFF
	export BINDINGS=ON
	export INSTALL_DIR=$(pwd)/examples/gprat_python
elif [[ "$1" == "cpp" ]]
then
	export EXAMPLES=ON
	export BINDINGS=OFF
	export INSTALL_DIR=$(pwd)/examples/gprat_cpp
else
    echo "Please specify input parameter: python/cpp"
    exit 1
fi

# Release:	release-linux
# Debug:	dev-linux
export PRESET=release-linux

################################################################################
# Compile code
################################################################################
cmake --preset $PRESET \
      -DGPRAT_BUILD_BINDINGS=$BINDINGS \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
      -DHPX_IGNORE_BOOST_COMPATIBILITY=ON \
      -DGPRAT_ENABLE_FORMAT_TARGETS=OFF \
      -DGPRAT_ENABLE_EXAMPLES=$EXAMPLES \
      -DGPRAT_ENABLE_MKL=$USE_MKL
cmake --build --preset $PRESET -- -j
cmake --install build/$PRESET

cd build/$PRESET
ctest --output-on-failure --no-tests=ignore -C Release -j 2
