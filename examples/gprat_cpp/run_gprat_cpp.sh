#!/bin/bash
################################################################################
set -e  # Exit immediately if a command exits with a non-zero status.
#set -x  # Print each command before executing it.

################################################################################
# Configurations
################################################################################
# Load compiler and dependencies
if [[ "$1" == "arm" ]]
then
    spack load gcc@14.2.0
    spack env activate gprat_cpu_gcc
    export LIB=lib64
elif [[ "$1" == "riscv" ]]
then
    spack load openblas arch=linux-fedora38-riscv64
    export HPX_CMAKE=$HOME/git_workspace/build-scripts/build/hpx/lib64/cmake/HPX
    export LIB=lib64
else
    module load gcc@14.2.0
    spack env activate gprat_cpu_gcc
    export LIB=lib
    export CC=gcc
    export CXX=g++
fi

# Configure APEX
export APEX_SCREEN_OUTPUT=0
export APEX_DISABLE=1

################################################################################
# Compile code
################################################################################
rm -rf build && mkdir build && cd build
# Configure the project
cmake .. -DCMAKE_BUILD_TYPE=Release -DGPRat_DIR=./$LIB/cmake/GPRat -DHPX_DIR=$HPX_CMAKE
 # Build the project
make -j

################################################################################
# Run code
################################################################################
./gprat_cpp
