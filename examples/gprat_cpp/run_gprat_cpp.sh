#!/bin/bash
# $1 cpu/gpu
# $2 x86/arm/riscv

################################################################################
set -e  # Exit immediately if a command exits with a non-zero status.
#set -x  # Print each command before executing it.

################################################################################
# Configurations
################################################################################

if [[ -z "$1" ]]; then
    echo "First parameter is missing. Using default: cpu"
    cpu=1
    gpu=0
elif [[ "$1" == "cpu" ]]; then
    cpu=1
    gpu=0
elif [[ "$1" == "gpu" ]]; then
    cpu=0
    gpu=1
else
    echo "Please specify input parameter: cpu/gpu"
    exit 1
fi

if [[ $cpu -eq 1 ]]; then
	# Load compiler and dependencies
	if [[ "$2" == "arm" ]]
	then
    		spack load gcc@14.2.0
    		spack env activate gprat_cpu_arm
    	export LIB=lib64
	elif [[ "$2" == "riscv" ]]
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

elif [[ $gpu -eq 1 ]]; then
    # Load Clang compiler and CUDA library
    module load clang/17.0.1
    module load cuda/12.0.1

    # Activate spack environment
    spack env activate gprat_gpu_clang

    if [[ -z "$2" ]]; then
        echo "Second parameter is missing. Using default: Run computations on CPU"
    elif [[ "$2" == "gpu" ]]; then
        use_gpu="--use_gpu"
    elif [[ "$2" != "cpu" ]]; then
        echo "Please specify second input parameter: cpu/gpu"
        exit 1
    fi
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
./gprat_cpp $use_gpu
