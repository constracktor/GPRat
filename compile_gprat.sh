#!/bin/bash
# $1: python/cpp
# $2: cpu/gpu
# $3: mkl/arm/riscv
################################################################################
set -ex  # Exit immediately if a command exits with a non-zero status.

################################################################################
# Configurations
################################################################################
# Select BLAS library
if [[ "$3" == "mkl" ]]
then
    export USE_MKL=ON
else
    export USE_MKL=OFF
fi

# Release: release-linux
# Debug: dev-linux
# Release for GPU: release-linux-gpu
# Debug for GPU: dev-linux-gpu
preset=release-linux-gpu

# Bindings and examples
if [[ "$1" == "python" ]]
then
	bindings=ON
	install_dir=$(pwd)/examples/gprat_python
elif [[ "$1" == "cpp" ]]
then
	bindings=OFF
	install_dir=$(pwd)/examples/gprat_cpp
else
    echo "Please specify input parameter: python/cpp"
    exit 1
fi

if [[ -z "$2" ]]; then
    echo "Second parameter is missing. Using default: cpu"
    cpu=1
    gpu=0
elif [[ "$2" == "cpu" ]]; then
    cpu=1
    gpu=0
elif [[ "$2" == "gpu" ]]; then
    cpu=0
    gpu=1
else
    echo "Please specify input parameter: cpu/gpu"
    exit 1
fi

if [[ $preset == "release-linux" || $preset == "dev-linux" ]]; then
	# Load compiler and dependencies
	if [[ "$3" == "arm" ]]
	then
    		spack load gcc@14.2.0
    		spack env activate gprat_cpu_arm
    		export USE_MKL=OFF
	elif [[ "$3" == "riscv" ]]
	then
    		#module load gcc/13.2.1
    		spack load openblas arch=linux-fedora38-riscv64
    		export HPX_CMAKE=$HOME/git_workspace/build-scripts/build/hpx/lib64/cmake/HPX
    		export USE_MKL=OFF
	else
    		# x86
    		module load gcc@14.2.0
    		spack env activate gprat_cpu_gcc
	fi

    cmake --preset $preset \
	-DGPRAT_BUILD_BINDINGS=$bindings \
	-DCMAKE_INSTALL_PREFIX=$install_dir \
	-DHPX_IGNORE_BOOST_COMPATIBILITY=ON \
	-DGPRAT_ENABLE_FORMAT_TARGETS=OFF

elif [[ $preset == "release-linux-gpu" || $preset == "dev-linux-gpu" ]]; then
    # Load Clang compiler and CUDA library
    module load clang/17.0.1
    module load cuda/12.0.1

    # Activate spack environment
    spack env activate gprat_gpu_clang

    cuda_arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk -F '.' '{print $1$2}')

    cmake --preset $preset \
	-DGPRAT_BUILD_BINDINGS=$bindings \
	-DCMAKE_INSTALL_PREFIX=$install_dir \
	-DHPX_IGNORE_BOOST_COMPATIBILITY=ON \
	-DGPRAT_ENABLE_FORMAT_TARGETS=OFF \
        -DCMAKE_C_COMPILER=$(which clang) \
        -DCMAKE_CXX_COMPILER=$(which clang++) \
        -DCMAKE_CUDA_COMPILER=$(which clang++) \
        -DCMAKE_CUDA_FLAGS=--cuda-path=${CUDA_HOME} \
        -DCMAKE_CUDA_ARCHITECTURES=$cuda_arch
fi

################################################################################
# Compile code
################################################################################
cmake --build --preset $PRESET -- -j
cmake --install build/$PRESET

cd build/$preset
ctest --output-on-failure --no-tests=ignore -C Release -j 2
