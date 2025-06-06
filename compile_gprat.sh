#!/bin/bash
# $1: python/cpp
# $2: cpu/gpu
# $3: mkl
################################################################################
set -e  # Exit immediately if a command exits with a non-zero status.

################################################################################
# Configurations
################################################################################
# Bindings
if [[ "$1" == "python" ]]
then
    BINDINGS=ON
    INSTALL_DIR=$(pwd)/examples/gprat_python
elif [[ "$1" == "cpp" ]]
then
    BINDINGS=OFF
    INSTALL_DIR=$(pwd)/examples/gprat_cpp
else
    echo "Please specify input parameter: python/cpp"
    exit 1
fi

# Select CMake preset
if [[ "$2" == "cpu" ]]; then
    # Release:
    PRESET=release-linux
    # Debug:
    #PRESET=dev-linux
elif [[ "$2" == "gpu" ]]; then
    # Release:
    PRESET=release-linux-gpu
    # Debug:
    #PRESET=dev-linux-gpu
elif [[ "$2" != "cpu" ]]; then
    echo "Input parameter is missing. Using default: Run computations on CPU in Release mode"
    PRESET=release-linux
fi

# Select BLAS library
if [[ "$3" == "mkl" ]]
then
    USE_MKL=ON
else
    USE_MKL=OFF
fi

if command -v spack &> /dev/null; then
    echo "Spack command found, checking for environments..."

    # Get current hostname
    HOSTNAME=$(hostname -s)

    if [[ "$HOSTNAME" == "ipvs-epyc1" ]]; then
	# Check if the gprat_cpu_gcc environment exists
    	if spack env list | grep -q "gprat_cpu_gcc"; then
	   echo "Found gprat_cpu_gcc environment, activating it."
	    module load gcc/14.2.0
	    export CXX=g++
	    export CC=gcc
	    spack env activate gprat_cpu_gcc
	    GPRAT_WITH_CUDA=OFF # whether GPRAT_WITH_CUDA is ON of OFF is irrelevant for this example
	fi
    elif [[ "$HOSTNAME" == "sven0"  ||  "$HOSTNAME" == "sven1" ]]; then
	#module load gcc/13.2.1
	spack load openblas arch=linux-fedora38-riscv64
	HPX_CMAKE=$HOME/git_workspace/build-scripts/build/hpx/lib64/cmake/HPX
	USE_MKL=OFF
    elif [[ $(uname -i) == "aarch64" ]]; then
	spack load gcc@14.2.0
	# Check if the gprat_cpu_arm environment exists
	if spack env list | grep -q "gprat_cpu_arm"; then
	    echo "Found gprat_cpu_arm environment, activating it."
	    spack env activate gprat_cpu_arm
	fi
	USE_MKL=OFF
    elif [[ "$HOSTNAME" == "simcl1n1" || "$HOSTNAME" == "simcl1n2" ]]; then
	# Check if the gprat_gpu_clang environment exists
	if spack env list | grep -q "gprat_gpu_clang"; then
	    echo "Found gprat_gpu_clang environment, activating it."
	    module load clang/17.0.1
	    export CXX=clang++
	    export CC=clang
	    module load cuda/12.0.1
	    spack env activate gprat_gpu_clang
	    GPRAT_WITH_CUDA=ON
	fi
    else
    	echo "Hostname is $HOSTNAME — no action taken."
    fi
else
    echo "Spack command not found. Building example without Spack."
    # Assuming that Spack is not required on given system
fi

if [[ $PRESET == "release-linux" || $PRESET == "dev-linux" ]]; then
    cmake --preset $PRESET \
	-DGPRAT_BUILD_BINDINGS=$BINDINGS \
	-DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
	-DHPX_IGNORE_BOOST_COMPATIBILITY=ON \
	-DHPX_DIR=$HPX_CMAKE \
	-DGPRAT_ENABLE_FORMAT_TARGETS=OFF \
	-DGPRAT_ENABLE_MKL=$USE_MKL
elif [[ $PRESET == "release-linux-gpu" || $PRESET == "dev-linux-gpu" ]]; then
    CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk -F '.' '{print $1$2}')

    cmake --preset $PRESET \
	-DGPRAT_BUILD_BINDINGS=$BINDINGS \
	-DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
	-DHPX_IGNORE_BOOST_COMPATIBILITY=ON \
	-DGPRAT_ENABLE_FORMAT_TARGETS=OFF \
        -DGPRAT_ENABLE_MKL=$USE_MKL \
	-DCMAKE_C_COMPILER=$(which clang) \
        -DCMAKE_CXX_COMPILER=$(which clang++) \
        -DCMAKE_CUDA_COMPILER=$(which clang++) \
        -DCMAKE_CUDA_FLAGS=--cuda-path=${CUDA_HOME} \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}
fi

################################################################################
# Compile code
################################################################################
cmake --build --preset $PRESET -- -j
cmake --install build/$PRESET

cd build/$PRESET
ctest --output-on-failure --no-tests=ignore -C Release -j 2
