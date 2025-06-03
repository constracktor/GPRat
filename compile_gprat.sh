#!/bin/bash
# $1: python/cpp
# $2: cpu/gpu
################################################################################
set -e  # Exit immediately if a command exits with a non-zero status.

################################################################################
# Configurations
################################################################################

# Bindings
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

# Select CMake preset
if [[ "$2" == "cpu" ]]; then
    # Release:
    preset=release-linux
    # Debug:
    #preset=dev-linux
elif [[ "$2" == "gpu" ]]; then
    # Release:
    preset=release-linux-gpu
    # Debug:
    #preset=dev-linux-gpu
elif [[ "$2" != "cpu" ]]; then
    echo "Input parameter is missing. Using default: Run computations on CPU in Release mode"
    preset=release-linux
fi

if command -v spack &> /dev/null; then
    echo "Spack command found, checking for environments..."

    # Get current hostname
    HOSTNAME=$(hostname)

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
    elif [[ "$HOSTNAME" == "simcl1n1" || "$HOSTNAME" == "simcl1n1" ]]; then
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

if [[ $preset == "release-linux" || $preset == "dev-linux" ]]; then
    cmake --preset $preset \
	-DGPRAT_BUILD_BINDINGS=$bindings \
	-DCMAKE_INSTALL_PREFIX=$install_dir \
	-DHPX_IGNORE_BOOST_COMPATIBILITY=ON \
	-DGPRAT_ENABLE_FORMAT_TARGETS=OFF

elif [[ $preset == "release-linux-gpu" || $preset == "dev-linux-gpu" ]]; then
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

cmake --build --preset $preset -- -j
cmake --install build/$preset

cd build/$preset
ctest --output-on-failure --no-tests=ignore -C Release -j 2
