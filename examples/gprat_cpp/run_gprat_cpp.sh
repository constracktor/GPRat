#!/bin/bash
# $1 cpu/cuda/sycl
# $2 mkl/none
# $3 SYCL with nvidia/amd/intel

################################################################################
set -e # Exit immediately if a command exits with a non-zero status.
#set -x  # Print each command before executing it.

# Resolve the example directory itself (independent of the caller's cwd),
# since we cd into build/run_gprat_cpp below and GPRat_DIR must still point at
# the already-installed GPRat package under this directory's lib/.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

################################################################################
# Configurations
################################################################################

# Set device for computation
if [[ -z "$1" ]]; then
  echo "Input parameter is missing. Using default: Run computations on CPU"
elif [[ "$1" == "cuda" || "$1" == "sycl" ]]; then
  use_gpu="--use-gpu"
elif [[ "$1" != "cpu" ]]; then
  echo "Please specify input parameter: cpu/cuda/sycl"
  exit 1
fi

if [[ "$3" == "nvidia" ]]; then

  CMAKE_PREFIX_PATH="/scratch-simcl1/grafml/Programs/oneMath_nvidia/oneMath/install/lib/cmake/oneMath:${CMAKE_PREFIX_PATH:-}"

elif [[ "$3" == "amd" ]]; then

  CMAKE_PREFIX_PATH="/scratch-simcl1/grafml/Programs/oneMath_amd/oneMath/install/lib/cmake/oneMath:${CMAKE_PREFIX_PATH:-}"

elif [[ "$3" == "intel" ]]; then

  # SITE-SPECIFIC: update this path to your local oneMath install prefix, or
  # set CMAKE_PREFIX_PATH before invoking this script to override it.
  CMAKE_PREFIX_PATH="/scratch/grafml/oneMath_intel_v0.9/oneMath/install:${CMAKE_PREFIX_PATH:-}"

fi

# Select BLAS library
if [[ "$2" == "mkl" ]]; then
  USE_MKL=ON
else
  USE_MKL=OFF
fi

# Set Spack if on simcl1n1, simcl1n2, simcl1n3, or simcl1n4
if [[ "$HOSTNAME" == "simcl1n1" || "$HOSTNAME" == "simcl1n2" || "$HOSTNAME" == "simcl1n3" || "$HOSTNAME" == "simcl1n4" ]]; then

  spack_destination="/scratch-simcl1/grafml/Programs/spack-fp2-simcl1n1"
  source $spack_destination/spack/share/spack/setup-env.sh

fi

# Set Spack if on pcsgs04
# SITE-SPECIFIC: spack_destination is hardcoded for the pcsgs04 cluster.
# Adjust this path to match your local Spack installation before running on a different machine.
if [[ "$HOSTNAME" == "pcsgs04" ]]; then

  spack_destination="/scratch/grafml/gprat-spack/spack/"
  source $spack_destination/share/spack/setup-env.sh

fi

if command -v spack &>/dev/null; then

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
      GPRAT_APEX_STEPS=OFF
      GPRAT_APEX_CHOLESKY=OFF
    fi

  elif [[ "$HOSTNAME" == "sven0" || "$HOSTNAME" == "sven1" ]]; then
    #module load gcc/13.2.1
    spack load openblas arch=linux-fedora38-riscv64
    HPX_CMAKE=$HOME/git_workspace/build-scripts/build/hpx/lib64/cmake/HPX
    GPRAT_WITH_CUDA=OFF
    GPRAT_APEX_STEPS=OFF
    GPRAT_APEX_CHOLESKY=OFF
    ADD=64
  elif [[ $(uname -i) == "aarch64" ]]; then
    spack load gcc@14.2.0
    # Check if the gprat_cpu_arm environment exists
    if spack env list | grep -q "gprat_cpu_arm"; then
      echo "Found gprat_cpu_arm environment, activating it."
      spack env activate gprat_cpu_arm
    fi
    GPRAT_WITH_CUDA=OFF
    GPRAT_APEX_STEPS=OFF
    GPRAT_APEX_CHOLESKY=OFF
    ADD=64

  elif [[ "$HOSTNAME" == "simcl1n1" || "$HOSTNAME" == "simcl1n2" ]]; then
    GPRAT_APEX_STEPS=OFF
    GPRAT_APEX_CHOLESKY=OFF

    # Check if the gprat_gpu_clang environment exists
    if spack env list | grep -q "gprat_gpu_clang"; then

      echo "Found gprat_gpu_clang environment, activating it."
      spack env activate gprat_gpu_clang

      if [[ "$1" == "cuda" ]]; then

        module load clang/17.0.1
        export CXX=clang++
        export CC=clang
        module load cuda/12.0.1
        GPRAT_WITH_CUDA=ON
        GPRAT_WITH_SYCL=OFF

      elif [[ "$1" == "sycl" ]]; then

        if command -v icpx --version &>/dev/null; then

          export CXX=icpx
          export CC=icx
          GPRAT_WITH_CUDA=OFF
          GPRAT_WITH_SYCL=ON

        else

          echo "DPC++ compiler not found. Please make sure that a DPC++ compiler is available in your PATH." 1>&2
          exit -1

        fi

      fi

    fi

  elif [[ "$HOSTNAME" == "simcl1n3" ]]; then
    GPRAT_APEX_STEPS=OFF
    GPRAT_APEX_CHOLESKY=OFF

    # Check if the gprat_gpu_clang environment exists
    if spack env list | grep -q "gprat_gpu_clang"; then

      echo "Found gprat_gpu_clang environment, activating it."
      spack env activate gprat_gpu_clang
      CMAKE_PREFIX_PATH="/scratch-simcl1/grafml/Programs/oneMath_nvidia/oneMath/install/lib/cmake/oneMath:${CMAKE_PREFIX_PATH:-}"

      if [[ "$1" == "sycl" ]]; then

        if command -v icpx --version &>/dev/null; then

          export CXX=icpx
          export CC=icx
          GPRAT_WITH_CUDA=OFF
          GPRAT_WITH_SYCL=ON

        else

          echo "DPC++ compiler not found. Please make sure that a DPC++ compiler is available in your PATH." 1>&2
          exit -1

        fi

      fi

    fi

  # pcsgs04 with Intel GPU (Arc B580) #############################################################
  elif [[ "$HOSTNAME" == "pcsgs04" ]]; then

    GPRAT_APEX_STEPS=OFF
    GPRAT_APEX_CHOLESKY=OFF

    # Check if the gprat_gpu_clang environment exists
    if spack env list | grep -q "gprat_gpu_clang"; then

      echo "Found gprat_gpu_clang environment, activating it."
      spack env activate gprat_gpu_clang

      if [[ "$1" == "sycl" ]]; then

        # icpx is not provided by the gprat_gpu_clang Spack environment on this host; it comes
        # from the system oneAPI install. Source it if icpx isn't already on PATH.
        # Pin to the compiler version the oneMath install below was built with (2025.3) -
        # newer icpx releases changed SYCL queue/BLAS header signatures and fail to compile
        # against this oneMath install.
        if ! command -v icpx &>/dev/null && [[ -f /opt/intel/oneapi/compiler/2025.3/env/vars.sh ]]; then
          source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
        fi

        # The Level-Zero GPU backend needs libumf (Unified Memory Framework) on
        # LD_LIBRARY_PATH; without it, GPU platform enumeration silently returns zero
        # devices and the example fails at runtime with "Requested GPU device is not available."
        if [[ -f /opt/intel/oneapi/umf/latest/env/vars.sh ]]; then
          source /opt/intel/oneapi/umf/latest/env/vars.sh
        fi

        # oneMath's Level-Zero libmkl_sycl_lapack/blas.so were built against the
        # system MKL 2025.3, not the older MKL 2024.2 bundled in the
        # gprat_gpu_clang Spack environment. Source it (and matching TBB) so
        # CMake's find_package(MKL) and the runtime linker resolve against the
        # matching version - otherwise linking fails with undefined references
        # like mkl_lapack_dpotrf_batch_strided.
        if [[ -f /opt/intel/oneapi/mkl/2025.3/env/vars.sh ]]; then
          source /opt/intel/oneapi/mkl/2025.3/env/vars.sh
        fi
        if [[ -f /opt/intel/oneapi/tbb/latest/env/vars.sh ]]; then
          source /opt/intel/oneapi/tbb/latest/env/vars.sh
        fi

        # The gprat_gpu_clang Spack environment bundles its own, older TBB
        # (2021.13) whose libtbb.so is missing symbols oneMath's MKL libraries
        # need (e.g. get_thread_reference_vertex). LD_LIBRARY_PATH alone isn't
        # enough since the linker's own -L search uses LIBRARY_PATH; prepend
        # the matching oneAPI TBB there too so it's found first at link time.
        if [[ -n "${TBBROOT:-}" ]]; then
          LIBRARY_PATH="$TBBROOT/lib:${LIBRARY_PATH:-}"
        fi

        if command -v icpx --version &>/dev/null; then

          export CXX=icpx
          export CC=icx
          GPRAT_WITH_CUDA=OFF
          GPRAT_WITH_SYCL=ON

          # SITE-SPECIFIC: update this path to your local oneMath install prefix, or
          # set CMAKE_PREFIX_PATH before invoking this script to override it.
          CMAKE_PREFIX_PATH="/scratch/grafml/oneMath_intel_v0.9/oneMath/install:${CMAKE_PREFIX_PATH:-}"

        else

          echo \
            "Intel oneAPI DPC++ compiler (icpx) not found. Please make sure that icpx is available in your PATH." 1>&2
          exit -1

        fi

      fi

    else

      echo \
        "Cannot find Spack environment gprat_gpu_clang. Please run spack-repo/environments/setup_gprat_gpu_clang.sh" 1>&2
      exit -1

    fi

  else

    echo "Hostname is $HOSTNAME — no action taken."
  fi

else

  echo "Spack command not found. Building example without Spack."
  # Assuming that Spack is not required on given system
fi

# Configure APEX
export APEX_SCREEN_OUTPUT=0
export APEX_DISABLE=1

################################################################################
# Compile code
################################################################################

cd "$SCRIPT_DIR"
rm -rf build && mkdir build && cd build && mkdir run_gprat_cpp && cd run_gprat_cpp

# Configure the project
#
# On pcsgs04, the gprat_gpu_clang Spack environment's RPATH is searched by the
# linker ahead of oneMath's own RPATH, so its bundled (older) TBB shadows the
# one oneMath's MKL libraries actually need. Force an explicit -L for the
# correct TBB (set above via TBBROOT when $1=sycl on pcsgs04) so the linker
# resolves the transitive libtbb.so.12 dependency there first.
if [[ -n "${TBBROOT:-}" ]]; then
  EXTRA_LINKER_FLAGS="-Wl,-rpath-link,${TBBROOT}/lib"
fi

cmake "$SCRIPT_DIR" -DCMAKE_BUILD_TYPE=Release \
  -DGPRat_DIR=${SCRIPT_DIR}/lib$ADD/cmake/GPRat \
  -DGPRAT_WITH_CUDA=${GPRAT_WITH_CUDA} \
  -DGPRAT_WITH_SYCL=${GPRAT_WITH_SYCL} \
  -DGPRAT_APEX_STEPS=${GPRAT_APEX_STEPS} \
  -DGPRAT_APEX_CHOLESKY=${GPRAT_APEX_CHOLESKY} \
  -DHPX_DIR=$HPX_CMAKE \
  -DUSE_MKL=$USE_MKL \
  -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH \
  -DCMAKE_EXE_LINKER_FLAGS="${EXTRA_LINKER_FLAGS:-}"

# Build the project
make -j

################################################################################
# Run code
################################################################################
echo "Running GPRat C++ example"

end_cores=$(python3 -c "import json; print(json.load(open('${SCRIPT_DIR}/config.json'))['END_CORES'])")
core_count=$((end_cores * 2))

taskset -c 0-$core_count:2 ./gprat_cpp $use_gpu

echo "Finished running GPRat C++ example"
