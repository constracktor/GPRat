#!/bin/bash
# Builds and runs the gprat_distributed benchmark against the CPU (OpenBLAS/MKL) backend.
# All arguments are forwarded to the gprat_distributed binary, e.g.:
#   ./run_gprat_distributed.sh --start 128 --end 4096 --step 2 --tiles 8 --loop 3
#
# NOTE: this script only launches a single HPX locality. Running across multiple
# localities/nodes requires additional HPX network configuration (parcelport,
# AGAS bootstrap, hostfile/mpirun setup) that is specific to the target cluster
# and is not set up here yet.

set -e # Exit immediately if a command exits with a non-zero status.

###################################################################################################
# Set Spack if on simcl1n1, simcl1n2, simcl1n3, or simcl1n4
###################################################################################################

if [[ \
  "$HOSTNAME" == "simcl1n1" || \
  "$HOSTNAME" == "simcl1n2" || \
  "$HOSTNAME" == "simcl1n3" || \
  "$HOSTNAME" == "simcl1n4" ]];
then

  spack_destination="/scratch-simcl1/grafml/Programs/spack-fp2-simcl1n1"
  source $spack_destination/spack/share/spack/setup-env.sh

fi

###################################################################################################
# Setup environment depending on the host
###################################################################################################

if command -v spack &>/dev/null; then

  echo "Spack command found, checking for environments..."

  HOSTNAME=$(hostname -s)

  # ipvs-epyc1 ####################################################################################
  if [[ "$HOSTNAME" == "ipvs-epyc1" ]]; then

    if spack env list | grep -q "gprat_cpu_gcc"; then
      echo "Found gprat_cpu_gcc environment, activating it."
      spack env activate gprat_cpu_gcc
      module load gcc/14.2.0
      export CXX=g++
      export CC=gcc
    fi

  # sven0 and sven1 ###############################################################################
  elif [[ "$HOSTNAME" == "sven0" || "$HOSTNAME" == "sven1" ]]; then

    spack load openblas arch=linux-fedora38-riscv64
    HPX_CMAKE=$HOME/git_workspace/build-scripts/build/hpx/lib64/cmake/HPX
    export LD_LIBRARY_PATH=$HOME/git_workspace/build-scripts/build/hpx/lib64:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$HOME/git_workspace/build-scripts/build/boost/lib:$LD_LIBRARY_PATH
    export LD_PRELOAD=$HOME/git_workspace/build-scripts/build/jemalloc/lib/libjemalloc.so.2

  # aarch64 #######################################################################################
  elif [[ $(uname -i) == "aarch64" ]]; then

    spack load gcc@14.2.0
    if spack env list | grep -q "gprat_cpu_arm"; then
      echo "Found gprat_cpu_arm environment, activating it."
      spack env activate gprat_cpu_arm
    fi

  # simcl1n1, simcl1n2, simcl1n3, simcl1n4 (CPU only) #############################################
  elif [[ \
    "$HOSTNAME" == "simcl1n1" || \
    "$HOSTNAME" == "simcl1n2" || \
    "$HOSTNAME" == "simcl1n3" || \
    "$HOSTNAME" == "simcl1n4" ]];
  then

    if spack env list | grep -q "gprat_cpu_gcc"; then
      echo "Found gprat_cpu_gcc environment, activating it."
      spack env activate gprat_cpu_gcc
      module load gcc/14.1.0
      export CXX=g++
      export CC=gcc
      LD_LIBRARY_PATH=$(spack location -i hpx)/lib:$LD_LIBRARY_PATH
      LD_LIBRARY_PATH=$(spack location -i openblas)/lib:$LD_LIBRARY_PATH
      LD_LIBRARY_PATH=$(spack location -i intel-oneapi-mkl)/lib:$LD_LIBRARY_PATH
    else
      echo "Cannot find Spack environment gprat_cpu_gcc. Please run spack-repo/environments/setup_gprat_cpu_gcc.sh" 1>&2
      exit 1
    fi

  # unknown host ##################################################################################
  else

    echo "Caution: This script does not cover host $HOSTNAME."

  fi

else

  echo "Spack command not found. Building example without Spack."

fi

###################################################################################################
# Configure APEX
###################################################################################################

export APEX_SCREEN_OUTPUT=0
export APEX_DISABLE=1

###################################################################################################
# Compile code
###################################################################################################

# Unlike examples/gprat_cpp, examples/gprat_distributed is only ever built in-tree
# (it has no standalone/out-of-tree CMake support), so we build it as part of the
# main GPRat build with GPRAT_WITH_DISTRIBUTED enabled.

# Resolve the script's own directory so cmake paths are always correct
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPRAT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$GPRAT_ROOT"

cmake --preset release-linux -DGPRAT_WITH_DISTRIBUTED=ON -DHPX_DIR=$HPX_CMAKE
cmake --build --preset release-linux --target gprat_distributed -j

###################################################################################################
# Run code
###################################################################################################

echo "Running GPRat distributed benchmark (single locality)"

# Run from GPRAT_ROOT so the default data/data_1024/... paths resolve.
"$GPRAT_ROOT/build/release-linux/examples/gprat_distributed/gprat_distributed" "$@"

echo "Finished running GPRat distributed benchmark"
