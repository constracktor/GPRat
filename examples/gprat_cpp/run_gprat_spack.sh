#!/bin/bash
# $1 cpu/gpu

################################################################################
set -e  # Exit immediately if a command exits with a non-zero status.
#set -x  # Print each command before executing it.

################################################################################
# Configurations
################################################################################

if command -v spack &> /dev/null; then
    echo "Spack command found."
    # Get current hostname
    HOSTNAME=$(hostname -s)

    if [[ "$HOSTNAME" == "ipvs-epyc1" ]]; then
	spack load gprat
	module load gcc/14.2.0
	export CXX=g++
	export CC=gcc
    else
    	echo "Hostname is $HOSTNAME — no action taken."
    fi
else
    echo "Spack command not found."
    exit 1
fi

# Configure APEX
export APEX_SCREEN_OUTPUT=0
export APEX_DISABLE=1

################################################################################
# Compile code
################################################################################
rm -rf build && mkdir build && cd build

# Configure the project
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
make -j

################################################################################
# Run code
################################################################################

./gprat_cpp $1
