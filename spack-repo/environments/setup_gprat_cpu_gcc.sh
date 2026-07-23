#!/usr/bin/env bash
set -e
# Script to setup CPU spack environment for GPRat using a recent gcc

# Load GCC compiler
if [[ "$1" == "arm" ]]
then
    spack load gcc@14.2.0
    env_name=gprat_cpu_arm
    env_yaml=spack_cpu_gcc_arm.yaml
elif [[ "$1" == "riscv" ]]
then
    echo "RISC-V not supported."
    exit 1
else
    module load gcc@14.2.0
    env_name=gprat_cpu_gcc
    env_yaml=spack_cpu_gcc.yaml
fi

# Find GCC compiler with spack
spack_destination="$HOME"
source $spack_destination/spack/share/spack/setup-env.sh
spack compiler find

# Create environment and copy config file
spack env create $env_name
cp $env_yaml $spack_destination/spack/var/spack/environments/$env_name/spack.yaml
spack env activate $env_name

# Use external python
spack external find python

# setup environment
spack concretize -f
spack install
