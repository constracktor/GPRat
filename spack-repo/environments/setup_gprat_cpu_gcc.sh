#!/usr/bin/env bash
set -e
# search for gcc compiler and install if necessary
# Load GCC compiler
if [[ "$1" == "arm" ]]
then
    spack load gcc@14.2.0
    env_name=gprat_cpu_arm
elif [[ "$1" == "riscv" ]]
then
    echo "RISC-V not supported."
    exit 1
else
    module load gcc@14.2.0
    env_name=gprat_cpu_gcc
fi

# Script to setup CPU spack environment for GPRat using a recent gcc
source $HOME/spack/share/spack/setup-env.sh
spack compiler find

# Create environment and copy config file
env_name=gprat_cpu_gcc
spack env create $env_name
cp spack_cpu_gcc.yaml $HOME/spack/var/spack/environments/$env_name/spack.yaml
spack env activate $env_name

# Use external python
spack external find python

# setup environment
spack concretize -f
spack install
