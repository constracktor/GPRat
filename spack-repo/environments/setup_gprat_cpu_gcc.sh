#!/usr/bin/env bash
# Script to setup CPU spack environment for GPRat using a recent gcc
set -e
# search for gcc compiler and install if necessary
# Load GCC compiler
if [[ "$1" == "arm" ]]
then
    spack load gcc@14.2.0
elif [[ "$1" == "riscv" ]]
then
    echo "TBD"
else
    module load gcc@14.2.0
fi

spack compiler find
# create environment and copy config file
export ENV_NAME=gprat_cpu_gcc
spack env create $ENV_NAME
cp spack_cpu_gcc.yaml $HOME/spack/var/spack/environments/$ENV_NAME/spack.yaml
spack env activate $ENV_NAME
# use external python
spack external find python
# setup environment
spack concretize -f
spack install
