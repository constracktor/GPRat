#!/usr/bin/env bash
set -e

# Script to setup CPU spack environment for GPRat using a recent gcc

# Search for gcc compiler and install if necessary
module load gcc/14.2.0
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
