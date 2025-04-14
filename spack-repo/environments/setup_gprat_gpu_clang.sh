#!/usr/bin/env bash
set -e

# Script to setup GPU spack environment for GPRat on simcl1n1-2

# Create environment and copy config file
source $HOME/spack/share/spack/setup-env.sh
spack env create gprat_gpu_clang
cp spack_gpu_clang.yaml $HOME/spack/var/spack/environments/gprat_gpu_clang/spack.yaml
spack env activate gprat_gpu_clang

# Find external compiler
module load clang/17.0.1
spack compiler find

# Find external packages
spack external find python
spack external find ninja
module load cuda/12.0.1
spack external find cuda

# Setup environment
spack concretize -f
spack install
