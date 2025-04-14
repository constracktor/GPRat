#!/usr/bin/env bash

# Script to install and setup spack

set -e

spack_repo_dir=$PWD

# Clone spack repository into $HOME/spack
cd
git clone -c feature.manyFiles=true --branch=v0.23.1 --depth=1 https://github.com/spack/spack.git

# Configure spack (add this to your .bashrc file)
source $HOME/spack/share/spack/setup-env.sh
# Find external compilers & software
spack compiler find
spack external find

# Add GPRat spack-repo to spack
spack repo add $spack_repo_dir
