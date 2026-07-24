#!/usr/bin/env bash

# Script to install and setup spack

set -e

spack_repo_dir=$PWD
spack_destination="$HOME"

# Clone spack repository
cd $spack_destination
git clone -c feature.manyFiles=true --branch=v0.23.1 --depth=1 https://github.com/spack/spack.git

# Configure spack (add this to your .bashrc file)
source $spack_destination/spack/share/spack/setup-env.sh
# Find external compilers & software
spack compiler find
spack external find

# Add GPRat spack-repo to spack
spack repo add $spack_repo_dir
