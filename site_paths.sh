#!/usr/bin/env bash
# Site-specific installation roots shared by compile_gprat.sh, create_python_package.sh,
# examples/gprat_cpp/run_gprat_cpp.sh, and examples/gprat_python/run_gprat_python.sh.
# Update the paths below if an installation moves; every script that sources this
# file picks up the change.

# Spack roots.
# SIMCL1_SPACK_ROOT is the parent directory of the cloned "spack" checkout
# (scripts source $SIMCL1_SPACK_ROOT/spack/share/spack/setup-env.sh).
SIMCL1_SPACK_ROOT="/scratch-simcl1/grafml/Programs/spack-fp2-simcl1n1"
# PCSGS04_SPACK_ROOT points directly at the "spack" checkout itself
# (scripts source $PCSGS04_SPACK_ROOT/share/spack/setup-env.sh).
PCSGS04_SPACK_ROOT="/scratch/grafml/gprat-spack/spack"

# oneMath install roots (each contains lib/ and lib/cmake/oneMath).
ONEMATH_NVIDIA_ROOT="/scratch-simcl1/grafml/Programs/oneMath_nvidia/oneMath/install"
ONEMATH_AMD_ROOT="/scratch-simcl1/grafml/Programs/oneMath_amd/oneMath/install"
ONEMATH_INTEL_ROOT="/scratch/grafml/oneMath_intel_v0.9/oneMath/install"
