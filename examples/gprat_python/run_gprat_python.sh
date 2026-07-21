#!/bin/bash
# Input $1: Specify how GPRat was compiled, options:	cpu/cuda/sycl
# Input $2: If GPRat was compiled with SYCL backend:	nvidia/amd/intel

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# Set --use-gpu flag
if [[ -z "$1" ]]; then
    echo "Input parameter is missing. Using default: Run computations on CPU"
	GPU=""
elif [[ "$1" == "cuda" || "$1" == "sycl" ]]; then
    GPU="--use-gpu"
	if [[ \
		"$HOSTNAME" != "simcl1n1" && \
		"$HOSTNAME" != "simcl1n2" && \
		"$HOSTNAME" != "simcl1n3" && \
		"$HOSTNAME" != "simcl1n4" && \
		"$HOSTNAME" != "pcsgs04" ]];
	then
		echo "GPU execution with this script is only supported on simcl1n1, simcl1n2, simcl1n3, simcl1n4, and pcsgs04." 1>&2
		exit 1
	fi
elif [[ "$1" != "cpu" ]]; then
    echo "Please specify input parameter: cpu/cuda/sycl"
    exit 1
fi

### SVEN0 AND SVEN1 ###############################################################################

# Setup LD_LIBRARY_PATH on sven0 and sven1
if [[ $(hostname -s) == "sven0" || $(hostname -s) == "sven1" ]]; then

	export LD_LIBRARY_PATH=$HOME/git_workspace/build-scripts/build/hpx/lib64:$LD_LIBRARY_PATH
	export LD_LIBRARY_PATH=$HOME/git_workspace/build-scripts/build/boost/lib:$LD_LIBRARY_PATH
	export LD_PRELOAD=$HOME/git_workspace/build-scripts/build/jemalloc/lib/libjemalloc.so.2

fi

### SIMCL1N1, SIMCL1N2, SIMCL1N3, SIMCL1N4 ########################################################

if [[ \
	"$HOSTNAME" == "simcl1n1" || \
	"$HOSTNAME" == "simcl1n2" || \
	"$HOSTNAME" == "simcl1n3" || \
	"$HOSTNAME" == "simcl1n4" ]]; 
then

	# Setup Spack
	spack_destination="/scratch-simcl1/grafml/Programs/spack-fp2-simcl1n1"
	source $spack_destination/spack/share/spack/setup-env.sh

	# GPU setup
	if [[ "$1" == "cuda" || "$1" == "sycl" ]]; then

		# simcl1n4 does not have a GPU
		if [[ "$HOSTNAME" == "simcl1n4" ]]; then
			echo "Machine $HOSTNAME does not have a GPU but you selected GPU execution." 1>&2
			exit 1
		fi

		# Check if the gprat_gpu_clang environment exists
		if spack env list | grep -q "gprat_gpu_clang"; then

			echo "Found gprat_gpu_clang environment, activating it."
			spack env activate gprat_gpu_clang
			LD_LIBRARY_PATH=$(spack location -i hpx)/lib:$LD_LIBRARY_PATH
			LD_LIBRARY_PATH=$(spack location -i openblas)/lib:$LD_LIBRARY_PATH
			LD_LIBRARY_PATH=$(spack location -i intel-oneapi-mkl)/lib:$LD_LIBRARY_PATH

		fi

		if [[ "$1" == "sycl" ]]; then

			# Add oneMath installation to LD_LIBRARY_PATH if gpu is specified
			if [[ "$2" == "nvidia" ]]; then

				ONEMATH_PATH="/scratch-simcl1/grafml/Programs/oneMath_nvidia/oneMath/install/lib/"
				LD_LIBRARY_PATH="$ONEMATH_PATH:$LD_LIBRARY_PATH"

			elif [[ "$2" == "amd" ]]; then

				ONEMATH_PATH="/scratch-simcl1/grafml/Programs/oneMath_amd/oneMath/install/lib/"
				LD_LIBRARY_PATH="$ONEMATH_PATH:$LD_LIBRARY_PATH"

			elif [[ "$2" == "intel" ]]; then

				echo "Machine $HOSTNAME does not have an Intel GPU." 1>&2
				exit 1
				
			elif [[ "$2" != "nvidia" ]]; then

				echo "Please specify gpu vendor: nvidia/amd/intel"
				exit 1

			fi

		fi

	# CPU setup
	elif [[ "$1" == "cpu" ]]; then

		if spack env list | grep -q "gprat_cpu_gcc"; then
			echo "Found gprat_cpu_gcc environment, activating it."
			spack env activate gprat_cpu_gcc
			module load gcc/14.2.0
			LD_LIBRARY_PATH=$(spack location -i hpx)/lib:$LD_LIBRARY_PATH
			LD_LIBRARY_PATH=$(spack location -i openblas)/lib:$LD_LIBRARY_PATH
			LD_LIBRARY_PATH=$(spack location -i intel-oneapi-mkl)/lib:$LD_LIBRARY_PATH
		fi

	fi

fi

### PCSGS04 #######################################################################################

# SITE-SPECIFIC: paths below are hardcoded for pcsgs04; adjust for other machines.
if [[ $(hostname) == "pcsgs04" ]]; then

	spack_destination="/scratch/grafml/gprat-spack/spack/"
	source $spack_destination/share/spack/setup-env.sh

	if [[ "$1" == "cuda" || "$1" == "sycl" ]]; then

		if spack env list | grep -q "gprat_gpu_clang"; then

			echo "Found gprat_gpu_clang environment, activating it."
			spack env activate gprat_gpu_clang
			LD_LIBRARY_PATH=$(spack location -i hpx)/lib:$LD_LIBRARY_PATH

			if [[ "$1" == "sycl" ]]; then

				if [[ "$2" != "intel" ]]; then

					echo "pcsgs04 only has an Intel GPU. Please specify gpu vendor: intel" 1>&2
					exit 1

				fi

				# icpx isn't in the gprat_gpu_clang env; source it from the system oneAPI install.
				if ! command -v icpx &>/dev/null && [[ -f /opt/intel/oneapi/compiler/2025.3/env/vars.sh ]]; then
					source /opt/intel/oneapi/compiler/2025.3/env/vars.sh
				fi

				# libumf is needed on LD_LIBRARY_PATH or GPU enumeration silently finds nothing.
				if [[ -f /opt/intel/oneapi/umf/latest/env/vars.sh ]]; then
					source /opt/intel/oneapi/umf/latest/env/vars.sh
				fi

				# The compiled extension needs system MKL 2025.3 and TBB 2022.3 on
				# LD_LIBRARY_PATH, not the older versions bundled in gprat_gpu_clang.
				if [[ -f /opt/intel/oneapi/mkl/2025.3/env/vars.sh ]]; then
					source /opt/intel/oneapi/mkl/2025.3/env/vars.sh
				fi
				LD_LIBRARY_PATH="/opt/intel/oneapi/tbb/2022.3/lib/intel64/gcc4.8:$LD_LIBRARY_PATH"

				# SITE-SPECIFIC: update to your local oneMath install prefix.
				ONEMATH_PATH="/scratch/grafml/oneMath_intel_v0.9/oneMath/install/lib"
				LD_LIBRARY_PATH="$ONEMATH_PATH:$LD_LIBRARY_PATH"

			fi

		else

			echo \
				"Cannot find Spack environment gprat_gpu_clang. Please run spack-repo/environments/setup_gprat_gpu_clang.sh" 1>&2
			exit 1

		fi

	fi

fi

### EXECUTION #####################################################################################

end_cores=$(python3 -c "import json; print(json.load(open('config.json'))['END_CORES'])")
core_count=$((end_cores * 2))

taskset -c 0-$core_count:2 python execute.py $GPU
