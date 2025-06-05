#!/bin/bash

if [[ $(hostname -s) == "sven0" || $(hostname -s) == "sven1" ]]
then
	export LD_LIBRARY_PATH=$HOME/git_workspace/build-scripts/build/hpx/lib64:$LD_LIBRARY_PATH
	export LD_LIBRARY_PATH=$HOME/git_workspace/build-scripts/build/boost/lib:$LD_LIBRARY_PATH
	export LD_PRELOAD=$HOME/git_workspace/build-scripts/build/jemalloc/lib/libjemalloc.so.2
elif [[ $(hostname) == "simcl1n1" || $(hostname) == "simcl1n2" ]]; then
	# Check if the gprat_gpu_clang environment exists
	if spack env list | grep -q "gprat_gpu_clang"; then
	    echo "Found gprat_gpu_clang environment, activating it."
	    module load cuda/12.0.1
	    spack env activate gprat_gpu_clang
	    LD_LIBRARY_PATH=$(spack location -i hpx)/lib:$LD_LIBRARY_PATH
	    LD_LIBRARY_PATH=$(spack location -i openblas)/lib:$LD_LIBRARY_PATH
            LD_LIBRARY_PATH=$(spack location -i intel-oneapi-mkl)/lib:$LD_LIBRARY_PATH
	fi
fi

python3 execute.py "$@"
