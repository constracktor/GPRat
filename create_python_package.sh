#!/bin/bash

if command -v spack &> /dev/null; then
    echo "Spack command found, checking for environments..."

    # Get current hostname
    HOSTNAME=$(hostname -s)

    if [[ "$HOSTNAME" == "ipvs-epyc1" ]]; then
	# Check if the gprat_cpu_gcc environment exists
    	if spack env list | grep -q "gprat_cpu_gcc"; then
	   echo "Found gprat_cpu_gcc environment, activating it."
	    module load gcc/14.2.0
	    export CXX=g++
	    export CC=gcc
	    spack env activate gprat_cpu_gcc
	fi

	# Create & Activate python environment
	if [ ! -d "pypi_env" ]; then
	    python -m venv pypi_env
	fi
	source pypi_env/bin/activate

	# Install requirements
	python -m ensurepip --upgrade
	pip install --upgrade pip
	pip install build
	pip install twine
	# Test: Manually install GPRat to pip
	pip uninstall -y gprat
	pip install .
	# Test: Check if import works
	python -c "import gprat"
	# Test: Run gprat_python example
	#cd examples/gprat_python
	#python execute.py
	#cd ../..

	# Build package
	rm -rf build dist
	# This command will generate a distribution archive
	# (.tar.gz) in the dist directory
	python -m build --sdist
	# Upload to Test PyPI
	twine upload --repository testpypi dist/*
    else
    	echo "Hostname is $HOSTNAME — no action taken. Packaging is currently only supported on ipvs-epyc1."
    fi
else
    echo "Spack command not found."
fi
