#!/bin/bash

# Create & Activate python environment
if [ ! -d "pypi_env" ]; then
    python -m venv pypi_env
fi
source pypi_env/bin/activate

# Install requirements
pip install --upgrade pip
pip install twine

# Load required libs
spack load gcc@14.2.0
export CXX=$(which g++)
spack load cmake
spack env activate gprat_cpu_gcc

# Test: Manually install GPRat to pip
pip uninstall -y gprat
pip install .
# Test: Check if import works
python -c "import gprat"
# Test: Run gprat_python example
cd examples/gprat_python
python execute.py
cd ../..

# Build package 
# This command will generate distribution archives 
# (.tar.gz and .whl) in the dist directory
#python3 setup.py sdist bdist_wheel
#rm -rf build
# Upload to PyPI
#python -m twine upload dist/*
