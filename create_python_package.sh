#!/bin/bash

# Create & Activate python environment
if [ ! -d "pypi_env" ]; then
    python3 -m venv pypi_env
fi
source pypi_env/bin/activate

# Install requirements
pip3 install --upgrade pip
pip3 install setuptools 
pip3 install twine
pip3 install --upgrade wheel
pip3 install --upgrade packaging

# Load required libs
spack load gcc@14.2.0
spack load cmake
spack env activate gprat_cpu_gcc

# Build package 
# This command will generate distribution archives 
# (.tar.gz and .whl) in the dist directory
python3 setup.py sdist bdist_wheel
rm -rf build

# Test: Manually install GPRat to pip
pip3 uninstall -y gprat
pip3 install .
# Test: Check if import works
python3 -c "import gprat"
# Test: Run gprat_python example
cd examples/gprat_python
python3 execute.py
cd ../..

# Upload to PyPI
python3 -m twine upload dist/*
