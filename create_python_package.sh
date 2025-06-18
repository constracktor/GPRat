#!/bin/bash

# Create & Activate python environment
if [ ! -d "pypi_env" ]; then
    python -m venv pypi_env
fi
source pypi_env/bin/activate

# Install requirements
pip install --upgrade pip
pip install build
pip install twine

# Load required libs
module load gcc/14.2.0
export CXX=$(which g++)
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
rm -rf build dist
# This command will generate a distribution archive
# (.tar.gz) in the dist directory
python -m build --sdist
# Upload to Test PyPI
twine upload --repository testpypi dist/*
