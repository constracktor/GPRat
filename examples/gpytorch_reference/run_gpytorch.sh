#!/bin/bash
# Input $1: Specify cpu/gpu/arm
if [[ "$1" == "gpu" ]]
then
    # Create & Activate python enviroment
    if [ ! -d "gpytorch_gpu_env" ]; then
        python -m venv gpytorch_gpu_env
    fi
    # Activate enviroment
    source gpytorch_gpu_env/bin/activate
    # Install requirements
    if ! python -c "import gpytorch"; then
        pip install gpytorch==1.13
    fi
    # Execute the python script
    python execute.py --use-gpu
elif [[ "$1" == "cpu" ]]
then
    # Create & Activate python enviroment
    if [ ! -d "gpytorch_cpu_env" ]; then
        python -m venv gpytorch_cpu_env
    fi
    # Activate enviroment
    source gpytorch_cpu_env/bin/activate
    # Install requirements
    if ! python -c "import gpytorch"; then
        pip install gpytorch==1.13
    fi
    # Execute the python script
    python execute.py
elif [[ "$1" == "arm" ]]
then
    spack load python@3.10
    # Create & Activate python enviroment
    if [ ! -d "gpytorch_arm_env" ]; then
        python -m venv gpytorch_arm_env
    fi
    # Activate enviroment
    source gpytorch_arm_env/bin/activate
    # Install requirements
    if ! python -c "import gpytorch"; then
        pip install gpytorch==1.13
    fi
    # Execute the python script
    python execute.py
else
    echo "Please specify input parameter: cpu/gpu/arm"
    exit 1
fi
