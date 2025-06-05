#!/bin/bash
# Input $1: Specify cpu/gpu/arm
if [[ "$1" == "gpu" ]]
then
    module load cuda/12.0.1
    # Create & Activate python environment
    if [ ! -d "gpflow_gpu_env" ]; then
        python -m venv gpflow_gpu_env
    fi
    source gpflow_gpu_env/bin/activate
    # Install gpflow if not already installed
    if ! python -c "import gpflow"; then
        pip install --no-cache-dir -r requirements_gpu.txt
    fi
    # Run on GPU
    export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME
    python execute.py --use-gpu
elif [[ "$1" == "cpu" ]]
then
    module load python/3.10.16
    # Create & Activate python environment
    if [ ! -d "gpflow_cpu_env" ]; then
        python -m venv gpflow_cpu_env
    fi
    source gpflow_cpu_env/bin/activate
    # Install gpflow if not already installed
    if ! python -c "import gpflow"; then
        pip install --no-cache-dir -r requirements_cpu.txt
        # manually install GPflow
        git clone https://github.com/GPflow/GPflow.git
        cd GPflow
        git checkout v2.10.0
        git apply ../gpflow_mkl.patch
        pip install -e .
        cd ..
    fi
    # Run on CPU
    python execute.py
elif [[ "$1" == "arm" ]]
then
    spack load python@3.10
    # Create & Activate python environment
    if [ ! -d "gpflow_arm_env" ]; then
        python -m venv gpflow_arm_env
    fi
    source gpflow_arm_env/bin/activate
    # Install gpflow if not already installed
    if ! python -c "import gpflow"; then
        pip install --no-cache-dir -r requirements_gpu.txt
    fi
    # Run on ARM
    python execute.py
else
    echo "Please specify input parameter: cpu/gpu/arm"
    exit 1
fi
