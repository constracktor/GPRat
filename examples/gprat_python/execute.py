'''
GPRat with Python bindings reference implementation
'''

# IMPORTS ###############################################################################


import time
import logging
import os
import sys
import subprocess
import argparse
from config import get_config
from hpx_logger import setup_logging
import gc

# GPRat
import lib.gprat as gprat
# import lib64.gprat as gprat   # depending on system
# import gprat                  # if installed with pip

# GLOBAL DEFINITIONS ####################################################################
logger = logging.getLogger()
log_filename = "./hpx_logs.log"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--use-gpu",
    action="store_true",
    help="Flag to use GPU (assuming available)",
)
parser.add_argument(
    "--optimize",
    action="store_true",
    help="Flag to use GPU (assuming available)",
)

args = parser.parse_args()

if args.use_gpu:
    sys.argv.remove("--use-gpu")
if args.optimize:
    sys.argv.remove("--optimize")

if args.optimize and args.use_gpu:
    print("Optimization is not implemented for GPU. Please run without --optimize flag.")
    sys.exit(1)

compiled_with_gpu = gprat.compiled_with_cuda() or gprat.compiled_with_sycl()

if not compiled_with_gpu and args.use_gpu:
    print("GPRat is not compiled with GPU support but GPU selected for execution.")
    sys.exit(1)

use_gpu = compiled_with_gpu and gprat.gpu_count() > 0 and args.use_gpu

# GPRAT_RUN #############################################################################
def gprat_run(
        config, 
        output_file, 
        size_train, 
        size_test,
        loop_index, 
        n_cores, 
        n_tiles,
        is_warmup=False
    ):

    print(f"Running GPRat with train size {size_train}, test size {size_test}, cores {n_cores}, tiles {n_tiles}, loop index {loop_index}.")

    target = "deadbeef"
    opti_t = -1
    gp = None

    total_t = time.perf_counter()

    # Load data
    load_t = time.perf_counter()
    train_in = gprat.GP_data(config["TRAIN_IN_FILE"], size_train, config["N_REG"])
    # n_reg=1 -> no offset: unlike the input, the training output has no
    # lookahead padding requirement.
    train_out = gprat.GP_data(config["TRAIN_OUT_FILE"], size_train, 1)
    test_in = gprat.GP_data(config["TEST_IN_FILE"], size_test, config["N_REG"])
    load_t = time.perf_counter() - load_t

    # CPU setup
    if not use_gpu:
        
        target_query_time = time.perf_counter()

        cmd = "lscpu | grep 'Model name' | cut -f 2 -d ':' | awk '{$1=$1}1'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        target = result.stdout.strip() + "@HPX"
        print("Running on: " + target)

        target_query_time = time.perf_counter() - target_query_time

        # Initialization
        init_t = time.perf_counter()

        ## Tiles
        n_tile_size = gprat.compute_train_tile_size(size_train, n_tiles)
        m_tiles, m_tile_size = gprat.compute_test_tiles(
            size_test, n_tiles, n_tile_size
        )

        ## Hyperparameters
        hpar = gprat.AdamParams(learning_rate=0.1, opt_iter=config["OPT_ITER"])

        ## GP object
        gp = gprat.GP(
            train_in.data,
            train_out.data,
            n_tiles,
            n_tile_size,
            kernel_params=[1.0, 1.0, 0.1],
            n_reg=config["N_REG"],
            trainable=[True, True, True]
        )

        init_t = time.perf_counter() - init_t

    # GPU setup
    elif use_gpu:

        target_query_time = time.perf_counter()

        cmd = "lshw -C display | grep \"product:\" | head -n1 | cut -d: -f2"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        target = result.stdout.strip()

        if gprat.compiled_with_cuda():
            target += "@CUDA"
        elif gprat.compiled_with_sycl():
            target += "@SYCL"

        print("Running on: " + target)

        target_query_time = time.perf_counter() - target_query_time

        # Initialization

        init_t = time.perf_counter()

        ## Tiles
        n_tile_size = gprat.compute_train_tile_size(size_train, n_tiles)
        m_tiles, m_tile_size = gprat.compute_test_tiles(
            size_test, n_tiles, n_tile_size
        )

        ## Hyperparameters
        hpar = gprat.AdamParams(learning_rate=0.1, opt_iter=config["OPT_ITER"])

        # GP object
        gp = gprat.GP(
            train_in.data,
            train_out.data,
            n_tiles,
            n_tile_size,
            kernel_params=[1.0, 1.0, 0.1],
            n_reg=config["N_REG"],
            trainable=[True, True, True],
            gpu_id=0,
            n_units=1
        )

        init_t = time.perf_counter() - init_t

    # Execution

    gprat.start_hpx(sys.argv, n_cores)

    # Perform optmization

    if args.optimize:
        opti_t = time.perf_counter()
        losses = gp.optimize(hpar)
        opti_t = time.perf_counter() - opti_t

    # Predict without uncertainty
    pred_t = time.perf_counter()
    pr_ = gp.predict(test_in.data, m_tiles, m_tile_size)
    pred_t = time.perf_counter() - pred_t

    # Predict with uncertainty
    pred_uncer_t = time.perf_counter()
    pr, var = gp.predict_with_uncertainty(
        test_in.data, m_tiles, m_tile_size
    )
    pred_uncer_t = time.perf_counter() - pred_uncer_t

    # Predict with full covariance
    pred_full_t = time.perf_counter()
    pr__, var__ = gp.predict_with_full_cov(
        test_in.data, m_tiles, m_tile_size
    )
    pred_full_t = time.perf_counter() - pred_full_t

    # Stop HPX runtime
    gprat.stop_hpx()

    total_t = time.perf_counter() - total_t - target_query_time

    # config and measurements

    row_data = \
    f"{target},{n_cores},{n_tiles},{size_train},{size_test},{config['N_REG']}," \
    f"{config['OPT_ITER']},{total_t},{load_t},{init_t},{opti_t},{pred_t}," \
    f"{pred_uncer_t},{pred_full_t},{loop_index}\n"

    if not is_warmup:
        output_file.write(row_data)
        output_file.flush()
        logger.info(row_data)

# EXECUTE ###############################################################################
def execute():
    """
    Execute the main process:
    - Set up logging.
    - Load configuration file.
    - Initialize output CSV file.
    - Write header to the output CSV file.
    - Iterate through different training sizes and for each training size
    """

    # load config
    logger.info("\n")
    logger.info("-" * 40)
    logger.info("Load config file.")
    config = get_config()

    file_path = "./output.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a") as output_file:
    
        if not file_exists or os.stat(file_path).st_size == 0:
            logger.info(
                "Target,Cores,N_tiles,N_train,N_test,N_regressor,Opt_iter,Total_time,Load_time,"\
                "Init_time,Opt_Time,Predict_time,Pred_Uncer_time,Pred_Full_time,N_loop"
            )
            header = \
                "Target,Cores,N_tiles,N_train,N_test,N_regressor,Opt_iter,Total_time,Load_time,"\
                "Init_time,Opt_Time,Predict_time,Pred_Uncer_time,Pred_Full_time,N_loop\n"
            output_file.write(header)

        # Perform warmup run
        gprat_run(config, output_file, config['TRAIN_SIZE_END'], config['TRAIN_SIZE_END'], 0, 
                  config['END_CORES'], config['N_TILES_END'], True)

        test_scale_factor = config['STEP'] if config['SCALE_TEST_WITH_TRAIN'] else 1

        cores = config["START_CORES"]

        while cores <= config['END_CORES']:

            n_tiles = config['N_TILES_START']

            while n_tiles <= config['N_TILES_END']:

                # Set train and test sizes
                data_size = config['TRAIN_SIZE_START'] if config['TRAIN_SIZE_START'] >= n_tiles \
                    else n_tiles
                test_size = config['TEST_SIZE'] if not config['SCALE_TEST_WITH_TRAIN'] \
                    else data_size

                # Loop over training data sizes
                while data_size <= config['TRAIN_SIZE_END']:

                    # Loop over different test iterations
                    for loop_index in range(config["LOOP"]):
                        logger.info("*" * 40)
                        logger.info(f"Cores: {cores}, Train Size: {data_size}, Loop: {loop_index}")
                        gc.collect()
                        gprat_run(config, output_file, data_size, test_size, loop_index, cores, n_tiles)

                    # Update sizes
                    data_size = data_size * config['STEP']
                    test_size = test_size * test_scale_factor

                n_tiles *= config['STEP_TILES']

            cores *= 2

    logger.info("Completed the program.")


# MAIN ##################################################################################
if __name__ == "__main__":
    setup_logging(log_filename, True, logger)
    execute()
