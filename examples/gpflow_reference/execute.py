import argparse
import logging
import os
import time

import gpflow
import numpy as np
import tensorflow as tf
import tensorflow.python.util._pywrap_util_port as tf_util

from config import get_config
from gpflow_logger import setup_logging
from utils import (
    init_model,
    load_data,
    optimize_model,
    predict,
    predict_with_var,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--use-gpu",
    action="store_true",
    help="Flag to use GPU (assuming available)",
)
args = parser.parse_args()

if not args.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logger = logging.getLogger()
log_filename = "./gpflow_logs.log"


def gpflow_run(target, config, output_file, size_train, l, cores):
    """
    Run the GPflow regression pipeline.

    Args:
        target (str): String of target (cpu/gpu) for logs.
        config (dict): Configuration parameters for the pipeline.
        output_csv_obj (csv.writer): CSV writer object for writing output data.
        size_train (int): Size of the training dataset.
        l (int): Loop index.

    Returns:
        None
    """
    total_t = time.time()

    X_train, Y_train, X_test, Y_test = load_data(
        train_in_path=config["train_in_file"],
        train_out_path=config["train_out_file"],
        test_in_path=config["test_in_file"],
        test_out_path=config["test_out_file"],
        size_train=size_train,
        size_test=config["N_TEST"],
        n_regressors=config["N_REG"],
    )

    # logger.info("Finished loading the data.")

    init_t = time.time()
    model = init_model(
        X_train,
        Y_train,
        k_var=1.0,
        k_lscale=1.0,
        noise_var=0.1,
        params_summary=False,
    )
    init_t = time.time() - init_t

    opti_t = time.perf_counter()
    optimize_model(model, training_iter=config["OPT_ITER"])
    opti_t = time.perf_counter() - opti_t
    # logger.info("Finished optimization.")

    pred_var_t = time.time()
    f_pred, f_var = predict_with_var(model, X_test)
    pred_var_t = time.time() - pred_var_t
    # logger.info("Finished making predictions.")

    pred_t = time.time()
    f_pred = predict(model, X_test)
    pred_t = time.time() - pred_t
    # logger.info("Finished making predictions.")

    TOTAL_TIME = time.time() - total_t
    INIT_TIME = init_t
    OPT_TIME = opti_t
    PRED_UNCER_TIME = pred_var_t
    PREDICTION_TIME = pred_t
    # ERROR = calculate_error(Y_test, y_pred).detach().cpu().numpy()

    row_data = f"{target},{cores},{size_train},{config['N_TEST']},{config['N_REG']},{config['OPT_ITER']},{TOTAL_TIME},{INIT_TIME},{OPT_TIME},{PRED_UNCER_TIME},{PREDICTION_TIME},{l}\n"
    output_file.write(row_data)

    logger.info(
        f"{target},{cores},{size_train},{config['N_TEST']},{config['N_REG']},{config['OPT_ITER']},{TOTAL_TIME},{INIT_TIME},{OPT_TIME},{PRED_UNCER_TIME},{PREDICTION_TIME},{l}"
    )
    # logger.info("Completed iteration.")


def execute():
    """
    Execute the main process:
        - Set up logging.
        - Load configuration file.
        - Initialize output CSV file.
        - Write header to the output CSV file.
        - Set up TensorFlow and GPflow configurations based on the loaded config.
        - Iterate through different training sizes and for each training size
        loop for a specified amount of times while executing `gpflow_run` function.
    """
    setup_logging(log_filename, True, logger)

    # Check if TensorFlow is using GPU
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        logger.info(f"GPUs available: {physical_devices}")
        target = "gpu"
    else:
        logger.info("No GPUs found. Using CPU.")
        target = "cpu"

    # logger.info("\n")
    # logger.info("-" * 40)
    # logger.info("Load config file.")
    config = get_config()

    file_path = "./output.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a") as output_file:
        if not file_exists or os.stat(file_path).st_size == 0:
            # logger.info("Write output file header")
            logger.info(
                "Target,Cores,N_train,N_test,N_reg,Opt_iter,Total_time,Init_time,Opt_Time,Pred_Var_time,Pred_time,N_loop"
            )
            header = "Target,Cores,N_train,N_test,N_regressor,Opt_iter,Total_time,Init_time,Opt_time,Pred_Uncer_time,Predict_time,N_loop\n"
            output_file.write(header)

        if config["PRECISION"] == "float32":
            gpflow.config.set_default_float(np.float32)
        else:
            gpflow.config.set_default_float(np.float64)

        # runs tests on exponentially increasing number of cores and
        # data size, for multiple loops (each loop starts with *s)
        tf.config.threading.set_intra_op_parallelism_threads(config["N_CORES"])
        tf.config.threading.set_inter_op_parallelism_threads(config["N_CORES"])
        cores = config["START_CORES"]
        while cores <= config["N_CORES"]:
            data_size = config["START"]
            while data_size <= config["END"]:
                for l in range(config["LOOP"]):
                    logger.info("*" * 40)
                    logger.info(f"Core: {cores}, Train Size: {data_size}, Loop: {l}")
                    gpflow_run(target,config, output_file, data_size, l, cores)
                data_size = data_size * config["STEP"]
            cores = cores * 2
    logger.info("Completed the program.")

def is_mkl_enabled():
    return tf_util.IsMklEnabled()

if __name__ == "__main__":
    # check if Intel oneAPI MKL is enabled
    print("","-" * 18, "\n", "MKL enabled:", is_mkl_enabled(), "\n", "-" * 18)
    execute()
