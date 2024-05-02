import sys
import time
import logging
from csv import writer
import tensorflow as tf
import gpflow
import numpy as np

from config import get_config
from gpflow_logger import setup_logging
from utils import load_data, train, optimize_model, predict, calculate_error

logger = logging.getLogger()
log_filename = "./gpflow_logs.log"


def gpflow_run(config, output_csv_obj, size_train, l):

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

    logger.info("Finished loading the data.")

    train_t = time.time()
    model = train(
        X_train, Y_train, k_var=1.0, k_lscale=1.0, noise_var=0.1, params_summary=False
    )
    train_t = time.time() - train_t
    
    opti_t = time.time()
    optimize_model(model)
    opti_t = time.time() - opti_t
    logger.info("Finished optimization.")

    pred_t = time.time()
    f_pred, f_var, y_pred, y_var = predict(model, X_test)
    pred_t = time.time() - pred_t
    logger.info("Finished model training.") 
    
    TOTAL_TIME = time.time() - total_t
    TRAIN_TIME = train_t
    OPTI_TIME = opti_t
    PREDICTION_TIME = pred_t
    ERROR = calculate_error(Y_test, f_pred)
    
    row_data = [config["N_CORES"], size_train, config["N_TEST"], config["N_REG"], 
                TOTAL_TIME, TRAIN_TIME, OPTI_TIME, PREDICTION_TIME, ERROR, l]
    output_csv_obj.writerow(row_data)
    
    logger.info("Completed iteration.")
    
    return f_pred, f_var, y_pred, y_var


def execute():
    """
    This function creates instance of active label class.
    Waits till the required amount of datasets collected in candidate pool.
    """
    setup_logging(log_filename, True, logger)
    logger.info("\n")
    logger.info("-" * 40)
    logger.info("Load config file.")
    config = get_config()
    output_file = open("./output.csv", "a", newline="")
    output_csv_obj = writer(output_file)
    
    logger.info("Write output file header")
    header = ["Cores", "N_train", "N_test", "N_regressor", "Total_time",
         "Train_time", "Optimization_Time", "Predict_time", "Error", "N_loop"]
    output_csv_obj.writerow(header)

    start = config["START"]
    end = config["END"]
    step = config["STEP"]
    tf.config.threading.set_inter_op_parallelism_threads(config["N_CORES"])
    if config["PRECISION"] == "float32":
        gpflow.config.set_default_float(np.float32)
    else:
        gpflow.config.set_default_float(np.float64)

    for i in range(start, end+step, step):
        for l in range(config["LOOP"]):
            logger.info("*" * 40)
            logger.info(f"Train Size: {i}, Loop: {l}")
            gpflow_run(config, output_csv_obj, i, l)

    output_file.close()
    
    logger.info("Completed the program.")


if __name__ == "__main__":
    execute()
