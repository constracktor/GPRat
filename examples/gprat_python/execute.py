import sys
import time
import os
import logging
from csv import writer
from config import get_config
from hpx_logger import setup_logging

import lib.gprat as gprat

logger = logging.getLogger()
log_filename = "./hpx_logs.log"

def gprat_run(config, output_csv_obj, n_train, l, cores):

    total_t = time.time()

    n_tile_size = gprat.compute_train_tile_size(n_train, config["N_TILES"])
    m_tiles, m_tile_size = gprat.compute_test_tiles(config["N_TEST"], config["N_TILES"], n_tile_size)
    hpar = gprat.AdamParams(learning_rate=0.1, opt_iter=config["OPT_ITER"])
    train_in = gprat.GP_data(config["train_in_file"], n_train, config["N_REG"])
    train_out = gprat.GP_data(config["train_out_file"], n_train, config["N_REG"])
    test_in = gprat.GP_data(config["test_in_file"], config["N_TEST"], config["N_REG"])

    ###### GP object ######
    init_t = time.time()
    gp = gprat.GP(train_in.data, train_out.data, config["N_TILES"], n_tile_size, kernel_params = [1.0,1.0,0.1], n_reg=config["N_REG"], trainable=[True, True, True])
    init_t = time.time() - init_t

    # Init hpx runtime but do not start it yet
    gprat.start_hpx(sys.argv, cores)

    # Perform optmization
    opti_t = time.time()
    losses = gp.optimize(hpar)
    opti_t = time.time() - opti_t
    logger.info("Finished optimization.")

    #gprat.suspend_hpx()
    #gprat.resume_hpx()

    # Predict
    pred_uncer_t = time.time()
    pr, var = gp.predict_with_uncertainty(test_in.data, m_tiles, m_tile_size)
    pred_uncer_t = time.time() - pred_uncer_t
    logger.info("Finished predictions.")

    # Predict
    pred_full_t = time.time()
    pr__, var__ = gp.predict_with_full_cov(test_in.data, m_tiles, m_tile_size)
    pred_full_t = time.time() - pred_full_t
    logger.info("Finished predictions with full cov.")

    # Predict
    pred_t = time.time()
    pr_ = gp.predict(test_in.data, m_tiles, m_tile_size)
    pred_t = time.time() - pred_t
    logger.info("Finished predictions.")

    # Stop HPX runtime
    gprat.stop_hpx()

    TOTAL_TIME = time.time() - total_t
    INIT_TIME = init_t
    OPTI_TIME = opti_t
    PRED_UNCER_TIME = pred_uncer_t
    PRED_FULL_TIME = pred_full_t
    PREDICTION_TIME = pred_t

    # config and measurements
    row_data = [cores, n_train, config["N_TEST"], config["N_TILES"], config["N_REG"], config["OPT_ITER"],
                TOTAL_TIME, INIT_TIME, OPTI_TIME, PRED_UNCER_TIME, PRED_FULL_TIME, PREDICTION_TIME, l]
    output_csv_obj.writerow(row_data)

    logger.info("Completed iteration.")

def execute():
    """
    Execute the main process:
    - Set up logging.
    - Load configuration file.
    - Initialize output CSV file.
    - Write header to the output CSV file.
    - Iterate through different training sizes and for each training size
    """

    # setup logging
    setup_logging(log_filename, True, logger)

    # load config
    logger.info("\n")
    logger.info("-" * 40)
    logger.info("Load config file.")
    config = get_config()

    # append log to ./output.csv
    file_exists = os.path.isfile("./output.csv")
    output_file = open("./output.csv", "a", newline="")
    output_csv_obj = writer(output_file)

    # write headers
    if not file_exists:
        logger.info("Write output file header")
        header = ["Cores", "N_train", "N_test", "N_TILES", "N_regressor",
                  "Opt_iter", "Total_time", "Init_time", "Optimization_Time",
                  "Pred_Var_time", "Pred_Full_time", "Predict_time", "N_loop"]
        output_csv_obj.writerow(header)

    # runs tests on exponentially increasing number of cores and
    # data size, for multiple loops (each loop starts with *s)
    cores = 2
    while cores <= config["N_CORES"]:
        data_size = config["START"]
        while data_size <= config["END"]:
            for l in range(config["LOOP"]):
                logger.info("*" * 40)
                logger.info(f"Core: {cores}, Train Size: {data_size}, Loop: {l}")
                gprat_run(config, output_csv_obj, data_size, l, cores)
            data_size = data_size * config["STEP"]
        cores = cores * 2

if __name__ == "__main__":
    execute()
