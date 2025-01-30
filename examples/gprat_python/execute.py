import logging
import os
import sys
import time
from csv import writer
import argparse

#import lib64.gprat as gprat
#import lib.gprat as gprat
import gprat

from config import get_config
from hpx_logger import setup_logging

logger = logging.getLogger()
log_filename = "./hpx_logs.log"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--use_gpu",
    action="store_true",
    help="Flag to use GPU (assuming available)",
)
args = parser.parse_args()
if args.use_gpu:
    sys.argv.remove("--use_gpu")

use_gpu = gprat.compiled_with_cuda() and gprat.gpu_count() > 0 and args.use_gpu


def gprat_run(config, output_csv_obj, n_train, l, cores):

    n_tile_size = gprat.compute_train_tile_size(n_train, config["N_TILES"])
    m_tiles, m_tile_size = gprat.compute_test_tiles(
        config["N_TEST"], config["N_TILES"], n_tile_size
    )
    hpar = gprat.AdamParams(learning_rate=0.1, opt_iter=config["OPT_ITER"])
    train_in = gprat.GP_data(config["train_in_file"], n_train, config["N_REG"])
    train_out = gprat.GP_data(
        config["train_out_file"], n_train, config["N_REG"]
    )
    test_in = gprat.GP_data(
        config["test_in_file"], config["N_TEST"], config["N_REG"]
    )

    total_t = time.time()

    if not use_gpu:

        target = "cpu"

        ###### GP object ######
        init_t = time.time()
        gp_cpu = gprat.GP(
            train_in.data,
            train_out.data,
            config["N_TILES"],
            n_tile_size,
            kernel_params=[1.0, 1.0, 0.1],
            n_reg=config["N_REG"],
            trainable=[True, True, True],
        )
        init_t = time.time() - init_t

        # Init hpx runtime but do not start it yet
        gprat.start_hpx(sys.argv, cores)

        # Perform optmization
        opti_t = time.time()
        losses = gp_cpu.optimize(hpar)
        opti_t = time.time() - opti_t
        logger.info("Finished optimization.")

        # gprat.suspend_hpx()
        # gprat.resume_hpx()

        # Predict
        pred_uncer_t = time.time()
        pr, var = gp_cpu.predict_with_uncertainty(
            test_in.data, m_tiles, m_tile_size
        )
        pred_uncer_t = time.time() - pred_uncer_t
        logger.info("Finished predictions.")

        # Predict
        pred_full_t = time.time()
        pr__, var__ = gp_cpu.predict_with_full_cov(
            test_in.data, m_tiles, m_tile_size
        )
        pred_full_t = time.time() - pred_full_t
        logger.info("Finished predictions with full cov.")

        # Predict
        pred_t = time.time()
        pr_ = gp_cpu.predict(test_in.data, m_tiles, m_tile_size)
        pred_t = time.time() - pred_t
        logger.info("Finished predictions.")

    else:

        target = "gpu"

        ###### GP object ######
        init_t = time.time()
        gp_gpu = gprat.GP(
            train_in.data,
            train_out.data,
            config["N_TILES"],
            n_tile_size,
            kernel_params=[1.0, 1.0, 0.1],
            n_reg=config["N_REG"],
            trainable=[True, True, True],
            gpu_id=0,
            n_streams=2,
        )
        init_t = time.time() - init_t

        # Init hpx runtime but do not start it yet
        gprat.start_hpx(sys.argv, cores)

        # NOTE: optimization is not implemented for GPU
        opti_t = -1

        # gprat.suspend_hpx()
        # gprat.resume_hpx()

        # Predict
        pred_uncer_t = time.time()
        pr, var = gp_gpu.predict_with_uncertainty(
            test_in.data, m_tiles, m_tile_size
        )
        pred_uncer_t = time.time() - pred_uncer_t
        logger.info("Finished predictions.")

        # Predict
        pred_full_t = time.time()
        pr__, var__ = gp_gpu.predict_with_full_cov(
            test_in.data, m_tiles, m_tile_size
        )
        pred_full_t = time.time() - pred_full_t
        logger.info("Finished predictions with full cov.")

        # Predict
        pred_t = time.time()
        pr_ = gp_gpu.predict(test_in.data, m_tiles, m_tile_size)
        pred_t = time.time() - pred_t
        logger.info("Finished predictions.")

    # Stop HPX runtime
    gprat.stop_hpx()

    total_t = time.time() - total_t

    # config and measurements
    row_data = [
        target,
        cores,
        n_train,
        config["N_TEST"],
        config["N_TILES"],
        config["N_REG"],
        config["OPT_ITER"],
        init_t,
        -1, # NOTE: optimization is not implemented for GPU
        total_t,
        pred_uncer_t,
        pred_full_t,
        pred_t,
        l,
    ]
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
        header = [
            "Target",
            "Cores",
            "N_train",
            "N_test",
            "N_TILES",
            "N_regressor",
            "Opt_iter",
            "Init_time",
            "Optimization_Time",
            "Pred_Var_time",
            "Pred_Full_time",
            "Predict_time",
            "N_loop",
        ]
        output_csv_obj.writerow(header)

    # runs tests on exponentially increasing number of cores and
    # data size, for multiple loops (each loop starts with *s)
    cores = 2
    while cores <= config["N_CORES"]:
        data_size = config["START"]
        while data_size <= config["END"]:
            for l in range(config["LOOP"]):
                logger.info("*" * 40)
                logger.info(
                    f"Core: {cores}, Train Size: {data_size}, Loop: {l}"
                )
                gprat_run(config, output_csv_obj, data_size, l, cores)
            data_size = data_size * config["STEP"]
        cores = cores * 2


if __name__ == "__main__":
    execute()
