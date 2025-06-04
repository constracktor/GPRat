import time
import logging
import torch
import gpytorch
import os
import argparse

from config import get_config
from gpytorch_logger import setup_logging
from utils import load_data, ExactGPModel, train, predict, predict_with_var

logger = logging.getLogger()
log_filename = "./gpytorch_logs.log"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--use-gpu",
    action="store_true",
    help="Flag to use GPU (assuming available)",
)
args = parser.parse_args()


def gpytorch_run(config, output_file, size_train, l, cores):
    """
    Run the Gaussian process regression pipeline.

    Args:
        config (dict): Configuration parameters for the pipeline.
        output_csv_obj (csv.writer): CSV writer object for writing output data.
        size_train (int): Size of the training dataset.
        l (int): Loop index.
    """
    total_t = time.time()
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    X_train, Y_train, X_test, Y_test = load_data(
        train_in_path=config["train_in_file"],
        train_out_path=config["train_out_file"],
        test_in_path=config["test_in_file"],
        test_out_path=config["test_out_file"],
        size_train=size_train,
        size_test=config["N_TEST"],
        n_regressors=config["N_REG"],
    )
    if args.use_gpu and torch.cuda.is_available():
        X_train, Y_train, X_test, Y_test = X_train.to(device), Y_train.to(device), X_test.to(device), Y_test.to(device)

    # logger.info("Finished loading the data.")

    init_t = time.time()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = 0.1
    model = ExactGPModel(X_train, Y_train, likelihood)
    if args.use_gpu and torch.cuda.is_available():
        model = model.to(device)
        likelihood = likelihood.to(device)
    init_t = time.time() - init_t
    # logger.info("Initialized model.")

    train_t = time.time()
    train(model, likelihood, X_train, Y_train, training_iter=config['OPT_ITER'])
    train_t = time.time() - train_t
    # logger.info("Trained model.")

    pred_var_t = time.time()
    f_pred, f_var = predict_with_var(model, likelihood, X_test)
    pred_var_t = time.time() - pred_var_t
    # logger.info("Finished making predictions.")

    pred_t = time.time()
    f_pred = predict(model, likelihood, X_test)
    pred_t = time.time() - pred_t
    # logger.info("Finished making predictions.")

    TOTAL_TIME = time.time() - total_t
    INIT_TIME = init_t
    OPT_TIME = train_t
    PRED_UNCER_TIME = pred_var_t
    PREDICTION_TIME = pred_t
    # ERROR = calculate_error(Y_test, y_pred).detach().cpu().numpy()

    row_data = f"{cores},{size_train},{config['N_TEST']},{config['N_REG']},{config['OPT_ITER']},{TOTAL_TIME},{INIT_TIME},{OPT_TIME},{PREDICTION_TIME},{l}\n"
    output_file.write(row_data)

    logger.info(f"{cores},{size_train},{config['N_TEST']},{config['N_REG']},{config['OPT_ITER']},{TOTAL_TIME},{INIT_TIME},{OPT_TIME},{PRED_UNCER_TIME},{PREDICTION_TIME},{l}")
    #logger.info("Completed iteration.")


def execute():
    """
    This function performs following steps:
        - Set up logging.
        - Load configuration file.
        - Write header for the output CSV file.
        - Set up PyTorch configurations based on the loaded config.
        - Iterate through different training sizes and for each training size
        loop for a specified amount of times while executing `gpytorch_run` function.
    """
    setup_logging(log_filename, True, logger)
    logger.info("\n")
    logger.info("-" * 40)
    logger.info("Load config file.")
    config = get_config()

    file_path = "./output.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a") as output_file:
        if not file_exists or os.stat(file_path).st_size == 0:
            # logger.info("Write output file header")
            logger.info("Cores,N_train,N_test,N_reg,Opt_iter,Total_time,Init_time,Opt_Time,Pred_Var_time,Pred_time,N_loop")
            header = "Cores,N_train,N_test,N_regressor,Opt_iter,Total_time,Init_time,Opt_time,Pred_Uncer_time,Predict_time,N_loop\n"
            output_file.write(header)

        if config["PRECISION"] == "float32":
            torch.set_default_dtype(torch.float32)
        else:
            torch.set_default_dtype(torch.float64)

        # runs tests on exponentially increasing number of cores and
        # data size, for multiple loops (each loop starts with *s)
        torch.set_num_threads(config["N_CORES"])
        cores = config["START_CORES"]
        while cores <= config["N_CORES"]:
            data_size = config["START"]
            while data_size <= config["END"]:
                for l in range(config["LOOP"]):
                    logger.info("*" * 40)
                    logger.info(f"Core: {cores}, Train Size: {data_size}, Loop: {l}")
                    gpytorch_run(config, output_file, data_size, l, cores)
                data_size = data_size * config["STEP"]
            cores = cores * 2
    logger.info("Completed the program.")

def is_mkl_enabled():
    torch_config = torch.__config__.show()
    index = torch_config.index("USE_MKL")
    value = torch_config[index+8:index+10]
    return True if value == 'ON' else False

if __name__ == "__main__":
    # check if Intel oneAPI MKL is enabled
    print("","-" * 18, "\n", "MKL enabled:", is_mkl_enabled(), "\n", "-" * 18)
    execute()
