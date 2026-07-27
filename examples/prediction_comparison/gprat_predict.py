"""
Predict with GPRat and print the result as one JSON line prefixed with
RESULT_JSON:, so compare.py can pick it out of the surrounding HPX/APEX
startup output.
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "lib"))
import gprat  # noqa: E402

DATA_DIR = SCRIPT_DIR / "../../data/data_1024"
TRAIN_SIZE = 512
TEST_SIZE = 64
N_REG = 8
N_TILES = 4
KERNEL_PARAMS = [1.0, 1.0, 0.1]  # lengthscale, vertical_lengthscale, noise_variance

train_in = gprat.GP_data(str(DATA_DIR / "training_input.txt"), TRAIN_SIZE, N_REG)
# n_reg=1 -> no offset: unlike the input, the training output has no
# lookahead padding requirement.
train_out = gprat.GP_data(str(DATA_DIR / "training_output.txt"), TRAIN_SIZE, 1)
test_in = gprat.GP_data(str(DATA_DIR / "test_input.txt"), TEST_SIZE, N_REG)

n_tile_size = gprat.compute_train_tile_size(TRAIN_SIZE, N_TILES)
m_tiles, m_tile_size = gprat.compute_test_tiles(TEST_SIZE, N_TILES, n_tile_size)

gp = gprat.GP(
    train_in.data,
    train_out.data,
    N_TILES,
    n_tile_size,
    kernel_params=KERNEL_PARAMS,
    n_reg=N_REG,
    # Predict with the given kernel_params as-is, no optimization: this keeps
    # the hyperparameters identical across all three backends so predictions
    # are directly comparable.
    trainable=[False, False, False],
)

gprat.start_hpx(sys.argv, 2)
mean, var = gp.predict_with_uncertainty(test_in.data, m_tiles, m_tile_size)
gprat.stop_hpx()

print("RESULT_JSON:" + json.dumps({"mean": list(mean), "var": list(var)}))
