"""
Run gprat_predict.py / gpflow_predict.py / gpytorch_predict.py, each under its
own interpreter (they depend on mutually incompatible environments), and
compare the predicted mean and variance across all three.

All three are configured with the same kernel hyperparameters and no
optimization, so their predictions should agree to floating-point precision.
"""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

BACKENDS = {
    "gprat": [sys.executable, str(SCRIPT_DIR / "gprat_predict.py")],
    "gpflow": [
        str(SCRIPT_DIR / "../gpflow_reference/gpflow_cpu_env/bin/python"),
        str(SCRIPT_DIR / "gpflow_predict.py"),
    ],
    "gpytorch": [
        str(SCRIPT_DIR / "../gpytorch_reference/gpytorch_cpu_env/bin/python"),
        str(SCRIPT_DIR / "gpytorch_predict.py"),
    ],
}

MARKER = "RESULT_JSON:"


def run_backend(name, cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_DIR)
    for line in result.stdout.splitlines():
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER) :])
    raise RuntimeError(
        f"{name}: no {MARKER} line found in output.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def max_diff(a, b):
    assert len(a) == len(b), f"length mismatch: {len(a)} vs {len(b)}"
    abs_diff = max(abs(x - y) for x, y in zip(a, b))
    rel_diff = max(abs(x - y) / max(abs(x), 1e-12) for x, y in zip(a, b))
    return abs_diff, rel_diff


def main():
    predictions = {name: run_backend(name, cmd) for name, cmd in BACKENDS.items()}

    print(f"Test size: {len(predictions['gprat']['mean'])}\n")

    rtol = 1e-6
    ok = True
    for field in ("mean", "var"):
        print(f"{field}:")
        for a, b in [("gprat", "gpflow"), ("gprat", "gpytorch"), ("gpflow", "gpytorch")]:
            abs_diff, rel_diff = max_diff(predictions[a][field], predictions[b][field])
            print(f"  {a:9s} vs {b:9s}: max_abs_diff={abs_diff:.3e}  max_rel_diff={rel_diff:.3e}")
            ok &= rel_diff < rtol

    verdict = "PASS: predictions agree" if ok else "FAIL: predictions do not agree"
    print(f"\n{verdict} within rtol={rtol:.0e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
