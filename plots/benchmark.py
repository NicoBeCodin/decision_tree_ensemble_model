#!/usr/bin/env python3
import re
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import numpy as np

REPEATS = 3  # Number of times to repeat each configuration for error bars

# Path to your executable; adjust relative to this script’s directory
import os
BIN = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build/MainEnsemble"))

# Model codes and names
models = {
    1: "DecisionTree",
    2: "Bagging",
    3: "Boosting",
    4: "LightGBM",
    5: "AdvancedGBDT",
}

# Hyperparameter grids for each model
param_grids = {
    1: {
        "max_depth": [10],
        "use_omp": [0, 1],
        "num_threads": [1, 2, 4, 8],
    },
    2: {
        "n_estimators": [200],
        "max_depth": [10],
        "use_omp": [0, 1],
        "num_threads": [1, 2, 4, 8],
    },
    3: {
        "n_estimators": [200],
        "max_depth": [10],
        "learning_rate": [0.1],
        "use_omp": [0, 1],
        "num_threads": [1, 2, 4, 8],
    },
    4: {
        "n_estimators": [200],
        "learning_rate": [0.1],
        "max_depth": [10],
        "num_leaves": [31],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "use_omp": [0, 1],
        "num_threads": [1, 2, 4, 8],
    },
    5: {
        "n_estimators": [200],
        "learning_rate": [0.1],
        "max_depth": [10],
        "min_data_leaf": [10],
        "num_bins": [128],
        "use_dart": [0, 1],
        "dropout_rate": [0.1],
        "skip_drop_rate": [0.5],
        "binning_method": [0, 1],
        "use_omp": [0, 1],
        "num_threads": [1, 2, 4, 8],
    },
}

results = []

for code, name in models.items():
    grid = param_grids[code]
    keys, values = zip(*grid.items())
    for combo in product(*values):
        flags = dict(zip(keys, combo))
        # Build command
        cmd = [BIN, str(code)] + [f"--{k}={v}" for k, v in flags.items()]
        env = os.environ.copy()
        if "num_threads" in flags:
            env["OMP_NUM_THREADS"] = str(flags["num_threads"])
        print("Running:", " ".join(cmd))

        # repeat the measurement REPEATS times
        t_trains, t_evals, mses, maes = [], [], [], []
        for _ in range(REPEATS):
            proc = subprocess.run(cmd, capture_output=True, text=True, input="0\n0\n", env=env)
            out = proc.stdout
            # parse metrics as before, naming parsed values t_t, t_e, m, a
            if code == 1:
                t_t = float(re.search(r"Training time: *([0-9\.]+)", out).group(1))
                t_e = float(re.search(r"Evaluation time: *([0-9\.]+)", out).group(1))
                m = float(re.search(r"Mean Squared Error.*: *([0-9\.eE+-]+)", out).group(1))
                a = float(re.search(r"Mean Absolute Error.*: *([0-9\.eE+-]+)", out).group(1))
            elif code == 2:
                t_t = float(re.search(r"Training time: *([0-9\.]+)", out).group(1))
                t_e = float(re.search(r"Evaluation time: *([0-9\.]+)", out).group(1))
                m = float(re.search(r"Mean Squared Error.*: *([0-9\.eE+-]+)", out).group(1))
                a = None
            elif code == 3:
                t_t = float(re.search(r"Training time: *([0-9\.]+)", out).group(1))
                t_e = float(re.search(r"Evaluation time: *([0-9\.]+)", out).group(1))
                m = float(re.search(r"Mean Square Error.*: *([0-9\.eE+-]+)", out).group(1))
                a = None
            elif code == 4:
                t_t = float(re.search(r"\[LightGBM\] Training time: *([0-9\.]+) s", out).group(1))
                t_e = float(re.search(r"\[LightGBM\] Prediction time: *([0-9\.]+) s", out).group(1))
                m = float(re.search(r"\[LightGBM\] MSE = *([0-9\.eE+-]+)", out).group(1))
                mae_match = re.search(r"MAE\s*=\s*([0-9\.eE+-]+)", out)
                a = float(mae_match.group(1)) if mae_match else None
            elif code == 5:
                t_t = float(re.search(r"\[AdvGBDT\] Training time: *([0-9\.]+) s", out).group(1))
                t_e = float(re.search(r"\[AdvGBDT\] Prediction time: *([0-9\.]+) s", out).group(1))
                m = float(re.search(r"\[AdvGBDT\] MSE= *([0-9\.eE+-]+)", out).group(1))
                adv_match = re.search(r"MAE\s*=\s*([0-9\.eE+-]+)", out)
                a = float(adv_match.group(1)) if adv_match else None
            t_trains.append(t_t)
            t_evals.append(t_e)
            mses.append(m)
            if a is not None:
                maes.append(a)

        # compute mean and std
        t_train_mean = float(np.mean(t_trains))
        t_train_std  = float(np.std(t_trains))
        t_eval_mean  = float(np.mean(t_evals))
        t_eval_std   = float(np.std(t_evals))
        mse_mean     = float(np.mean(mses))
        mse_std      = float(np.std(mses))
        mae_mean     = float(np.nanmean(maes)) if maes else None
        mae_std      = float(np.nanstd(maes)) if maes else None

        entry = {
            "model": name,
            **flags,
            "train_time_mean": t_train_mean,
            "train_time_std" : t_train_std,
            "eval_time_mean": t_eval_mean,
            "eval_time_std" : t_eval_std,
            "mse_mean"      : mse_mean,
            "mse_std"       : mse_std,
            "mae_mean"      : mae_mean,
            "mae_std"       : mae_std,
        }
        results.append(entry)

# Create DataFrame and save to CSV
df = pd.DataFrame(results)
df.to_csv("benchmark_results_extended.csv", index=False)
print("Results written to benchmark_results_extended.csv")
print(df)
