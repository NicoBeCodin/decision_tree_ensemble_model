#!/usr/bin/env python3
import re
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

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
    1: {"max_depth": [5, 10, 20], "use_omp": [0, 1]},
    2: {"n_estimators": [10, 50, 100], "max_depth": [5, 10], "use_omp": [0, 1]},
    3: {"n_estimators": [50, 100], "max_depth": [5, 8], "learning_rate": [0.01, 0.1], "use_omp": [0, 1]},
    4: {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [5, 10], "num_leaves": [31, 64], "subsample": [0.8], "colsample_bytree": [0.8], "use_omp": [0, 1]},
    5: {"n_estimators": [100, 200], "learning_rate": [0.01], "max_depth": [8], "min_data_leaf": [20], "num_bins": [256], "use_dart": [0, 1], "dropout_rate": [0.1], "skip_drop_rate": [0.5], "binning_method": [0, 1], "use_omp": [0, 1]},
}

results = []

for code, name in models.items():
    grid = param_grids[code]
    keys, values = zip(*grid.items())
    for combo in product(*values):
        flags = dict(zip(keys, combo))
        # Build command
        cmd = [BIN, str(code)] + [f"--{k}={v}" for k, v in flags.items()]
        print("Running:", " ".join(cmd))
        # Automatically answer "no" (0) twice for saveModel and visualization prompts
        proc = subprocess.run(cmd, capture_output=True, text=True, input="0\n0\n")
        out = proc.stdout

        # Initialize metrics
        t_train = t_eval = mse = mae = None

        if code == 1:
            # DecisionTree parsing
            t_train = float(re.search(r"Training time: *([0-9\.]+)", out).group(1))
            t_eval = float(re.search(r"Evaluation time: *([0-9\.]+)", out).group(1))
            mse = float(re.search(r"Mean Squared Error.*: *([0-9\.eE+-]+)", out).group(1))
            mae = float(re.search(r"Mean Absolute Error.*: *([0-9\.eE+-]+)", out).group(1))
        elif code == 2:
            # Bagging parsing
            t_train = float(re.search(r"Training time: *([0-9\.]+)", out).group(1))
            t_eval = float(re.search(r"Evaluation time: *([0-9\.]+)", out).group(1))
            mse = float(re.search(r"Mean Squared Error.*: *([0-9\.eE+-]+)", out).group(1))
            mae = None
        elif code == 3:
            # Boosting parsing (note 'Mean Square Error' without 'd')
            t_train = float(re.search(r"Training time: *([0-9\.]+)", out).group(1))
            t_eval = float(re.search(r"Evaluation time: *([0-9\.]+)", out).group(1))
            mse = float(re.search(r"Mean Square Error.*: *([0-9\.eE+-]+)", out).group(1))
            mae = None
        elif code == 4:
            # LightGBM parsing
            t_train = float(re.search(r"\[LightGBM\] Training time: *([0-9\.]+) s", out).group(1))
            t_eval = float(re.search(r"\[LightGBM\] Prediction time: *([0-9\.]+) s", out).group(1))
            mse = float(re.search(r"\[LightGBM\] MSE = *([0-9\.eE+-]+)", out).group(1))
            # Parse MAE from the same line as MSE
            mae_match = re.search(r"MAE\s*=\s*([0-9\.eE+-]+)", out)
            mae = float(mae_match.group(1)) if mae_match else None
        elif code == 5:
            # Advanced GBDT parsing
            t_train = float(re.search(r"\[AdvGBDT\] Training time: *([0-9\.]+) s", out).group(1))
            t_eval = float(re.search(r"\[AdvGBDT\] Prediction time: *([0-9\.]+) s", out).group(1))
            mse = float(re.search(r"\[AdvGBDT\] MSE= *([0-9\.eE+-]+)", out).group(1))
            # Parse MAE even if on same line
            adv_match = re.search(r"MAE\s*=\s*([0-9\.eE+-]+)", out)
            mae = float(adv_match.group(1)) if adv_match else None

        entry = {
            "model": name,
            **flags,
            "train_time_s": t_train,
            "eval_time_s": t_eval,
            "mse": mse,
            "mae": mae,
        }
        results.append(entry)

# Create DataFrame and save to CSV
df = pd.DataFrame(results)
df.to_csv("benchmark_results_extended.csv", index=False)
print("Results written to benchmark_results_extended.csv")
print(df)

# Plot scalability and performance for each model
plt.figure()
for name, group in df.groupby("model"):
    # Choose primary hyperparameter for x-axis
    if "n_estimators" in group:
        x = group["n_estimators"]
    elif "max_depth" in group:
        x = group["max_depth"]
    else:
        x = range(len(group))
    plt.plot(x, group["train_time_s"], marker='o', label=name)
plt.xlabel("Hyperparameter")
plt.ylabel("Training time (s)")
plt.title("Scalability: Training Time")
plt.legend()

plt.figure()
for name, group in df.groupby("model"):
    if "n_estimators" in group:
        x = group["n_estimators"]
    elif "max_depth" in group:
        x = group["max_depth"]
    else:
        x = range(len(group))
    plt.plot(x, group["mse"], marker='o', label=name)
plt.xlabel("Hyperparameter")
plt.ylabel("MSE")
plt.title("Performance: MSE")
plt.legend()

plt.tight_layout()
plt.show()