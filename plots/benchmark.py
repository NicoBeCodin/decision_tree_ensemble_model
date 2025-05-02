#!/usr/bin/env python3
import os, re, subprocess, pandas as pd, numpy as np
from itertools import product

REPEATS   = 3
ROOT_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR  = os.path.join(ROOT_DIR, "datasets", "processed")
BIN       = os.path.join(ROOT_DIR, "build", "MainEnsemble")

# ❶  *** modèles conservés ***
models = {
    1: "DecisionTree",
    2: "Bagging",
    3: "Boosting",
    4: "LightGBM",
    # 5 retiré (seg-fault)
}

# ❷  mapping jeux de données (weak scaling)
data_map = {
    1: "cleaned_data_5k_ga_adaptative.csv",
    2: "cleaned_data_10k_ga_adaptative.csv",
    3: "cleaned_data_15k_ga_adaptative.csv",
    4: "cleaned_data_20k_ga_adaptative.csv",
    6: "cleaned_data_30k_ga_adaptative.csv",
}

strong_data_file = "cleaned_data_30k_ga_adaptative.csv"
thread_counts    = [1, 2, 4, 6]

param_template = {
    1: {"max_depth": [10]},
    2: {"n_estimators": [200], "max_depth": [10]},
    3: {"n_estimators": [200], "max_depth": [10], "learning_rate": [0.1]},
    4: {"n_estimators": [200], "learning_rate": [0.1], "max_depth": [10],
        "num_leaves": [31], "subsample": [0.8], "colsample_bytree": [0.8]},
    # 5 supprimé
}

# ---------- bench ----------
results = []

for scaling in ["strong", "weak"]:
    for code, name in models.items():
        keys, vals = zip(*param_template[code].items())
        for combo in product(*vals):
            base = dict(zip(keys, combo))
            for nt in thread_counts:

                cfg = base | {"num_threads": nt, "use_omp": 1}
                data_file = strong_data_file if scaling == "strong" else data_map.get(nt, strong_data_file)
                abs_data  = os.path.join(DATA_DIR, data_file)

                cmd = [BIN, str(code), f"--data={abs_data}"] + [f"--{k}={v}" for k, v in cfg.items()]
                env = os.environ.copy(); env["OMP_NUM_THREADS"] = str(nt)

                print(f"[{scaling.upper()}] {name:12s}  nt={nt}  →  {' '.join(cmd)}")

                t_tr, t_ev, mse, mae = [], [], [], []
                for _ in range(REPEATS):
                    run = subprocess.run(cmd, capture_output=True, text=True, input="0\n0\n", env=env)
                    out = run.stdout

                    try:
                        if code == 1:  # DecisionTree
                            t_tr.append(float(re.search(r"Training time:\s*([0-9.eE+-]+)", out).group(1)))
                            t_ev.append(float(re.search(r"Evaluation time:\s*([0-9.eE+-]+)", out).group(1)))
                            mse.append(float(re.search(r"MSE\):\s*([0-9.eE+-]+)",  out).group(1)))
                            mae.append(float(re.search(r"MAE\):\s*([0-9.eE+-]+)",  out).group(1)))

                        elif code in (2, 3):  # Bagging / Boosting
                            t_tr.append(float(re.search(r"Training time:\s*([0-9.eE+-]+)", out).group(1)))
                            t_ev.append(float(re.search(r"Evaluation time:\s*([0-9.eE+-]+)", out).group(1)))
                            mse.append(float(re.search(r"MSE\):?\s*([0-9.eE+-]+)", out).group(1)))

                        elif code == 4:       # LightGBM
                            t_tr.append(float(re.search(r"\[LightGBM\] Training time:\s*([0-9.eE+-]+)", out).group(1)))
                            t_ev.append(float(re.search(r"\[LightGBM\] Prediction time:\s*([0-9.eE+-]+)", out).group(1)))
                            mse.append(float(re.search(r"\[LightGBM\] MSE\s*=\s*([0-9.eE+-]+)", out).group(1)))
                            m2 = re.search(r"MAE\s*=\s*([0-9.eE+-]+)", out)
                            if m2: mae.append(float(m2.group(1)))

                    except AttributeError:
                        print("⚠️  parse fail – run ignoré"); continue

                results.append({
                    "scaling"        : scaling,
                    "model"          : name,
                    **cfg,
                    "train_time_mean": np.mean(t_tr)  if t_tr else np.nan,
                    "train_time_std" : np.std (t_tr)  if t_tr else np.nan,
                    "eval_time_mean" : np.mean(t_ev)  if t_ev else np.nan,
                    "eval_time_std"  : np.std (t_ev)  if t_ev else np.nan,
                    "mse_mean"       : np.mean(mse)   if mse else np.nan,
                    "mse_std"        : np.std (mse)   if mse else np.nan,
                    "mae_mean"       : np.mean(mae)   if mae else np.nan,
                    "mae_std"        : np.std (mae)   if mae else np.nan,
                })

# ---------- export ----------
out_csv = "scaling_results_no_advgbdt.csv"
pd.DataFrame(results).to_csv(out_csv, index=False)
print(f"\n✔︎ Benchmark fini – résultats dans {out_csv}")
