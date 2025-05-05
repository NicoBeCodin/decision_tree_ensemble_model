#!/usr/bin/env python3
# bench_adv_gbdt.py
import os, re, subprocess, numpy as np, pandas as pd
from itertools import product
import signal 

# --------------------------------------------------------------------------
# Config générale
# --------------------------------------------------------------------------
REPEATS      = 3                            # mesures / config
ROOT_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR     = os.path.join(ROOT_DIR, "datasets", "processed")
BIN          = os.path.join(ROOT_DIR, "build", "MainEnsemble")

thread_counts      = [1, 2, 4, 6]           # cœurs à tester
strong_data_file   = "cleaned_data_30k_ga_adaptative.csv"
data_map = {                                # threads ➜ CSV (weak scaling)
    1: "cleaned_data_5k_ga_adaptative.csv",
    2: "cleaned_data_10k_ga_adaptative.csv",
    4: "cleaned_data_20k_ga_adaptative.csv",
    6: "cleaned_data_30k_ga_adaptative.csv",
}

# hyper-paramètres du modèle 5
param_grid = {
    "n_estimators"    : [200],
    "learning_rate"   : [0.1],
    "max_depth"       : [10],
    "min_data_leaf"   : [20],
    "num_bins"        : [255],
    "use_dart"        : [1],
    "dropout_rate"    : [0.5],
    "skip_drop_rate"  : [0.3],
    "binning_method"  : [1],
}

# --------------------------------------------------------------------------
# Regex pré-compilées pour AdvancedGBDT
# --------------------------------------------------------------------------
NUM = r"([0-9.+\-eE]+)"        # nombre générique
re_train = re.compile(r"\[AdvGBDT\]\s*Training time:\s*"   + NUM)
re_pred  = re.compile(r"\[AdvGBDT\]\s*Prediction time:\s*" + NUM)
re_perf  = re.compile(r"\[AdvGBDT\].*MSE\s*=\s*" + NUM + r"\s*,\s*MAE\s*=\s*" + NUM)

# --------------------------------------------------------------------------
# Bench
# --------------------------------------------------------------------------
def all_param_combos(grid):
    keys, values = zip(*grid.items())
    for combo in product(*values):
        yield dict(zip(keys, combo))

results = []

for scaling in ("strong", "weak"):
    for combo in all_param_combos(param_grid):
        for nt in thread_counts:

            # dataset selon le type de scaling
            data_name = strong_data_file if scaling == "strong" else data_map[nt]
            data_abs  = os.path.join(DATA_DIR, data_name)

            # cmd complète
            flags = combo | {"num_threads": nt, "use_omp": 1}
            cmd = [BIN, "5", f"--data={data_abs}"] + \
                  [f"--{k}={v}" for k, v in flags.items()]

            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(nt)

            print(f"[{scaling.upper()}] {' '.join(cmd)}")

            tt, te, mse, mae = [], [], [], []

            status_list = []              # <-- nouveau tableau

            for _ in range(REPEATS):
                proc = subprocess.run(cmd, text=True, capture_output=True,
                                    input="0\n0\n", env=env)
                if proc.returncode != 0:          # -------- ERREUR / SIGNAUX ----------
                    if proc.returncode == -signal.SIGSEGV:
                        print("⚠️  SEGFAULT")
                        status_list.append("segfault")
                    else:
                        print(f"⚠️  crash (code {proc.returncode})")
                        status_list.append(f"err_{proc.returncode}")
                    continue      # on ne parse pas stdout dans ce cas

                out = proc.stdout
                status_list.append("ok")

                # -------------- extraction regex comme avant --------------
                m_train = re_train.search(out)
                m_pred  = re_pred .search(out)
                m_perf  = re_perf .search(out)
                if not (m_train and m_pred and m_perf):
                    print("⚠️  parsing failed – run skipped")
                    status_list[-1] = "parse_fail"
                    continue

            if not tt:          # aucun run valide
                continue

            results.append({
                "scaling"         : scaling,
                "num_threads"     : nt,
                **combo,
                "train_time_mean" : np.mean(tt),  "train_time_std": np.std(tt),
                "eval_time_mean"  : np.mean(te),  "eval_time_std" : np.std(te),
                "mse_mean"        : np.mean(mse), "mse_std"       : np.std(mse),
                "mae_mean"        : np.mean(mae), "mae_std"       : np.std(mae),
            })

# --------------------------------------------------------------------------
# Sauvegarde
# --------------------------------------------------------------------------
df = pd.DataFrame(results)
df.to_csv("adv_gbdt_scaling.csv", index=False)
print("\n✅ Résultats écrits dans adv_gbdt_scaling.csv")
