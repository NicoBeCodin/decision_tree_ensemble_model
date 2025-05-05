#!/usr/bin/env python3
# bench_bagging_hybrid.py
import os, re, subprocess, numpy as np, pandas as pd
from itertools import product
import signal

# --------------------------------------------------------------------------
# Config générale
# --------------------------------------------------------------------------
REPEATS      = 3
ROOT_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR     = os.path.join(ROOT_DIR, "datasets", "processed")
BIN          = os.path.join(ROOT_DIR, "build", "MainEnsemble")

mpi_ranks    = [1, 2, 3, 4, 5, 6]
omp_threads  = [1, 2, 3, 4, 5, 6]
strong_file  = "cleaned_data_30k_ga_adaptative.csv"

data_map = {  # pour weak scaling
    1: "cleaned_data_5k_ga_adaptative.csv",
    2: "cleaned_data_10k_ga_adaptative.csv",
    3: "cleaned_data_15k_ga_adaptative.csv",
    4: "cleaned_data_20k_ga_adaptative.csv",
    5: "cleaned_data_25k_ga_adaptative.csv",
    6: "cleaned_data_30k_ga_adaptative.csv",
}

# Paramètres du modèle Bagging
param_grid = {
    "n_estimators": [200],
    "max_depth": [10],
    "criteria": [0],
    "which_loss_function": [0],
}

# --------------------------------------------------------------------------
# Génère toutes les combinaisons
# --------------------------------------------------------------------------
def all_param_combos(grid):
    keys, values = zip(*grid.items())
    for combo in product(*values):
        yield dict(zip(keys, combo))

# --------------------------------------------------------------------------
# Regex MSE/MAE + durée pour Bagging
# --------------------------------------------------------------------------
NUM = r"([0-9.+\-eE]+)"
re_train = re.compile(r"Training time: *" + NUM)
re_eval  = re.compile(r"Evaluation time: *" + NUM)
re_mse   = re.compile(r"Mean Square[d]? Error.*: *" + NUM)
re_mae   = re.compile(r"Mean Absolute Error.*: *" + NUM)

# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------
results = []

for scaling in ("strong", "weak"):
    for combo in all_param_combos(param_grid):
        for rank in mpi_ranks:
            for thread in omp_threads:
                dataset   = strong_file if scaling == "strong" else data_map.get(rank, strong_file)
                data_path = os.path.join(DATA_DIR, dataset)

                flags = combo | {"num_threads": thread, "use_omp": 1}
                args = [f"--{k}={v}" for k, v in flags.items()]
                cmd = [
                    "mpiexec", "-n", str(rank),
                    "--env", "OMP_NUM_THREADS", str(thread),
                    "--env", "OMP_PROC_BIND", "close",
                    "--env", "OMP_PLACES", "cores",
                    BIN, "2", "-p", "0", "0", "20", "60", "2", "0.000001", "1", "4",
                    f"--data={data_path}"
                ] + args

                print(f"[{scaling.upper()}] MPI={rank}  OMP={thread} → {' '.join(cmd)}")

                tt, te, mse, mae = [], [], [], []
                status = []

                for _ in range(REPEATS):
                    proc = subprocess.run(cmd, capture_output=True, text=True, input="0\n0\n")
                    if proc.returncode != 0:
                        if proc.returncode == -signal.SIGSEGV:
                            print("⚠️  SEGFAULT")
                            status.append("segfault")
                        else:
                            print(f"⚠️  crash (code {proc.returncode})")
                            status.append(f"err_{proc.returncode}")
                        continue

                    out = proc.stdout
                    status.append("ok")

                    m_tr = re_train.search(out)
                    m_ev = re_eval.search(out)
                    m_mse = re_mse.search(out)
                    m_mae = re_mae.search(out)

                    if not (m_tr and m_ev and m_mse and m_mae):
                        print("⚠️  Parse fail – run ignoré")
                        status[-1] = "parse_fail"
                        continue

                    tt.append(float(m_tr.group(1)))
                    te.append(float(m_ev.group(1)))
                    mse.append(float(m_mse.group(1)))
                    mae.append(float(m_mae.group(1)))

                if not tt: continue  # skip si aucun run valide

                results.append({
                    "scaling": scaling,
                    "mpi_ranks": rank,
                    "omp_threads": thread,
                    **combo,
                    "train_time_mean": np.mean(tt), "train_time_std": np.std(tt),
                    "eval_time_mean":  np.mean(te), "eval_time_std":  np.std(te),
                    "mse_mean":        np.mean(mse), "mse_std":        np.std(mse),
                    "mae_mean":        np.mean(mae), "mae_std":        np.std(mae),
                })

# --------------------------------------------------------------------------
# Sauvegarde
# --------------------------------------------------------------------------
df = pd.DataFrame(results)
df.to_csv("bagging_hybrid_scaling.csv", index=False)
print("\n✅ Résultats écrits dans bagging_hybrid_scaling.csv")
