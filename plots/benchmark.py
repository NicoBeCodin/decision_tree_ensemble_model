#!/usr/bin/env python3
import os, re, subprocess, pandas as pd, numpy as np
from itertools import product

REPEATS   = 3
ROOT_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR  = os.path.join(ROOT_DIR, "datasets", "processed")
BIN       = os.path.join(ROOT_DIR, "build", "MainEnsemble")

# ---------- modèles ----------
models = {
    1: "DecisionTree",
    2: "Bagging",
    3: "Boosting",
    4: "LightGBM",
}

# ---------- data ----------
data_map = {        # weak-scaling
    1: "cleaned_data_5k_ga_adaptative.csv",
    2: "cleaned_data_10k_ga_adaptative.csv",
    3: "cleaned_data_15k_ga_adaptative.csv",
    4: "cleaned_data_20k_ga_adaptative.csv",
    6: "cleaned_data_30k_ga_adaptative.csv",
}
strong_data_file = "cleaned_data_30k_ga_adaptative.csv"

# ---------- paramètres ----------
thread_counts = [1, 2, 4, 6]
mpi_sets      = {
    1: [1],          # DecisionTree : pas de MPI
    2: [1, 2, 4, 6], # Bagging      : oui
    3: [1],          # Boosting     : pas de MPI
    4: [1],          # LightGBM     : pas de MPI dans ton projet
}

# pour Bagging (2) et Boosting (3) on teste MSE *et* MAE
param_template = {
    1: {"max_depth": [10]},                           # DecisionTree
    2: {"n_estimators": [200], "max_depth": [10],     # Bagging
        "criteria":            [0, 1],                #   0=MSE  1=MAE
        "which_loss_function": [0, 1]},               #   idem
    3: {"n_estimators": [200], "max_depth": [10],     # Boosting
        "learning_rate":       [0.1],
        "criteria":            [0, 1],
        "which_loss_function": [0, 1]},
    4: {"n_estimators": [200], "learning_rate": [0.1], "max_depth": [10],
        "num_leaves": [31], "subsample": [0.8], "colsample_bytree": [0.8]},
}

# ---------- utilitaire : construit la ligne de commande ----------
def make_cmd(model_code: int,
             mpi_rank  : int,
             nt        : int,
             data_file : str,
             flags     : dict[str, str]) -> list[str]:

    """Construit la commande finale"""
    base = [BIN, str(model_code), f"--data={data_file}"] + \
           [f"--{k}={v}" for k, v in flags.items()]

    if mpi_rank == 1:                      # pas de MPI
        return base

    mpiexec = ["mpiexec",
               "-n", str(mpi_rank),
               "--env", "OMP_NUM_THREADS", str(nt),
               "--env", "OMP_PROC_BIND",  "close",
               "--env", "OMP_PLACES",     "cores"]
    return mpiexec + base


# ---------- bench ----------
results = []

for scaling in ["strong", "weak"]:
    for code, name in models.items():

        mpi_ranks = mpi_sets[code]

        keys, vals = zip(*param_template[code].items())
        for combo in product(*vals):

            base_flags = dict(zip(keys, combo))

            # ─── NOUVEAU : ne garder que (0,0) ou (1,1) ──────────────
            if code in (2, 3):               # Bagging ou Boosting
                if base_flags["criteria"] != base_flags["which_loss_function"]:
                    continue                  # saute la combinaison incorrecte
            # ──────────────────────────────────────────────────────────

            for nt in thread_counts:
                for mp in mpi_ranks:

                    cfg       = base_flags | {"num_threads": nt, "use_omp": 1}
                    data_file = strong_data_file if scaling == "strong" \
                                else data_map.get(nt, strong_data_file)
                    abs_data  = os.path.join(DATA_DIR, data_file)

                    cmd = make_cmd(code, mp, nt, abs_data, cfg)
                    print(f"[{scaling.upper()}] {name:12s} "
                          f"mpi={mp:2d}  nt={nt}  → {' '.join(cmd)}")

                    t_tr = []; t_ev = []; mse = []; mae = []
                    for _ in range(REPEATS):
                        run = subprocess.run(cmd, capture_output=True,
                                             text=True, input="0\n0\n")
                        out = run.stdout   # seul le rank-0 parle

                        try:
                            if code == 1:   # DecisionTree
                                t_tr.append(float(re.search(r"Training time: *([0-9\.]+)", out).group(1)))
                                t_ev.append(float(re.search(r"Evaluation time: *([0-9\.]+)", out).group(1)))
                                mse.append(float(re.search(r"Mean Squared Error \(MSE\): *([0-9\.eE+-]+)", out).group(1)))
                                mae.append(float(re.search(r"Mean Absolute Error \(MAE\): *([0-9\.eE+-]+)", out).group(1)))

                            elif code in (2, 3):                    # Bagging / Boosting
                                t_tr.append(float(re.search(r"Training time: *([0-9\.]+)", out).group(1)))
                                t_ev.append(float(re.search(r"Evaluation time: *([0-9\.]+)", out).group(1)))

                                m_mse = re.search(r"Mean Square[d]? Error.*: *([0-9\.eE+-]+)", out)
                                m_mae = re.search(r"Mean Absolute Error.*: *([0-9\.eE+-]+)", out)

                                if m_mse:
                                    mse.append(float(m_mse.group(1)))
                                if m_mae:
                                    mae.append(float(m_mae.group(1)))

                            elif code == 4:        # LightGBM
                                t_tr.append(float(re.search(r"\[LightGBM\] Training time: *([0-9\.]+)", out).group(1)))
                                t_ev.append(float(re.search(r"\[LightGBM\] Prediction time: *([0-9\.]+)", out).group(1)))
                                mse.append(float(re.search(r"\[LightGBM\] MSE = *([0-9\.eE+-]+)", out).group(1)))
                                m2 = re.search(r"MAE\s*=\s*([0-9\.eE+-]+)", out)
                                if m2: mae.append(float(m2.group(1)))

                        except AttributeError:
                            print("⚠️  parse fail – run ignoré"); continue

                    results.append({
                        "scaling"   : scaling,
                        "model"     : name,
                        "mpi_rank"  : mp,
                        **cfg,
                        "train_time_mean": np.mean(t_tr) if t_tr else np.nan,
                        "train_time_std" : np.std (t_tr) if t_tr else np.nan,
                        "eval_time_mean" : np.mean(t_ev) if t_ev else np.nan,
                        "eval_time_std"  : np.std (t_ev) if t_ev else np.nan,
                        "mse_mean"       : np.mean(mse) if mse else np.nan,
                        "mse_std"        : np.std (mse) if mse else np.nan,
                        "mae_mean"       : np.mean(mae) if mae else np.nan,
                        "mae_std"        : np.std (mae) if mae else np.nan,
                    })

# ---------- export ----------
out_csv = "scaling_results_with_mpi.csv"
pd.DataFrame(results).to_csv(out_csv, index=False)
print(f"\n✔︎ Bench terminé – résultats dans {out_csv}")
