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
    5: "AdvGBDT",
}

# ---------- data ----------
data_map = {        # weak-scaling
    1: "cleaned_data_5k_ga_adaptative.csv",
    2: "cleaned_data_10k_ga_adaptative.csv",
    3: "cleaned_data_15k_ga_adaptative.csv",
    4: "cleaned_data_20k_ga_adaptative.csv",
    5: "cleaned_data_25k_ga_adaptative.csv",
    6: "cleaned_data_30k_ga_adaptative.csv",
}
strong_data_file = "cleaned_data_30k_ga_adaptative.csv"

# ---------- paramètres ----------
thread_counts = [1, 2, 3, 4, 5, 6]

# pour Bagging (2) et Boosting (3) on teste MSE *et* MAE
param_template = {
    1: {"max_depth": [10]},                           # DecisionTree
    2: {"n_estimators": [200], "max_depth": [10],     # Bagging
        "criteria":            [0],                #   0=MSE  1=MAE
        "which_loss_function": [0]},               #   idem
    3: {"n_estimators": [200], "max_depth": [10],     # Boosting
        "learning_rate":       [0.1],
        "criteria":            [0],
        "which_loss_function": [0]},
    4: {"n_estimators": [200], "learning_rate": [0.1], "max_depth": [10],
        "num_leaves": [31], "subsample": [0.8], "colsample_bytree": [0.8]},
    5: {"n_estimators": [200], "learning_rate": [0.1], "max_depth": [10],
        "min_data_leaf": [20], "num_bins": [255], "use_dart": [1],
        "dropout_rate": [0.5], "skip_drop_rate": [0.3], "binning_method": [1]
    },
}

# ---------- utilitaire : construit la ligne de commande ----------
def make_cmd(model_code: int,
             nt        : int,
             data_file : str,
             flags     : dict[str, str]) -> list[str]:

    """Construit la commande finale"""
    base = [BIN, str(model_code), f"--data={data_file}"] + \
           [f"--{k}={v}" for k, v in flags.items()]

    return base


# ---------- bench ----------
results = []

for scaling in ["strong", "weak"]:
    for code, name in models.items():

        keys, vals = zip(*param_template[code].items())
        for combo in product(*vals):

            base_flags = dict(zip(keys, combo))

            # ─── NOUVEAU : ne garder que (0,0) ou (1,1) ──────────────
            if code in (2, 3):               # Bagging ou Boosting
                if base_flags["criteria"] != base_flags["which_loss_function"]:
                    continue                  # saute la combinaison incorrecte
            # ──────────────────────────────────────────────────────────

            for nt in thread_counts:
                    cfg       = base_flags | {"num_threads": nt, "use_omp": 1}
                    data_file = strong_data_file if scaling == "strong" \
                                else data_map.get(nt, strong_data_file)
                    abs_data  = os.path.join(DATA_DIR, data_file)

                    cmd = make_cmd(code, nt, abs_data, cfg)
                    print(f"[{scaling.upper()}] {name:12s} "
                          f"nt={nt}  → {' '.join(cmd)}")

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
                                mse.append(float(re.search(r"Mean Square[d]? Error.*: *([0-9\.eE+-]+)", out).group(1)))
                                mae.append(float(re.search(r"Mean Absolute Error.*: *([0-9\.eE+-]+)", out).group(1)))

                            elif code == 4:        # LightGBM
                                t_tr.append(float(re.search(r"\[LightGBM\] Training time: *([0-9\.]+)", out).group(1)))
                                t_ev.append(float(re.search(r"\[LightGBM\] Prediction time: *([0-9\.]+)", out).group(1)))
                                mse.append(float(re.search(r"\[LightGBM\] MSE = *([0-9\.eE+-]+)", out).group(1)))
                                mae.append(float(re.search(r"MAE\s*=\s*([0-9\.eE+-]+)", out).group(1)))
                            
                            elif code == 5:
                                m_tr   = re.search(r"\[AdvGBDT\]\s*Training time:\s*([0-9\.eE+-]+)", out)
                                m_pred = re.search(r"\[AdvGBDT\]\s*Prediction time:\s*([0-9\.eE+-]+)", out)
                                m_perf = re.search(r"\[AdvGBDT\].*MSE\s*=\s*([0-9\.eE+-]+)\s*,\s*MAE\s*=\s*([0-9\.eE+-]+)", out)

                                if not (m_tr and m_pred and m_perf):
                                    print("⚠️  parse fail (AdvGBDT) – run ignoré");  continue

                                t_tr.append(float(m_tr.group(1)))
                                t_ev.append(float(m_pred.group(1)))
                                mse .append(float(m_perf.group(1)))
                                mae .append(float(m_perf.group(2)))

                        except AttributeError:
                            print("⚠️  parse fail – run ignoré"); continue

                    results.append({
                        "scaling"   : scaling,
                        "model"     : name,
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
out_csv = "scaling_results_all.csv"
pd.DataFrame(results).to_csv(out_csv, index=False)
print(f"\n✔︎ Bench terminé – résultats dans {out_csv}")
