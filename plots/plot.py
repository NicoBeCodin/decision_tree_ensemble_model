#!/usr/bin/env python3
import os 
import pandas as pd
import matplotlib.pyplot as plt

# hyperparameter grids per model
param_grids = {
    "DecisionTree": {"max_depth": [2, 5, 10, 20, 40, 80, 160], "use_omp": [0, 1]},
    "Bagging": {
        "n_estimators": [10, 50, 100, 200, 400],
        "max_depth": [2, 5, 10, 20, 40, 80],
        "use_omp": [0, 1]
    },
    "Boosting": {
        "n_estimators": [50, 100, 200, 400, 800],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.001, 0.01, 0.05, 0.1],
        "use_omp": [0, 1]
    },
    "LightGBM": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20],
        "learning_rate": [0.05, 0.1],
        "use_omp": [0, 1]
    },
    "AdvancedGBDT": {
        "n_estimators": [100, 200, 400],
        "max_depth": [8, 16],
        "learning_rate": [0.01, 0.05],
        "use_omp": [0, 1]
    },
}

def plot_time_scalability(df, hp_map):
    """
    For each model in the dataframe, plot training time vs hyperparameters.
    DecisionTree: Training time vs max_depth (use_omp curves).
    Ensemble models: 
        - For fixed max_depth values, plot training time vs n_estimators.
        - For fixed n_estimators values, plot training time vs max_depth.
    """
    for model, group in df.groupby('model'):
        if 'DecisionTree' in model:
            # decision tree only depends on max_depth
            plt.figure()
            for omp_flag in sorted(group['use_omp'].unique()):
                sub = group[group['use_omp'] == omp_flag].sort_values('max_depth')
                plt.errorbar(
                    sub['max_depth'], sub['train_time_mean'], yerr=sub['train_time_std'],
                    marker='o', markersize=4, linewidth=1, capsize=3, elinewidth=1, label=f"use_omp={int(omp_flag)}"
                )
            plt.xlabel('max_depth')
            plt.ylabel("Temps d'entraînement moyen (s)")
            plt.title(f"{model}: Training Time vs max_depth")
            plt.legend()
            plt.tight_layout()
        elif model == "Bagging":
            depths = param_grids["Bagging"]["max_depth"]
            estims = param_grids["Bagging"]["n_estimators"]
            # fixed max_depth plots
            for depth in depths:
                sub_depth = group[group['max_depth'] == depth]
                if sub_depth.empty:
                    continue
                plt.figure()
                for omp_flag in sorted(sub_depth['use_omp'].unique()):
                    sub = sub_depth[sub_depth['use_omp'] == omp_flag].sort_values('n_estimators')
                    plt.errorbar(
                        sub['n_estimators'], sub['train_time_mean'], yerr=sub['train_time_std'],
                        marker='o', markersize=4, linewidth=1, capsize=3, elinewidth=1, label=f"use_omp={int(omp_flag)}"
                    )
                plt.xlabel('n_estimators')
                plt.ylabel("Temps d'entraînement moyen (s)")
                plt.title(f"{model}: Training Time vs n_estimators (max_depth={depth})")
                plt.legend()
                plt.tight_layout()
            # fixed n_estimators plots
            for n in estims:
                sub_n = group[group['n_estimators'] == n]
                if sub_n.empty:
                    continue
                plt.figure()
                for omp_flag in sorted(sub_n['use_omp'].unique()):
                    sub = sub_n[sub_n['use_omp'] == omp_flag].sort_values('max_depth')
                    plt.errorbar(
                        sub['max_depth'], sub['train_time_mean'], yerr=sub['train_time_std'],
                        marker='o', markersize=4, linewidth=1, capsize=3, elinewidth=1, label=f"use_omp={int(omp_flag)}"
                    )
                plt.xlabel('max_depth')
                plt.ylabel("Temps d'entraînement moyen (s)")
                plt.title(f"{model}: Training Time vs max_depth (n_estimators={n})")
                plt.legend()
                plt.tight_layout()
        elif model == "Boosting":
            depths = param_grids["Boosting"]["max_depth"]
            estims = param_grids["Boosting"]["n_estimators"]
            lrs = param_grids["Boosting"]["learning_rate"]
            # fixed max_depth → train_time vs n_estimators
            for depth in depths:
                sub_depth = group[group['max_depth'] == depth]
                if sub_depth.empty: continue
                plt.figure()
                for omp_flag in sorted(sub_depth['use_omp'].unique()):
                    sub = sub_depth[sub_depth['use_omp'] == omp_flag]
                    mean_t = sub.groupby('n_estimators')['train_time_mean'].mean()
                    std_t = sub.groupby('n_estimators')['train_time_mean'].std().fillna(0)
                    plt.errorbar(
                        mean_t.index, mean_t.values, yerr=std_t.values,
                        marker='o', markersize=4, linewidth=1, capsize=3, elinewidth=1,
                        label=f"use_omp={int(omp_flag)}"
                    )
                plt.xlabel('n_estimators')
                plt.ylabel("Temps d'entraînement moyen (s)")
                plt.title(f"{model}: Training Time vs n_estimators (max_depth={depth})")
                plt.legend()
                plt.tight_layout()
            # fixed n_estimators → train_time vs max_depth
            for n in estims:
                sub_n = group[group['n_estimators'] == n]
                if sub_n.empty: continue
                plt.figure()
                for omp_flag in sorted(sub_n['use_omp'].unique()):
                    sub = sub_n[sub_n['use_omp'] == omp_flag]
                    mean_t = sub.groupby('max_depth')['train_time_mean'].mean()
                    std_t = sub.groupby('max_depth')['train_time_mean'].std().fillna(0)
                    plt.errorbar(
                        mean_t.index, mean_t.values, yerr=std_t.values,
                        marker='o', markersize=4, linewidth=1, capsize=3, elinewidth=1,
                        label=f"use_omp={int(omp_flag)}"
                    )
                plt.xlabel('max_depth')
                plt.ylabel("Temps d'entraînement moyen (s)")
                plt.title(f"{model}: Training Time vs max_depth (n_estimators={n})")
                plt.legend()
                plt.tight_layout()
            # learning_rate variations at max_depth=9
            target_depth = depths[-1]
            for lr in lrs:
                sub_lr = group[(group['learning_rate'] == lr) & (group['max_depth'] == target_depth)]
                if sub_lr.empty: continue
                plt.figure()
                for omp_flag in sorted(sub_lr['use_omp'].unique()):
                    sub = sub_lr[sub_lr['use_omp'] == omp_flag]
                    mean_t = sub.groupby('n_estimators')['train_time_mean'].mean()
                    std_t = sub.groupby('n_estimators')['train_time_mean'].std().fillna(0)
                    plt.errorbar(
                        mean_t.index, mean_t.values, yerr=std_t.values,
                        marker='o', markersize=4, linewidth=1, capsize=3, elinewidth=1,
                        label=f"lr={lr}, omp={int(omp_flag)}"
                    )
                plt.xlabel('n_estimators')
                plt.ylabel("Temps d'entraînement moyen (s)")
                plt.title(f"{model}: Training Time vs n_estimators (max_depth={target_depth}, lr={lr})")
                plt.legend()
                plt.tight_layout()
            # learning_rate variations at n_estimators=400
            target_n = 400
            for lr in lrs:
                sub_lr = group[(group['learning_rate'] == lr) & (group['n_estimators'] == target_n)]
                if sub_lr.empty: continue
                plt.figure()
                for omp_flag in sorted(sub_lr['use_omp'].unique()):
                    sub = sub_lr[sub_lr['use_omp'] == omp_flag]
                    mean_t = sub.groupby('max_depth')['train_time_mean'].mean()
                    std_t = sub.groupby('max_depth')['train_time_mean'].std().fillna(0)
                    plt.errorbar(
                        mean_t.index, mean_t.values, yerr=std_t.values,
                        marker='o', markersize=4, linewidth=1, capsize=3, elinewidth=1,
                        label=f"lr={lr}, omp={int(omp_flag)}"
                    )
                plt.xlabel('max_depth')
                plt.ylabel("Temps d'entraînement moyen (s)")
                plt.title(f"{model}: Training Time vs max_depth (n_estimators={target_n}, lr={lr})")
                plt.legend()
                plt.tight_layout()
        elif model == "LightGBM":
            depths = param_grids["LightGBM"]["max_depth"]
            estims = param_grids["LightGBM"]["n_estimators"]
            lrs = param_grids["LightGBM"]["learning_rate"]
            # fixed max_depth → train_time vs n_estimators
            for depth in depths:
                sub_depth = group[group['max_depth'] == depth]
                if sub_depth.empty: continue
                plt.figure()
                for omp_flag in sorted(sub_depth['use_omp'].unique()):
                    sub = sub_depth[sub_depth['use_omp'] == omp_flag].sort_values('n_estimators')
                    plt.errorbar(
                        sub['n_estimators'], sub['train_time_mean'], yerr=sub['train_time_std'],
                        marker='o', markersize=4, linewidth=1, capsize=3, elinewidth=1,
                        label=f"use_omp={int(omp_flag)}"
                    )
                plt.xlabel('n_estimators')
                plt.ylabel("Temps d'entraînement moyen (s)")
                plt.title(f"{model}: Training Time vs n_estimators (max_depth={depth})")
                plt.legend()
                plt.tight_layout()
            # fixed n_estimators → train_time vs max_depth
            for n in estims:
                sub_n = group[group['n_estimators'] == n]
                if sub_n.empty: continue
                plt.figure()
                for omp_flag in sorted(sub_n['use_omp'].unique()):
                    sub = sub_n[sub_n['use_omp'] == omp_flag].sort_values('max_depth')
                    plt.errorbar(
                        sub['max_depth'], sub['train_time_mean'], yerr=sub['train_time_std'],
                        marker='o', markersize=4, linewidth=1, capsize=3, elinewidth=1,
                        label=f"use_omp={int(omp_flag)}"
                    )
                plt.xlabel('max_depth')
                plt.ylabel("Temps d'entraînement moyen (s)")
                plt.title(f"{model}: Training Time vs max_depth (n_estimators={n})")
                plt.legend()
                plt.tight_layout()
            # learning_rate variations at max_depth = max value
            target_depth = depths[-1]
            for lr in lrs:
                sub_lr = group[(group['learning_rate'] == lr) & (group['max_depth'] == target_depth)]
                if sub_lr.empty: continue
                plt.figure()
                for omp_flag in sorted(sub_lr['use_omp'].unique()):
                    sub = sub_lr[sub_lr['use_omp'] == omp_flag]
                    mean_t = sub.groupby('n_estimators')['train_time_mean'].mean()
                    std_t = sub.groupby('n_estimators')['train_time_mean'].std().fillna(0)
                    plt.errorbar(
                        mean_t.index, mean_t.values, yerr=std_t.values,
                        marker='o', markersize=4, linewidth=1, capsize=3, elinewidth=1,
                        label=f"lr={lr}, omp={int(omp_flag)}"
                    )
                plt.xlabel('n_estimators')
                plt.ylabel("Temps d'entraînement moyen (s)")
                plt.title(f"{model}: Training Time vs n_estimators (max_depth={target_depth}, lr={lr})")
                plt.legend()
                plt.tight_layout()
            # learning_rate variations at n_estimators = max value
            target_n = estims[-1]
            for lr in lrs:
                sub_lr = group[(group['learning_rate'] == lr) & (group['n_estimators'] == target_n)]
                if sub_lr.empty: continue
                plt.figure()
                for omp_flag in sorted(sub_lr['use_omp'].unique()):
                    sub = sub_lr[sub_lr['use_omp'] == omp_flag]
                    mean_t = sub.groupby('max_depth')['train_time_mean'].mean()
                    std_t = sub.groupby('max_depth')['train_time_mean'].std().fillna(0)
                    plt.errorbar(
                        mean_t.index, mean_t.values, yerr=std_t.values,
                        marker='o', markersize=4, linewidth=1, capsize=3, elinewidth=1,
                        label=f"lr={lr}, omp={int(omp_flag)}"
                    )
                plt.xlabel('max_depth')
                plt.ylabel("Temps d'entraînement moyen (s)")
                plt.title(f"{model}: Training Time vs max_depth (n_estimators={target_n}, lr={lr})")
                plt.legend()
                plt.tight_layout()
        elif model == "AdvancedGBDT":
            depths = param_grids["AdvancedGBDT"]["max_depth"]
            estims = param_grids["AdvancedGBDT"]["n_estimators"]
            lrs = param_grids["AdvancedGBDT"]["learning_rate"]
            # fixed max_depth → train_time vs n_estimators
            for depth in depths:
                sub_depth = group[group['max_depth'] == depth]
                if sub_depth.empty: continue
                plt.figure()
                for omp_flag in sorted(sub_depth['use_omp'].unique()):
                    sub = sub_depth[sub_depth['use_omp'] == omp_flag].sort_values('n_estimators')
                    plt.errorbar(
                        sub['n_estimators'], sub['train_time_mean'], yerr=sub['train_time_std'],
                        marker='o', markersize=4, linewidth=1, capsize=3, elinewidth=1,
                        label=f"use_omp={int(omp_flag)}"
                    )
                plt.xlabel('n_estimators')
                plt.ylabel("Temps d'entraînement moyen (s)")
                plt.title(f"{model}: Training Time vs n_estimators (max_depth={depth})")
                plt.legend()
                plt.tight_layout()
            # fixed n_estimators → train_time vs max_depth
            for n in estims:
                sub_n = group[group['n_estimators'] == n]
                if sub_n.empty: continue
                plt.figure()
                for omp_flag in sorted(sub_n['use_omp'].unique()):
                    sub = sub_n[sub_n['use_omp'] == omp_flag].sort_values('max_depth')
                    plt.errorbar(
                        sub['max_depth'], sub['train_time_mean'], yerr=sub['train_time_std'],
                        marker='o', markersize=4, linewidth=1, capsize=3, elinewidth=1,
                        label=f"use_omp={int(omp_flag)}"
                    )
                plt.xlabel('max_depth')
                plt.ylabel("Temps d'entraînement moyen (s)")
                plt.title(f"{model}: Training Time vs max_depth (n_estimators={n})")
                plt.legend()
                plt.tight_layout()
            # learning_rate variations at max_depth = max value
            target_depth = depths[-1]
            for lr in lrs:
                sub_lr = group[(group['learning_rate'] == lr) & (group['max_depth'] == target_depth)]
                if sub_lr.empty: continue
                plt.figure()
                for omp_flag in sorted(sub_lr['use_omp'].unique()):
                    sub = sub_lr[sub_lr['use_omp'] == omp_flag]
                    mean_t = sub.groupby('n_estimators')['train_time_mean'].mean()
                    std_t = sub.groupby('n_estimators')['train_time_mean'].std().fillna(0)
                    plt.errorbar(
                        mean_t.index, mean_t.values, yerr=std_t.values,
                        marker='o', markersize=4, linewidth=1, capsize=3, elinewidth=1,
                        label=f"lr={lr}, omp={int(omp_flag)}"
                    )
                plt.xlabel('n_estimators')
                plt.ylabel("Temps d'entraînement moyen (s)")
                plt.title(f"{model}: Training Time vs n_estimators (max_depth={target_depth}, lr={lr})")
                plt.legend()
                plt.tight_layout()
            # learning_rate variations at n_estimators = max value
            target_n = estims[-1]
            for lr in lrs:
                sub_lr = group[(group['learning_rate'] == lr) & (group['n_estimators'] == target_n)]
                if sub_lr.empty: continue
                plt.figure()
                for omp_flag in sorted(sub_lr['use_omp'].unique()):
                    sub = sub_lr[sub_lr['use_omp'] == omp_flag]
                    mean_t = sub.groupby('max_depth')['train_time_mean'].mean()
                    std_t = sub.groupby('max_depth')['train_time_mean'].std().fillna(0)
                    plt.errorbar(
                        mean_t.index, mean_t.values, yerr=std_t.values,
                        marker='o', markersize=4, linewidth=1, capsize=3, elinewidth=1,
                        label=f"lr={lr}, omp={int(omp_flag)}"
                    )
                plt.xlabel('max_depth')
                plt.ylabel("Temps d'entraînement moyen (s)")
                plt.title(f"{model}: Training Time vs max_depth (n_estimators={target_n}, lr={lr})")
                plt.legend()
                plt.tight_layout()

def plot_mse_quality(df, hp_map):
    plt.figure()
    for model, group in df.groupby('model'):
        hp = hp_map[model]
        for omp_flag in sorted(group['use_omp'].unique()):
            sub = group[group['use_omp'] == omp_flag].sort_values(hp)
            plt.plot(
                sub[hp],
                sub['mse_mean'],
                marker='o', markersize=4, linewidth=1,
                label=f"{model} omp={int(omp_flag)}"
            )
    plt.xlabel("Valeur de l'hyperparamètre")
    plt.ylabel("MSE moyen")
    plt.title("Qualité (MSE) vs Hyperparamètre")
    plt.legend()
    plt.tight_layout()

def plot_speedup(df, hp_map):
    plt.figure()
    # For each model, compute speedup as mean(T1)/mean(Tp) over primary hyperparam values
    for model, group in df.groupby('model'):
        hp = hp_map[model]
        # average train_time_mean for mono-thread and multi-thread across other params
        mono = group[group['use_omp'] == 0].groupby(hp)['train_time_mean'].mean()
        multi = group[group['use_omp'] == 1].groupby(hp)['train_time_mean'].mean()
        # find hyperparam values present in both
        common = mono.index.intersection(multi.index).sort_values()
        if len(common) > 0:
            speedup = mono.loc[common] / multi.loc[common]
            plt.plot(common, speedup, marker='o', markersize=4, linewidth=1, label=model)
    plt.xlabel("Valeur de l'hyperparamètre")
    plt.ylabel("Speedup (T₁ / Tₚ)")
    plt.title("Accélération OpenMP vs Hyperparamètre")
    plt.legend()
    plt.tight_layout()

def main():
    # Ajuste ici le chemin vers ton fichier CSV si nécessaire
    df = pd.read_csv("benchmark_results_extended.csv")

    # Détermine l'hyperparamètre principal par modèle
    hp_map = {}
    for model in df['model'].unique():
        if 'DecisionTree' in model:
            hp_map[model] = 'max_depth'
        else:
            hp_map[model] = 'n_estimators'

    # Dessine les 3 graphiques clés
    plot_time_scalability(df, hp_map)
    plot_mse_quality(df, hp_map)
    plot_speedup(df, hp_map)

    plt.show()

if __name__ == "__main__":
    main()