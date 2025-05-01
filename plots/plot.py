#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

# Ensure output directory exists
os.makedirs("plots", exist_ok=True)

# 1. Load benchmark results
# Locate the benchmark CSV in expected locations
csv_candidates = [
    os.path.join(os.path.dirname(__file__), "..", "benchmark_results_extended.csv"),
    os.path.join(os.path.dirname(__file__), "..", "plots", "benchmark_results_extended.csv")
]
csv_path = None
for candidate in csv_candidates:
    if os.path.isfile(candidate):
        csv_path = candidate
        break
if csv_path is None:
    raise FileNotFoundError(
        "benchmark_results_extended.csv not found; checked: "
        + ", ".join(csv_candidates)
    )
df = pd.read_csv(csv_path)

# Primary hyperparameter per model name
primary_dict = {
    "DecisionTree": "max_depth",
    "Bagging":      "n_estimators",
    "Boosting":     "n_estimators",
    "LightGBM":     "n_estimators",
    "AdvancedGBDT": "n_estimators",
}

# 1. Scalabilité temporelle (Training Time vs Hyperparamètre principal)
for model_name, primary in primary_dict.items():
    model_df = df[df["model"] == model_name]
    if model_df.empty or primary not in model_df.columns:
        continue
    fig, ax = plt.subplots()
    for omp_flag in sorted(model_df["use_omp"].unique()):
        subdf = model_df[model_df["use_omp"] == omp_flag]
        # average training times per hyperparam value
        mean_df = subdf.groupby(primary)["train_time_s"].mean().reset_index()
        ax.plot(mean_df[primary], mean_df["train_time_s"],
                marker="o", label=f"use_omp={omp_flag}")
    ax.set_xlabel(primary)
    ax.set_ylabel("Training time (s)")
    ax.set_title(f"{model_name}: Training Time vs {primary}")
    ax.legend()
    fig.savefig(f"plots/{model_name}_scaling_time.png")
    plt.close(fig)

# 4. Speed-up OpenMP (Speed-up = T1 / Tp)
for model_name, group in df.groupby("model"):
    primary = primary_dict.get(model_name)
    if primary is None or primary not in group.columns:
        continue
    single = group[group["use_omp"] == 0][["train_time_s", primary]].set_index(primary)
    multi = group[group["use_omp"] == 1][["train_time_s", primary]].set_index(primary)
    shared = single.join(multi, lsuffix="_1", rsuffix="_p", how="inner")
    shared["speedup"] = shared["train_time_s_1"] / shared["train_time_s_p"]
    plt.figure()
    plt.plot(shared.index, shared["speedup"], marker='o')
    plt.xlabel(primary)
    plt.ylabel("Speed-up (T1/Tp)")
    plt.title(f"{model_name}: OpenMP Speed-up vs {primary}")
    plt.savefig(f"plots/{model_name}_speedup.png")
    plt.close()

# 5. Comparaison à hyperparamètre constant (bar charts)
# Choose a common value (first encountered) per model
const_vals = {
    model: primary_dict.get(model)
    for model in df["model"].unique()
    if primary_dict.get(model) in df.columns
}

# Training time bar chart
plt.figure()
names, times = [], []
for model_name, primary in const_vals.items():
    row = df[(df["model"] == model_name) & (df[primary] == df[df["model"]==model_name][primary].unique()[0]) & (df["use_omp"]==0)]
    if not row.empty:
        names.append(model_name)
        times.append(row["train_time_s"].iloc[0])
plt.bar(names, times)
plt.ylabel("Training time (s)")
plt.title(f"Training Time at constant hyperparameter")
plt.xticks(rotation=45)
plt.savefig("plots/comparison_time_bar.png")
plt.close()

# MSE bar chart
plt.figure()
names, errs = [], []
for model_name, primary in const_vals.items():
    row = df[(df["model"] == model_name) & (df[primary] == df[df["model"]==model_name][primary].unique()[0]) & (df["use_omp"]==0)]
    if not row.empty:
        names.append(model_name)
        errs.append(row["mse"].iloc[0])
plt.bar(names, errs)
plt.ylabel("MSE")
plt.title(f"MSE at constant hyperparameter")
plt.xticks(rotation=45)
plt.savefig("plots/comparison_mse_bar.png")
plt.close()

# 6. Pareto front (Training time vs MSE)
plt.figure()
for model_name, group in df.groupby("model"):
    plt.scatter(group["train_time_s"], group["mse"], label=model_name)
plt.xlabel("Training time (s)")
plt.ylabel("MSE")
plt.title("Pareto front: Time vs MSE")
plt.legend()
plt.savefig("plots/pareto_time_mse.png")
plt.close()

# 7. Sensibilité aux hyperparamètres (heatmap) example for DecisionTree
tree_df = df[df["model"] == "DecisionTree"]
# Only plot heatmap if both hyperparameters are present in data
if not tree_df.empty and "max_depth" in tree_df.columns and "min_samples_split" in tree_df.columns:
    pivot = tree_df.pivot(index="max_depth", columns="min_samples_split", values="mse")
    plt.figure()
    plt.imshow(pivot, aspect='auto', origin='lower')
    plt.colorbar(label="MSE")
    plt.xlabel("min_samples_split")
    plt.ylabel("max_depth")
    plt.title("DecisionTree: MSE heatmap")
    plt.savefig("plots/heatmap_tree_mse.png")
    plt.close()
else:
    print("Skipping DecisionTree heatmap: 'min_samples_split' not found in data.")

# 8. Évolution de la perte (Boosting only)
# If original loss logs were saved, else skip

# 9. Importance des caractéristiques
# Assuming importance columns exist: feature_importance.<feature>
# We'll skip if not present

# 10. Comparaison multi-modèles radar (example simplified)
# skip

# 11. Distribution des temps (boxplot)
plt.figure()
data = [group["train_time_s"] for _, group in df.groupby("model")]
labels = df["model"].unique()
plt.boxplot(data, labels=labels)
plt.ylabel("Training time (s)")
plt.title("Training time distribution")
plt.xticks(rotation=45)
plt.savefig("plots/boxplot_time.png")
plt.close()

# 12. 3D surface (for Boosting: n_estimators vs learning_rate vs mse)
boost_df = df[df["model"]=="Boosting"]
if not boost_df.empty and "learning_rate" in boost_df:
    X = boost_df["n_estimators"].values
    Y = boost_df["learning_rate"].values
    Z = boost_df["mse"].values
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("learning_rate")
    ax.set_zlabel("MSE")
    plt.title("Boosting: MSE surface")
    plt.savefig("plots/3d_boost_mse.png")
    plt.close()

plt.tight_layout()
print("All plots generated in ./plots/")