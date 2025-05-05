# Script: make_fig01_strong_scaling_split.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.ticker as mticker

df = pd.read_csv("scaling_results_all2.csv")
group1 = ["Bagging", "Boosting"]
group2 = ["DecisionTree", "LightGBM", "AdvGBDT"]

COLORS = {
    "Bagging": "#117733",
    "Boosting": "#44AA99",
    "DecisionTree": "#332288",
    "LightGBM": "#88CCEE",
    "AdvGBDT": "#DDCC77",
}
NICE = {
    "Bagging": "Bagging",
    "Boosting": "Boosting",
    "DecisionTree": "Tree",
    "LightGBM": "LightGBM",
    "AdvGBDT": "AdvGBDT",
}

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7.3, 5.5),
                        gridspec_kw={"height_ratios": (1.2, 1)})
df_strong = df[df.scaling == "strong"]

def plot_group(models, marker):
    for model in models:
        g = df_strong[df_strong.model == model].sort_values("num_threads")
        base = g[g.num_threads == 1].train_time_mean.iloc[0]
        speedup = base / g.train_time_mean
        err_rt = g.train_time_std
        err_sp = speedup * (g.train_time_std / g.train_time_mean)
        axs[0].errorbar(g.num_threads, g.train_time_mean, yerr=err_rt,
                        marker=marker, lw=1.8, ms=5, label=NICE[model],
                        color=COLORS[model])
        axs[1].errorbar(g.num_threads, speedup, yerr=err_sp,
                        marker=marker, lw=1.8, ms=5, color=COLORS[model])
        for x, s in zip(g.num_threads, speedup):
            eta = s / x
            axs[1].annotate(f"{eta:.2f}", (x, s), textcoords="offset points",
                            xytext=(0, 5), ha="center", fontsize=8, color=COLORS[model])

plot_group(group1, "o")
plot_group(group2, "^")

p_vals = np.array(sorted(df_strong.num_threads.unique()))
axs[1].plot(p_vals, p_vals, "--", color="grey", lw=1, label="Idéal")

axs[0].set_ylabel("Temps d’entraînement [s]")
axs[0].set_title("Strong scaling – runtime")
axs[1].set_xlabel("Nombre de threads")
axs[1].set_ylabel("Speed-up\n(baseline 1 thread)")
axs[1].set_xlim(0.8)
axs[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
axs[0].legend(ncol=3, fontsize=9)
fig.tight_layout()

fig.savefig("figures/fig01_strong_scaling_split.png", dpi=300)
