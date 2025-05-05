#!/usr/bin/env python3
# file: plot_accuracy_vs_threads.py
"""
Trace séparément MSE et MAE en fonction du nombre de threads pour :
  • strong-scaling      (dataset fixe)
  • weak-scaling        (dataset × p)

→  figures/accuracy_strong_mse.png
    figures/accuracy_strong_mae.png
    figures/accuracy_weak_mse.png
    figures/accuracy_weak_mae.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path         
import matplotlib.ticker as mt

df = pd.read_csv("adv_gbdt_scaling.csv")
outdir = Path("figures")
outdir.mkdir(exist_ok=True)

# Palette
cols = {"DecisionTree":"#1f77b4", "Bagging":"#ff7f0e",
        "Boosting":"#2ca02c",     "LightGBM":"#d62728", 
        "AdvGBDT": "#44aa99"}

def plot_metric(kind, metric, fname):
    sub = df[df.scaling == kind]
    fig, ax = plt.subplots()

    for mod, g in sub.groupby("model"):
        if g[f"{metric}_mean"].isna().all():
            continue
        g = g.sort_values("num_threads")
        ax.errorbar(g.num_threads, g[f"{metric}_mean"],
                    yerr=g[f"{metric}_std"],
                    marker="o" if metric == "mse" else "x",
                    linestyle="-" if metric == "mse" else "--",
                    lw=1.6,
                    color=cols.get(mod, "#444444"),
                    label=mod)

    ax.set_xlabel("Nombre de threads")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{kind.capitalize()} scaling – {metric.upper()}")
    ax.set_xbound(0.8, sub.num_threads.max() + 0.2)
    ax.xaxis.set_major_locator(mt.MaxNLocator(integer=True))
    ax.grid(ls=":", alpha=.6)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / fname, dpi=300)
    plt.close(fig)

# Génération des 4 courbes
plot_metric("strong", "mse", "accuracy_strong_mse.png")
plot_metric("strong", "mae", "accuracy_strong_mae.png")
plot_metric("weak",   "mse", "accuracy_weak_mse.png")
plot_metric("weak",   "mae", "accuracy_weak_mae.png")

print("✔︎ Figures MSE / MAE strong & weak sauvegardées dans 'figures/'")
