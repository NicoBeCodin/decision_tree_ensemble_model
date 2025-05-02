#!/usr/bin/env python3
# file: plot_accuracy_vs_threads.py
"""
Trace MSE & MAE en fonction du nombre de threads pour
  • strong-scaling      (dataset fixe)
  • weak-scaling        (dataset × p)
-->
  figures/accuracy_strong.png
  figures/accuracy_weak.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path         
import matplotlib.ticker as mt

df   = pd.read_csv("scaling_results.csv")
cols = {"DecisionTree":"#1f77b4", "Bagging":"#ff7f0e",
        "Boosting":"#2ca02c",     "LightGBM":"#d62728"}

def plot_kind(kind, fname):
    sub = df[df.scaling == kind]
    fig, ax = plt.subplots()
    for mod, g in sub.groupby("model"):
        g = g.sort_values("num_threads")
        ax.errorbar(g.num_threads, g.mse_mean, yerr=g.mse_std,
                    marker="o", lw=1.6,  color=cols.get(mod),
                    label=f"{mod} – MSE")
        if not g.mae_mean.isna().all():
            ax.errorbar(g.num_threads, g.mae_mean, yerr=g.mae_std,
                        marker="x", lw=1.2, ls="--",
                        color=cols.get(mod), alpha=.6,
                        label=f"{mod} – MAE")

    ax.set_xlabel("Nombre de threads")
    ax.set_ylabel("Erreur")
    ax.set_title(f"{kind.capitalize()} scaling – MSE / MAE")
    ax.set_xbound(0.8, sub.num_threads.max()+0.2)
    ax.xaxis.set_major_locator(mt.MaxNLocator(integer=True))
    ax.grid(ls=":", alpha=.6)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    (Path("figures").mkdir(exist_ok=True) or True)
    fig.savefig(f"figures/{fname}", dpi=300)
    plt.close(fig)

plot_kind("strong", "accuracy_strong.png")
plot_kind("weak",   "accuracy_weak.png")
print("✔︎ Précision strong+weak sauvegardée dans figures/")
