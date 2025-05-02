#!/usr/bin/env python3
# file: make_scaling_plots.py
"""
Génère des figures haute définition (PNG + PDF) :
  • fig01_strong_scaling.png / .pdf   (runtime + speed-up)
  • fig02_weak_scaling_runtime.png / .pdf
  • fig03_weak_scaling_efficiency.png / .pdf
à partir de scaling_results.csv.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.ticker as mticker

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Réglages d’apparence global : police + couleurs
# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize": (7.3, 4.6),
    "figure.dpi": 110,
    "savefig.dpi": 320,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.right": False,
    "axes.spines.top":   False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.55,
    "legend.frameon": False,
})

# Palette Tol Muted – color-blind-safe
TOL = ["#332288", "#117733", "#44AA99", "#88CCEE",
       "#DDCC77", "#CC6677", "#AA4499", "#882255"]
COLORS = {
    "DecisionTree":  TOL[0],
    "Bagging":       TOL[1],
    "Boosting":      TOL[2],
    "LightGBM":      TOL[3],
    # "AdvancedGBDT":  TOL[4],  # on l’ignore pour le moment
}
NICE = {  # étiquettes courtes
    "DecisionTree": "Tree",
    "Bagging":      "Bagging",
    "Boosting":     "Boosting",
    "LightGBM":     "LightGBM",
}

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Lecture du CSV
# ──────────────────────────────────────────────────────────────────────────────
CSV = Path("scaling_results_without_mpi.csv")
if not CSV.exists():
    raise SystemExit(f"❌ {CSV} introuvable")

df = pd.read_csv(CSV, dtype={"model": str})
df = df[df.model.isin(COLORS)]        # supprime AdvGBDT s’il reste
outdir = Path("figures")
outdir.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 3.  STRONG SCALING  – runtime + speed-up
# ──────────────────────────────────────────────────────────────────────────────
strong = df[df.scaling == "strong"].copy()

fig, (ax_rt, ax_sp) = plt.subplots(2, 1, sharex=True,
                                   gridspec_kw={"height_ratios": (1.2, 1)})

for model, grp in strong.groupby("model"):
    g = grp.sort_values("num_threads")
    base = g[g.num_threads == 1].train_time_mean.iloc[0]
    speedup = base / g.train_time_mean
    err_rt  = g.train_time_std
    err_sp  = speedup * (g.train_time_std / g.train_time_mean)

    # runtime
    ax_rt.errorbar(g.num_threads, g.train_time_mean, yerr=err_rt,
                   marker="o", lw=1.8, ms=5,
                   color=COLORS[model], label=NICE[model])

    # speed-up
    ax_sp.errorbar(g.num_threads, speedup, yerr=err_sp,
                   marker="o", lw=1.8, ms=5,
                   color=COLORS[model])

    # étiquette d’efficacité η=S/p au-dessus du point
    for x, s in zip(g.num_threads, speedup):
        eta = s / x
        ax_sp.annotate(f"{eta:.2f}", (x, s), textcoords="offset points",
                       xytext=(0, 5), ha="center", fontsize=8, color=COLORS[model])

# ligne idéale
p_vals = np.array(sorted(strong.num_threads.unique()))
ax_sp.plot(p_vals, p_vals, "--", color="grey", lw=1, label="Idéal")

# habillage
ax_rt.set_ylabel("Temps d’entraînement [s]")
ax_rt.set_title("Strong scaling")

ax_sp.set_xlabel("Nombre de threads")
ax_sp.set_ylabel("Speed-up\n(baseline 1 thread)")
ax_sp.set_xlim(0.8)
ax_sp.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

ax_rt.legend(ncol=2, fontsize=9)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(outdir / f"fig01_strong_scaling.{ext}")
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# 4.  WEAK SCALING – runtime
# ──────────────────────────────────────────────────────────────────────────────
weak = df[df.scaling == "weak"].copy()

fig, ax = plt.subplots()
for model, grp in weak.groupby("model"):
    g = grp.sort_values("num_threads")
    ax.errorbar(g.num_threads, g.train_time_mean, yerr=g.train_time_std,
                marker="s", lw=1.8, ms=5,
                color=COLORS[model], label=NICE[model])

ax.set_xlabel("Nombre de threads  (taille × p)")
ax.set_ylabel("Temps d’entraînement [s]")
ax.set_title("Weak scaling – runtime")
ax.set_ylim(bottom=0)
ax.set_xlim(0.8)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.legend(ncol=2, fontsize=9)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(outdir / f"fig02_weak_scaling_runtime.{ext}")
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# 5.  WEAK SCALING – efficacité parallèle
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots()
for model, grp in weak.groupby("model"):
    g = grp.sort_values("num_threads")
    T1 = g[g.num_threads == 1].train_time_mean.iloc[0]
    eff = (T1 / g.train_time_mean) / g.num_threads
    ax.plot(g.num_threads, eff, marker="^", lw=1.8, ms=5,
            color=COLORS[model], label=NICE[model])
    # annotation
    for x, e in zip(g.num_threads, eff):
        ax.annotate(f"{e:.2f}", (x, e), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8)

ax.set_xlabel("Nombre de threads")
ax.set_ylabel("Efficacité  η = S/p")
ax.set_title("Weak scaling – efficacité")
ax.set_ylim(0, 1.05)
ax.set_xlim(0.8)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.legend(ncol=2, fontsize=9)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(outdir / f"fig03_weak_scaling_efficiency.{ext}")
plt.close(fig)

print("✔︎  Figures enregistrées dans :", outdir.resolve())
