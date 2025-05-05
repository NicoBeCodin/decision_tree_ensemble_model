#!/usr/bin/env python3
"""
Bagging MPI – WEAK SCALING
==========================

Produit deux figures HD ( PNG + PDF ) :
  • fig_bagging_mpi_weak_runtime
  • fig_bagging_mpi_weak_efficiency

Chaque courbe ≙ un nombre de processus MPI (1‒6).
Abscisse = threads OpenMP (p) utilisés par *chaque* processus.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.ticker as mt

# ───────────────────────────
# 1.  Apparence générale
# ───────────────────────────
plt.rcParams.update({
    "figure.figsize": (7.0, 4.4),
    "savefig.dpi": 320,
    "font.size": 11,
    "axes.spines.right": False,
    "axes.spines.top":   False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.55,
    "legend.frameon": False,
})

COLORS = ["#332288", "#117733", "#44AA99", "#88CCEE", "#CC6677", "#AA4499"]

# ───────────────────────────
# 2.  Lecture du CSV
# ───────────────────────────
CSV = Path("bagging_hybrid_scaling.csv")          # ← adapte si besoin
if not CSV.exists():
    raise SystemExit(f"❌  {CSV} introuvable")

df = pd.read_csv(CSV)
weak = df[df.scaling == "weak"].copy()

out = Path("figures")
out.mkdir(exist_ok=True)

# ───────────────────────────
# 3.  Figure RUNTIME
# ───────────────────────────
fig_rt, ax_rt = plt.subplots()

for idx, (ranks, grp) in enumerate(weak.groupby("mpi_ranks")):
    g = grp.sort_values("omp_threads")
    ax_rt.errorbar(
        g.omp_threads, g.train_time_mean, yerr=g.train_time_std,
        marker="s", lw=1.8, ms=5, color=COLORS[idx % len(COLORS)],
        label=f"{ranks} proc."
    )

ax_rt.set_xlabel("Threads OpenMP  (taille × p)")
ax_rt.set_ylabel("Temps d’entraînement [s]")
ax_rt.set_title("Bagging MPI – weak scaling : runtime")
ax_rt.set_xlim(0.8)
ax_rt.set_ylim(bottom=0)
ax_rt.xaxis.set_major_locator(mt.MaxNLocator(integer=True))
ax_rt.legend(ncol=3, fontsize=9)
fig_rt.tight_layout()

for ext in ("png", "pdf"):
    fig_rt.savefig(out / f"fig_bagging_mpi_weak_runtime.{ext}")
plt.close(fig_rt)

# ───────────────────────────
# 4.  Figure EFFICACITÉ
# ───────────────────────────
fig_eff, ax_eff = plt.subplots()

for idx, (ranks, grp) in enumerate(weak.groupby("mpi_ranks")):
    g  = grp.sort_values("omp_threads")
    T1 = g[g.omp_threads == 1].train_time_mean.iloc[0]     # baseline même n_ranks
    eta = (T1 / g.train_time_mean)                        # η = T1 / Tp

    ax_eff.plot(
        g.omp_threads, eta, marker="^", lw=1.8, ms=5,
        color=COLORS[idx % len(COLORS)], label=f"{ranks} proc."
    )
    # annotation de la valeur
    for x, e in zip(g.omp_threads, eta):
        ax_eff.annotate(f"{e:.2f}", (x, e), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=8)

ax_eff.set_xlabel("Threads OpenMP  (p)")
ax_eff.set_ylabel("Efficacité η = T₁ / Tₚ")
ax_eff.set_title("Bagging MPI – weak scaling : efficacité")
ax_eff.set_ylim(0, 1.05)
ax_eff.set_xlim(0.8)
ax_eff.xaxis.set_major_locator(mt.MaxNLocator(integer=True))
ax_eff.legend(ncol=3, fontsize=9)
fig_eff.tight_layout()

for ext in ("png", "pdf"):
    fig_eff.savefig(out / f"fig_bagging_mpi_weak_efficiency.{ext}")
plt.close(fig_eff)

print("✔︎  Figures enregistrées dans :", out.resolve())