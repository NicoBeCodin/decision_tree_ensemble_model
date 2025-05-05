#!/usr/bin/env python3
"""
fig_bagging_mpi_strong.png / .pdf
 - runtime   (haut)
 - speed-up  (bas)
à partir d’un CSV du type :

scaling,mpi_ranks,omp_threads,n_estimators,max_depth,criteria,...
strong,1,1,200,10,0,0,27.53, ...
strong,1,2,200,10,0,0,14.04, ...
strong,2,1,200,10,0,0,27.84, ...
...
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.ticker as mt

# ───────────────────────────
# 1.  Paramètres d’apparence
# ───────────────────────────
plt.rcParams.update({
    "figure.figsize": (7.0, 4.6),
    "savefig.dpi": 320,
    "font.size": 11,
    "axes.spines.right": False,
    "axes.spines.top":   False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.55,
    "legend.frameon": False,
})

PALETTE = ["#332288", "#117733", "#44AA99", "#88CCEE", "#CC6677", "#AA4499"]

# ───────────────────────────
# 2.  Lecture du CSV
# ───────────────────────────
CSV = Path("bagging_hybrid_scaling.csv")          # ← adapte le nom si besoin
if not CSV.exists():
    raise SystemExit(f"❌  {CSV} introuvable")

df = pd.read_csv(CSV)
strong = df[df.scaling == "strong"].copy()

# ───────────────────────────
# 3.  Figure runtime + speed-up
# ───────────────────────────
fig, (ax_rt, ax_sp) = plt.subplots(
    2, 1, sharex=True, gridspec_kw={"height_ratios": (1.25, 1)}
)

for idx, (ranks, grp) in enumerate(strong.groupby("mpi_ranks")):
    g = grp.sort_values("omp_threads")
    base = g[g.omp_threads == 1].train_time_mean.iloc[0]
    speedup = base / g.train_time_mean
    err_rt  = g.train_time_std
    err_sp  = speedup * (g.train_time_std / g.train_time_mean)

    color = PALETTE[idx % len(PALETTE)]
    label = f"{ranks} proc."

    # Temps d’entraînement
    ax_rt.errorbar(
        g.omp_threads, g.train_time_mean, yerr=err_rt,
        marker="o", lw=1.8, ms=5, color=color, label=label
    )

    # Speed-up
    ax_sp.errorbar(
        g.omp_threads, speedup, yerr=err_sp,
        marker="o", lw=1.8, ms=5, color=color
    )

    # Affiche l’efficacité η = S/p
    for x, s in zip(g.omp_threads, speedup):
        eta = s / x
        ax_sp.annotate(f"{eta:.2f}", (x, s), textcoords="offset points",
                       xytext=(0, 5), ha="center", fontsize=8, color=color)

# Ligne idéale (1 procès) — même abscisse pour comparaison
p_vals = np.sort(strong.omp_threads.unique())
ax_sp.plot(p_vals, p_vals, "--", color="grey", lw=1, label="Idéal 1 proc.")

# ───────────────
# 4.  Habillage
# ───────────────
ax_rt.set_ylabel("Temps d’entraînement [s]")
ax_rt.set_title("Bagging MPI – strong scaling")

ax_sp.set_xlabel("Threads OpenMP par processus")
ax_sp.set_ylabel("Speed-up\n(baseline = 1 thread)")
ax_sp.set_xlim(0.8)
ax_sp.xaxis.set_major_locator(mt.MaxNLocator(integer=True))

ax_rt.legend(ncol=3, fontsize=9)
fig.tight_layout()

# ───────────────
# 5.  Export
# ───────────────
out = Path("figures")
out.mkdir(exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(out / f"fig_bagging_mpi_strong.{ext}")

print("✔︎  Figure enregistrée :", out / "fig_bagging_mpi_strong.png")