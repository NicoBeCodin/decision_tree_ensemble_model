#!/usr/bin/env python3
import pandas as pd, sys, pathlib as pl

src = pl.Path(sys.argv[1] if len(sys.argv) > 1 else "scaling_results.csv")
dst = src.with_stem(src.stem + "_nompi")

df = pd.read_csv(src)
out = df[df.mpi_rank <= 1]            # <=1  ⇢ on garde 0 ou 1   
out.to_csv(dst, index=False)

print(f"✔︎  {len(df)-len(out)} lignes retirées – nouveau fichier : {dst}")
