"""Diagnostic (main-agent): is the B3c eps=0 'win' a robust edge or a gamma=0.04
tuning artifact? Sweep gamma at the winning cells (eps=0) and a drift cell (eps=0.2).
A legitimate cost win should (a) not require an extreme hand-tuned gamma and
(b) ideally appear where drift actually exists. gamma=1.0 == pooled+inflation.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import experiments.b3c_weighted as B
from experiments.b3c_weighted import run_seed, DELTA

def sweep(ell, eps, gammas, n_seeds=100):
    print(f"\n--- ell={ell} eps={eps} (n={n_seeds}) ---")
    print(f"{'gamma':>6} | {'pooled v/fr':>12} | {'SAVEDw v/fr':>12} | win vs pooled?")
    for g in gammas:
        B.GAMMA = g
        agg = {m: {"fresh":0,"breach":0} for m in ("pooled","SAVEDw")}
        for s in range(n_seeds):
            tot = run_seed(s, eps, ell)
            for m in agg:
                agg[m]["fresh"] += tot[m]["fresh"]; agg[m]["breach"] += int(tot[m]["breach"]>0)
        r = {m:{"valid":(agg[m]["breach"]/n_seeds)<=DELTA,"fresh":agg[m]["fresh"]/n_seeds} for m in agg}
        win = r["pooled"]["valid"] and r["SAVEDw"]["valid"] and r["SAVEDw"]["fresh"] < r["pooled"]["fresh"]-1e-9
        print(f"{g:>6} | {str(r['pooled']['valid'])[0]:>3}/{r['pooled']['fresh']:>7.0f} | "
              f"{str(r['SAVEDw']['valid'])[0]:>3}/{r['SAVEDw']['fresh']:>7.0f} | {win}")

if __name__ == "__main__":
    gammas = [1.0, 0.7, 0.5, 0.2, 0.04]
    sweep(0.9, 0.0, gammas)
    sweep(0.5, 0.0, gammas)
    sweep(0.9, 0.2, gammas)
