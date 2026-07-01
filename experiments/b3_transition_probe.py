"""Verification probe (main-agent): fine ε grid across the pooled valid->invalid
transition at high locality. This is the ONLY place a C2 sweet spot could hide:
a band where pooled-CS is still just-valid but loosening, while SAVED is valid
and cheaper. If no such cell exists here, the empty-sweet-spot verdict is robust.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.b3_amortization import run_seed, DELTA

def main():
    n_seeds = 160
    ells = [0.9, 0.7, 0.5]
    epss = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    print(f"\nB3 TRANSITION PROBE (delta={DELTA}, n_seeds={n_seeds}) — hunting a hidden sweet spot")
    hdr = f"{'ell':>4} {'eps':>5} | {'p_valid':>7} {'p_fresh':>8} | {'s_valid':>7} {'s_fresh':>8} | verdict"
    print(hdr); print("-"*len(hdr))
    found = False
    for ell in ells:
        for eps in epss:
            agg = {m: {"fresh":0,"correct":0,"breach_seeds":0} for m in ("pooled","saved")}
            for seed in range(n_seeds):
                tot = run_seed(seed, eps, ell)
                for m in agg:
                    agg[m]["fresh"] += tot[m]["fresh"]
                    agg[m]["breach_seeds"] += int(tot[m]["breach"]>0)
            r = {}
            for m in ("pooled","saved"):
                r[m] = {"valid": (agg[m]["breach_seeds"]/n_seeds)<=DELTA,
                        "fresh": agg[m]["fresh"]/n_seeds}
            if not r["pooled"]["valid"]:
                v = "VOID(pooled invalid)"
            elif not r["saved"]["valid"]:
                v = "saved invalid"
            elif r["saved"]["fresh"] < r["pooled"]["fresh"] - 1e-9:
                v = "*** C2 WIN ***"; found = True
            else:
                v = "no win"
            print(f"{ell:>4} {eps:>5} | {str(r['pooled']['valid']):>7} {r['pooled']['fresh']:>8.1f} "
                  f"| {str(r['saved']['valid']):>7} {r['saved']['fresh']:>8.1f} | {v}")
        print("-"*len(hdr))
    print(f"\nHIDDEN C2 SWEET SPOT FOUND? {found}")

if __name__ == "__main__":
    main()
