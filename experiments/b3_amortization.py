"""B3 — label-amortization sweet spot (C2 go/no-go gate).

Honest, falsifiable design (built to LET THE SWEET SPOT BE EMPTY):
- A workload of N queries over a (possibly drifting) snapshot reuses one shared
  label pool. Each query must certify recall >= TARGET at level delta by drawing
  oracle labels (each = +1 cost) until its lower bound >= TARGET, reusing
  applicable pooled labels for free. Locality `ell` = prob a pooled label is
  applicable/reusable for the next query (inter-query selected-set overlap).
- TARGET is satisfiable for EARLY snapshots but, under drift, the LATE snapshot
  true recall falls below TARGET -> those queries must be REFUSED. A method that
  still certifies them commits a false certification (BREACH = invalid).
- Cost is counted ONLY among methods VALID in a cell. Cost = total fresh oracle
  labels over the stream (reused labels are free => already net of ESS). We also
  report certified-queries-per-oracle-call (correct certifications only).
- C2 win cell := pooled-CS (stream-ReDD) is VALID there AND SAVED uses strictly
  fewer labels / higher certified-per-call. Cells where pooled is INVALID are
  marked VOID (a "win" there is just C1 validity, not a cost win).

Methods:
- pooled  : streaming-ReDD. Betting CS over ALL applicable pooled + fresh labels,
            no drift inflation (uses maximal data => most power, but stale under drift).
- saved   : SAVED. Betting CS over recent-W applicable + fresh, MINUS drift inflation.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.recall_cs import RecallCS
from saved.baselines import PHI_MAX
from saved.drift import true_recall

TOL = 1e-9
GRID = 401
TAU = 0.5
MU0 = 1.6
TARGET = 0.80          # certifiable at eps=0 (true recall = Phi(1.1)=0.864)
CAP = 120              # per-query fresh-label cap
WINDOW = 70
DELTA = 0.10


def _certify(reuse_xs, fresh, inflation_fn):
    """Feed reuse (free), then draw fresh until lower-inflation >= TARGET or CAP.
    Returns (certified, used_fresh)."""
    cs = RecallCS(delta=DELTA, grid=GRID)
    for x in reuse_xs:
        cs.update(x)
    used = 0
    L = cs.lower() - inflation_fn()
    while L < TARGET and used < CAP:
        cs.update(fresh[used]); used += 1
        L = cs.lower() - inflation_fn()
    return (L >= TARGET), used


def run_seed(seed, eps, ell, N=30, eps_bound=None):
    rng = np.random.default_rng(seed)
    eps_bound = eps if eps_bound is None else eps_bound
    pool = []  # list of (x_indicator, t_norm)
    tot = {m: {"fresh": 0, "correct": 0, "breach": 0, "refused": 0}
           for m in ("pooled", "saved")}
    for q in range(N):
        t = q / (N - 1)
        r_q = true_recall(t, TAU, MU0, eps)
        certifiable = r_q >= TARGET - 1e-9
        # applicable reuse from the shared pool (Bernoulli(ell) per pooled label)
        if pool:
            mask = rng.random(len(pool)) < ell
            applicable = [pool[i] for i in range(len(pool)) if mask[i]]
        else:
            applicable = []
        fresh = (rng.random(CAP) < r_q).astype(float)  # indicators at CURRENT snapshot

        # pooled: all applicable, no inflation
        pooled_reuse = [x for (x, _) in applicable]
        cert_p, used_p = _certify(pooled_reuse, fresh, lambda: 0.0)

        # saved: recent-W applicable + inflation over t-span
        appl_sorted = sorted(applicable, key=lambda z: z[1])[-WINDOW:]
        saved_reuse = [x for (x, _) in appl_sorted]
        tmin = min((tt for (_, tt) in appl_sorted), default=t)
        infl = lambda: eps_bound * PHI_MAX * max(0.0, t - tmin)
        cert_s, used_s = _certify(saved_reuse, fresh, infl)

        for name, cert, used in (("pooled", cert_p, used_p), ("saved", cert_s, used_s)):
            tot[name]["fresh"] += used
            if cert:
                if certifiable:
                    tot[name]["correct"] += 1
                else:
                    tot[name]["breach"] += 1  # false certification (invalid)
            else:
                tot[name]["refused"] += 1
        # grow shared pool with labels physically drawn at this snapshot
        kmax = max(used_p, used_s)
        for j in range(kmax):
            pool.append((float(fresh[j]), t))
    return tot


def main():
    ells = [0.0, 0.5, 0.9]
    epss = [0.0, 0.4, 0.8]
    n_seeds = 120
    out = {"delta": DELTA, "target": TARGET, "n_seeds": n_seeds, "cells": []}
    print(f"\nB3 label-amortization gate (delta={DELTA}, target recall={TARGET}, "
          f"N=30 queries, n_seeds={n_seeds})")
    print("Cost = total fresh oracle labels over stream (reused=free => net-of-ESS).")
    print("VALID := breach-seed-rate <= delta. C2 cost comparison only where pooled VALID.\n")
    hdr = (f"{'ell':>4} {'eps':>4} | {'pooled:valid':>12} {'p_fresh':>8} {'p_cpc':>6} "
           f"| {'saved:valid':>11} {'s_fresh':>8} {'s_cpc':>6} | {'C2 verdict':>22}")
    print(hdr); print("-" * len(hdr))
    any_c2 = False
    for ell in ells:
        for eps in epss:
            agg = {m: {"fresh": 0, "correct": 0, "breach_seeds": 0} for m in ("pooled", "saved")}
            for seed in range(n_seeds):
                tot = run_seed(seed, eps, ell)
                for m in agg:
                    agg[m]["fresh"] += tot[m]["fresh"]
                    agg[m]["correct"] += tot[m]["correct"]
                    agg[m]["breach_seeds"] += int(tot[m]["breach"] > 0)
            res = {}
            for m in ("pooled", "saved"):
                valid = (agg[m]["breach_seeds"] / n_seeds) <= DELTA
                fresh = agg[m]["fresh"] / n_seeds
                cpc = agg[m]["correct"] / max(1, agg[m]["fresh"])  # certified-per-oracle-call
                res[m] = {"valid": valid, "fresh": fresh, "cpc": cpc,
                          "breach_rate": agg[m]["breach_seeds"] / n_seeds}
            # C2 verdict
            if not res["pooled"]["valid"]:
                verdict = "VOID(pooled invalid)"
            elif not res["saved"]["valid"]:
                verdict = "saved invalid"
            elif res["saved"]["fresh"] < res["pooled"]["fresh"] - 1e-9 and \
                    res["saved"]["cpc"] > res["pooled"]["cpc"] + 1e-9:
                verdict = "C2 WIN"; any_c2 = True
            else:
                verdict = "no win (pooled <=)"
            print(f"{ell:>4} {eps:>4} | {str(res['pooled']['valid']):>12} "
                  f"{res['pooled']['fresh']:>8.1f} {res['pooled']['cpc']:>6.3f} "
                  f"| {str(res['saved']['valid']):>11} {res['saved']['fresh']:>8.1f} "
                  f"{res['saved']['cpc']:>6.3f} | {verdict:>22}")
            out["cells"].append({"ell": ell, "eps": eps, "pooled": res["pooled"],
                                  "saved": res["saved"], "verdict": verdict})
        print("-" * len(hdr))
    out["c2_sweet_spot_nonempty"] = any_c2
    print(f"\nC2 sweet spot non-empty? {any_c2}")
    p = os.path.join(os.path.dirname(__file__), "..", "results", "b3_amortization.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {os.path.abspath(p)}")


if __name__ == "__main__":
    main()
