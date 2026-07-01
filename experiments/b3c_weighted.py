"""B3c — the REAL SAVED mechanism: drift-REWEIGHTED FULL-POOL confidence sequence.

The windowed prototype (B3/B3b) hard-discards data => weak power and even fragile
validity. The proposed SAVED uses the FULL applicable pool with a recency weight
w_i = gamma^(age_i) (predictable => still anytime-valid via update_w), minus a
drift-inflation eps_bound*PHI_MAX*(weighted mean age). It keeps far more effective
data than a window while staying valid under bounded drift.

Honest head-to-head at the same per-query cap, under drift + reuse:
  - pooled  : full pool, NO weighting, NO inflation (stream-ReDD; invalid under drift).
  - perquery: fresh current-snapshot only, no reuse (valid; the honest drift baseline).
  - SAVEDw  : reweighted full pool + inflation (proposed; should be valid AND cheaper).
Metrics: validity (breach-seed-rate<=delta), mean fresh labels, correct certs.
C2 WIN := SAVEDw valid AND fresh < strongest-valid-baseline fresh.
Lets the sweet spot be empty.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.recall_cs import RecallCS
from saved.baselines import PHI_MAX
from saved.drift import true_recall

GRID = 401
TAU = 0.5
MU0 = 1.6
TARGET = 0.80
CAP = 120
DELTA = 0.10
GAMMA = 0.04   # recency decay base: w = GAMMA**age_norm (age in [0,1]); strong decay


def certify_unweighted(reuse_xs, fresh):
    cs = RecallCS(delta=DELTA, grid=GRID)
    for x in reuse_xs:
        cs.update(x)
    used = 0
    while cs.lower() < TARGET and used < CAP:
        cs.update(fresh[used]); used += 1
    return cs.lower() >= TARGET, used


def certify_weighted(reuse, t_now, eps_bound, fresh):
    """reuse = list of (x, t_i). Recency-weight, seed once, then fresh (w=1)."""
    cs = RecallCS(delta=DELTA, grid=GRID)
    sw = 0.0; swa = 0.0
    for (x, ti) in reuse:
        age = max(0.0, t_now - ti)
        w = GAMMA ** age
        cs.update_w(x, w)
        sw += w; swa += w * age
    used = 0
    def infl():
        wage = (swa / sw) if sw > 0 else 0.0
        return eps_bound * PHI_MAX * wage
    L = cs.lower() - infl()
    while L < TARGET and used < CAP:
        cs.update(fresh[used]); used += 1  # fresh = current snapshot, w=1, age 0
        L = cs.lower() - infl()
    return L >= TARGET, used


def run_seed(seed, eps, ell, N=30):
    rng = np.random.default_rng(seed)
    pool = []
    tot = {m: {"fresh": 0, "correct": 0, "breach": 0} for m in ("pooled", "perquery", "SAVEDw")}
    for q in range(N):
        t = q / (N - 1)
        r_q = true_recall(t, TAU, MU0, eps)
        certifiable = r_q >= TARGET - 1e-9
        if pool:
            mask = rng.random(len(pool)) < ell
            applicable = [pool[i] for i in range(len(pool)) if mask[i]]
        else:
            applicable = []
        fresh = (rng.random(CAP) < r_q).astype(float)
        cert_p, used_p = certify_unweighted([x for (x, _) in applicable], fresh)
        cert_r, used_r = certify_unweighted([], fresh)
        cert_s, used_s = certify_weighted(applicable, t, eps, fresh)
        for name, cert, used in (("pooled", cert_p, used_p), ("perquery", cert_r, used_r),
                                  ("SAVEDw", cert_s, used_s)):
            tot[name]["fresh"] += used
            if cert:
                tot[name]["correct" if certifiable else "breach"] += 1
        kmax = max(used_p, used_r, used_s)
        for j in range(kmax):
            pool.append((float(fresh[j]), t))
    return tot


def main():
    ells = [0.5, 0.9]
    epss = [0.0, 0.2, 0.4, 0.8]
    n_seeds = 100
    out = {"delta": DELTA, "target": TARGET, "gamma": GAMMA, "n_seeds": n_seeds, "cells": []}
    print(f"\nB3c REWEIGHTED full-pool SAVED (delta={DELTA}, target={TARGET}, gamma={GAMMA}, "
          f"N=30, seeds={n_seeds})\n")
    hdr = (f"{'ell':>4} {'eps':>4} | {'pooled v/fr':>13} | {'perQ v/fr':>11} "
           f"| {'SAVEDw v/fr':>13} | {'strongest valid':>15} | {'verdict':>16}")
    print(hdr); print("-" * len(hdr))
    any_c2 = False
    for ell in ells:
        for eps in epss:
            agg = {m: {"fresh": 0, "breach_seeds": 0} for m in ("pooled", "perquery", "SAVEDw")}
            for seed in range(n_seeds):
                tot = run_seed(seed, eps, ell)
                for m in agg:
                    agg[m]["fresh"] += tot[m]["fresh"]
                    agg[m]["breach_seeds"] += int(tot[m]["breach"] > 0)
            res = {m: {"valid": (agg[m]["breach_seeds"] / n_seeds) <= DELTA,
                       "fresh": agg[m]["fresh"] / n_seeds} for m in agg}
            if res["pooled"]["valid"]:
                sv, svf = "pooled", res["pooled"]["fresh"]
            elif res["perquery"]["valid"]:
                sv, svf = "perquery", res["perquery"]["fresh"]
            else:
                sv, svf = "none", float("inf")
            if not res["SAVEDw"]["valid"]:
                verdict = "SAVEDw invalid"
            elif sv == "none":
                verdict = "no valid base"
            elif res["SAVEDw"]["fresh"] < svf - 1e-9:
                verdict = f"C2 WIN vs {sv}"; any_c2 = True
            else:
                verdict = f"no win ({sv}<=)"
            vf = lambda m: f"{str(res[m]['valid'])[0]}/{res[m]['fresh']:.0f}"
            print(f"{ell:>4} {eps:>4} | {vf('pooled'):>13} | {vf('perquery'):>11} "
                  f"| {vf('SAVEDw'):>13} | {sv:>15} | {verdict:>16}")
            out["cells"].append({"ell": ell, "eps": eps, **{k: res[k] for k in res},
                                  "strongest_valid": sv, "verdict": verdict})
        print("-" * len(hdr))
    out["c2_sweet_spot_nonempty"] = any_c2
    print(f"\nC2 sweet spot non-empty? {any_c2}")
    p = os.path.join(os.path.dirname(__file__), "..", "results", "b3c_weighted.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {os.path.abspath(p)}")


if __name__ == "__main__":
    main()
