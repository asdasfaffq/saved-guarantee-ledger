"""B3b — amortization vs the STRONGEST VALID baseline per regime (honest C2).

Fix to B3: cost may only be compared among methods VALID in a cell. The strongest
valid baseline DEPENDS ON THE REGIME:
  - low/no drift  : pooled-CS (stream-ReDD) is valid and uses all data => strongest.
  - under drift   : pooled-CS / SUPG / LOTUS / ReDD are INVALID (false certs), so the
                    strongest VALID baseline is PER-QUERY RECALIBRATION (draw fresh
                    labels from the current snapshot each query, no cross-query reuse).
SAVED's claim is an amortization win in the DRIFT regime, where it is valid and reuses
labels across queries, against per-query-recal (the only other valid option there).

Verdict per (locality ell x drift eps) cell:
  strongest_valid = pooled if pooled valid else (per-query-recal if valid else none)
  C2 WIN := SAVED valid AND SAVED fresh-labels < strongest_valid fresh-labels.
This still LETS THE SWEET SPOT BE EMPTY (if SAVED never beats the strongest valid baseline).
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
TARGET = 0.80
CAP = 120
WINDOW = 70
DELTA = 0.10


def _certify(reuse_xs, fresh, inflation_fn):
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
    pool = []
    tot = {m: {"fresh": 0, "correct": 0, "breach": 0}
           for m in ("pooled", "perquery", "saved")}
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

        # pooled: all applicable, no inflation (invalid under drift)
        cert_p, used_p = _certify([x for (x, _) in applicable], fresh, lambda: 0.0)
        # per-query recal: NO reuse, only current-snapshot fresh, no inflation (valid)
        cert_r, used_r = _certify([], fresh, lambda: 0.0)
        # SAVED: recent-W applicable + drift inflation (valid)
        appl_sorted = sorted(applicable, key=lambda z: z[1])[-WINDOW:]
        tmin = min((tt for (_, tt) in appl_sorted), default=t)
        cert_s, used_s = _certify([x for (x, _) in appl_sorted], fresh,
                                  lambda: eps_bound * PHI_MAX * max(0.0, t - tmin))

        for name, cert, used in (("pooled", cert_p, used_p),
                                  ("perquery", cert_r, used_r),
                                  ("saved", cert_s, used_s)):
            tot[name]["fresh"] += used
            if cert:
                if certifiable:
                    tot[name]["correct"] += 1
                else:
                    tot[name]["breach"] += 1
        kmax = max(used_p, used_r, used_s)
        for j in range(kmax):
            pool.append((float(fresh[j]), t))
    return tot


def main():
    ells = [0.0, 0.5, 0.9]
    epss = [0.0, 0.4, 0.8]
    n_seeds = 120
    out = {"delta": DELTA, "target": TARGET, "n_seeds": n_seeds, "cells": []}
    print(f"\nB3b amortization vs strongest VALID baseline (delta={DELTA}, target={TARGET}, "
          f"N=30, seeds={n_seeds})")
    print("valid := breach-seed-rate <= delta. fresh = mean total fresh oracle labels.\n")
    hdr = (f"{'ell':>4} {'eps':>4} | {'pooled(v/fresh)':>16} | {'perQ(v/fresh)':>15} "
           f"| {'SAVED(v/fresh)':>15} | {'strongest valid':>15} | {'C2 verdict':>16}")
    print(hdr); print("-" * len(hdr))
    any_c2 = False
    for ell in ells:
        for eps in epss:
            agg = {m: {"fresh": 0, "breach_seeds": 0} for m in ("pooled", "perquery", "saved")}
            for seed in range(n_seeds):
                tot = run_seed(seed, eps, ell)
                for m in agg:
                    agg[m]["fresh"] += tot[m]["fresh"]
                    agg[m]["breach_seeds"] += int(tot[m]["breach"] > 0)
            res = {}
            for m in agg:
                valid = (agg[m]["breach_seeds"] / n_seeds) <= DELTA
                res[m] = {"valid": valid, "fresh": agg[m]["fresh"] / n_seeds}
            # strongest valid baseline (exclude SAVED)
            if res["pooled"]["valid"]:
                sv, sv_fresh = "pooled", res["pooled"]["fresh"]
            elif res["perquery"]["valid"]:
                sv, sv_fresh = "perquery", res["perquery"]["fresh"]
            else:
                sv, sv_fresh = "none", float("inf")
            if not res["saved"]["valid"]:
                verdict = "SAVED invalid"
            elif sv == "none":
                verdict = "no valid baseline"
            elif res["saved"]["fresh"] < sv_fresh - 1e-9:
                verdict = f"C2 WIN vs {sv}"; any_c2 = True
            else:
                verdict = f"no win ({sv}<=)"

            def vf(m):
                return f"{str(res[m]['valid'])[0]}/{res[m]['fresh']:.0f}"
            print(f"{ell:>4} {eps:>4} | {vf('pooled'):>16} | {vf('perquery'):>15} "
                  f"| {vf('saved'):>15} | {sv:>15} | {verdict:>16}")
            out["cells"].append({"ell": ell, "eps": eps,
                                  "pooled": res["pooled"], "perquery": res["perquery"],
                                  "saved": res["saved"], "strongest_valid": sv,
                                  "verdict": verdict})
        print("-" * len(hdr))
    out["c2_sweet_spot_nonempty"] = any_c2
    print(f"\nC2 sweet spot (vs strongest VALID baseline) non-empty? {any_c2}")
    p = os.path.join(os.path.dirname(__file__), "..", "results", "b3b_amortization_valid.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {os.path.abspath(p)}")


if __name__ == "__main__":
    main()
