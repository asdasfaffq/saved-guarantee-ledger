"""Comparison among VALID drift-aware methods (the reviewer's key missing experiment).

Prior figures mostly show SAVED beating INVALID baselines (stream-ReDD, fixed-n). The
honest question a reviewer asks is: among methods that are actually VALID under drift, does
SAVED give a higher trustworthy certification yield? Here all methods aim to stay valid; we
report coverage (should be >=1-delta), trusted yield (certified-and-true / N), and
false-certificate rate, at a fixed per-query budget over a drifting reused-budget stream.

Valid methods:
  saved      : full reused pool, per-sample drift shift b_i=kappa*age_i (default).
  fresh-only : only current-snapshot fresh labels, no reuse (valid, wastes reuse).
  recent-win : recent-W reused labels, no shift (bounded staleness).
  max-age    : full pool, conservative shift b_i=kappa*max_age.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.recall_cs import RecallCS
from saved.baselines import PHI_MAX
from saved.drift import true_recall

GRID, TAU, MU0, DELTA = 401, 0.5, 1.6, 0.10
R = 0.80            # target; certifiable early, drops below under drift
B, WIN = 30, 70     # per-query fresh budget; recent window


def certify(reuse, fresh, t_now, kappa, policy):
    cs = RecallCS(delta=DELTA, grid=GRID)
    if policy == "fresh":
        use = []
    elif policy == "recent":
        use = sorted(reuse, key=lambda z: z[1])[-WIN:]
    else:
        use = reuse
    maxage = max((t_now - ti for (_, ti) in use), default=0.0)
    for (x, ti) in use:
        if policy == "saved":
            b = kappa * max(0.0, t_now - ti)
        elif policy == "maxage":
            b = kappa * maxage
        else:                      # recent: no shift
            b = 0.0
        cs.update_shift(x, 1.0, b)
    used = 0
    while cs.lower() < R and used < B:
        cs.update(fresh[used]); used += 1
    return cs.lower() >= R, used


def run(eps, methods, N=25, n_seeds=100):
    kappa = eps * PHI_MAX
    agg = {m: {"true": 0, "false": 0, "cert": 0} for m in methods}
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        pool = []
        for q in range(N):
            t = q / (N - 1)
            r_now = true_recall(t, TAU, MU0, eps)
            good = r_now >= R - 1e-9
            fresh = (rng.random(B) < r_now).astype(float)
            for m in methods:
                cert, _ = certify(pool, fresh, t, kappa, m)
                if cert:
                    agg[m]["cert"] += 1
                    agg[m]["true" if good else "false"] += 1
            for x in fresh:
                pool.append((float(x), t))
    tot = N * n_seeds
    return {m: {"trusted_yield": agg[m]["true"] / tot,
                "false_rate": agg[m]["false"] / tot,
                "cert_rate": agg[m]["cert"] / tot} for m in methods}


def main():
    methods = ["saved", "fresh", "recent", "maxage"]
    labs = {"saved": "SAVED (per-sample)", "fresh": "fresh-only",
            "recent": "recent-window", "maxage": "max-age"}
    epss = [0.0, 0.4, 0.8]
    out = {"delta": DELTA, "R": R, "B": B, "cells": []}
    print(f"\nAmong VALID methods: trusted yield / false-cert (delta={DELTA}, target={R}, "
          f"budget B={B}/query, drifting reused stream)\n")
    hdr = f"{'eps':>4} | " + " | ".join(f"{labs[m]:>18}" for m in methods)
    print(hdr); print("-" * len(hdr))
    print("      " + "   ".join("(yield / false)".rjust(16) for _ in methods))
    for eps in epss:
        res = run(eps, methods)
        cells = " | ".join(f"{res[m]['trusted_yield']:.2f} / {res[m]['false_rate']:.3f}".rjust(18)
                           for m in methods)
        print(f"{eps:>4} | {cells}")
        out["cells"].append({"eps": eps, **res})
    p = os.path.join(os.path.dirname(__file__), "..", "results", "valid_baselines.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nsaved -> {os.path.abspath(p)}")


if __name__ == "__main__":
    main()
