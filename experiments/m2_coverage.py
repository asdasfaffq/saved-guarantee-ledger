"""M2 main result: simultaneous coverage under drift + budget reuse.

A stream of T queried steps over a drifting snapshot. Each step the analyst asks
for the recall lower bound of the CURRENT snapshot. We record, per method, whether
its lower bound EVER exceeds the current true recall (an anytime/simultaneous
breach). Coverage = 1 - breach_rate over many seeds. We sweep drift eps.

Methods: FixedNSplitConformal (SUPG/ReDD fixed-n), PooledCS (streaming-ReDD),
WindowedSAVED (SAVED). Also report mean lower-bound (power) so the honest
width/power tradeoff is visible.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.drift import draw_item, true_recall
from saved.baselines import (FixedNSplitConformal, PooledCS, WindowedSAVED,
                              SAVEDDriftRobust)

TOL = 1e-9


def run_seed(seed, eps, delta=0.1, T=800, tau=0.5, mu0=1.6, pos_rate=0.3,
             query_every=25, grid=801):
    rng = np.random.default_rng(seed)
    # SAVED uses the assumed drift bound eps (bounded-TV assumption).
    saved_dr = SAVEDDriftRobust(delta, eps_bound=eps, window=70, grid=grid)
    methods = {
        "FixedN(SUPG/ReDD)": FixedNSplitConformal(delta, k=50),
        "PooledCS(streamReDD)": PooledCS(delta, grid=grid),
        "SAVED-window(no-tilt abl)": WindowedSAVED(delta, window=70, grid=grid),
    }
    breached = {k: False for k in methods}
    breached["SAVED(drift-robust)"] = False
    last_L = {k: 0.0 for k in methods}
    last_L["SAVED(drift-robust)"] = 0.0
    n_q = 0
    for t in range(T):
        t_norm = t / (T - 1)
        g, s = draw_item(t_norm, rng, mu0, eps, pos_rate)
        x = 1.0 if (g == 1 and s >= tau) else 0.0
        if g == 1:  # only oracle-positive items feed the recall estimand
            for m in methods.values():
                m.update(x)
            saved_dr.update(x, t_norm)
        if t % query_every != 0 and t != T - 1:
            continue
        r_now = true_recall(t_norm, tau, mu0, eps)
        n_q += 1
        for name, m in methods.items():
            L = m.lower()
            last_L[name] = L
            if L > r_now + TOL:
                breached[name] = True
        Ldr = saved_dr.lower(t_norm)
        last_L["SAVED(drift-robust)"] = Ldr
        if Ldr > r_now + TOL:
            breached["SAVED(drift-robust)"] = True
    calls = {k: methods[k].calls for k in methods}
    calls["SAVED(drift-robust)"] = saved_dr.calls
    return breached, last_L, calls


def main():
    delta = 0.1
    eps_grid = [0.0, 0.4, 0.8, 1.2]
    n_seeds = 200
    out = {"delta": delta, "n_seeds": n_seeds, "eps_grid": eps_grid, "rows": []}
    print(f"\nM2 coverage under drift+reuse  (delta={delta}, nominal coverage={1-delta:.2f}, "
          f"n_seeds={n_seeds})\n")
    header = f"{'eps':>5} | {'method':<22} | {'coverage':>9} | {'mean_LB(power)':>14} | {'calls':>6}"
    print(header); print("-" * len(header))
    for eps in eps_grid:
        agg = {}
        for seed in range(n_seeds):
            br, lastL, calls = run_seed(seed, eps, delta=delta)
            for name in br:
                a = agg.setdefault(name, {"breach": 0, "lb": 0.0, "calls": 0})
                a["breach"] += int(br[name])
                a["lb"] += lastL[name]
                a["calls"] = calls[name]
        for name, a in agg.items():
            cov = 1.0 - a["breach"] / n_seeds
            mean_lb = a["lb"] / n_seeds
            ok = "OK " if cov >= (1 - delta) - 0.03 else "BREACH"
            print(f"{eps:>5} | {name:<22} | {cov:>9.3f} | {mean_lb:>14.3f} | {a['calls']:>6}  {ok}")
            out["rows"].append({"eps": eps, "method": name, "coverage": cov,
                                 "mean_lb": mean_lb, "calls": a["calls"]})
        print("-" * len(header))
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    p = os.path.join(os.path.dirname(__file__), "..", "results", "m2_coverage.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {os.path.abspath(p)}")


if __name__ == "__main__":
    main()
