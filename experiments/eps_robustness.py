"""eps-misspecification robustness (load-bearing caveat).

SAVED's drift-inflation uses an assumed drift bound eps_bound. In M2 it was
handed the TRUE eps. Here we sweep eps_bound = c * eps_true and report coverage
+ power (mean lower bound) vs c. Expectation:
  c < 1 (under-specify drift) -> inflation too small -> coverage BREAKS.
  c > 1 (over-specify)        -> inflation too large -> valid but power lost.
This quantifies the fragility and must be a headline caveat, not buried.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.drift import draw_item, true_recall
from saved.baselines import SAVEDDriftRobust

TOL = 1e-9


def run_seed(seed, eps_true, c, delta=0.1, T=800, tau=0.5, mu0=1.6,
             pos_rate=0.3, query_every=25, grid=801, window=70):
    rng = np.random.default_rng(seed)
    saved = SAVEDDriftRobust(delta, eps_bound=c * eps_true, window=window, grid=grid)
    breached = False
    last_L = 0.0
    for t in range(T):
        t_norm = t / (T - 1)
        g, s = draw_item(t_norm, rng, mu0, eps_true, pos_rate)
        if g == 1:
            saved.update(1.0 if s >= tau else 0.0, t_norm)
        if t % query_every != 0 and t != T - 1:
            continue
        r_now = true_recall(t_norm, tau, mu0, eps_true)
        L = saved.lower(t_norm)
        last_L = L
        if L > r_now + TOL:
            breached = True
    return breached, last_L


def main():
    delta = 0.1
    cs = [0.5, 0.75, 1.0, 1.5, 2.0]
    eps_list = [0.4, 0.8]
    n_seeds = 200
    out = {"delta": delta, "n_seeds": n_seeds, "rows": []}
    print(f"\neps-misspecification robustness  (delta={delta}, nominal coverage={1-delta:.2f}, "
          f"n_seeds={n_seeds})")
    print("c = eps_bound / eps_true ;  c<1 under-specify, c>1 over-specify\n")
    header = f"{'eps_true':>8} | {'c':>4} | {'coverage':>9} | {'mean_LB(power)':>14}"
    print(header); print("-" * len(header))
    for eps in eps_list:
        for c in cs:
            br = 0; lb = 0.0
            for seed in range(n_seeds):
                b, L = run_seed(seed, eps, c, delta=delta)
                br += int(b); lb += L
            cov = 1.0 - br / n_seeds
            mean_lb = lb / n_seeds
            flag = "OK " if cov >= (1 - delta) - 0.03 else "BREACH"
            print(f"{eps:>8} | {c:>4} | {cov:>9.3f} | {mean_lb:>14.3f}  {flag}")
            out["rows"].append({"eps_true": eps, "c": c, "coverage": cov, "mean_lb": mean_lb})
        print("-" * len(header))
    p = os.path.join(os.path.dirname(__file__), "..", "results", "eps_robustness.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {os.path.abspath(p)}")


if __name__ == "__main__":
    main()
