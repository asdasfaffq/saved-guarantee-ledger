"""Clean drift-correction ablation (fixes the epsilon=0 inconsistency).

The earlier 'no drift-shift' row was a WINDOWED variant, so it also changed the
estimator (window vs full pool), confounding the comparison. Here EVERY variant
uses the SAME full-pool adaptive-lambda betting CS; only the per-sample shift b_i
changes. At eps=0 (kappa=0) all shift policies coincide (b_i=0), so 'none' and
'per-sample' MUST match -- exactly the sanity check the earlier figure failed.

Variants (all on the identical full-pool adaptive-EB CS):
  none       : b_i = 0                       (unshifted)
  per-sample : b_i = kappa * age_i           (Prop 3a, default)
  max-age    : b_i = kappa * max_age         (conservative, valid)
  mean-age   : report L_unshifted - kappa*mean_age  (aggregate heuristic, 3b)
  fixed-lam  : per-sample but lambda fixed = 0.5
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.recall_cs import RecallCS
from saved.baselines import PHI_MAX
from saved.drift import draw_item, true_recall

TOL = 1e-9


def lower(samples, t_now, kappa, policy, delta=0.1, grid=401):
    if not samples:
        return 0.0
    lam = 0.5 if policy == "fixedlam" else "adaptive"
    cs = RecallCS(delta=delta, lam=lam, grid=grid)
    ages = [max(0.0, t_now - ti) for (_, ti) in samples]
    if policy in ("persample", "fixedlam"):
        bs = [kappa * a for a in ages]
    elif policy == "maxage":
        mx = max(ages); bs = [kappa * mx] * len(ages)
    else:  # none, meanage -> unshifted feed
        bs = [0.0] * len(ages)
    for (x, _), b in zip(samples, bs):
        cs.update_shift(x, 1.0, b)
    L = cs.lower()
    if policy == "meanage":
        L = max(0.0, L - kappa * (sum(ages) / len(ages)))
    return L


def run(eps, delta=0.1, T=500, tau=0.5, mu0=1.6, qe=25, n_seeds=100):
    kappa = eps * PHI_MAX
    variants = ["none", "persample", "maxage", "meanage", "fixedlam"]
    breach = {v: 0 for v in variants}
    lbsum = {v: 0.0 for v in variants}
    nq = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        buf = []
        seed_breach = {v: False for v in variants}
        for t in range(T):
            tn = t / (T - 1)
            g, s = draw_item(tn, rng, mu0, eps)
            if g == 1:
                buf.append((1.0 if s >= tau else 0.0, tn))
            if t % qe and t != T - 1:
                continue
            r = true_recall(tn, tau, mu0, eps)
            nq += 1 if seed == 0 else 0
            for v in variants:
                L = lower(buf, tn, kappa, v)
                lbsum[v] += L
                if L > r + TOL:
                    seed_breach[v] = True
        for v in variants:
            breach[v] += int(seed_breach[v])
    nchk = len(range(0,T,qe)) + 1
    cov = {v: 1 - breach[v] / n_seeds for v in variants}
    pw = {v: lbsum[v] / (n_seeds * nchk) for v in variants}
    return cov, pw


def main():
    delta = 0.1
    epss = [0.0, 0.4, 0.8]
    out = {"delta": delta, "rows": []}
    print(f"\nClean drift-correction ablation (same full-pool adaptive-EB CS; only b_i changes)")
    print(f"delta={delta}, nominal {1-delta:.2f}. Sanity: at eps=0 all should ~coincide.\n")
    hdr = f"{'eps':>4} | " + " | ".join(f"{v:>10}" for v in
                                        ["none", "persample", "maxage", "meanage", "fixedlam"])
    print(hdr); print("-" * len(hdr))
    for eps in epss:
        cov, pw = run(eps)
        row = {"eps": eps, "coverage": cov, "power": pw}
        out["rows"].append(row)
        print(f"{eps:>4} | " + " | ".join(f"{cov[v]:>10.3f}" for v in
              ["none", "persample", "maxage", "meanage", "fixedlam"]))
    p = os.path.join(os.path.dirname(__file__), "..", "results", "ablation_clean.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nsaved -> {os.path.abspath(p)}")


if __name__ == "__main__":
    main()
