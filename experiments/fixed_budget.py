"""Fixed-budget trustworthy-certification dominance (the headline validity figure).

At a FIXED oracle budget B fresh labels per query (reuse free for pool methods),
each method certifies recall>=target or refuses. Under drift+reuse we report,
per method, the rates over the N-query stream:
  - correct  : certified AND true recall >= target  (trustworthy guarantee)
  - FALSE    : certified BUT true recall <  target  (a LIE -- the dangerous failure)
  - refused  : not certified
A guarantee system is only useful if FALSE ~ 0. Trustworthy throughput := correct
rate of a method whose FALSE rate <= delta; methods exceeding delta are UNTRUSTWORTHY.

Methods: pooled-CS (stream-ReDD), fixed-N (SUPG/LOTUS), per-query recal, SAVEDw (reweighted).
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
TAU, MU0, TARGET, DELTA = 0.5, 1.6, 0.80, 0.10
GAMMA = 0.04


def cert_unweighted(reuse_xs, fresh, B):
    cs = RecallCS(delta=DELTA, grid=GRID)
    for x in reuse_xs:
        cs.update(x)
    for j in range(B):
        cs.update(fresh[j])
    return cs.lower() >= TARGET


def cert_weighted(reuse, t_now, eps_bound, fresh, B):
    cs = RecallCS(delta=DELTA, grid=GRID)
    sw = swa = 0.0
    for (x, ti) in reuse:
        age = max(0.0, t_now - ti); w = GAMMA ** age
        cs.update_w(x, w); sw += w; swa += w * age
    for j in range(B):
        cs.update(fresh[j])
    infl = eps_bound * PHI_MAX * ((swa / sw) if sw > 0 else 0.0)
    return cs.lower() - infl >= TARGET


def run_seed(seed, eps, ell, B, N=30):
    rng = np.random.default_rng(seed)
    pool = []
    out = {m: {"correct": 0, "false": 0, "refused": 0} for m in
           ("pooled", "fixedN", "perquery", "SAVEDw")}
    fixedN_frozen = {"L": 0.0, "set": False, "buf": []}
    for q in range(N):
        t = q / (N - 1)
        r_q = true_recall(t, TAU, MU0, eps)
        good = r_q >= TARGET - 1e-9
        applicable = [pool[i] for i in range(len(pool)) if rng.random() < ell] if pool else []
        fresh = (rng.random(max(B, 60)) < r_q).astype(float)

        certs = {}
        certs["pooled"] = cert_unweighted([x for (x, _) in applicable], fresh, B)
        certs["perquery"] = cert_unweighted([], fresh, B)
        certs["SAVEDw"] = cert_weighted(applicable, t, eps, fresh, B)
        # fixedN: calibrate once on first query's B labels, reuse static bound
        if not fixedN_frozen["set"]:
            fixedN_frozen["buf"].extend(list(fresh[:B]))
            m = float(np.mean(fixedN_frozen["buf"]))
            from math import log, sqrt
            fixedN_frozen["L"] = max(0.0, m - sqrt(log(1 / DELTA) / (2 * len(fixedN_frozen["buf"]))))
            fixedN_frozen["set"] = True
        certs["fixedN"] = fixedN_frozen["L"] >= TARGET

        for m, c in certs.items():
            if c:
                out[m]["correct" if good else "false"] += 1
            else:
                out[m]["refused"] += 1
        for j in range(B):
            pool.append((float(fresh[j]), t))
        if len(pool) > 400:        # cap reuse pool (recent labels) for speed
            pool = pool[-400:]
    return out


def main():
    B = 25
    ell = 0.7
    epss = [0.0, 0.4, 0.8]
    n_seeds, N = 300, 25
    out = {"B": B, "ell": ell, "delta": DELTA, "n_seeds": n_seeds, "rows": []}
    print(f"\nFixed-budget trustworthy certification (B={B} fresh/query, ell={ell}, "
          f"target={TARGET}, delta={DELTA}, N={N}, seeds={n_seeds})")
    print("FALSE-rate is the dangerous failure (a lie). Trustworthy := FALSE-rate <= delta.\n")
    hdr = f"{'eps':>4} | {'method':>9} | {'correct%':>8} {'FALSEperq%':>10} {'SESSION-lie%':>12} | {'trustworthy?':>12}"
    print(hdr); print("-" * len(hdr))
    for eps in epss:
        agg = {m: {"correct": 0, "false": 0, "refused": 0, "bad_sessions": 0} for m in
               ("pooled", "fixedN", "perquery", "SAVEDw")}
        for seed in range(n_seeds):
            o = run_seed(seed, eps, ell, B, N=N)
            for m in agg:
                for k in ("correct", "false", "refused"):
                    agg[m][k] += o[m][k]
                agg[m]["bad_sessions"] += int(o[m]["false"] > 0)
        tot = n_seeds * N
        for m in ("pooled", "fixedN", "perquery", "SAVEDw"):
            cr = agg[m]["correct"] / tot
            fr = agg[m]["false"] / tot
            sess = agg[m]["bad_sessions"] / n_seeds  # session-level: ANY false cert (M2-style)
            trust = "YES" if sess <= DELTA else "NO (lies)"
            print(f"{eps:>4} | {m:>9} | {cr*100:>7.1f}% {fr*100:>9.1f}% {sess*100:>11.1f}% | {trust:>12}")
            out["rows"].append({"eps": eps, "method": m, "correct": cr, "false_perq": fr,
                                 "session_lie_rate": sess, "trustworthy": sess <= DELTA})
        print("-" * len(hdr))
    p = os.path.join(os.path.dirname(__file__), "..", "results", "fixed_budget.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {os.path.abspath(p)}")


if __name__ == "__main__":
    main()
