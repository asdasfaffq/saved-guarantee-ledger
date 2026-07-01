"""Ablation: per-stratum (ABae-style) recall CS under HETEROGENEOUS strata.

When proxy-score strata of positives have different recalls / drift rates, pooling
them into one confidence sequence is unsafe: an individual sample's conditional mean
r_{k}(t_i) deviates from the overall current recall by the STRATUM GAP, which the
per-sample drift shift does NOT bound. So the pooled bound can exceed the true overall
recall (lie). A STRATIFIED estimator -- one per-sample-shift CS per stratum at level
delta/K, combined by known prevalences -- keeps each stratum homogeneous and is valid.

We sweep a heterogeneity knob = the recall gap between strata, and report coverage of:
  pooled  (one CS over all positives, shift by the overall drift rate), vs
  stratified (per-stratum CS, per-stratum kappa, delta/K, prevalence-combined).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.recall_cs import RecallCS

TOL = 1e-9
DELTA = 0.10
PHI = 1.0  # we feed selection indicators directly, so kappa = drift rate of recall


def run(gap, method, n_seeds=200, T=500, qe=25):
    # two strata, prevalence 1/2 each. Stratum recalls drift down at different rates.
    r0_0, rate0 = 0.80 + gap / 2, 0.50      # "hard": higher start, fast drift
    r1_0, rate1 = 0.80 - gap / 2, 0.05      # "easy": lower start, slow drift
    pi = 0.5
    def rk(r0, rate, t):
        return min(1.0, max(0.0, r0 - rate * t))
    def overall(t):
        return pi * rk(r0_0, rate0, t) + pi * rk(r1_0, rate1, t)
    kappa_avg = pi * rate0 + pi * rate1
    kappa_max = max(rate0, rate1)
    br = 0; lb_sum = 0.0; lb_n = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        if method.startswith("pooled"):
            cs = RecallCS(delta=DELTA, grid=801)
        else:
            cs0 = RecallCS(delta=DELTA / 2, grid=801)
            cs1 = RecallCS(delta=DELTA / 2, grid=801)
        kap = kappa_max if method == "pooled_max" else kappa_avg
        bad = False
        for t in range(T):
            tn = t / (T - 1)
            k = 0 if rng.random() < pi else 1
            x = 1.0 if rng.random() < rk(r0_0 if k == 0 else r1_0,
                                         rate0 if k == 0 else rate1, tn) else 0.0
            if method.startswith("pooled"):
                cs.update_shift(x, 1.0, kap * tn)
            else:
                (cs0 if k == 0 else cs1).update_shift(x, 1.0, (rate0 if k == 0 else rate1) * tn)
            if t % qe and t != T - 1:
                continue
            r_now = overall(tn)
            L = cs.lower() if method.startswith("pooled") else pi * cs0.lower() + pi * cs1.lower()
            if L > r_now + TOL:
                bad = True
            lb_sum += L; lb_n += 1
        br += int(bad)
    return 1 - br / n_seeds, lb_sum / lb_n


def main():
    print(f"\nPer-stratum ablation under heterogeneous strata (delta={DELTA}, nominal {1-DELTA:.2f})\n")
    hdr = (f"{'gap':>4} | {'pooled-avgK cov/pow':>20} | {'pooled-maxK cov/pow':>20} "
           f"| {'stratified cov/pow':>19}")
    print(hdr); print("-" * len(hdr))
    for gap in [0.0, 0.2, 0.4]:
        ca, pa = run(gap, "pooled")
        cm, pm = run(gap, "pooled_max")
        cs, ps = run(gap, "stratified")
        def f(c):
            return "OK" if c >= (1 - DELTA) - 0.04 else "LIE"
        print(f"{gap:>4} | {ca:.3f}/{pa:.2f} {f(ca):>3} | {cm:.3f}/{pm:.2f} {f(cm):>3} "
              f"| {cs:.3f}/{ps:.2f} {f(cs):>3}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
