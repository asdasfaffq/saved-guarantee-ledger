"""Ablation: proxy-correlated oracle bias (the rho axis) and the anchor correction.

No drift (eps=0); we isolate ORACLE bias. A true positive with proxy score s is
correctly kept positive by the oracle with prob increasing in s when rho>0 (the
oracle drops HARD, low-proxy positives) -- exactly the proxy-correlated error a
biased LLM oracle exhibits. Then the surviving oracle-positives skew high-score,
so the naive recall CS (treating oracle labels as truth, as SUPG/LOTUS implicitly
do) OVER-estimates recall and LIES. A small clean human anchor (true labels)
restores validity. (Combining anchor+oracle via PPI for extra power is future work.)
"""
import os
import sys
from math import erf, sqrt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.recall_cs import RecallCS

TOL = 1e-9
MU0, TAU, DELTA, TARGET = 1.6, 0.5, 0.10, 0.70


def phi(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def run(rho, method, n_seeds=200, T=400, qe=25, anchor_rate=0.15):
    true_recall = phi(MU0 - TAU)               # exact recall at fixed snapshot
    p0 = 0.7                                    # base keep rate
    br = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        cs = RecallCS(delta=DELTA, grid=801)
        bad = False
        for t in range(T):
            s = rng.normal(MU0, 1.0)            # a true positive's proxy score
            x = 1.0 if s >= TAU else 0.0
            keep = (1 - rho) * p0 + rho * sigmoid(3.0 * (s - TAU))  # correlated keep
            if method == "naive":
                if rng.random() < keep:         # oracle keeps it as positive
                    cs.update(x)
            else:  # anchor: only clean-labeled positives (true), small budget
                if rng.random() < anchor_rate:
                    cs.update(x)
            if t % qe and t != T - 1:
                continue
            if cs.lower() > true_recall + TOL:  # certified above true recall = a lie
                bad = True
        br += int(bad)
    return 1 - br / n_seeds


def main():
    print(f"\nProxy-correlated oracle bias ablation (no drift; delta={DELTA}, "
          f"true recall={phi(MU0-TAU):.3f})")
    print("coverage = 1 - P(CS lower bound ever exceeds true recall). lie if < 1-delta.\n")
    hdr = f"{'rho':>5} | {'naive (oracle=truth)':>20} | {'anchor-corrected':>16}"
    print(hdr); print("-" * len(hdr))
    for rho in [0.0, 0.5, 1.0]:
        cn = run(rho, "naive"); ca = run(rho, "anchor")
        fn = "OK " if cn >= (1 - DELTA) - 0.04 else "LIES"
        fa = "OK " if ca >= (1 - DELTA) - 0.04 else "LIES"
        print(f"{rho:>5} | {cn:>14.3f} {fn:>5} | {ca:>11.3f} {fa:>4}")


if __name__ == "__main__":
    main()
