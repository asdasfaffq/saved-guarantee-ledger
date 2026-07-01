"""Per-sample shift (Prop 3a, rigorous) vs mean-age inflation (3b, implemented heuristic).
Streaming M2-style coverage. Expect: 3a fixes the eps=0 anti-conservatism (coverage ~>=0.90)
and keeps validity under drift; report power (mean LB) for the honest tradeoff."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.drift import draw_item, true_recall
from saved.baselines import SAVEDDriftRobust, PerSampleSAVED

TOL = 1e-9


def run(seed, eps, method, delta=0.1, T=800, tau=0.5, mu0=1.6, qe=25):
    rng = np.random.default_rng(seed)
    if method == "meanage":
        m = SAVEDDriftRobust(delta, eps_bound=eps, window=70, grid=801)
        windowed = True
    else:
        m = PerSampleSAVED(delta, eps_bound=eps, grid=801)
        windowed = False
    breached = False; lastL = 0.0
    for t in range(T):
        tn = t / (T - 1)
        g, s = draw_item(tn, rng, mu0, eps)
        if g == 1:
            m.update(1.0 if s >= tau else 0.0, tn)
        if t % qe and t != T - 1:
            continue
        r = true_recall(tn, tau, mu0, eps)
        L = m.lower(tn); lastL = L
        if L > r + TOL:
            breached = True
    return breached, lastL


def main():
    delta = 0.1; epss = [0.0, 0.4, 0.8, 1.2]; n = 200
    print(f"\nPer-sample (3a, rigorous) vs mean-age (3b, heuristic) — coverage & power "
          f"(delta={delta}, nominal {1-delta:.2f}, seeds={n})\n")
    hdr = f"{'eps':>4} | {'method':>10} | {'coverage':>9} | {'mean_LB':>8}"
    print(hdr); print("-" * len(hdr))
    for eps in epss:
        for method in ("meanage", "persample"):
            br = 0; lb = 0.0
            for seed in range(n):
                b, L = run(seed, eps, method, delta=delta)
                br += int(b); lb += L
            cov = 1 - br / n
            flag = "OK " if cov >= (1 - delta) - 0.03 else "BREACH"
            print(f"{eps:>4} | {method:>10} | {cov:>9.3f} | {lb/n:>8.3f}  {flag}")
        print("-" * len(hdr))


if __name__ == "__main__":
    main()
