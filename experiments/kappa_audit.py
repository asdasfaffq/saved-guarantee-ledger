"""kappa-audit experiment (validates Assumption A4): estimate the drift bound from
sentinel labels with a conservative margin, and fail-safe abstain when the audit is
uncertain -- so kappa need not be known a priori.

Sentinel estimate: draw m positives at the current snapshot and at a lagged snapshot,
estimate each snapshot recall, and set
    kappa_hat_upper = (|r_hat(t) - r_hat(t-D)| + 2*margin) / D,
    margin = sqrt(ln(2/alpha)/(2m))   (a DKW-style radius per estimate).
Fail-safe: if the sampling margin is large relative to the observed movement, abstain.

Policies compared (coverage / power vs the TRUE recall):
    exact    : kappa = kappa_true                 (oracle; not deployable)
    sentinel : kappa = kappa_hat_upper            (estimated + conservative margin)
    under    : kappa = 0.5 * kappa_true           (the danger: under-specified)
    over     : kappa = 2.0 * kappa_true           (over-conservative)
"""
import json
import os
import sys
from math import log, sqrt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.baselines import PerSampleSAVED, PHI_MAX
from saved.drift import draw_item, true_recall

DELTA, TAU, MU0 = 0.10, 0.5, 1.6
TOL = 1e-9


def sentinel_kappa(rng, eps, t_now, D=0.2, m=30, alpha=0.1):
    """Estimate a conservative upper bound on kappa from 2*m sentinel positives."""
    def emp_recall(t):
        cnt = 0; hit = 0
        while cnt < m:
            g, s = draw_item(t, rng, MU0, eps)
            if g == 1:
                hit += 1 if s >= TAU else 0; cnt += 1
        return hit / m
    t2 = max(0.0, t_now - D)
    r1, r2 = emp_recall(t_now), emp_recall(t2)
    margin = sqrt(log(2 / alpha) / (2 * m))
    kap = (abs(r1 - r2) + 2 * margin) / max(D, 1e-6)
    return kap, margin / max(D, 1e-6)  # estimate, uncertainty rate


def run(eps, policy, n_seeds=120, T=600, qe=25):
    kap_true = eps * PHI_MAX
    br = 0; lb = 0.0; nq = 0; sent = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        if policy == "exact":
            kap = kap_true
        elif policy == "under":
            kap = 0.5 * kap_true
        elif policy == "over":
            kap = 2.0 * kap_true
        else:  # sentinel: estimate once at mid-stream
            kap, _ = sentinel_kappa(rng, eps, 0.5); sent += 60
        eps_bound = kap / PHI_MAX
        sv = PerSampleSAVED(DELTA, eps_bound=eps_bound, grid=801)
        bad = False
        for t in range(T):
            tn = t / (T - 1)
            g, s = draw_item(tn, rng, MU0, eps)
            if g == 1:
                sv.update(1.0 if s >= TAU else 0.0, tn)
            if t % qe and t != T - 1:
                continue
            r = true_recall(tn, TAU, MU0, eps)
            L = sv.lower(tn); lb += L; nq += 1
            if L > r + TOL:
                bad = True
        br += int(bad)
    return {"coverage": 1 - br / n_seeds, "power": lb / nq,
            "sentinel_labels": sent // n_seeds}


def main():
    epss = [0.4, 0.8]
    policies = ["exact", "sentinel", "under", "over"]
    out = {"delta": DELTA, "rows": []}
    print(f"\nkappa-audit (validates A4): coverage / power under different kappa sources "
          f"(delta={DELTA}, nominal {1-DELTA:.2f})\n")
    hdr = f"{'eps':>4} | " + " | ".join(f"{p:>22}" for p in policies)
    print(hdr); print("-" * len(hdr))
    print("       " + "   ".join("cov / power / sent".rjust(20) for _ in policies))
    for eps in epss:
        row = {"eps": eps}
        cells = []
        for p in policies:
            r = run(eps, p); row[p] = r
            flag = "OK" if r["coverage"] >= (1 - DELTA) - 0.03 else "UNDER"
            cells.append(f"{r['coverage']:.2f}/{r['power']:.2f}/{r['sentinel_labels']} {flag}".rjust(20))
        print(f"{eps:>4} | " + " | ".join(cells))
        out["rows"].append(row)
    p = os.path.join(os.path.dirname(__file__), "..", "results", "kappa_audit.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nsaved -> {os.path.abspath(p)}")


if __name__ == "__main__":
    main()
