"""Safe-adaptivity / post-selection stress test (validates Assumption A3 + Prop 4).

An analyst wants to certify recall>=r and has several candidate predicates. If they
pick the predicate that looks best ON THE VERY LABELS used to certify it, that is
post-selection double-dipping: the winner's empirical recall is upward-biased, so its
lower bound over-certifies -> false certificates, even though each per-predicate bound
is individually valid. We compare:
  unsafe    : pick argmax empirical recall on the certification labels, certify with them.
  firewall  : pick by a PUBLIC signal independent of the certification labels (A3), certify
              with the hidden labels.
  splitting : split labels; pick on half, certify on the other half.
Metric: session false-certificate rate (selected predicate certified while true recall < r).
Expectation: unsafe >> delta; firewall and splitting <= delta.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.recall_cs import RecallCS

DELTA, R = 0.10, 0.85


def lower(xs, delta=DELTA, grid=401):
    cs = RecallCS(delta=delta, grid=grid)
    for x in xs:
        cs.update(float(x))
    return cs.lower()


def run(method, n_pred=40, n=90, n_seeds=2000, sigma_pub=0.05):
    false_cert = 0
    certified = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        mu = rng.uniform(0.82, 0.90, n_pred)               # some below r=0.85, some above
        X = (rng.random((n_pred, n)) < mu[:, None]).astype(float)
        if method == "unsafe":
            emp = X.mean(1)
            sel = int(np.argmax(emp))
            L = lower(X[sel])                               # certify on the SAME labels
        elif method == "firewall":
            pub = mu + rng.normal(0, sigma_pub, n_pred)     # public signal, independent of X
            sel = int(np.argmax(pub))
            L = lower(X[sel])                               # certify on hidden labels
        else:  # splitting
            h = n // 2
            emp = X[:, :h].mean(1)
            sel = int(np.argmax(emp))
            L = lower(X[sel, h:])                           # certify on held-out half
        if L >= R:
            certified += 1
            if mu[sel] < R - 1e-9:
                false_cert += 1
    return false_cert / n_seeds, certified / n_seeds


def main():
    print(f"\nSafe-adaptivity / post-selection stress test (delta={DELTA}, target r={R}, "
          f"10 candidate predicates, 40 labels each)\n")
    hdr = f"{'method':>10} | {'session false-cert rate':>24} | {'certification rate':>18}"
    print(hdr); print("-" * len(hdr))
    for m in ("unsafe", "firewall", "splitting"):
        fc, cr = run(m)
        flag = "VALID" if fc <= DELTA + 0.01 else "INVALID (inflated)"
        print(f"{m:>10} | {fc:>23.3f} | {cr:>18.3f}   {flag}")


if __name__ == "__main__":
    main()
