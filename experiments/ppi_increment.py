"""PPI power-recovery increment for the biased-oracle case (fixed-sample CLT version).

Construction (prove-then-code). Recall = N/D with unconditional means
    N = E[1{s>=tau} 1{g=1}],   D = E[1{g=1}].
A small clean anchor A (true g) and a large biased-oracle set L (oracle label o).
Predictions Zhat=1{s>=tau}1{o=1}, What=1{o=1}. Prediction-powered estimators:
    N_ppi = mean_L(Zhat) + mean_A(Z - Zhat),   D_ppi = mean_L(What) + mean_A(W - What),
both unbiased for N, D (the anchor rectifier removes the oracle bias). Recall_hat = N_ppi/D_ppi,
with a one-sided delta-method lower bound at level 1-delta. Compared to:
  naive    : oracle as truth on L (biased -> lies),
  anchor   : anchor-only means (valid but high variance / loose),
  ppi      : the above (valid AND tighter when the oracle is informative).
Guarantee is fixed-sample (CLT); an anytime prediction-powered CS is future work.
"""
import os
import sys
import numpy as np

Z90 = 1.2815515594  # one-sided normal quantile for delta=0.1
DELTA = 0.10
MU0, TAU_Q, PREV = 1.6, 0.20, 0.30


def gen(rng, n, drop_k=3.0, fp=0.15):
    """n records: g~Bernoulli(PREV); positives s~N(MU0,1), negatives s~N(0,1).
    Oracle o: proxy-correlated. Positives kept (o=1) w.p. sigmoid(drop_k*(s-tau)) (drops
    low-s positives); negatives labeled o=1 w.p. fp (false positives)."""
    g = (rng.random(n) < PREV).astype(int)
    s = rng.normal(0, 1, n) + MU0 * g
    return g, s


def ratio_lo(N, D, varN, varD, covND):
    """One-sided (1-delta) lower bound on N/D via the delta method."""
    if D <= 1e-9:
        return 0.0
    r = N / D
    grad_var = (varN + r * r * varD - 2 * r * covND) / (D * D)
    se = np.sqrt(max(grad_var, 0.0))
    return max(0.0, r - Z90 * se)


def run(method, drop_k, fp=0.0, n_anchor=120, n_large=4000, n_seeds=400):
    # fixed tau from a large reference draw so true recall is stable
    ref = np.random.default_rng(999)
    g0, s0 = gen(ref, 200000)
    tau = float(np.quantile(s0[g0 == 1], TAU_Q))
    true_recall = float((s0[g0 == 1] >= tau).mean())
    cov = 0; los = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        # anchor (true labels) + large (oracle only), disjoint draws from same dist
        ga, sa = gen(rng, n_anchor)
        gl, sl = gen(rng, n_large)
        # oracle: drop low-proxy positives (strength drop_k); fp false positives on negatives

        def oracle(g, s):
            keep = 1 / (1 + np.exp(-drop_k * (s - tau)))
            o = np.where(g == 1, (rng.random(len(g)) < keep).astype(int),
                         (rng.random(len(g)) < fp).astype(int))
            return o
        oa, ol = oracle(ga, sa), oracle(gl, sl)
        # integrands
        Za = ((sa >= tau) & (ga == 1)).astype(float); Wa = (ga == 1).astype(float)
        Zha = ((sa >= tau) & (oa == 1)).astype(float); Wha = (oa == 1).astype(float)
        Zhl = ((sl >= tau) & (ol == 1)).astype(float); Whl = (ol == 1).astype(float)
        na, M = len(ga), len(gl)
        if method == "naive":
            N, D = Zhl.mean(), Whl.mean()
            C = np.cov(np.vstack([Zhl, Whl])) / M
            lo = ratio_lo(N, D, C[0, 0], C[1, 1], C[0, 1])
        elif method == "anchor":
            N, D = Za.mean(), Wa.mean()
            C = np.cov(np.vstack([Za, Wa])) / na
            lo = ratio_lo(N, D, C[0, 0], C[1, 1], C[0, 1])
        else:  # ppi
            N = Zhl.mean() + (Za - Zha).mean()
            D = Whl.mean() + (Wa - Wha).mean()
            Cl = np.cov(np.vstack([Zhl, Whl])) / M
            Cr = np.cov(np.vstack([Za - Zha, Wa - Wha])) / na
            C = Cl + Cr
            lo = ratio_lo(N, D, C[0, 0], C[1, 1], C[0, 1])
        cov += int(lo <= true_recall + 1e-9)
        los.append(lo)
    return true_recall, cov / n_seeds, float(np.mean(los))


def main():
    print(f"\nPPI power-recovery increment (biased oracle, fixed-sample CLT, delta={DELTA})")
    print("no false positives (fp=0) so the naive/oracle-as-truth bound can LIE. anchor n=120.")
    print("coverage=P(LB<=true recall); power=mean LB (higher=tighter). true recall=0.80.\n")
    hdr = f"{'drop_k(bias)':>12} | {'naive cov/pow':>14} | {'anchor cov/pow':>15} | {'ppi cov/pow':>13}"
    print(hdr); print("-" * len(hdr))
    for dk in [0.5, 1.5, 3.0]:
        _, cn, pn = run("naive", dk)
        _, ca, pa = run("anchor", dk)
        _, cp, pp = run("ppi", dk)
        def f(c):
            return "OK" if c >= (1 - DELTA) - 0.04 else "LIE"
        print(f"{dk:>12} | {cn:.3f}/{pn:.2f} {f(cn):>3} | {ca:.3f}/{pa:.2f} {f(ca):>3} "
              f"| {cp:.3f}/{pp:.2f} {f(cp):>3}")


if __name__ == "__main__":
    main()
