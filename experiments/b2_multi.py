"""Full-length expansion: multi-predicate external validity on real text.
Runs the drift+reuse coverage experiment for EACH of the 4 20newsgroups classes as the
target predicate (not just sci.med), with a real MiniLM proxy. Broadens external validity
from a single predicate to a suite. Oracle = dataset label; deterministic truth.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.baselines import FixedNSplitConformal, PooledCS, PerSampleSAVED, PHI_MAX

TOL = 1e-9
CATS = ['rec.sport.baseball', 'sci.med', 'sci.space', 'talk.politics.guns']


def proxies():
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.linear_model import LogisticRegression
    from sentence_transformers import SentenceTransformer
    tr = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=CATS)
    te = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=CATS)
    enc = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    Etr = enc.encode(tr.data, batch_size=64, normalize_embeddings=True)
    Ete = enc.encode(te.data, batch_size=64, normalize_embeddings=True)
    out = {}
    for c in CATS:
        yt = (np.array(tr.target) == CATS.index(c)).astype(int)
        ye = (np.array(te.target) == CATS.index(c)).astype(int)
        clf = LogisticRegression(max_iter=3000, C=2.0).fit(Etr, yt)
        s = clf.predict_proba(Ete)[:, 1]
        out[c] = s[ye == 1]      # positives' proxy scores
    return out


def coverage(pos, rate, delta=0.1, T=500, qe=25, n_seeds=100):
    tau = float(np.quantile(pos, 0.20))
    med = float(np.median(pos))
    easy = pos[pos >= med]; hard = pos[pos < med]
    re, rh = float((easy >= tau).mean()), float((hard >= tau).mean())
    kappa = rate * abs(re - rh); eps_bound = kappa / PHI_MAX

    def rec(tn):
        w = max(0.0, 1.0 - rate * tn); return w * re + (1 - w) * rh
    agg = {m: 0 for m in ("FixedN", "PooledCS", "SAVED")}
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        fn = FixedNSplitConformal(delta, k=50); pc = PooledCS(delta, grid=801)
        sv = PerSampleSAVED(delta, eps_bound=eps_bound, grid=801)
        br = {m: False for m in agg}
        for t in range(T):
            tn = t / (T - 1); w = max(0.0, 1.0 - rate * tn)
            src = easy if rng.random() < w else hard
            x = 1.0 if src[rng.integers(len(src))] >= tau else 0.0
            fn.update(x); pc.update(x); sv.update(x, tn)
            if t % qe and t != T - 1:
                continue
            r = rec(tn)
            for name, mth in (("FixedN", fn.lower()), ("PooledCS", pc.lower()), ("SAVED", sv.lower(tn))):
                if mth > r + TOL:
                    br[name] = True
        for m in agg:
            agg[m] += int(br[m])
    return {m: 1 - agg[m] / n_seeds for m in agg}, rec(0.0), rec(1.0)


def main():
    P = proxies()
    out = {"dataset": "20newsgroups", "proxy": "MiniLM+LR", "rate": 1.0, "rows": []}
    print("\nMulti-predicate external validity (real text, MiniLM proxy, strong drift rate=1.0)\n")
    hdr = f"{'predicate':>24} | {'recall drift':>14} | {'FixedN':>7} {'Pooled':>7} {'SAVED':>7}"
    print(hdr); print("-" * len(hdr))
    for c in CATS:
        cov, r0, r1 = coverage(P[c], 1.0)
        print(f"{c:>24} | {r0:.2f}->{r1:.2f}     | {cov['FixedN']:>7.3f} "
              f"{cov['PooledCS']:>7.3f} {cov['SAVED']:>7.3f}")
        out["rows"].append({"predicate": c, "recall_start": r0, "recall_end": r1, "coverage": cov})
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "b2_multi.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved -> results/b2_multi.json")


if __name__ == "__main__":
    main()
