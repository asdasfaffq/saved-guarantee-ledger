"""Cross-dataset external validity: a SECOND real dataset (AG News, 4 topic classes).
Same drift+reuse coverage protocol + real MiniLM proxy. Shows the result is not specific
to 20newsgroups. Oracle = dataset label; deterministic truth. Runs on CPU."""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.baselines import FixedNSplitConformal, PooledCS, PerSampleSAVED, PHI_MAX

TOL = 1e-9
LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def proxies(n_train=6000, n_test=4000):
    from datasets import load_dataset
    from sklearn.linear_model import LogisticRegression
    from sentence_transformers import SentenceTransformer
    tr = load_dataset("ag_news", split=f"train[:{n_train}]")
    te = load_dataset("ag_news", split=f"test[:{n_test}]")
    enc = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    Etr = enc.encode(tr["text"], batch_size=128, normalize_embeddings=True)
    Ete = enc.encode(te["text"], batch_size=128, normalize_embeddings=True)
    ytr = np.array(tr["label"]); yte = np.array(te["label"])
    out = {}
    for c in LABELS:
        clf = LogisticRegression(max_iter=3000, C=2.0).fit(Etr, (ytr == c).astype(int))
        s = clf.predict_proba(Ete)[:, 1]
        out[c] = s[yte == c]
    return out


def coverage(pos, rate, delta=0.1, T=500, qe=25, n_seeds=100):
    tau = float(np.quantile(pos, 0.20)); med = float(np.median(pos))
    easy = pos[pos >= med]; hard = pos[pos < med]
    re, rh = float((easy >= tau).mean()), float((hard >= tau).mean())
    eps_bound = rate * abs(re - rh) / PHI_MAX

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
    out = {"dataset": "ag_news", "proxy": "MiniLM+LR", "rate": 1.0, "rows": []}
    print("\nCross-dataset external validity: AG News (MiniLM proxy, strong drift rate=1.0)\n")
    hdr = f"{'predicate':>10} | {'recall drift':>14} | {'FixedN':>7} {'Pooled':>7} {'SAVED':>7}"
    print(hdr); print("-" * len(hdr))
    for c in LABELS:
        cov, r0, r1 = coverage(P[c], 1.0)
        print(f"{LABELS[c]:>10} | {r0:.2f}->{r1:.2f}     | {cov['FixedN']:>7.3f} "
              f"{cov['PooledCS']:>7.3f} {cov['SAVED']:>7.3f}")
        out["rows"].append({"predicate": LABELS[c], "recall_start": r0, "recall_end": r1, "coverage": cov})
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "b2_agnews.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved -> results/b2_agnews.json")


if __name__ == "__main__":
    main()
