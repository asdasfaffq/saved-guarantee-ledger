"""B2 (neural proxy): real text + REAL transformer-embedding proxy (MiniLM) on RTX 4090.
Upgrades the proxy from TF-IDF (bag-of-words) to a real learned sentence embedding
(all-MiniLM-L6-v2) + logistic regression -- a genuine LLM-family proxy. Same drift +
coverage protocol as b2_real.py; oracle = dataset label (deterministic truth)."""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.baselines import FixedNSplitConformal, PooledCS, PerSampleSAVED, PHI_MAX

TOL = 1e-9


def build_proxy(target="sci.med"):
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.linear_model import LogisticRegression
    from sentence_transformers import SentenceTransformer
    cats = ['rec.sport.baseball', 'sci.med', 'sci.space', 'talk.politics.guns']
    tr = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=cats)
    te = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=cats)
    yt = (np.array(tr.target) == cats.index(target)).astype(int)
    ye = (np.array(te.target) == cats.index(target)).astype(int)
    enc = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
    Etr = enc.encode(tr.data, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    Ete = enc.encode(te.data, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    clf = LogisticRegression(max_iter=3000, C=2.0).fit(Etr, yt)
    s = clf.predict_proba(Ete)[:, 1]
    return s, ye


def main():
    s, y = build_proxy()
    pos = s[y == 1]
    tau = float(np.quantile(pos, 0.20))
    med = float(np.median(pos))
    easy = pos[pos >= med]; hard = pos[pos < med]
    r_easy = float((easy >= tau).mean()); r_hard = float((hard >= tau).mean())
    print(f"MiniLM proxy: #pos={len(pos)}, tau={tau:.3f}, recall_easy={r_easy:.3f}, "
          f"recall_hard={r_hard:.3f}")
    delta, target, T, qe, n_seeds = 0.1, 0.70, 600, 25, 120

    def recall_t(tn, rate):
        w = max(0.0, 1.0 - rate * tn)
        return w * r_easy + (1 - w) * r_hard

    out = {"proxy": "all-MiniLM-L6-v2 + LogisticRegression", "dataset": "20newsgroups sci.med",
           "n_pos": int(len(pos)), "tau": tau, "recall_easy": r_easy, "recall_hard": r_hard,
           "delta": delta, "target": target, "rows": []}
    for rate in [0.0, 0.6, 1.0]:
        kappa = rate * abs(r_easy - r_hard); eps_bound = kappa / PHI_MAX
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
                r = recall_t(tn, rate)
                for name, mth in (("FixedN", fn.lower()), ("PooledCS", pc.lower()), ("SAVED", sv.lower(tn))):
                    if mth > r + TOL:
                        br[name] = True
            for m in agg:
                agg[m] += int(br[m])
        cov = {m: 1 - agg[m] / n_seeds for m in agg}
        print(f"[rate={rate}] recall {recall_t(0,rate):.3f}->{recall_t(1,rate):.3f}, kappa={kappa:.3f}: "
              + ", ".join(f"{m}={cov[m]:.3f}" for m in ("FixedN", "PooledCS", "SAVED")))
        out["rows"].append({"rate": rate, "recall_start": recall_t(0, rate),
                            "recall_end": recall_t(1, rate), "kappa": kappa, "coverage": cov})
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "b2_neural.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved -> results/b2_neural.json")


if __name__ == "__main__":
    main()
