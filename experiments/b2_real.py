"""B2: external validity on REAL text (20newsgroups) with a REAL proxy model.

Upgrades the coverage experiment from synthetic Gaussian to real data:
- proxy = TF-IDF + LogisticRegression P(positive) (a real, imperfect proxy);
- ground truth g = true topic label (deterministic => coverage exactly measurable);
- oracle = true label (model-agnostic abstraction; no LLM API needed);
- REAL drift: split true positives into easy/hard by proxy score; the snapshot
  mixes them with a time-varying weight so the snapshot recall at a fixed threshold
  genuinely drifts down over the stream. kappa (the drift Lipschitz bound) is computed
  exactly from the real score arrays and handed to SAVED.

Measures anytime simultaneous coverage of SAVED (per-sample, Prop 3a) vs SUPG fixed-n
and pooled-CS (stream-ReDD) under real-text drift + budget reuse.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.recall_cs import RecallCS
from saved.baselines import FixedNSplitConformal, PooledCS, PerSampleSAVED, PHI_MAX

TOL = 1e-9


def build_proxy(target="sci.med"):
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    cats = ['rec.sport.baseball', 'sci.med', 'sci.space', 'talk.politics.guns']
    tr = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=cats)
    te = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=cats)
    yt = (np.array(tr.target) == cats.index(target)).astype(int)
    ye = (np.array(te.target) == cats.index(target)).astype(int)
    vec = TfidfVectorizer(max_features=20000, min_df=2, stop_words='english')
    Xtr = vec.fit_transform(tr.data)
    Xte = vec.transform(te.data)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, yt)
    # score the TEST split (proxy is imperfect on unseen data -> realistic)
    s = clf.predict_proba(Xte)[:, 1]
    return s, ye  # proxy scores, true labels (oracle/truth)


def main():
    s, y = build_proxy()
    pos = s[y == 1]            # real positives' proxy scores
    # threshold so base recall ~ 0.80
    tau = float(np.quantile(pos, 0.20))
    med = float(np.median(pos))
    easy = pos[pos >= med]     # high-score positives
    hard = pos[pos < med]      # low-score positives
    r_easy = float((easy >= tau).mean())
    r_hard = float((hard >= tau).mean())
    print(f"real proxy built: #pos={len(pos)}, tau={tau:.3f}, "
          f"recall_easy={r_easy:.3f}, recall_hard={r_hard:.3f}")

    delta, target = 0.1, 0.70
    T, qe, n_seeds = 600, 25, 120

    def recall_t(tn, rate):
        w = max(0.0, 1.0 - rate * tn)          # easy-positive weight decays over stream
        return w * r_easy + (1 - w) * r_hard

    for rate in [0.0, 0.6, 1.0]:               # drift strength (0 = no drift)
        kappa = rate * abs(r_easy - r_hard)    # exact Lipschitz bound of recall_t
        eps_bound = kappa / PHI_MAX            # so SAVED's kappa = real kappa
        agg = {m: 0 for m in ("FixedN", "PooledCS", "SAVED")}
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            fn = FixedNSplitConformal(delta, k=50)
            pc = PooledCS(delta, grid=801)
            sv = PerSampleSAVED(delta, eps_bound=eps_bound, grid=801)
            br = {m: False for m in agg}
            for t in range(T):
                tn = t / (T - 1)
                w = max(0.0, 1.0 - rate * tn)
                # draw a positive's proxy score from the snapshot mixture
                src = easy if rng.random() < w else hard
                x = 1.0 if src[rng.integers(len(src))] >= tau else 0.0
                fn.update(x); pc.update(x); sv.update(x, tn)
                if t % qe and t != T - 1:
                    continue
                r = recall_t(tn, rate)
                for name, mth in (("FixedN", fn.lower()), ("PooledCS", pc.lower()),
                                  ("SAVED", sv.lower(tn))):
                    if mth > r + TOL:
                        br[name] = True
            for m in agg:
                agg[m] += int(br[m])
        cov = {m: 1 - agg[m] / n_seeds for m in agg}
        print(f"\n[rate={rate}] real-text drift, true recall {recall_t(0,rate):.3f}->"
              f"{recall_t(1,rate):.3f}, target={target}, kappa={kappa:.3f}")
        for m in ("FixedN", "PooledCS", "SAVED"):
            flag = "OK " if cov[m] >= (1 - delta) - 0.04 else "BREACH"
            print(f"   {m:>9}: coverage={cov[m]:.3f}  {flag}")


if __name__ == "__main__":
    main()
