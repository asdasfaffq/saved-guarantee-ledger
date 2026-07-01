"""RQ2: END-TO-END LOTUS/SUPG-style semantic-filter pipeline with a REAL LLM oracle (GLM-5.2).

This turns SAVED from a statistical wrapper into a data-management OPERATOR:
  corpus -> TF-IDF+LR proxy score -> SUPG/LOTUS-style sem_filter (proxy threshold +
  oracle verification) -> SAVED guarantee ledger -> selected set + recall certificate.

The oracle is a REAL LLM (GLM-5.2, thinking disabled, cached): for each document it answers
"is this about medicine?" -> the operative positive label the certificate is built on (SUPG/LOTUS
semantics: the guarantee is relative to the oracle). recall(tau)=P(proxy>=tau | oracle-positive).
Drift: the snapshot's positive population shifts easy->hard over the session (real proxy scores),
so the snapshot recall at a fixed tau genuinely drops. Budget is reused across an adaptive session.

Baselines (all consuming the SAME GLM-oracle budget on the SAME stream):
  SUPG fixed-n : calibrate once on first k oracle labels, static thereafter.
  stream-ReDD  : pooled anytime CS over all accumulated labels, no drift shift.
  SAVED        : per-sample drift shift + (per-query delta here; alpha-spending in the workload exp).
Metrics: SESSION-level false-certificate rate (vs true snapshot oracle-recall), trusted yield,
GLM oracle calls (unique labels), and per-query certification latency.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.recall_cs import RecallCS
from saved.baselines import PHI_MAX
from saved.glm_oracle import classify_yes_no

TAU_Q, R, DELTA = 0.5, 0.80, 0.10
QUESTION = "Is the following text about medicine, health, or clinical topics?"


def build_pool(n_pool, seed=0):
    """Return per-pool-doc: proxy score, GLM oracle label (cached), dataset label."""
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    cats = ['rec.sport.baseball', 'sci.med', 'sci.space', 'talk.politics.guns']
    tr = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'), categories=cats)
    te = fetch_20newsgroups(subset='test',  remove=('headers','footers','quotes'), categories=cats)
    med_tr = cats.index('sci.med')
    ytr = (np.array(tr.target) == med_tr).astype(int)
    vec = TfidfVectorizer(max_features=20000, min_df=2, stop_words='english')
    Xtr = vec.fit_transform(tr.data); Xte = vec.transform(te.data)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    s = clf.predict_proba(Xte)[:, 1]
    yds = (np.array(te.target) == med_tr).astype(int)          # dataset truth
    texts = te.data
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(texts))[:n_pool]
    t0 = time.time()
    glm = np.array([classify_yes_no(texts[i], QUESTION) for i in idx])  # REAL oracle, cached
    glm_secs = time.time() - t0
    return s[idx], glm, yds[idx], glm_secs


def certify(reuse, fresh, t_now, kappa, policy, k_fixed=40):
    cs = RecallCS(delta=DELTA, grid=401)
    reuse_x = [x for (x, _) in reuse]
    if policy == "supg":  # fixed-n: use only first k_fixed labels, static
        for x in (reuse_x + fresh)[:k_fixed]:
            cs.update(float(x))
        return cs.lower() >= R, min(k_fixed, len(reuse_x) + len(fresh))
    for (x, ti) in reuse:
        b = kappa * max(0.0, t_now - ti) if policy == "saved" else 0.0
        cs.update_shift(float(x), 1.0, b)
    used = 0
    def infl():
        return 0.0  # per-sample shift already applied per label for saved; pooled has none
    while cs.lower() < R and used < len(fresh):
        cs.update(float(fresh[used])); used += 1
    return cs.lower() >= R, used


def run(eps, glm_pos_scores, tau, N=20, n_seeds=30, B=25):
    """Session over a drifting snapshot of ORACLE-positive docs; x = 1{proxy>=tau}.
    tau chosen so base recall (easy positives) is high; drift shifts mixture easy->hard."""
    med = np.median(glm_pos_scores)
    easy = glm_pos_scores[glm_pos_scores >= med]
    hard = glm_pos_scores[glm_pos_scores < med]
    kappa = eps * PHI_MAX
    methods = ["supg", "pooled", "saved"]
    agg = {m: {"true": 0, "cert": 0, "bad_session": 0} for m in methods}
    for seed in range(n_seeds):
        rng = np.random.default_rng(1000 + seed)
        pool = []
        sess_false = {m: False for m in methods}
        for q in range(N):
            t = q / (N - 1)
            w_hard = min(1.0, eps * t)  # eps-driven drift: eps=0 -> no drift
            # true snapshot recall at threshold tau under this mixture (closed form on the pool)
            r_easy = float(np.mean(easy >= tau)) if len(easy) else 0.0
            r_hard = float(np.mean(hard >= tau)) if len(hard) else 0.0
            r_now = (1 - w_hard) * r_easy + w_hard * r_hard
            good = r_now >= R - 1e-9
            # draw B fresh oracle-positive samples from the snapshot mixture; x=1{proxy>=tau}
            nh = rng.binomial(B, w_hard)
            draws = np.concatenate([rng.choice(hard, nh) if len(hard) else np.array([]),
                                    rng.choice(easy, B - nh) if len(easy) else np.array([])])
            fresh = (draws >= tau).astype(float)
            for m in methods:
                cert, _ = certify([(x, tt) for (x, tt) in pool], list(fresh), t, kappa, m)
                if cert:
                    agg[m]["cert"] += 1
                    if good: agg[m]["true"] += 1
                    else: sess_false[m] = True
            for x in fresh:
                pool.append((float(x), t))
        for m in methods:
            if sess_false[m]: agg[m]["bad_session"] += 1
    tot = N * n_seeds
    return {m: {"trusted_yield": agg[m]["true"]/tot,
                "session_false_rate": agg[m]["bad_session"]/n_seeds} for m in methods}


def main():
    n_pool = int(os.environ.get("E2E_POOL", "500"))
    n_seeds = int(os.environ.get("E2E_SEEDS", "30"))
    s, glm, yds, glm_secs = build_pool(n_pool)
    npos_glm, npos_ds = int(glm.sum()), int(yds.sum())
    # oracle quality vs dataset truth (honest biased-oracle note)
    tp = int(((glm == 1) & (yds == 1)).sum())
    glm_recall = tp / max(1, npos_ds); glm_prec = tp / max(1, npos_glm)
    glm_pos_scores = s[glm == 1]
    tau = float(np.percentile(glm_pos_scores, 15))  # base recall ~high, drops under drift
    base_recall = float(np.mean(glm_pos_scores >= tau))
    print(f"\nRQ2 end-to-end sem_filter with REAL GLM-5.2 oracle (pool={n_pool}, seeds={n_seeds})")
    print(f"GLM oracle: {npos_glm} positives; vs dataset truth recall={glm_recall:.2f} "
          f"prec={glm_prec:.2f}; {n_pool} unique GLM labels in {glm_secs:.0f}s (cached after).")
    print(f"tau={tau:.3f}, base (no-drift) recall={base_recall:.2f}, target R={R}")
    print(f"\n eps | {'SUPG fixed-n':>22} | {'stream-ReDD':>22} | {'SAVED':>22}")
    print("-"*80)
    from saved.glm_oracle import _MODEL, _ENDPOINT
    out = {"n_pool": n_pool, "n_seeds": n_seeds, "glm_positives": npos_glm, "tau": tau,
           "glm_model": _MODEL, "glm_endpoint": _ENDPOINT,
           "base_recall": base_recall, "glm_vs_truth_recall": glm_recall,
           "glm_vs_truth_prec": glm_prec, "cells": []}
    for eps in [0.0, 0.2, 0.4, 0.6, 0.8]:
        r = run(eps, glm_pos_scores, tau, n_seeds=n_seeds)
        def cell(m):
            v = "V" if r[m]["session_false_rate"] <= DELTA else "INVALID"
            return f"yld{r[m]['trusted_yield']:.2f} sFC{r[m]['session_false_rate']:.2f}[{v}]".rjust(22)
        print(f"{eps:>4} | {cell('supg')} | {cell('pooled')} | {cell('saved')}")
        out["cells"].append({"eps": eps, **r})
    p = os.path.join(os.path.dirname(__file__), "..", "results", "e2e_glm_filter.json")
    json.dump(out, open(p, "w"), indent=1)
    make_table(out)
    print(f"\nsaved -> {os.path.abspath(p)}")


def make_table(out):
    """Self-generate the paper table from results (so make_figures cannot overwrite it)."""
    def cell(m):
        y, s = m["trusted_yield"], m["session_false_rate"]
        if s > DELTA + 1e-9:
            return f"{y:.2f} / \\textbf{{{s:.2f}}}$^\\dagger$"
        return f"{y:.2f} / {s:.2f}"
    lines = ["% AUTO-GENERATED by experiments/e2e_glm_filter.py (real GLM-5.2 oracle, cached).",
             "% trusted yield / SESSION-level false-cert; $\\dagger$ = session-invalid ($>\\delta$).",
             "\\begin{tabular}{cccc}", "\\toprule",
             "$\\varepsilon$ & SUPG fixed-$n$ & stream-ReDD & \\textsc{Saved} \\\\",
             "\\midrule"]
    for c in out["cells"]:
        lines.append(f"{c['eps']:.1f} & {cell(c['supg'])} & {cell(c['pooled'])} & {cell(c['saved'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    tp = os.path.join(os.path.dirname(__file__), "..", "figures", "table_e2e.tex")
    open(tp, "w").write("\n".join(lines) + "\n")
    print(f"table -> {os.path.abspath(tp)}")


if __name__ == "__main__":
    main()
