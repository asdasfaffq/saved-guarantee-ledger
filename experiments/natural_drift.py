"""RQ5-natural: coverage under NATURALLY time-ordered drift (real arXiv corpus).

Unlike the induced easy/hard mixture, the drift here is REAL: a TF-IDF+LR proxy is
trained on early years (< SPLIT) and then applied to later years, whose cs.LG
abstracts have naturally drifted (classical ML -> deep learning -> LLMs), so the
snapshot recall of the fixed threshold genuinely falls over time. Truth is the arXiv
category label (deterministic), so coverage is exactly measurable.

We stream over time bins (a query per bin), reusing a label budget. A single-invocation
fixed-n certificate calibrated on the early bin, and a pooled stream-ReDD certificate,
are compared against SAVED, whose per-sample drift shift uses an AUDITED bound kappa
estimated from the data (A4), never the unknown true drift.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.recall_cs import RecallCS

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "arxiv_timed.json")
R, DELTA, GRID = 0.68, 0.10, 401   # R sits inside the natural drift band (early >R, late <R)
SPLIT = "2019"   # proxy trained + certificate calibrated on abstracts before this year
B_CAL, B_FRESH = 80, 8             # one-time calibration budget; per-query fresh top-up


def build():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    rows = json.load(open(DATA))
    rows.sort(key=lambda r: r["date"])
    early = [r for r in rows if r["date"][:4] < SPLIT]
    Xtr = [r["text"] for r in early]; ytr = np.array([r["label"] for r in early])
    vec = TfidfVectorizer(max_features=20000, min_df=2, stop_words="english")
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(vec.fit_transform(Xtr), ytr)
    for r in rows:
        pass
    s = clf.predict_proba(vec.transform([r["text"] for r in rows]))[:, 1]
    for r, sc in zip(rows, s):
        r["score"] = float(sc)
    return rows


def bins_by_halfyear(rows):
    """Group POSITIVE (cs.LG) items into ordered time bins; return list of score-arrays."""
    pos = [r for r in rows if r["label"] == 1]
    def key(r):
        y, m = int(r["date"][:4]), int(r["date"][5:7])
        return f"{y}H{1 if m <= 6 else 2}"
    order = sorted({key(r) for r in pos})
    return order, {b: np.array([r["score"] for r in pos if key(r) == b]) for b in order}


def cert_pooled(cal, reuse, ages, kappa, dq, shift):
    """Anytime-valid lower bound over calibration + reused labels.
    shift=False -> stream-ReDD (no drift discount); shift=True -> SAVED (per-sample kappa*age)."""
    cs = RecallCS(delta=dq, grid=GRID)
    for (x, age) in list(cal) + list(zip(reuse, ages)):
        b = kappa * age if shift else 0.0
        cs.update_shift(float(x), 1.0, b)
    return cs.lower() >= R


def run(rows, tau, n_seeds=200):
    order, binmap = bins_by_halfyear(rows)
    stream_bins = [b for b in order if b.split("H")[0] >= SPLIT]
    true_recall = {b: float(np.mean(binmap[b] >= tau)) for b in order}
    tr = [true_recall[b] for b in stream_bins]
    # audited conservative drift bound (A4): max observed per-bin recall drop x1.3 margin
    kappa = 1.3 * max((abs(tr[i] - tr[i-1]) for i in range(1, len(tr))), default=0.05)
    N = len(stream_bins)
    methods = ["fixedn", "pooled", "saved"]
    agg = {m: {"cert": 0, "true": 0, "bad": 0} for m in methods}
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        # one-time calibration at session start, drawn from the FIRST stream bin (early)
        cal0 = (rng.choice(binmap[stream_bins[0]], B_CAL) >= tau).astype(float)
        cs_fixed = RecallCS(delta=DELTA, grid=GRID)
        for x in cal0:
            cs_fixed.update(float(x))
        fixed_certified = cs_fixed.lower() >= R      # static SUPG verdict, frozen for the session
        cal_aged = [(x, 0) for x in cal0]             # same labels feed pooled/SAVED, aged by bin
        pool = []                                     # fresh top-ups: (indicator, bin_index)
        sess_false = {m: False for m in methods}
        for qi, b in enumerate(stream_bins):
            scores = binmap[b]
            if len(scores) < 5:
                continue
            good = true_recall[b] >= R - 1e-9
            fresh = (rng.choice(scores, B_FRESH) >= tau).astype(float)
            reuse = [x for (x, _) in pool] + list(fresh)
            ages = [qi - bi for (_, bi) in pool] + [0] * len(fresh)
            cal_ap = [(x, qi) for (x, _) in cal_aged]   # calibration labels aged to current bin
            verdict = {
                "fixedn": fixed_certified,               # frozen; never re-audited
                "pooled": cert_pooled(cal_ap, reuse, ages, 0.0, DELTA, shift=False),
                "saved":  cert_pooled(cal_ap, reuse, ages, kappa, DELTA / N, shift=True),
            }
            for m in methods:
                if verdict[m]:
                    agg[m]["cert"] += 1
                    if good: agg[m]["true"] += 1
                    else: sess_false[m] = True
            for x in fresh:
                pool.append((float(x), qi))
        for m in methods:
            if sess_false[m]: agg[m]["bad"] += 1
    tot = N * n_seeds
    return {m: {"yield": agg[m]["true"]/tot, "sfc": agg[m]["bad"]/n_seeds} for m in methods}, kappa, true_recall, stream_bins


def sweep(rows):
    """Robustness: the validity verdict must not hinge on one (tau, R)."""
    pos_early = np.array([r["score"] for r in rows if r["label"] == 1 and r["date"][:4] < SPLIT])
    global R
    print("\nrobustness sweep (session false-cert; [INVALID] if > delta=0.10):")
    print(f"{'tau_pct':>7} {'R':>5} | {'fixedn':>8} {'pooled':>8} {'saved':>8}")
    grid = []
    for pct in (10, 15, 20):
        tau = float(np.percentile(pos_early, pct))
        for Rv in (0.66, 0.68, 0.70):
            R = Rv
            res, *_ = run(rows, tau, n_seeds=120)
            grid.append({"tau_pct": pct, "R": Rv,
                         "sfc": {m: res[m]["sfc"] for m in ("fixedn", "pooled", "saved")}})
            def tag(m): return f"{res[m]['sfc']:.2f}{'!' if res[m]['sfc']>DELTA else ' '}"
            print(f"{pct:>7} {Rv:>5.2f} | {tag('fixedn'):>8} {tag('pooled'):>8} {tag('saved'):>8}")
    R = 0.68
    return grid


def main():
    rows = build()
    npos = sum(r["label"] for r in rows)
    print(f"arXiv corpus: {len(rows)} abstracts, {npos} cs.LG (positive), "
          f"{rows[0]['date']}..{rows[-1]['date']}; proxy trained on <{SPLIT}")
    pos_early = np.array([r["score"] for r in rows if r["label"] == 1 and r["date"][:4] < SPLIT])
    tau = float(np.percentile(pos_early, 15))  # early recall high; drifts down naturally
    res, kappa, tr, sb = run(rows, tau)
    print(f"tau={tau:.3f}, audited kappa={kappa:.3f}")
    print("natural recall trajectory (cs.LG, fixed tau) over time bins:")
    print("  " + "  ".join(f"{b}:{tr[b]:.2f}" for b in sb))
    print(f"\n{'method':>10} | {'trusted yield':>13} | {'session false-cert':>18}")
    print("-"*48)
    for m, name in [("fixedn","SUPG fixed-n"),("pooled","stream-ReDD"),("saved","SAVED")]:
        r = res[m]; inv = "INVALID" if r["sfc"] > DELTA else "valid"
        print(f"{name:>10} | {r['yield']:>13.2f} | {r['sfc']:.3f} [{inv}]")
    grid = sweep(rows)
    out = {"tau": tau, "R": R, "kappa": kappa, "n": len(rows), "npos": npos,
           "B_cal": B_CAL, "B_fresh": B_FRESH, "split": SPLIT,
           "recall_traj": {b: tr[b] for b in sb}, "results": res, "sweep": grid}
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "natural_drift.json"), "w"), indent=1)
    print("\nsaved -> results/natural_drift.json")


if __name__ == "__main__":
    main()
