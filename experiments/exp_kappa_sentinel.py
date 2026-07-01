"""RQ6 (Phase B): kappa SENTINEL AUDIT as a MAIN experiment (promoted from footnote).

Can the drift bound SAVED needs be AUDITED in practice, rather than assumed?
x = sentinel positives m in {20,40,60,100,200}
report per source of kappa: empirical coverage, trusted yield (mean lower bound),
and inconclusive/fallback rate.
sources: true kappa (oracle) | under-spec (0.5x) | over-spec (2x)
         | sentinel-estimated kappa + DKW margin | fresh-only fallback

Sentinel audit: draw m positives at t and at a lagged snapshot t-D, estimate each
snapshot recall, set kappa_hat = (|r1-r2| + 2*margin)/D, margin = sqrt(ln(2/alpha)/2m).
FAIL-SAFE: if the margin is large relative to the observed movement (margin > |r1-r2|),
the audit is INCONCLUSIVE and the ledger falls back to fresh-only rather than certify --
so an unreliable drift estimate yields abstention/fallback, never a silent false cert.
Auto-generates its own figure.
"""
import json, os, sys
from math import log, sqrt
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.baselines import PerSampleSAVED, PHI_MAX
from saved.drift import draw_item, true_recall

DELTA, TAU, MU0, TOL = 0.10, 0.5, 1.6, 1e-9
SOURCES = ["exact", "under", "over", "sentinel", "fresh"]


def sentinel_kappa(rng, eps, t_now, m, D=0.2, alpha=0.1):
    def emp_recall(t):
        cnt = hit = 0
        while cnt < m:
            g, s = draw_item(t, rng, MU0, eps)
            if g == 1:
                hit += int(s >= TAU); cnt += 1
        return hit / m
    r1, r2 = emp_recall(t_now), emp_recall(max(0.0, t_now - D))
    margin = sqrt(log(2 / alpha) / (2 * m))
    inconclusive = margin > abs(r1 - r2)          # fail-safe trigger
    kap = (abs(r1 - r2) + 2 * margin) / max(D, 1e-6)
    return kap, inconclusive


def fresh_only_lower(rng, tn, eps, delta_q, k=60):
    """VALID fresh-only bound: draw k positive labels at the CURRENT snapshot only
    (no reuse across drift). Uses alpha-spending delta_q=delta/#queries so that the
    per-query independent bounds are SESSION-valid (FWER), matching the reuse methods'
    single anytime-valid CS. This is the fair valid, no-reuse baseline."""
    from saved.baselines import _betting_lower
    xs = []
    while len(xs) < k:
        g, s = draw_item(tn, rng, MU0, eps)
        if g == 1:
            xs.append(1.0 if s >= TAU else 0.0)
    return _betting_lower(xs, delta_q, grid=801)


def run(eps, source, m, n_seeds=120, T=600, qe=25):
    kap_true = eps * PHI_MAX
    bad = 0; lb = 0.0; nq = 0; inconclusive_sessions = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        if source == "exact":
            kap = kap_true
        elif source == "under":
            kap = 0.5 * kap_true
        elif source == "over":
            kap = 2.0 * kap_true
        elif source == "fresh":
            kap = None                     # valid fresh-only baseline (no reuse)
        else:                              # sentinel: conservative kappa_hat (margin = safety)
            kap, inc = sentinel_kappa(rng, eps, 0.5, m)
            inconclusive_sessions += int(inc)   # diagnostic only; does not change behavior
        use_reuse = kap is not None
        eps_bound = (kap / PHI_MAX) if kap is not None else 0.0
        sv = PerSampleSAVED(DELTA, eps_bound=eps_bound, grid=801) if use_reuse else None
        n_eval = len([t for t in range(T) if (t % qe == 0) or t == T - 1])
        delta_q = DELTA / n_eval                # alpha-spending for the fresh-only baseline
        sess_bad = False
        for t in range(T):
            tn = t / (T - 1)
            g, s = draw_item(tn, rng, MU0, eps)
            if use_reuse and g == 1:
                sv.update(1.0 if s >= TAU else 0.0, tn)
            if t % qe and t != T - 1:
                continue
            r = true_recall(tn, TAU, MU0, eps)
            L = sv.lower(tn) if use_reuse else fresh_only_lower(rng, tn, eps, delta_q)
            lb += L; nq += 1
            if L > r + TOL:
                sess_bad = True
        bad += int(sess_bad)
    return {"coverage": 1 - bad / n_seeds, "power": lb / nq,
            "inconclusive_rate": inconclusive_sessions / n_seeds}


def main():
    ms = [20, 40, 60, 100, 200]
    eps = 0.8                    # strong drift = the dangerous regime for kappa
    out = {"delta": DELTA, "eps": eps, "ms": ms, "curves": {}}
    print(f"\nkappa SENTINEL AUDIT (main RQ6; eps={eps}, delta={DELTA}, nominal {1-DELTA:.2f})")
    print("source     " + "  ".join(f"m={m}" for m in ms) + "   (coverage / power / inconclusive)")
    for src in SOURCES:
        cov, pw, ic = [], [], []
        for m in ms:
            r = run(eps, src, m)
            cov.append(round(r["coverage"], 3)); pw.append(round(r["power"], 3)); ic.append(round(r["inconclusive_rate"], 3))
        out["curves"][src] = {"coverage": cov, "power": pw, "inconclusive": ic}
        print(f"  {src:>8}  cov " + " ".join(f"{c:.2f}" for c in cov)
              + " | pow " + " ".join(f"{p:.2f}" for p in pw)
              + " | inc " + " ".join(f"{f:.2f}" for f in ic))
    rp = os.path.join(os.path.dirname(__file__), "..", "results", "kappa_sentinel.json")
    json.dump(out, open(rp, "w"), indent=1)
    print(f"\nsaved -> {os.path.abspath(rp)}")
    make_fig(out)


def make_fig(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ms = out["ms"]
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.1))
    col = {"exact": "C0", "under": "C3", "over": "C1", "sentinel": "C2", "fresh": "C4"}
    handles = []
    for src in SOURCES:
        c = out["curves"][src]
        h, = ax[0].plot(ms, c["coverage"], "o-", color=col[src], label=src, ms=4, lw=1.5)
        handles.append(h)
        ax[1].plot(ms, c["power"], "o-", color=col[src], ms=4, lw=1.5)
        ax[2].plot(ms, c["inconclusive"], "o-", color=col[src], ms=4, lw=1.5)
    hn = ax[0].axhline(1 - out["delta"], ls="--", color="gray", lw=0.9, label=r"nominal $1-\delta$")
    handles.append(hn)
    ax[0].set_ylabel("empirical coverage", fontsize=10); ax[0].set_title("coverage", fontsize=10)
    ax[0].set_ylim(0.55, 1.06)
    ax[1].set_ylabel("trusted power", fontsize=10); ax[1].set_title("power (mean lower bound)", fontsize=10)
    ax[2].set_ylabel("inconclusive rate", fontsize=10); ax[2].set_title("sentinel inconclusive", fontsize=10)
    for a in ax:
        a.set_xlabel("sentinel positives $m$", fontsize=10); a.grid(alpha=0.25); a.tick_params(labelsize=8)
    # one shared legend below all panels, in clear whitespace (no overlap)
    fig.legend(handles=handles, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=6, frameon=True)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fp = os.path.join(os.path.dirname(__file__), "..", "figures", "fig_kappa_sentinel.pdf")
    fig.savefig(fp, bbox_inches="tight", pad_inches=0.02); print(f"figure -> {os.path.abspath(fp)}")


if __name__ == "__main__":
    main()
