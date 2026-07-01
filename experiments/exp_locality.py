"""RQ4 (Phase B): WORKLOAD LOCALITY sweep. Where does reuse pay?

x = workload locality ell in {0,0.3,0.5,0.7,0.9} (prob. a past label is eligible/applicable
    to the current query; high ell = overlapping selected sets = reuse should pay)
y1 = trusted yield (session-level), y2 = fresh oracle labels per certified query
methods = SAVED (per-sample), max-age, fresh-only, recent-window
panels  = eps in {0.4, 0.8}

Honesty: at LOW locality the reuse pool is mostly irrelevant, so SAVED should collapse
toward fresh-only (or abstain); at HIGH locality reuse should let SAVED certify at a lower
fresh-label cost. We report labels-per-certified-query so a "yield" gain that just spends
more labels is visible. Auto-generates its own figure.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.recall_cs import RecallCS
from saved.baselines import PHI_MAX
from saved.drift import true_recall

GRID, TAU, MU0, R, DELTA, WIN, B = 401, 0.5, 1.6, 0.80, 0.10, 70, 60
METHODS = ["saved", "maxage", "fresh", "recent"]
LABELS = {"saved": "SAVED (per-sample)", "maxage": "max-age",
          "fresh": "fresh-only", "recent": "recent-window"}


def certify(reuse, fresh, t_now, kappa, policy):
    cs = RecallCS(delta=DELTA, grid=GRID)
    if policy == "fresh":
        use = []
    elif policy == "recent":
        use = sorted(reuse, key=lambda z: z[1])[-WIN:]
    else:
        use = reuse
    maxage = max((t_now - ti for (_, ti) in use), default=0.0)
    for (x, ti) in use:
        if policy == "saved":
            b = kappa * max(0.0, t_now - ti)
        elif policy == "maxage":
            b = kappa * maxage
        else:
            b = 0.0
        cs.update_shift(x, 1.0, b)
    used = 0
    while cs.lower() < R and used < B:
        cs.update(fresh[used]); used += 1
    return cs.lower() >= R, used


def run(eps, ell, n_seeds=150, N=25):
    kappa = eps * PHI_MAX
    agg = {m: {"true": 0, "cert": 0, "bad": 0, "labels": 0} for m in METHODS}
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        pool = []
        sess_false = {m: False for m in METHODS}
        for q in range(N):
            t = q / (N - 1)
            r_now = true_recall(t, TAU, MU0, eps)
            good = r_now >= R - 1e-9
            fresh = (rng.random(B) < r_now).astype(float)
            # locality: each past label is eligible for this query w.p. ell
            applicable = [p for p in pool if rng.random() < ell]
            for m in METHODS:
                cert, used = certify(applicable, fresh, t, kappa, m)
                agg[m]["labels"] += used
                if cert:
                    agg[m]["cert"] += 1
                    agg[m]["true"] += int(good)
                    if not good:
                        sess_false[m] = True
            for x in fresh:
                pool.append((float(x), t))
        for m in METHODS:
            agg[m]["bad"] += int(sess_false[m])
    tot = N * n_seeds
    return {m: {"yield": agg[m]["true"] / tot,
                "sfc": agg[m]["bad"] / n_seeds,
                "labels_per_cert": (agg[m]["labels"] / agg[m]["cert"]) if agg[m]["cert"] else float("nan")}
            for m in METHODS}


def main():
    ells = [0.0, 0.3, 0.5, 0.7, 0.9]
    epss = [0.4, 0.8]
    out = {"delta": DELTA, "R": R, "B": B, "ells": ells, "panels": {}}
    print(f"\nLOCALITY sweep (session-level; delta={DELTA}, target={R}, B={B})")
    for eps in epss:
        print(f"\n eps={eps}   ell = " + "  ".join(str(e) for e in ells))
        panel = {m: {"yield": [], "sfc": [], "lpc": []} for m in METHODS}
        for ell in ells:
            r = run(eps, ell)
            for m in METHODS:
                panel[m]["yield"].append(round(r[m]["yield"], 3))
                panel[m]["sfc"].append(round(r[m]["sfc"], 3))
                panel[m]["lpc"].append(round(r[m]["labels_per_cert"], 1) if r[m]["labels_per_cert"] == r[m]["labels_per_cert"] else None)
        for m in METHODS:
            print(f"  {LABELS[m]:>18} yld " + " ".join(f"{v:.2f}" for v in panel[m]["yield"])
                  + " | lpc " + " ".join((f"{v:.0f}" if v is not None else "--") for v in panel[m]["lpc"]))
        out["panels"][str(eps)] = panel
    rp = os.path.join(os.path.dirname(__file__), "..", "results", "locality.json")
    json.dump(out, open(rp, "w"), indent=1)
    print(f"\nsaved -> {os.path.abspath(rp)}")
    make_fig(out)


def make_fig(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ells = out["ells"]; epss = ["0.4", "0.8"]
    colors = {"saved": "C0", "maxage": "C1", "fresh": "C2", "recent": "C3"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    for j, eps in enumerate(epss):
        panel = out["panels"][eps]
        for m in METHODS:
            axes[0, j].plot(ells, panel[m]["yield"], "o-", color=colors[m], label=LABELS[m], ms=4)
            lpc = [v if v is not None else np.nan for v in panel[m]["lpc"]]
            axes[1, j].plot(ells, lpc, "o-", color=colors[m], ms=4)
        axes[0, j].set_title(f"$\\varepsilon={eps}$")
        axes[1, j].set_xlabel("workload locality $\\ell$")
    axes[0, 0].set_ylabel("trusted yield")
    axes[1, 0].set_ylabel("labels / cert. query")
    axes[0, 0].legend(fontsize=6, loc="upper left")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fp = os.path.join(os.path.dirname(__file__), "..", "figures", "fig_locality.pdf")
    fig.savefig(fp, bbox_inches="tight"); print(f"figure -> {os.path.abspath(fp)}")


if __name__ == "__main__":
    main()
