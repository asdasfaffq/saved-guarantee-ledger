"""RQ4 (Phase B): YIELD-vs-BUDGET. At what oracle budget does SAVED become useful?

x = fresh oracle budget B per query in {25,50,100,200,400,800}
y1 = trusted certified-query rate (certified AND true)     [session-level honest yield]
y2 = SESSION-level false-certificate rate (>=1 false cert in a 25-query session)
methods = SAVED (per-sample), max-age, fresh-only, recent-window, stream-ReDD
panels  = eps in {0.0, 0.4, 0.8}

Honesty: metrics are SESSION-level (FWER). stream-ReDD is the invalid streaming
baseline (full pool, no drift shift). We expect SAVED's yield to be LOW at small
budget under drift and to climb with B, while stream-ReDD's yield is high but its
session false-cert rate stays > delta under drift (invalid). Auto-generates its own
figure so make_figures cannot overwrite it.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.recall_cs import RecallCS
from saved.baselines import PHI_MAX
from saved.drift import true_recall

GRID, TAU, MU0, R, DELTA, WIN = 201, 0.5, 1.6, 0.80, 0.10, 70
POOLCAP, CHUNK = 250, 10        # cap reuse pool; check lower() every CHUNK fresh (conservative)
METHODS = ["saved", "maxage", "fresh", "recent", "streamReDD"]
LABELS = {"saved": "SAVED (per-sample)", "maxage": "max-age", "fresh": "fresh-only",
          "recent": "recent-window", "streamReDD": "stream-ReDD"}


def certify(reuse, fresh, t_now, kappa, policy, B):
    cs = RecallCS(delta=DELTA, grid=GRID)
    if policy == "fresh":
        use = []
    elif policy == "recent":
        use = sorted(reuse, key=lambda z: z[1])[-WIN:]
    else:                       # saved, maxage, streamReDD: full pool
        use = reuse
    maxage = max((t_now - ti for (_, ti) in use), default=0.0)
    for (x, ti) in use:
        if policy == "saved":
            b = kappa * max(0.0, t_now - ti)
        elif policy == "maxage":
            b = kappa * maxage
        else:                   # recent, streamReDD, fresh: no shift
            b = 0.0
        cs.update_shift(x, 1.0, b)
    if cs.lower() >= R:
        return True, 0
    used = 0
    while used < B:             # consume fresh in chunks, checking lower() every CHUNK
        stop = min(used + CHUNK, B)
        for j in range(used, stop):
            cs.update(fresh[j])
        used = stop
        if cs.lower() >= R:
            break
    return cs.lower() >= R, used


def run(eps, B, n_seeds=40, N=25):
    kappa = eps * PHI_MAX
    agg = {m: {"true": 0, "cert": 0, "bad": 0} for m in METHODS}
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        pool = []
        sess_false = {m: False for m in METHODS}
        for q in range(N):
            t = q / (N - 1)
            r_now = true_recall(t, TAU, MU0, eps)
            good = r_now >= R - 1e-9
            fresh = (rng.random(B) < r_now).astype(float)
            for m in METHODS:
                cert, _ = certify(pool, fresh, t, kappa, m, B)
                if cert:
                    agg[m]["cert"] += 1
                    if good:
                        agg[m]["true"] += 1
                    else:
                        sess_false[m] = True
            for x in fresh:
                pool.append((float(x), t))
            if len(pool) > POOLCAP:
                pool = pool[-POOLCAP:]
        for m in METHODS:
            agg[m]["bad"] += int(sess_false[m])
    tot = N * n_seeds
    return {m: {"yield": agg[m]["true"] / tot,
                "sfc": agg[m]["bad"] / n_seeds} for m in METHODS}


def main():
    Bs = [25, 50, 100, 200, 400]
    epss = [0.0, 0.4, 0.8]
    out = {"delta": DELTA, "R": R, "Bs": Bs, "panels": {}}
    print(f"\nYIELD vs BUDGET (session-level; delta={DELTA}, target={R})")
    for eps in epss:
        print(f"\n eps={eps}   " + "  ".join(f"B={B}" for B in Bs))
        panel = {m: {"yield": [], "sfc": []} for m in METHODS}
        for B in Bs:
            r = run(eps, B)
            for m in METHODS:
                panel[m]["yield"].append(round(r[m]["yield"], 3))
                panel[m]["sfc"].append(round(r[m]["sfc"], 3))
        for m in METHODS:
            print(f"  {LABELS[m]:>18} yld " + " ".join(f"{v:.2f}" for v in panel[m]["yield"]))
            print(f"  {'':>18} sFC " + " ".join(f"{v:.2f}" for v in panel[m]["sfc"]))
        out["panels"][str(eps)] = panel
    rp = os.path.join(os.path.dirname(__file__), "..", "results", "yield_budget.json")
    json.dump(out, open(rp, "w"), indent=1)
    print(f"\nsaved -> {os.path.abspath(rp)}")
    make_fig(out)


def make_fig(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Bs = out["Bs"]; epss = ["0.0", "0.4", "0.8"]
    colors = {"saved": "C0", "maxage": "C1", "fresh": "C2", "recent": "C3", "streamReDD": "C4"}
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.2), sharex=True)
    for j, eps in enumerate(epss):
        panel = out["panels"][eps]
        for m in METHODS:
            axes[0, j].plot(Bs, panel[m]["yield"], "o-", color=colors[m], label=LABELS[m], ms=4)
            axes[1, j].plot(Bs, panel[m]["sfc"], "o-", color=colors[m], ms=4)
        axes[0, j].set_title(f"$\\varepsilon={eps}$"); axes[0, j].set_xscale("log")
        axes[1, j].set_xscale("log")
        axes[1, j].axhline(out["delta"], ls="--", color="gray", lw=0.8)
        axes[1, j].set_xlabel("oracle budget $B$/query")
    axes[0, 0].set_ylabel("trusted yield")
    axes[1, 0].set_ylabel("session false-cert")
    axes[0, 0].legend(fontsize=6, loc="upper left")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fp = os.path.join(os.path.dirname(__file__), "..", "figures", "fig_yield_budget.pdf")
    fig.savefig(fp, bbox_inches="tight"); print(f"figure -> {os.path.abspath(fp)}")


if __name__ == "__main__":
    main()
