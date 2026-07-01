"""RQ5 fairness fix (expert review item 7): compare against a GENUINELY session-valid
fresh-only baseline (fresh-only + alpha-spending), not just naive per-query-delta fresh-only.

Session-level (FWER) comparison at delta=0.1 over an N-query adaptive drifting reused stream.
Methods:
  fresh_naive   : fresh labels each query, per-query delta (NO alpha-spending) -> multiplicity failure.
  fresh_alpha   : fresh labels each query, delta_q = delta/N (alpha-spending) -> valid by design, low power.
  recent        : recent-window reuse, no drift shift, per-query delta -> staleness failure.
  maxage_alpha  : reuse + conservative max-age shift + alpha-spending -> valid.
  saved_alpha   : reuse + per-sample drift shift + alpha-spending -> valid (SAVED).
Report SESSION-level false-cert rate (<=delta => valid) and trusted yield. The honest point:
even against a valid fresh-only (alpha), SAVED's reuse gives higher trusted yield.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.recall_cs import RecallCS
from saved.baselines import PHI_MAX
from saved.drift import true_recall

TAU, MU0, R, DELTA, GRID = 0.5, 1.6, 0.80, 0.10, 401
WINDOW = 70


def certify(reuse, fresh, t_now, kappa, policy, dq):
    cs = RecallCS(delta=dq, grid=GRID)
    if policy.startswith("fresh"):
        used = 0
        while cs.lower() < R and used < len(fresh):
            cs.update(float(fresh[used])); used += 1
        return cs.lower() >= R, used
    ruse = reuse[-WINDOW:] if policy == "recent" else reuse
    tmin = min((ti for (_, ti) in ruse), default=t_now)
    for (x, ti) in ruse:
        if policy == "saved":
            b = kappa * max(0.0, t_now - ti)
        elif policy == "maxage_alpha":
            b = kappa * max(0.0, t_now - tmin)
        else:  # recent: no shift
            b = 0.0
        cs.update_shift(float(x), 1.0, b)
    used = 0
    while cs.lower() < R and used < len(fresh):
        cs.update(float(fresh[used])); used += 1
    return cs.lower() >= R, used


def run(eps, methods, N=25, n_seeds=200, B=30):
    kappa = eps * PHI_MAX
    dq = {m: (DELTA / N if m.endswith("alpha") else DELTA) for m in methods}
    agg = {m: {"true": 0, "cert": 0, "bad": 0} for m in methods}
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        pool = []
        sess_false = {m: False for m in methods}
        for q in range(N):
            t = q / (N - 1)
            r_now = true_recall(t, TAU, MU0, eps)
            good = r_now >= R - 1e-9
            fresh = (rng.random(B) < r_now).astype(float)
            for m in methods:
                cert, _ = certify(pool, fresh, t, kappa, m, dq[m])
                if cert:
                    agg[m]["cert"] += 1
                    if good: agg[m]["true"] += 1
                    else: sess_false[m] = True
            for x in fresh:
                pool.append((float(x), t))
        for m in methods:
            if sess_false[m]: agg[m]["bad"] += 1
    tot = N * n_seeds
    return {m: {"yield": agg[m]["true"]/tot, "sfc": agg[m]["bad"]/n_seeds} for m in methods}


def main():
    methods = ["fresh_naive", "fresh_alpha", "recent", "maxage_alpha", "saved_alpha"]
    labels = {"fresh_naive": "fresh-only, per-query $\\delta$",
              "fresh_alpha": "fresh-only $+\\alpha$-spend",
              "recent": "recent-window, no shift",
              "maxage_alpha": "max-age $+\\alpha$-spend",
              "saved_alpha": "\\textbf{SAVED (per-sample) $+\\alpha$}"}
    print(f"\nRQ5 valid-baselines w/ alpha-spending (delta={DELTA}, N=25, 200 sessions)")
    print(f"{'eps':>4} | " + " | ".join(f"{m:>16}" for m in methods)); print("-"*100)
    res = {}
    for eps in [0.0, 0.4, 0.8]:
        r = run(eps, methods); res[eps] = r
        cells = [f"y{r[m]['yield']:.2f} f{r[m]['sfc']:.2f}{'!' if r[m]['sfc']>DELTA else ''}".rjust(16) for m in methods]
        print(f"{eps:>4} | " + " | ".join(cells))
    # auto-gen table
    lines = ["% AUTO from experiments/valid_baselines_alpha.py (session-level, 200 sessions).",
             "\\begin{tabular}{lccc}", "\\toprule",
             "method & $\\epsilon{=}0.0$ & $\\epsilon{=}0.4$ & $\\epsilon{=}0.8$ \\\\", "\\midrule"]
    for m in methods:
        cells = []
        for e in [0.0, 0.4, 0.8]:
            y, f = res[e][m]["yield"], res[e][m]["sfc"]
            ftxt = f"\\textbf{{{f:.2f}}}$^\\dagger$" if f > DELTA + 1e-9 else f"{f:.2f}"
            cells.append(f"{y:.2f} / {ftxt}")
        lines.append(f"{labels[m]:<34}& " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\\\[2pt]{\\scriptsize yield / session false-cert; $^\\dagger$ ${>}\\delta{=}0.1$ (session-invalid). "
              "fresh-only becomes valid with $\\alpha$-spending but at low yield; SAVED's reuse gives the highest valid yield.}"]
    tp = os.path.join(os.path.dirname(__file__), "..", "figures", "table_valid_alpha.tex")
    open(tp, "w").write("\n".join(lines) + "\n")
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "results", "valid_baselines_alpha.json"), "w"), indent=1)
    print(f"\ntable -> {os.path.abspath(tp)}")


if __name__ == "__main__":
    main()
