"""RQ (Phase B): PRECISION certificate small experiment (backs the symmetry claim).

Precision = E[1{g=1} | selected] is the symmetric conditional mean to recall, so the
identical betting CS applies with the roles of 'selected' and 'positive' swapped. We
verify this on the same drift/reuse setting: target precision >= 0.9, compare
static-calibration (SUPG fixed-n), stream-ReDD (pooled CS, no shift), and SAVED
(per-sample shift). Metric is SESSION-level false-certificate rate + trusted yield.

Drift model (closed form, so we can hand SAVED the exact precision-movement bound kappa):
  selected := s >= tau. positives ~ N(mu_pos(t),1), mu_pos(t)=mu0-eps*t; negatives ~ N(0,1).
  precision(t) = p*Phi(mu_pos(t)-tau) / (p*Phi(mu_pos(t)-tau) + (1-p)*Phi(-tau)).
Drift lowers mu_pos, so precision falls -> a stale precision certificate over-claims,
exactly like recall. Auto-generates its own table.
"""
import json, os, sys
from math import erf, sqrt, log
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from saved.recall_cs import RecallCS

DELTA, GRID = 0.10, 401
TAU, MU0, POS = 1.5, 2.6, 0.5
RP = 0.90                       # target precision
METHODS = ["fixedN", "streamReDD", "saved"]
LAB = {"fixedN": "SUPG fixed-$n$", "streamReDD": "stream-ReDD", "saved": "\\textsc{Saved}"}


def _phi(z):
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def precision(t, eps):
    mp = MU0 - eps * t
    num = POS * _phi(mp - TAU)
    den = num + (1 - POS) * _phi(-TAU)
    return num / den


def kappa_prec(eps, n=201):
    ts = np.linspace(0, 1, n)
    pr = np.array([precision(t, eps) for t in ts])
    return float(np.max(np.abs(np.diff(pr)) / (ts[1] - ts[0])))


def draw_selected(t, eps, rng, k):
    """Draw k oracle-labelled SELECTED items; return their g indicators (1=positive)."""
    out = []
    while len(out) < k:
        g = int(rng.random() < POS)
        s = rng.normal(0, 1) + ((MU0 - eps * t) if g == 1 else 0.0)
        if s >= TAU:                       # selected
            out.append(float(g))
    return out


def certify(reuse, fresh, t_now, kappa, policy, B):
    cs = RecallCS(delta=DELTA, grid=GRID)
    if policy == "streamReDD" or policy == "saved":
        for (x, ti) in reuse:
            b = kappa * max(0.0, t_now - ti) if policy == "saved" else 0.0
            cs.update_shift(x, 1.0, b)
    used = 0
    while cs.lower() < RP and used < B:
        cs.update(fresh[used]); used += 1
    return cs.lower() >= RP, used


def run(eps, n_seeds=200, N=25, B=40):
    kappa = kappa_prec(eps)
    agg = {m: {"true": 0, "cert": 0, "bad": 0} for m in METHODS}
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        pool = []
        fixedN = {"L": 0.0, "set": False, "buf": []}
        sess_false = {m: False for m in METHODS}
        for q in range(N):
            t = q / (N - 1)
            pr = precision(t, eps)
            good = pr >= RP - 1e-9
            fresh = draw_selected(t, eps, rng, B)
            # fixed-n: calibrate once on first query's B labels (Hoeffding), static after
            if not fixedN["set"]:
                fixedN["buf"] = fresh[:B]
                m = float(np.mean(fixedN["buf"]))
                fixedN["L"] = max(0.0, m - sqrt(log(1 / DELTA) / (2 * B)))
                fixedN["set"] = True
            certs = {"fixedN": fixedN["L"] >= RP}
            certs["streamReDD"] = certify(pool, fresh, t, kappa, "streamReDD", B)[0]
            certs["saved"] = certify(pool, fresh, t, kappa, "saved", B)[0]
            for m in METHODS:
                if certs[m]:
                    agg[m]["cert"] += 1
                    if good:
                        agg[m]["true"] += 1
                    else:
                        sess_false[m] = True
            for x in fresh:
                pool.append((float(x), t))
        for m in METHODS:
            agg[m]["bad"] += int(sess_false[m])
    tot = N * n_seeds
    return {m: {"yield": agg[m]["true"] / tot, "sfc": agg[m]["bad"] / n_seeds} for m in METHODS}


def main():
    epss = [0.0, 1.6]           # strong enough that precision drops below target under drift
    out = {"delta": DELTA, "target_precision": RP, "rows": []}
    print(f"\nPRECISION certificate (session-level; target precision={RP}, delta={DELTA})")
    print(f"{'eps':>4} | " + " | ".join(f"{LAB[m][:12]:>12}" for m in METHODS) + "   (yield / session-FC)")
    tex_rows = []
    for eps in epss:
        r = run(eps)
        cells = " | ".join(f"{r[m]['yield']:.2f} / {r[m]['sfc']:.2f}" for m in METHODS)
        print(f"{eps:>4} | " + cells)
        out["rows"].append({"eps": eps, **{m: r[m] for m in METHODS}})
        tex_cells = []
        for m in METHODS:
            fc = r[m]["sfc"]
            s = f"{r[m]['yield']:.2f} / {fc:.2f}"
            if fc > DELTA + 1e-9:
                s = f"{r[m]['yield']:.2f} / \\textbf{{{fc:.2f}}}$^\\dagger$"
            tex_cells.append(s)
        tex_rows.append(f"{eps:.1f} & " + " & ".join(tex_cells) + " \\\\")
    rp = os.path.join(os.path.dirname(__file__), "..", "results", "precision.json")
    json.dump(out, open(rp, "w"), indent=1)
    tex = ("% AUTO-GENERATED by experiments/exp_precision.py\n"
           "\\begin{tabular}{lccc}\n\\toprule\n"
           "$\\varepsilon$ & " + " & ".join(LAB[m] for m in METHODS) + " \\\\\n"
           "& \\multicolumn{3}{c}{trusted yield / session false-cert} \\\\\n\\midrule\n"
           + "\n".join(tex_rows) + "\n\\bottomrule\n\\end{tabular}\n")
    tp = os.path.join(os.path.dirname(__file__), "..", "figures", "table_precision.tex")
    open(tp, "w").write(tex)
    print(f"\nsaved -> {os.path.abspath(rp)}\ntable -> {os.path.abspath(tp)}\n(dagger = session-invalid, >delta)")


if __name__ == "__main__":
    main()
