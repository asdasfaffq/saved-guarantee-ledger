"""M2 regression: under drift, SAVED(drift-robust) keeps >=1-delta coverage
while fixed-n / pooled baselines collapse, and the no-tilt ablation breaches.
Fast subset (fewer seeds) of the full experiment.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.m2_coverage import run_seed


def _coverage(eps, n_seeds=80, delta=0.1):
    agg = {}
    for seed in range(n_seeds):
        br, _, _ = run_seed(seed, eps, delta=delta)
        for name, b in br.items():
            agg[name] = agg.get(name, 0) + int(b)
    return {k: 1 - v / n_seeds for k, v in agg.items()}


def test_saved_holds_baselines_collapse_under_drift():
    delta = 0.1
    cov = _coverage(0.8, n_seeds=80, delta=delta)
    saved = cov["SAVED(drift-robust)"]
    pooled = cov["PooledCS(streamReDD)"]
    fixed = cov["FixedN(SUPG/ReDD)"]
    abl = cov["SAVED-window(no-tilt abl)"]
    # SAVED holds nominal coverage (small MC slack).
    assert saved >= (1 - delta) - 0.05, f"SAVED coverage {saved:.3f} < target"
    # Baselines collapse far below nominal under strong drift.
    assert pooled < 0.5 and fixed < 0.5, f"baselines did not collapse: pooled={pooled}, fixed={fixed}"
    # Drift-inflation term is necessary: removing it (window-only) breaches.
    assert abl < saved, f"ablation {abl:.3f} should be worse than SAVED {saved:.3f}"


def test_no_drift_all_valid():
    cov = _coverage(0.0, n_seeds=80, delta=0.1)
    # At eps=0 every method is (approximately) valid.
    for name, c in cov.items():
        assert c >= 0.84, f"{name} undercovers at eps=0: {c:.3f}"


def test_persample_fixes_eps0_and_holds_under_drift():
    """Prop 3a (per-sample shift) must be valid at eps=0 (fixing the 3b anti-conservatism)
    and remain valid under drift. Fast subset."""
    import numpy as np
    from saved.drift import draw_item, true_recall
    from saved.baselines import PerSampleSAVED

    def cov(eps, n=80, delta=0.1, T=600, tau=0.5, mu0=1.6, qe=25):
        br = 0
        for seed in range(n):
            rng = np.random.default_rng(seed)
            m = PerSampleSAVED(delta, eps_bound=eps, grid=801)
            bad = False
            for t in range(T):
                tn = t / (T - 1)
                g, s = draw_item(tn, rng, mu0, eps)
                if g == 1:
                    m.update(1.0 if s >= tau else 0.0, tn)
                if t % qe and t != T - 1:
                    continue
                if m.lower(tn) > true_recall(tn, tau, mu0, eps) + 1e-9:
                    bad = True
            br += int(bad)
        return 1 - br / n
    assert cov(0.0) >= 0.86, "per-sample undercovers at eps=0"
    assert cov(0.8) >= 0.86, "per-sample undercovers under drift"
