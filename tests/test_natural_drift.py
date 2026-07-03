"""Natural-drift regression (real arXiv corpus). Skips if the corpus file is absent
(fetch it with `python3 data/fetch_arxiv.py`). Asserts the honest coverage story:
under naturally time-ordered drift SAVED holds session-level coverage everywhere on a
(tau, R) grid, while the naive baselines breach it exactly where drift crosses R.
Fast subset (fewer seeds) of experiments/natural_drift.py.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "arxiv_timed.json")


@pytest.mark.skipif(not os.path.exists(DATA), reason="arxiv_timed.json not fetched")
def test_saved_valid_everywhere_baselines_breach_on_crossing():
    import experiments.natural_drift as nd

    rows = nd.build()
    pos_early = np.array([r["score"] for r in rows if r["label"] == 1 and r["date"][:4] < nd.SPLIT])
    saved_worst = 0.0
    baseline_breached = False
    for pct in (15, 20):
        tau = float(np.percentile(pos_early, pct))
        for Rv in (0.66, 0.68, 0.70):
            nd.R = Rv
            res, *_ = nd.run(rows, tau, n_seeds=60)
            saved_worst = max(saved_worst, res["saved"]["sfc"])
            if res["fixedn"]["sfc"] > nd.DELTA or res["pooled"]["sfc"] > nd.DELTA:
                baseline_breached = True
    nd.R = 0.68
    # SAVED never breaches delta (small MC slack) anywhere on the grid.
    assert saved_worst <= nd.DELTA + 0.03, f"SAVED breached: worst sfc={saved_worst:.3f}"
    # There exists a regime where a naive baseline breaches — the drift-crossing regime.
    assert baseline_breached, "expected a baseline to breach coverage under natural drift"


@pytest.mark.skipif(not os.path.exists(DATA), reason="arxiv_timed.json not fetched")
def test_prospective_audit_never_false_certifies():
    """The deployable variant: kappa estimated online from PAST bins only must still hold
    session-level coverage (audit-or-abstain), never a silent false certificate."""
    import experiments.natural_drift as nd

    rows = nd.build()
    pos_early = np.array([r["score"] for r in rows if r["label"] == 1 and r["date"][:4] < nd.SPLIT])
    tau = float(np.percentile(pos_early, 15))
    nd.R = 0.68
    p = nd.run_prospective(rows, tau, n_seeds=80)
    nd.R = 0.68
    assert p["sfc"] <= nd.DELTA, f"prospective audit breached coverage: sfc={p['sfc']:.3f}"


@pytest.mark.skipif(not os.path.exists(DATA), reason="arxiv_timed.json not fetched")
def test_natural_recall_actually_drifts_down():
    """Sanity: the real corpus must exhibit a downward recall drift, else the test is vacuous."""
    import experiments.natural_drift as nd

    rows = nd.build()
    pos_early = np.array([r["score"] for r in rows if r["label"] == 1 and r["date"][:4] < nd.SPLIT])
    tau = float(np.percentile(pos_early, 15))
    order, binmap = nd.bins_by_halfyear(rows)
    stream = [b for b in order if b.split("H")[0] >= nd.SPLIT]
    traj = [float(np.mean(binmap[b] >= tau)) for b in stream]
    assert traj[0] - traj[-1] >= 0.05, f"insufficient natural drift: {traj[0]:.2f}->{traj[-1]:.2f}"
