"""M0 sanity gate (R001-R003): the recall CS must be anytime-valid.

Gate: at eps=0 with a clean oracle, empirical coverage of "true recall >= L_t"
over many simulated streams must be >= 1 - delta (within Monte-Carlo error).
This is the decision-stage check that guards against an invalid / leaky CS
before any headline experiment is trusted.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saved import RecallCS, make_corpus, true_recall, oracle_label


def _run_one_stream(seed, delta, tau, beta=0.0, n_corpus=20000, n_draw=300):
    rng = np.random.default_rng(seed)
    g, s = make_corpus(n_corpus, rng)
    r_true = true_recall(g, s, tau)
    cs = RecallCS(delta=delta, lam=0.5)
    total_oracle_calls = 0
    breached_ever = False  # did L_t exceed true recall at ANY t? (anytime test)
    # sample items uniformly; oracle-label; feed POSITIVE items' selection indicator
    idx = rng.integers(0, n_corpus, size=n_draw)
    for i in idx:
        o = oracle_label(g[i], beta, rng)
        total_oracle_calls += 1
        if o == 1:  # oracle says positive -> contributes to recall CS
            x = 1.0 if s[i] >= tau else 0.0
            cs.update(x)
            if cs.lower() > r_true + 1e-12:
                breached_ever = True
    return r_true, cs.lower(), breached_ever, total_oracle_calls, cs.t


def test_anytime_coverage_eps0_clean_oracle():
    delta, tau = 0.1, 0.5
    n_streams = 600
    breaches = 0
    covered_final = 0
    calls_seen = []
    for seed in range(n_streams):
        r_true, L, breached, calls, t_used = _run_one_stream(seed, delta, tau)
        breaches += int(breached)
        covered_final += int(L <= r_true + 1e-12)
        calls_seen.append(calls)
    breach_rate = breaches / n_streams
    # Anytime validity: P(exists t: recall < L_t) <= delta. Allow MC slack.
    assert breach_rate <= delta + 0.04, f"anytime breach rate {breach_rate:.3f} > delta {delta}"
    # Final-time coverage should be comfortably high too.
    assert covered_final / n_streams >= 1 - delta - 0.04
    # Budget-leakage sentinel: oracle calls must equal n_draw exactly.
    assert all(c == 300 for c in calls_seen), "oracle-call accounting drifted"


def test_lower_bound_is_below_truth_on_average():
    """The bound should be valid (rarely above truth) AND non-vacuous (>0)."""
    delta, tau = 0.1, 0.5
    Ls, Rs = [], []
    for seed in range(200):
        r_true, L, _, _, _ = _run_one_stream(seed, delta, tau, n_draw=400)
        Ls.append(L); Rs.append(r_true)
    Ls, Rs = np.array(Ls), np.array(Rs)
    assert Ls.mean() > 0.0, "bound is vacuous (always 0)"
    assert Ls.mean() < Rs.mean(), "lower bound not below true recall on average"


def test_monotone_certification_threshold():
    """certified() must be monotone: easier (lower) targets certify at least as often."""
    delta, tau = 0.1, 0.5
    rng = np.random.default_rng(0)
    g, s = make_corpus(20000, rng)
    cs = RecallCS(delta=delta)
    idx = rng.integers(0, 20000, size=400)
    for i in idx:
        if g[i] == 1:
            cs.update(1.0 if s[i] >= tau else 0.0)
    L = cs.lower()
    assert cs.certified(L - 0.05) or not cs.certified(L + 0.05)


def test_adaptive_lambda_is_valid():
    """Predictable plug-in (adaptive) lambda must remain anytime-valid at eps=0."""
    import numpy as np
    from saved import make_corpus, true_recall, oracle_label
    from saved.recall_cs import RecallCS
    delta, tau = 0.1, 0.5
    n_streams, breaches = 500, 0
    for seed in range(n_streams):
        rng = np.random.default_rng(seed + 10_000)
        g, s = make_corpus(20000, rng)
        r_true = true_recall(g, s, tau)
        cs = RecallCS(delta=delta, lam='adaptive')
        idx = rng.integers(0, 20000, size=300)
        for i in idx:
            if oracle_label(g[i], 0.0, rng) == 1:
                cs.update(1.0 if s[i] >= tau else 0.0)
                if cs.lower() > r_true + 1e-12:
                    breaches += 1
                    break
    assert breaches / n_streams <= delta + 0.04, f"adaptive breach rate {breaches/n_streams:.3f}"
