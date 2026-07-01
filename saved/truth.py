"""Deterministic-truth corpus generator with controlled knobs (M0).

Knobs (per FINAL_PROPOSAL / EXPERIMENT_PLAN):
    eps   : per-step drift strength (mixture-weight shift). M0 sanity uses eps=0.
    beta  : oracle bias magnitude (flip prob of oracle vs ground truth g).
    rho   : proxy-oracle error correlation (M0 sanity uses 0).
Ground truth g(x) is a hidden rule label; proxy score s(x) correlates with g;
oracle o(x) is a (possibly biased) labeller. Used ONLY to MEASURE coverage --
the method never sees g except via oracle/anchor draws.
"""
from __future__ import annotations

import numpy as np


def make_corpus(n: int, rng: np.random.Generator, pos_rate: float = 0.3,
                proxy_auc_gap: float = 1.2):
    """Return (g, s): hidden labels g in {0,1} and proxy scores s in R.

    Positives get scores ~ N(+proxy_auc_gap,1), negatives ~ N(0,1): a proxy
    that is informative but imperfect (the realistic SUPG/LOTUS regime).
    """
    g = (rng.random(n) < pos_rate).astype(int)
    s = rng.normal(0.0, 1.0, size=n) + proxy_auc_gap * g
    return g, s


def true_recall(g: np.ndarray, s: np.ndarray, tau: float) -> float:
    """recall(tau) = P(s >= tau | g=1) on the full corpus snapshot."""
    pos = g == 1
    if pos.sum() == 0:
        return 1.0
    return float((s[pos] >= tau).mean())


def oracle_label(g_i: int, beta: float, rng: np.random.Generator) -> int:
    """Oracle = ground truth flipped with prob beta (bias). beta=0 => clean."""
    if beta > 0 and rng.random() < beta:
        return 1 - int(g_i)
    return int(g_i)
