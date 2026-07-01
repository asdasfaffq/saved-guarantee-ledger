"""Drifting corpus stream (M2).

At stream step t (t_norm in [0,1]), the current snapshot has
    positives' proxy scores ~ N(mu_pos(t), 1),  mu_pos(t) = mu0 - eps*t_norm
    negatives' proxy scores ~ N(0, 1).
A larger `eps` makes positives progressively harder to detect, so the
snapshot recall at a fixed threshold tau DECREASES over the stream:
    true_recall(t) = P(N(mu_pos(t),1) >= tau) = Phi(mu_pos(t) - tau).
This is the canonical failure setting: methods calibrated on early
(high-recall) data over-state the *current* snapshot recall.
"""
from __future__ import annotations

import numpy as np
from math import erf, sqrt


def _phi(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def mu_pos(t_norm: float, mu0: float, eps: float) -> float:
    return mu0 - eps * t_norm


def true_recall(t_norm: float, tau: float, mu0: float, eps: float) -> float:
    return _phi(mu_pos(t_norm, mu0, eps) - tau)


def draw_item(t_norm: float, rng: np.random.Generator, mu0: float, eps: float,
              pos_rate: float = 0.3):
    """Draw one (g, s) from the current snapshot."""
    g = int(rng.random() < pos_rate)
    s = rng.normal(0.0, 1.0) + (mu_pos(t_norm, mu0, eps) if g == 1 else 0.0)
    return g, s
