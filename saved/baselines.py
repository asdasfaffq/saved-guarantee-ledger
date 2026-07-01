"""Baselines + SAVED's drift-handling variant for the M2 coverage study.

All produce a recall LOWER bound at level delta for the CURRENT snapshot,
fed a stream of selection indicators of oracle-labelled positives.

- FixedNSplitConformal : SUPG/ReDD-style. Calibrate ONCE on the first k
  labelled positives (fixed-n Hoeffding lower bound), then report it
  statically. Invalid under drift (stale) and ignores multiplicity.
- PooledCS            : streaming-ReDD strong baseline. An anytime-valid
  betting CS over the ENTIRE accumulating pool (no drift reweighting). Valid
  under reuse, but estimates the time-AVERAGED recall, so it over-states a
  DECREASING current snapshot recall under drift.
- WindowedSAVED       : the SAVED minimal drift variant. Betting CS over a
  recency WINDOW (snapshot-pinned estimand). Tracks current recall; O(eps*W)
  bias is the windowing-vs-power tradeoff in the proposal.
"""
from __future__ import annotations

from collections import deque
from math import log, sqrt

import numpy as np

from .recall_cs import RecallCS


def _betting_lower(xs, delta: float, lam='adaptive', grid: int = 2001) -> float:
    """Anytime-valid betting lower bound on a [0,1] mean from samples xs."""
    if len(xs) == 0:
        return 0.0
    cs = RecallCS(delta=delta, lam=lam, grid=grid)
    for x in xs:
        cs.update(x)
    return cs.lower()


class FixedNSplitConformal:
    """Calibrate once on first k positives; static thereafter (fixed-n)."""

    def __init__(self, delta: float, k: int = 60):
        self.delta = delta
        self.k = k
        self._buf = []
        self._L = 0.0
        self._frozen = False
        self.calls = 0

    def update(self, x: float):
        self.calls += 1
        if self._frozen:
            return
        self._buf.append(float(x))
        if len(self._buf) >= self.k:
            m = float(np.mean(self._buf))
            # fixed-n Hoeffding lower bound (NOT anytime-valid)
            self._L = max(0.0, m - sqrt(log(1.0 / self.delta) / (2.0 * self.k)))
            self._frozen = True

    def lower(self) -> float:
        return self._L


class PooledCS:
    """Streaming-ReDD: anytime-valid CS over the whole accumulating pool."""

    def __init__(self, delta: float, lam='adaptive', grid: int = 801):
        self.cs = RecallCS(delta=delta, lam=lam, grid=grid)
        self.calls = 0

    def update(self, x: float):
        self.calls += 1
        self.cs.update(x)

    def lower(self) -> float:
        return self.cs.lower()


class WindowedSAVED:
    """SAVED minimal drift variant: betting CS over a recency window."""

    def __init__(self, delta: float, window: int = 80, lam='adaptive', grid: int = 801):
        self.delta = delta
        self.window = window
        self.lam = lam
        self.grid = grid
        self.buf = deque(maxlen=window)
        self.calls = 0

    def update(self, x: float):
        self.calls += 1
        self.buf.append(float(x))

    def lower(self) -> float:
        return _betting_lower(list(self.buf), self.delta, self.lam, grid=self.grid)


# Max of the standard-normal pdf: |d recall / d t_norm| <= eps * PHI_MAX, so the
# snapshot recall moves at most eps*PHI_MAX per unit normalized time.
PHI_MAX = 0.3989422804014327


class SAVEDDriftRobust:
    """SAVED: windowed betting CS MINUS an explicit drift-inflation term.

    Snapshot-pinned estimand. Within the recency window the labelled positives
    span a normalized-time range [t_min, t_now]; under a per-unit-time drift
    bound `eps_bound` the snapshot recall of any window sample differs from the
    current snapshot recall by at most eps_bound*PHI_MAX*(t_now - t_min). We
    subtract that inflation from the windowed lower bound to restore validity
    (cap->validity, window size->power). This is the O(eps*t) inflation term.
    """

    def __init__(self, delta: float, eps_bound: float, window: int = 70,
                 lam='adaptive', grid: int = 801):
        self.delta = delta
        self.eps_bound = eps_bound
        self.window = window
        self.lam = lam
        self.grid = grid
        self.buf = deque(maxlen=window)   # (x, t_norm)
        self.calls = 0

    def update(self, x: float, t_norm: float):
        self.calls += 1
        self.buf.append((float(x), float(t_norm)))

    def lower(self, t_now: float) -> float:
        if not self.buf:
            return 0.0
        xs = [b[0] for b in self.buf]
        t_min = min(b[1] for b in self.buf)
        base = _betting_lower(xs, self.delta, self.lam, grid=self.grid)
        infl = self.eps_bound * PHI_MAX * max(0.0, t_now - t_min)
        return max(0.0, base - infl)


class PerSampleSAVED:
    """SAVED with the RIGOROUS per-sample drift shift (THEORY.md Prop 3a).
    Each reused sample i bets against m_i = min(m + kappa*age_i, 1); fresh samples
    (age 0) are unshifted. No aggregate inflation. Anytime-valid by construction."""

    def __init__(self, delta: float, eps_bound: float, lam='adaptive', grid: int = 801):
        self.delta = delta
        self.eps_bound = eps_bound      # kappa = eps_bound * PHI_MAX
        self.lam = lam
        self.grid = grid
        self.buf = []                   # (x, t_norm)
        self.calls = 0

    def update(self, x: float, t_norm: float):
        self.calls += 1
        self.buf.append((float(x), float(t_norm)))

    def lower(self, t_now: float) -> float:
        if not self.buf:
            return 0.0
        cs = RecallCS(delta=self.delta, lam=self.lam, grid=self.grid)
        kappa = self.eps_bound * PHI_MAX
        for (x, ti) in self.buf:
            cs.update_shift(x, 1.0, kappa * max(0.0, t_now - ti))
        return cs.lower()
