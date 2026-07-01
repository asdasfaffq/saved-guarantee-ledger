"""Anytime-valid recall confidence sequence (M0 minimal, global-weight).

Load-bearing insight (resolves the "unknown corpus denominator" objection):
    recall(tau) = P(selected | positive) = E[ 1{s(x) >= tau} | g(x)=1 ].
This is a CONDITIONAL Bernoulli mean over the positive subpopulation, so the
unknown corpus #positives cancels. A betting confidence sequence on the
selection indicators of oracle-labelled positives therefore yields an
ANYTIME-VALID lower bound on recall for a FIXED snapshot + threshold tau.

Betting CS (Waudby-Smith & Ramdas style, simplified fixed-lambda for M0):
    For testing H0: mu <= m, define the capital
        K_t^-(m) = prod_{i<=t} (1 + lambda * (X_i - m)),  X_i, m in [0,1], lambda in (0,1].
    Under mu <= m, E[1 + lambda(X_i - m)] = 1 + lambda(mu - m) <= 1, and each
    factor is >= 1 - lambda >= 0, so K_t^-(m) is a nonnegative supermartingale.
    By Ville's inequality P(exists t: K_t^-(m) >= 1/delta) <= delta.
    Lower confidence bound:  L_t = sup{ m : K_t^-(m) >= 1/delta }.
    => P(exists t: recall < L_t) <= delta  (anytime-valid).

This is the global-weight, no-PPI, no-stratification minimal version (the
de-risking lemma of the FINAL_PROPOSAL). Drift tilt / PPI increment / per-stratum
extensions plug into the same wealth process later.
"""
from __future__ import annotations

import numpy as np


class RecallCS:
    """Streaming anytime-valid lower confidence sequence on a [0,1] mean (=recall).

    Feed selection indicators X_i in {0,1} of oracle-labelled POSITIVE items.
    `lower()` returns the current anytime-valid recall lower bound.
    """

    def __init__(self, delta: float = 0.1, lam='adaptive', grid: int = 2001,
                 lam_cap: float = 0.6):
        assert 0 < delta < 1
        self.delta = delta
        self.adaptive = (lam == 'adaptive')
        if not self.adaptive:
            assert 0 < lam <= 1
            self.lam = lam
        self.lam_cap = lam_cap
        self.thresh = np.log(1.0 / delta)
        self.m_grid = np.linspace(0.0, 1.0, grid)
        # log-capital per candidate m, for numerical stability
        self.logK = np.zeros_like(self.m_grid)
        self.t = 0
        # running stats for the predictable plug-in (empirical-Bernstein) bet
        self._sx = 0.0
        self._sxx = 0.0

    def _lambda(self) -> float:
        """Predictable plug-in lambda (Waudby-Smith & Ramdas, empirical Bernstein).
        Uses ONLY data seen so far (x_{1..t-1}) => predictable => anytime-valid.
        Larger when variance is low / early; shrinks ~1/sqrt(t)."""
        if not self.adaptive:
            return self.lam
        n = self.t
        if n < 1:
            return min(self.lam_cap, 0.5)
        mu = self._sx / n
        var = max(self._sxx / n - mu * mu, 1e-3)
        lam = np.sqrt(2.0 * self.thresh / (var * (n + 1)))
        return float(min(self.lam_cap, max(1e-3, lam)))

    def update_shift(self, x: float, w: float, b: float) -> None:
        """Per-sample drift-shift bet (THEORY.md Prop 3a, the rigorous construction).
        Tests recall_now <= m by betting against the shifted target m_i = min(m+b,1),
        where b = kappa*(age) bounds |recall_{t_i} - recall_now|. Predictable w,b =>
        anytime-valid lower bound on recall_now (no aggregate inflation needed)."""
        x = float(x); w = float(w); b = float(b)
        lam = self._lambda()
        mi = np.minimum(self.m_grid + b, 1.0)
        factors = 1.0 + w * lam * (x - mi)
        factors = np.clip(factors, 1e-300, None)
        self.logK += np.log(factors)
        self.t += 1
        self._sx += x; self._sxx += x * x

    def update(self, x: float) -> None:
        """Process one bounded observation x in [0,1] (selection indicator)."""
        x = float(x)
        assert 0.0 - 1e-9 <= x <= 1.0 + 1e-9
        lam = self._lambda()
        factors = 1.0 + lam * (x - self.m_grid)  # >= 1 - lam >= 0
        factors = np.clip(factors, 1e-300, None)
        self.logK += np.log(factors)
        self.t += 1
        self._sx += x; self._sxx += x * x

    def update_w(self, x: float, w: float) -> None:
        """Weighted observation: bet scaled by a PREDICTABLE weight w in [0,1].
        factor = 1 + w*lam*(x-m) is a valid supermartingale under H0: mu<=m
        (E[.] = 1 + w*lam*(mu-m) <= 1; factor >= 1 - w*lam >= 0). Lets stale
        labels be down-weighted while preserving anytime-validity.
        """
        x = float(x); w = float(w)
        lam = self._lambda()
        factors = 1.0 + w * lam * (x - self.m_grid)
        factors = np.clip(factors, 1e-300, None)
        self.logK += np.log(factors)
        self.t += 1
        self._sx += x; self._sxx += x * x

    def lower(self) -> float:
        """Anytime-valid lower bound L_t = sup{ m : K_t^-(m) >= 1/delta }."""
        if self.t == 0:
            return 0.0
        rejected = self.logK >= self.thresh  # these m are rejected (too low)
        if not rejected.any():
            return 0.0
        # logK is non-increasing in m; rejection region is a low-m prefix.
        idx = np.where(rejected)[0]
        return float(self.m_grid[idx.max()])

    def certified(self, target_recall: float) -> bool:
        return self.lower() >= target_recall
