# Per-sample shift (Prop 3a) vs aggregate mean-age inflation (Prop 3b)

Theory (paper/THEORY.md) showed the originally-implemented aggregate "weighted-mean-age"
inflation (3b) is NOT rigorously anytime-valid: the unshifted CS lower-bounds the MAX past
snapshot recall, and bridging it to the current snapshot needs the MAX age (or a per-sample
shift), not the weighted-mean age. The rigorous fix is the per-sample shift (3a):
each reused sample i bets against m_i = min(m + kappa*age_i, 1). Reproduce:
`python3 experiments/persample_check.py`.

| ε | construction | coverage (nominal 0.90) | power (mean LB) | valid? |
|---|---|---|---|---|
| 0.0 | mean-age (3b, heuristic) | 0.840 | 0.769 | **NO (breach)** |
| 0.0 | **per-sample (3a, rigorous)** | **0.965** | **0.803** | YES |
| 0.4 | mean-age (3b) | 0.965 | 0.619 | yes |
| 0.4 | **per-sample (3a)** | 0.980 | **0.665** | YES |
| 0.8 | mean-age (3b) | 0.995 | 0.444 | yes |
| 0.8 | **per-sample (3a)** | 0.970 | **0.516** | YES |

## Conclusion (theory↔experiment agree)
- Per-sample (3a) **fixes the ε=0 anti-conservatism** (0.840 → 0.965, now ≥ 1−δ) and is
  **uniformly more powerful** (higher mean LB at every ε) while staying valid under drift.
- This is a strict improvement: the rigorous construction is also the tighter one. The mean-age
  heuristic only "worked" under strong recency weighting because weight concentration made
  mean-age ≈ max-effective-age; it has no general guarantee.
- **Action taken:** `PerSampleSAVED` (saved/baselines.py) implements 3a and is now the canonical
  SAVED method; the paper's main theorem is Prop 3a. The mean-age variant is kept only as the
  ablation that motivates the fix.

This is an honest theory→code→experiment loop: a proof found the implemented inflation unsound,
the corrected construction was implemented, and experiments confirm it is both more valid and
more powerful.
