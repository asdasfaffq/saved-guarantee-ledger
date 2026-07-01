# PPI power-recovery increment (resolves the last open ablation)

Recall = N/D as a ratio of unconditional means; PPI estimators N_ppi = mean_L(Zhat)+mean_A(Z-Zhat),
D_ppi likewise, with a one-sided delta-method lower bound (fixed-sample CLT). Oracle drops low-proxy
positives (bias strength drop_k, no false positives so naive can lie). Anchor n=120, large=4000,
400 seeds, delta=0.1, true recall=0.80. Reproduce: `python3 experiments/ppi_increment.py`.

| drop_k (bias) | naive cov/pow | anchor cov/pow | PPI cov/pow |
|---|---|---|---|
| 0.5 | 0.007 / 0.84 **LIES** | 0.870 / 0.72 OK | 0.865 / 0.74 OK |
| 1.5 | 0.000 / 0.90 **LIES** | 0.870 / 0.72 OK | 0.870 / 0.74 OK |
| 3.0 | 0.000 / 0.93 **LIES** | 0.870 / 0.72 OK | 0.865 / 0.74 OK |

## Honest reading
- The PPI increment **works**: it is valid (coverage ~0.87) and recovers **modest** power over
  anchor-only (mean lower bound 0.72 -> 0.74) while the naive oracle-as-truth bound lies
  catastrophically (coverage -> 0, it over-estimates because kept positives skew high-proxy).
- The power gain is **modest under strong proxy-correlated bias**: the oracle is only weakly
  informative, so the anchor rectifier variance stays high. PPI helps more when the oracle is
  a better predictor (standard PPI behavior).
- Caveats (honest): guarantee is **fixed-sample CLT**, not anytime; both anchor and PPI show mild
  finite-sample undercoverage (~0.87 vs nominal 0.90) at n=120 — a betting-CS / prediction-powered
  confidence-sequence version would tighten this and is the remaining future work.
