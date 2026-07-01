# kappa-audit (validates Assumption A4): drift bound need not be known a priori

coverage / power (mean lower bound) / sentinel-labels, delta=0.1. Reproduce:
`python3 experiments/kappa_audit.py`.

| eps | exact (oracle) | sentinel (estimated) | under (0.5x) | over (2x) |
|---|---|---|---|---|
| 0.4 | 1.00 / 0.63 / 0 | 1.00 / 0.18 / 60 | 0.95 / 0.65 | 1.00 / 0.59 |
| 0.8 | 0.98 / 0.56 / 0 | 1.00 / 0.15 / 60 | **0.69 / 0.60 (UNDER)** | 1.00 / 0.48 |

## Honest reading
- **The danger is real:** UNDER-specifying kappa (0.5x) silently breaks coverage under strong
  drift (0.69 << 0.90 at eps=0.8) -- exactly why kappa must be a conservative bound, not a guess.
- **kappa need not be known a priori:** a sentinel-based estimate (draw ~60 sentinel positives,
  estimate recall movement, add a DKW-style margin) stays valid (coverage 1.00) WITHOUT the true
  kappa. The cost is (i) ~60 extra sentinel labels and (ii) low power (0.15-0.18): the DKW margin
  is conservative, so the audited bound over-inflates. This is the honest deployable option --
  safe without oracle knowledge, at a power price.
- **over-specifying** (2x) is valid with a moderate power loss (0.48-0.59), matching the
  eps-misspecification sweep.
- Fail-safe: when the sentinel margin is large relative to the observed movement, the audit is
  inconclusive and the ledger abstains / falls back to fresh-only rather than certify (A4) --
  so an unreliable drift audit yields abstention, never a silent false certificate.

Takeaway: A4 is deployable (sentinel audit + fail-safe), the under-specification danger is
demonstrated, and the honest cost is conservativeness/power plus a small sentinel budget.
