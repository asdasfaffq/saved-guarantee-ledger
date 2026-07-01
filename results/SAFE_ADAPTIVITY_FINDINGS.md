# Safe-adaptivity / post-selection stress test (validates Assumption A3)

10-40 candidate predicates, analyst certifies recall>=0.85 at delta=0.1. Reproduce:
`python3 experiments/safe_adaptivity.py`.

| method | session false-cert rate | certification rate |
|---|---|---|
| unsafe (pick on the certification labels, certify with them) | 0.029 | 0.874 |
| firewall (pick by public signal, certify with hidden labels; A3) | 0.002 | 0.090 |
| sample-splitting (pick on half, certify on the other half) | 0.001 | 0.025 |

## Honest reading
- The post-selection effect is REAL and directional: unsafe reuse cherry-picks, certifying
  0.874 of sessions (vs 0.090 for the firewall) and inflating the false-certificate rate ~15x
  (0.029 vs 0.002). Much of unsafe's apparent yield is illegitimate (it certifies borderline
  predicates that merely looked good on the certification labels).
- BUT: SAVED's conservative anytime confidence sequence keeps even the unsafe false-cert rate
  BELOW delta (0.029 < 0.1) at these settings -- the conservatism partially protects. We do NOT
  claim unsafe reuse catastrophically breaks here.
- The label firewall (A3) and sample-splitting are the principled guarantees that hold
  regardless of the query family size or certification tightness; they also correctly refuse
  the illegitimate cherry-picked certifications (low, honest yield). This is why we adopt the
  firewall model rather than relying on the CS's conservatism.
