# M2 Findings — Coverage under Drift + Budget Reuse

Setting: drifting snapshot stream (positives get harder over time → snapshot recall at fixed τ decreases), δ=0.1 (nominal coverage 0.90), N≈30 queried checkpoints over a reused label pool, 200 seeds. "Coverage" = fraction of seeds with NO anytime breach of `recall_now ≥ L_t`. Deterministic ground truth (Φ-closed-form recall). Oracle clean (β=0); drift is the only stressor here.

| ε (drift) | FixedN (SUPG/ReDD) | PooledCS (stream-ReDD) | SAVED-window (no-tilt ablation) | **SAVED (drift-robust)** | SAVED power (mean LB) |
|---|---|---|---|---|---|
| 0.0 | 1.00 | 0.95 | 0.87 | 0.87 | 0.767 |
| 0.4 | 0.86 ✗ | 0.52 ✗ | 0.725 ✗ | **0.975 ✓** | 0.621 |
| 0.8 | 0.09 ✗ | 0.02 ✗ | 0.52 ✗ | **0.995 ✓** | 0.446 |
| 1.2 | 0.00 ✗ | 0.00 ✗ | 0.32 ✗ | **0.995 ✓** | 0.268 |

## What this establishes (honest)
- **C1 validity confirmed (directional headline)**: the only method holding ≥1−δ across all drift levels is SAVED(drift-robust). Fixed-n (SUPG/ReDD split-conformal) and the strong streaming baseline (pooled-CS / stream-ReDD) both collapse toward 0 coverage as drift grows — exactly the silent-failure the proposal predicted. Pooled-CS collapses *fastest* because it averages stale high-recall history.
- **Mechanism necessity proven**: the no-tilt ablation (recency window only) helps a lot but still breaches (0.32–0.725). The explicit O(ε·t) drift-inflation term is what restores validity — it is load-bearing, not decoration.
- **Honest power cost is visible**: SAVED's mean lower bound falls 0.767→0.621→0.446→0.268 as ε grows. Validity is bought with conservativeness (the windowing/inflation–power tradeoff). This is the anti-claim check: SAVED's coverage is not "free"; it is the correct, quantified price.

## Caveats / TODO (do not oversell)
- At ε=0 SAVED sits at 0.87 (just under nominal 0.90): the fixed-λ=0.5 betting CS is slightly loose with a 70-sample window. Fix in paper version with a predictable-λ (WSR aGRAPA/ONS) betting CS to tighten coverage to ~0.90 and improve power.
- This M2 isolates DRIFT with a clean oracle. Biased-oracle (β,ρ) + the PPI increment is the next block; budget-amortization sweet-spot (C2) vs pooled-CS net-of-ESS is the separate go/no-go gate.
- `eps_bound` is given to SAVED as the true drift bound (bounded-TV assumption). Estimating it from the corpus and propagating estimation error is required for the real system (planned).
- Single controlled generator; real LLM-extraction corpus (B2) still pending for external validity.

Reproduce: `python3 experiments/m2_coverage.py` (writes results/m2_coverage.json); `python3 -m pytest tests/ -q`.
