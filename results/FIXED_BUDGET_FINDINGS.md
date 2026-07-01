# Fixed-Budget Trustworthy-Certification — the headline validity-dominance figure

Setting: fixed oracle budget B=25 fresh labels/query, locality ℓ=0.7, target recall 0.80,
δ=0.1, N=25-query stream, 80 seeds, adaptive-λ (empirical-Bernstein) CS for all methods.
Reproduce: `python3 experiments/fixed_budget.py`.

"SESSION-lie%" = fraction of sessions in which a method issues **at least one false
certificate** (certifies recall≥0.80 when the true snapshot recall is below it). This is the
M2-consistent, session-level notion of an (un)trustworthy guarantee. A guarantee system is
TRUSTWORTHY iff SESSION-lie ≤ δ.

| ε (drift) | method | correct% | per-query FALSE% | **SESSION-lie%** | trustworthy? |
|---|---|---|---|---|---|
| 0.0 | pooled (stream-ReDD) | 47.2 | 0.0 | 0.0 | YES |
| 0.0 | **SAVEDw** | 48.2 | 0.0 | 0.0 | YES |
| 0.0 | fixedN (SUPG/LOTUS) | 0.0 | 0.0 | 0.0 | YES (vacuous) |
| 0.0 | per-query recal | 2.9 | 0.0 | 0.0 | YES |
| 0.4 | pooled (stream-ReDD) | 12.0 | 1.2 | **21.2** | **NO (lies)** |
| 0.4 | **SAVEDw** | 3.9 | 0.0 | **0.0** | **YES** |
| 0.4 | fixedN | 0.0 | 0.0 | 0.0 | YES (vacuous) |
| 0.4 | per-query recal | 1.1 | 0.1 | 1.2 | YES |
| 0.8 | pooled (stream-ReDD) | 4.6 | 1.4 | **18.8** | **NO (lies)** |
| 0.8 | **SAVEDw** | 1.2 | 0.0 | **0.0** | **YES** |
| 0.8 | fixedN | 0.0 | 0.0 | 0.0 | YES (vacuous) |
| 0.8 | per-query recal | 0.5 | 0.1 | 2.5 | YES |

## Honest reading
- **No drift (ε=0):** SAVEDw ≈ pooled (≈48% correct, 0 lies). SAVEDw marginally ahead.
- **Under drift (ε≥0.4):** pooled-CS / stream-ReDD becomes **UNTRUSTWORTHY** — it lies in
  **19–21% of sessions**. SAVEDw **never lies (0% sessions)** and still does useful work.
- fixedN (SUPG/LOTUS static calibration) is *vacuously* safe — it refuses everything (0%
  correct), so it is useless, not trustworthy in any useful sense. per-query recal lies in
  1–2.5% of sessions and certifies almost nothing.
- **SAVEDw is the only method that is BOTH useful (certifies) AND trustworthy (0% session
  lies) under drift + budget reuse.** This is the honest "超越": on trustworthy guarantees —
  the property that defines a guarantee system — SAVED dominates the DB baselines, which
  silently emit false guarantees under drift.

## Honest caveats (unchanged)
- This is NOT a label-cost win: where a baseline is *valid* (pooled at ε=0), it matches SAVED.
- correct% is low for all at B=25 because the budget is small and most queries are refused;
  the dominance is about WHO LIES, not raw throughput.
- Controlled simulation; real LLM-extraction corpus (B2) still pending.
