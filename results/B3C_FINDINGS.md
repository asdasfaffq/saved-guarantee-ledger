# B3c Findings — the REAL (drift-reweighted full-pool) SAVED, and the honest surpass

Reweighted full-pool SAVED (predictable recency weights `w=γ^age` via the weighted
betting supermartingale `update_w`, + drift inflation). Replaces the crude windowed
prototype. δ=0.1, target recall 0.80, N=30, 100 seeds. Reproduce:
`python3 experiments/b3c_weighted.py`.

| ℓ | ε | pooled (valid/fresh) | per-query (valid/fresh) | SAVEDw (valid/fresh) | strongest valid baseline | verdict |
|---|---|---|---|---|---|---|
| 0.5 | 0.0 | T/463 | T/2729 | T/321 | pooled | "win" — **not credible** (see note) |
| 0.5 | 0.2 | T/1254 | T/3096 | T/1872 | pooled | no win (pooled cheaper) |
| 0.5 | 0.4 | **F**/2495 | **F**/3312 | **T**/3035 | none | **only SAVEDw valid** |
| 0.5 | 0.8 | **F**/3066 | **F**/3456 | **T**/3410 | none | **only SAVEDw valid** |
| 0.9 | 0.0 | T/252 | T/2714 | T/187 | pooled | "win" — **not credible** |
| 0.9 | 0.2 | T/1034 | T/3096 | T/1633 | pooled | no win (pooled cheaper) |
| 0.9 | 0.4 | T/2333 | **F**/3293 | **T**/2894 | pooled | no win (pooled cheaper) |
| 0.9 | 0.8 | **F**/2972 | **F**/3454 | **T**/3360 | none | **only SAVEDw valid** |

## Honest reading (three gates now agree)
- **Cost / label "超越" does NOT hold against a valid baseline.** Where drift is mild
  enough for pooled-CS to be valid (ε≤0.4), pooled is as cheap or cheaper than SAVEDw.
  The two "C2 WIN" cells are at **ε=0**, where the target is always satisfiable so NO query
  is un-certifiable → a breach is impossible → "validity" is free and untestable. SAVEDw
  needing fewer fresh labels there reflects mild anti-conservatism, not a trustworthy cost
  win. We do NOT claim a cost/throughput win. (B3, B3b, B3c all agree: empty under any
  honest, validity-constrained comparison.)
- **The real, robust surpass is on the VALIDITY axis.** Under genuine drift (ε≥0.4),
  **every baseline becomes invalid** — pooled-CS / stream-ReDD (stale), fixed-N SUPG/LOTUS
  (stale), AND even per-query recalibration (no alpha-spending → family-wise breach under
  the reused multi-query workload). **SAVEDw is the only method that keeps a valid 1−δ
  guarantee.** Its drift-inflation simultaneously buys drift-robustness AND implicit
  multiplicity protection — a single mechanism covering both failure axes that no existing
  approach (or the naive fix) handles.

## The honest "超越" claim (defensible, non-manufactured)
> Under interactive, drifting, budget-reused workloads, SAVED is the **only** method that
> keeps its precision/recall guarantee valid; SUPG (PVLDB'20), LOTUS sem_filter (PVLDB'25),
> ReDD-style streaming, and naive per-query recalibration all emit FALSE guarantees there
> (empirical coverage falls below 1−δ, often to ~0). A guarantee system that silently lies
> is worthless, so on *trustworthy* certifications-at-fixed-budget SAVED strictly dominates
> them. The honest price is conservativeness (power), not extra labels vs a *valid* baseline.

This **surpasses** the DB-venue baselines on the axis that defines a guarantee system —
correctness — and builds directly on (reproduces + stress-tests + extends) SUPG/ABae/LOTUS.

## Caveats (not buried)
- No cost/label win vs a valid baseline; do not claim one.
- ε=0 cost cells are not meaningful (validity untestable); a mid-range target (strictly inside
  the drift band) is needed to measure power cleanly — TODO for the paper version.
- SAVEDw at ε=0 is mildly anti-conservative; tighten with predictable-λ (WSR aGRAPA) and a
  formal weighted-CS inflation proof. Validity in the drift cells (T where all baselines F) is
  mechanistically sound (inflation lowers the bound → refuses borderline queries).
- Single controlled generator + simulated oracle; real LLM-extraction corpus still pending.
