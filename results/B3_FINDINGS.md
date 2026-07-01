# B3 + eps-Robustness Findings — the make-or-break gate

Deterministic-truth simulation only. δ=0.1. Reproduce:
`python3 experiments/b3_amortization.py`, `python3 experiments/eps_robustness.py`.

---

## (1) B3 — label-amortization sweet spot (C2 go/no-go): **C2 FAILS — sweet spot is EMPTY**

Workload: N=30 queries reusing one shared label pool; certify recall≥0.80 at δ=0.1 by drawing
oracle labels until the lower bound clears the target (reused labels free => net-of-ESS).
Locality `ell` = prob a pooled label is reusable for the next query. TARGET is satisfiable for
early snapshots but, under drift, late-snapshot true recall < TARGET => those queries must be
REFUSED; a method that still certifies them commits a false certification (invalid). Cost is
compared ONLY where pooled-CS (stream-ReDD) is itself VALID. 120 seeds.

| ell | eps | pooled valid | pooled fresh | pooled cpc | SAVED valid | SAVED fresh | SAVED cpc | C2 verdict |
|----|----|----|----|----|----|----|----|----|
| 0.0 | 0.0 | True  | 2705.6 | 0.006 | True  | 2705.6 | 0.006 | no win (pooled ≤) |
| 0.0 | 0.4 | False | —      | —     | False | —      | —     | VOID (pooled invalid) |
| 0.0 | 0.8 | False | —      | —     | False | —      | —     | VOID (pooled invalid) |
| 0.5 | 0.0 | True  | 566.3  | 0.048 | True  | 1050.4 | 0.024 | no win (pooled ≤) |
| 0.5 | 0.4 | False | —      | —     | False | —      | —     | VOID (pooled invalid) |
| 0.5 | 0.8 | False | —      | —     | True  | 3408.1 | 0.001 | VOID (pooled invalid) |
| 0.9 | 0.0 | True  | 253.7  | 0.114 | True  | 325.4  | 0.088 | no win (pooled ≤) |
| 0.9 | 0.4 | False | —      | —     | False | —      | —     | VOID (pooled invalid) |
| 0.9 | 0.8 | False | —      | —     | False | —      | —     | VOID (pooled invalid) |

**C2 sweet spot non-empty? FALSE.**

### Why it is empty (the structural reason, not a bug)
- The locality knob works (ℓ: 0→0.9 cuts fresh labels 2705→253), so reuse is real and the
  harness is sound. But it helps **pooled-CS at least as much as SAVED**.
- Where pooled-CS is **valid** (no/low drift), it uses the *entire* accumulating pool => more
  effective samples => more power => it certifies with **fewer-or-equal** fresh labels than
  SAVED, whose windowing + drift-inflation deliberately *discards* data and *widens* the bound.
  So SAVED never beats a valid pooled-CS on cost.
- Where SAVED's machinery pays off (drift), pooled-CS is **invalid** => any "win" there is just
  C1 validity, not a label-cost win. Comparing cost against an invalid method is void.
- Validity and label-efficiency trade off in **opposite directions** across drift; there is no
  band where SAVED is both valid AND strictly cheaper than a valid baseline.

### Honest consequence (per FINAL_PROPOSAL honesty clause)
The "performance-strong = fewer oracle labels than a valid DB baseline" framing (P2/P3) is
**not supported** by the data. Recommended **pivot**: recenter the contribution on **P1 —
validity under drift + budget reuse**. The defensible, honest one-line claim becomes:

> *SAVED is the only method that retains its 1−δ precision/recall guarantee under corpus drift
> and reused-budget interactive querying; existing guaranteed systems (SUPG/LOTUS/ReDD,
> stream-ReDD) silently become invalid there. The price is power/conservativeness, not validity.*

This is a correctness/soundness contribution, not a cost-win contribution. **It does NOT satisfy
the user's stated "method must be performance-strong (beat DB baselines on cost/throughput)"
requirement** — that tension needs the user's verdict (see options below).

---

## (2) eps-misspecification robustness — load-bearing caveat (relatively benign)

SAVED's drift-inflation uses an assumed bound eps_bound = c·eps_true. 200 seeds.

| eps_true | c=0.5 | c=0.75 | c=1.0 | c=1.5 | c=2.0 |
|---|---|---|---|---|---|
| 0.4 coverage | 0.875 (mild breach) | 0.955 | 0.975 | 0.995 | 1.000 |
| 0.4 power(mean LB) | 0.644 | 0.632 | 0.621 | 0.597 | 0.574 |
| 0.8 coverage | 0.905 | 0.955 | 0.995 | 1.000 | 1.000 |
| 0.8 power(mean LB) | 0.492 | 0.469 | 0.446 | 0.400 | 0.353 |

- **Under-specification (c<1)**: only mild undercoverage (0.875 at ε=0.4; still OK at ε=0.8).
  Not catastrophic, but eps_bound should be a **conservative upper bound** to stay valid.
- **Over-specification (c>1)**: validity preserved, power cost ~7–12% of mean LB at c=2.
- Net: SAVED is **reasonably robust** to drift-bound misspecification; this is a manageable
  caveat, not a fragility that sinks the method. (Estimating eps_bound from the live corpus and
  propagating its error remains future work.)

---

## Bottom line for the user verdict
- **C1 (validity) — CONFIRMED & strong** (M2): SAVED holds ≥1−δ under drift+reuse; all baselines collapse.
- **C2 (cost/throughput win) — FAILS** (B3): empty sweet spot; SAVED never beats a *valid* baseline on labels.
- **eps-robustness — OK** (mild under-spec undercoverage; bounded over-spec power cost).

Options for the user: (A) **recenter on validity** (honest, strong soundness paper; drop the
"performance-strong cost" claim) — recommended; (B) keep hunting a genuine efficiency edge with a
NEW mechanism (e.g., drift-adaptive variance reduction that beats pooled even at low drift) —
speculative, no guarantee; (C) stop-loss. Per coordinator scope, STOPPING here for the verdict.
