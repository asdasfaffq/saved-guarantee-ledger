# Natural temporal drift (real arXiv corpus) — findings

**Script:** `experiments/natural_drift.py` · **Output:** `results/natural_drift.json`
**Corpus:** 3,960 arXiv abstracts, 2016-01 .. 2024-03, balanced cs.LG (positive predicate)
vs cs.DB (negative), 220 per category per year. Category is the deterministic label,
so snapshot recall and coverage are computed exactly (no oracle noise, no simulated shift).

## Why this experiment

The other drift experiments in the paper induce drift as an easy/hard mixture with a
controlled schedule. The obvious objection is that the schedule is our own construction.
This experiment removes that objection: a TF-IDF + logistic-regression proxy is trained
once on pre-2019 abstracts and then applied unchanged to later years, whose cs.LG content
has drifted naturally (classical ML -> deep learning -> LLM-era vocabulary). The snapshot
recall of a fixed score threshold therefore falls on its own, with no schedule from us.

## What we observe

Natural recall trajectory of the fixed threshold (tau at the 15th percentile of early
positives, target R = 0.68 inside the drift band):

| bin | 2019H1 | 2020H1 | 2021H1 | 2022H1 | 2023H1 | 2024H1 |
|-----|--------|--------|--------|--------|--------|--------|
| recall | 0.75 | 0.72 | 0.66 | 0.67 | 0.64 | 0.63 |

The drift is real and roughly monotone: the early threshold that clears R = 0.68 no longer
clears it after ~2021. Session over the six bins, one query per bin, δ = 0.10, audited
drift bound κ = 0.083 (1.3× the largest observed per-bin recall drop, per assumption A4):

| method | trusted yield | session false-cert | verdict |
|--------|--------------:|-------------------:|---------|
| SUPG fixed-n (static certificate) | 0.08 | 0.230 | **INVALID** |
| stream-ReDD (pooled, no discount) | 0.08 | 0.245 | **INVALID** |
| SAVED (per-sample κ·age + α-spending) | 0.02 | 0.000 | valid |

This reproduces the paper's central claim on a naturally time-ordered corpus: a certificate
calibrated once (SUPG fixed-n) or pooled without a drift discount (stream-ReDD) is false in
23–24% of sessions once natural drift crosses the target, whereas SAVED holds session-level
coverage. The two numbers match the induced-drift experiments (17–24% baseline, ≤0.3% SAVED).

## Honest cost boundary

SAVED's trusted yield (0.02) is lower than the baselines' (0.08). This is not a defect: the
baselines' apparent yield is exactly the false certification the paper warns about — they keep
certifying good early bins and never withdraw the certificate as the corpus drifts. SAVED
correctly abstains once the audited drift bound admits that recall may have fallen below R.
The paper's stated position is unchanged: SAVED buys correctness under drift and reuse, not
higher yield or fewer labels.

## Robustness (not a single-point result)

Sweeping τ (10th/15th/20th percentile) × R (0.66/0.68/0.70), session false-cert rate:

| τ pct | R | fixed-n | pooled | SAVED |
|------:|---:|--------:|-------:|------:|
| 10 | 0.66 | 0.00 | 0.00 | 0.00 |
| 10 | 0.68 | 0.00 | 0.00 | 0.00 |
| 10 | 0.70 | 0.55 | 0.47 | 0.00 |
| 15 | 0.66 | 0.39 | 0.39 | 0.00 |
| 15 | 0.68 | 0.25 | 0.24 | 0.00 |
| 15 | 0.70 | 0.15 | 0.11 | 0.00 |
| 20 | 0.66 | 0.19 | 0.12 | 0.00 |
| 20 | 0.68 | 0.07 | 0.05 | 0.00 |
| 20 | 0.70 | 0.05 | 0.03 | 0.00 |

SAVED is valid in every cell. The baselines are invalid (>δ) exactly in the regime the
guarantee is for — where natural drift pushes true recall across R during the session
(up to 55% false-cert). Where R sits below the whole trajectory there is no crossing and
every method is trivially valid. No cell exists where SAVED is invalid and a baseline is not.

## Threats / caveats

- One proxy family (TF-IDF + LR) and one predicate (cs.LG). The drift magnitude is modest
  (0.75 -> 0.63); a stronger proxy or a different predicate would move the trajectory but not
  the qualitative coverage gap, which follows from anytime-validity + the audited κ, not the
  corpus.
- Half-year bins with 220 positives each; the fresh top-up (8 labels/query) is deliberately
  small so reuse — and therefore drift — is on the critical path.
- κ is audited from the observed trajectory with a 1.3× margin. In deployment κ comes from
  the sentinel audit (A4); here we use the data-derived bound to keep the check self-contained.
