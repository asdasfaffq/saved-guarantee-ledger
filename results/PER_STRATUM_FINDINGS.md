# Per-stratum ablation (resolves the open item — honest negative)

Heterogeneous strata: two proxy-score strata of positives drift at different rates
(fast 0.50 vs slow 0.05), recall-level gap swept. δ=0.1, 200 seeds. Reproduce:
`python3 experiments/per_stratum.py`.

| gap | pooled-avgκ (cov/pow) | pooled-maxκ (cov/pow) | stratified (cov/pow) |
|---|---|---|---|
| 0.0 | 0.560 / 0.56 **LIES** | 0.975 / 0.51 OK | 0.990 / 0.50 OK |
| 0.2 | 0.585 / 0.56 **LIES** | 0.970 / 0.51 OK | 1.000 / 0.50 OK |
| 0.4 | 0.580 / 0.56 **LIES** | 0.975 / 0.51 OK | 1.000 / 0.50 OK |

## Honest reading (this resolves "per-stratum: pending" with a clear NO)
- Pooling heterogeneous-drift strata with the **average** drift rate is INVALID (lies, 0.56):
  individual samples deviate from the overall recall by more than the average-rate shift bounds.
- The simple conservative fix — pool but inflate by the **max** rate — is valid (0.97).
- **Per-stratum stratification is also valid but does NOT improve power** over the max-rate
  fix (0.50 vs 0.51): the α-split (δ/K per stratum) offsets the per-stratum-κ benefit, and the
  stratified bound slightly over-covers.
- Conclusion: per-stratum stratification is **not warranted** in this regime; use the global
  max-rate construction. This is an honest negative that avoids over-engineering.
