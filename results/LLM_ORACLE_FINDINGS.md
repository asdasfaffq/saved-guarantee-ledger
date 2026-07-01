# A: Live-LLM oracle on real text (Qwen2.5-0.5B-Instruct)

A real local LLM (Qwen2.5-0.5B-Instruct, run on CPU) labels each real 20newsgroups document
("is it about medicine? yes/no" via next-token yes/no logits); truth = dataset label; proxy =
MiniLM embeddings + LR. This instantiates the biased-oracle ablation with REAL LLM errors.
Reproduce: `LD_LIBRARY_PATH=$CONDA/lib python3 experiments/llm_oracle.py` -> results/llm_oracle.json.

| quantity | value |
|---|---|
| LLM oracle accuracy (mixed set) | 0.463 (over-labels "yes" -> low precision; recall side fine) |
| LLM recall (keeps true positives) | 0.904 |
| corr(LLM-keep, proxy score \| positive) | **0.110 (mild)** |
| true recall / naive (LLM-kept) recall | 0.801 / 0.818 |
| coverage: naive (LLM-oracle as truth) | **0.880** (slightly < nominal 0.90) |
| coverage: anchor-corrected | **0.970** |

## Honest reading
- The framework runs with a real LLM oracle. For THIS small model the errors are only **mildly
  proxy-correlated (ρ≈0.11)**, so the naive bound degrades only mildly (0.88, just below nominal)
  rather than collapsing; the clean anchor restores 0.97.
- The dramatic lying (coverage 0.23/0.00) appears only at high proxy-correlation (the simulated
  ρ-sweep, Fig. biased-oracle). So the biased-oracle failure is REAL but its severity is
  **model-dependent**; the simulation bounds the worst case, the live LLM gives one real point
  on that axis. We do NOT overclaim that every real LLM lies dramatically.
