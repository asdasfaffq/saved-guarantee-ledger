# SAVED: A Guarantee Ledger for Drift-Robust Semantic Filtering — Reproducibility Package

Code, data, and scripts to reproduce the experiments in the PVLDB submission
*"SAVED: A Guarantee Ledger for Drift-Robust Semantic Filtering in Interactive Data Systems."*

SAVED is a SUPG/LOTUS-compatible **guarantee-ledger operator**: it keeps a recall
certificate for an LLM semantic filter **session-valid** when a small oracle/human
label budget is reused across an adaptive, drifting query stream. Its contribution
is *correctness* under drift and reuse, not fewer oracle labels; this repository
lets a reviewer verify every headline number.

## Quick start

```bash
pip install numpy scipy scikit-learn matplotlib   # core deps
python3 -m pytest tests/ -q                        # sanity gates
bash reproduce.sh                                  # regenerate all results/figures
```

No API key and no network are required for the core results: all confidence-sequence,
coverage, ablation, and cost experiments run on deterministic-truth or cached data.

## Layout

- `saved/` — the guarantee-ledger primitives: `recall_cs.py` (anytime-valid betting
  confidence sequence with the per-sample drift shift), `baselines.py` (reproduced
  SUPG fixed-*n*, pooled stream-ReDD, recent-window, max-age), `drift.py`,
  `glm_oracle.py` (optional real-LLM oracle; reads its key from an env var, never stored here).
- `experiments/` — one script per result (coverage-vs-drift, fixed-budget trustworthy
  certification, clean drift-shift ablation, valid-baseline comparison with α-spending,
  κ-audit, prevalence/cost accounting, safe-adaptivity, biased-oracle/PPI, precision,
  systems overhead, the end-to-end GLM-5.2 `sem_filter` integration check, and
  `natural_drift.py` — the naturally time-ordered corpus check on real arXiv abstracts,
  2016–2024, where drift arises from real chronology rather than an imposed schedule).
- `data/` — `fetch_arxiv.py` builds the naturally time-ordered corpus (cs.LG vs cs.DB
  abstracts, 220/category/year) and `arxiv_timed.json` is the exact cached snapshot used
  in the paper, so the numbers reproduce without re-hitting the arXiv API.
- `results/` — the JSON outputs and honest `*_FINDINGS.md` notes behind each table/figure.
- `tests/` — pytest gates (anytime coverage ≤ δ, non-vacuous bounds, exact oracle-call
  accounting / budget-leak sentinel).
- `THEORY.md` — statements and proofs (anytime-valid lower bound; per-sample drift
  shift; α-spending simultaneous coverage; why the aggregate mean-age inflation is not valid).

## Real-LLM oracle (optional)

The end-to-end integration check (`experiments/e2e_glm_filter.py`) uses a real GLM-5.2
oracle. To re-run it, set your own key in the environment (it is read from `GLM_API_KEY`
and is **never** committed to this repository):

```bash
export GLM_API_KEY=<your-key>
python3 experiments/e2e_glm_filter.py
```

Responses are cached locally (gitignored), so a re-run needs no further API calls.

## Honesty note

The paper reports an explicit **cost boundary**: SAVED does not reduce oracle-label
cost relative to a valid baseline. A budget-leakage sentinel in every run asserts the
oracle-call count equals the intended budget, so this negative result cannot be an
accounting artifact. The coverage result is also reproduced on a naturally time-ordered corpus
(`experiments/natural_drift.py`): a proxy trained once on pre-2019 arXiv abstracts drifts
on its own, the fixed-threshold recall falls 0.75→0.63 with no imposed schedule, and the
naive baselines issue false certificates in 23–25% of sessions while SAVED holds coverage.
Remaining future work (human-adjudicated gold, a full learned-proxy LOTUS cascade,
aggregate/join operators) is stated as open in the paper, not claimed here.
