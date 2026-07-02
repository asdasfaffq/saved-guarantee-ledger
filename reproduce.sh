#!/usr/bin/env bash
# One-command reproduction of all SAVED results, figures, and the paper PDF.
# Deterministic-truth simulations + real-text (20newsgroups) experiments. No API keys needed.
set -e
cd "$(dirname "$0")"

echo "== tests (anytime-validity + drift coverage gates) =="
python3 -m pytest tests/ -q

echo "== core experiments =="
python3 experiments/m2_coverage.py            # validity under drift+reuse (Fig 2)
python3 experiments/fixed_budget.py           # trustworthy certification (Fig 1, Table 1)
python3 experiments/b3c_weighted.py           # honest empty cost sweet-spot (Fig 3)
python3 experiments/eps_robustness.py         # eps-misspecification (Fig 4)
python3 experiments/persample_check.py        # Prop 3a vs 3b (per-sample fix)
python3 experiments/ablation_clean.py           # clean drift-shift ablation (Fig 8)
python3 experiments/safe_adaptivity.py          # post-selection / label-firewall (A3)
python3 experiments/valid_baselines.py          # among-valid-methods yield comparison
python3 experiments/kappa_audit.py              # drift-bound sentinel audit (A4)          # among-valid-methods yield comparison          # post-selection / label-firewall (A3)
python3 experiments/biased_oracle.py          # biased-oracle ablation (Fig 6)
python3 experiments/b2_real.py                # external validity, TF-IDF proxy (Fig 5)
python3 experiments/biased_oracle.py       # biased-oracle ablation (Fig 6)
python3 experiments/per_stratum.py          # per-stratum ablation (honest negative)
python3 experiments/ppi_increment.py        # PPI power-recovery increment
python3 experiments/b2_multi.py             # multi-predicate external validity
python3 experiments/throughput.py           # systems overhead micro-benchmark
python3 experiments/b2_agnews.py || echo "(ag_news needs HF datasets; skipped)"   # cross-dataset
python3 experiments/b2_neural.py || echo "(b2_neural needs sentence-transformers + GPU/HF; skipped)"

echo "== Phase B (session-level; each self-generates its table/figure) =="
python3 experiments/e2e_glm_filter.py            # RQ2 end-to-end sem_filter, REAL GLM-5.2 oracle (cached, instant)
python3 experiments/natural_drift.py             # naturally time-ordered corpus (real arXiv 2016-2024; Table natural)
python3 experiments/valid_baselines_alpha.py     # RQ4 fair alpha-spent valid-methods table (paper Table)
# python3 experiments/exp_locality.py            # (superseded by valid_baselines_alpha; not cited in paper)
python3 experiments/exp_kappa_sentinel.py        # RQ6 kappa sentinel-audit main experiment
python3 experiments/exp_prevalence.py            # RQ6 prevalence / oracle-call ledger
python3 experiments/exp_safe_adapt_grid.py       # RQ6 safe-adaptivity grid
python3 experiments/exp_precision.py             # precision certificate (symmetry check)
# python3 experiments/exp_yield_budget.py        # RQ4 yield-vs-budget (heavy; deferred to extended version)

echo "== figures + tables =="
python3 figures/make_figures.py                  # NOTE: fixed_budget.py now uses 300 seeds

echo "== paper =="
cd paper
pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
echo "Built paper/main.pdf"
